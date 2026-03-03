"""Standalone FFT plot GUI for Saleae — watches a shared JSON file and auto-refreshes.

Usage: python fft_plot_gui.py [path_to_fft_data.json]
Defaults to %TEMP%/saleae_fft_data.json.
"""

import sys
import os
import json
import tempfile
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("tkinter is required but not found.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy is required. Run pip install numpy")
    sys.exit(1)

try:
    from scipy.signal import find_peaks
except ImportError:
    print("scipy is required. Run pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    print("matplotlib is required. Run pip install matplotlib")
    sys.exit(1)

DEFAULT_DATA_FILE = os.path.join(tempfile.gettempdir(), 'saleae_fft_data.json')
POLL_INTERVAL_MS = 1000  # Check for file changes every second


class FFTPlotApp:
# Tkinter FFT magnitude plot

    def __init__(self, root: tk.Tk, data_path: str):
        self.root = root
        self.data_path = data_path
        self.fft_data = None
        self._last_mtime = 0.0

        self._show_peaks = False
        self._peak_artists = []
        self._freqs_db = None
        self._freqs = None
        self._refreshing_peaks = False
        self._max_data_freq = 1.0  # updated when data arrives

        self.root.title("FFT Viewer")
        self.root.geometry("960x640")
        self.root.minsize(640, 400)

        # Info bar
        self.info_bar = ttk.Frame(self.root, padding=6)
        self.info_bar.pack(fill=tk.X, side=tk.TOP)

        self.lbl_fundamental = ttk.Label(
            self.info_bar, text="Fundamental Frequency: —",
            font=("Segoe UI", 11, "bold"))
        self.lbl_fundamental.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_sample_rate = ttk.Label(self.info_bar, text="Sample Rate: —")
        self.lbl_sample_rate.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_samples = ttk.Label(self.info_bar, text="Samples: —")
        self.lbl_samples.pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(self.info_bar, text="", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT)

        self.btn_peaks = ttk.Button(
            self.info_bar, text="Peaks: OFF", command=self._toggle_peaks)
        self.btn_peaks.pack(side=tk.RIGHT, padx=(0, 6))

        self._peak_threshold = tk.DoubleVar(value=20.0)
        self.lbl_threshold = ttk.Label(self.info_bar, text="Threshold: 20 dB")
        self.lbl_threshold.pack(side=tk.RIGHT, padx=(0, 4))

        self.slider_threshold = ttk.Scale(
            self.info_bar, from_=1, to=60, orient=tk.HORIZONTAL,
            variable=self._peak_threshold, command=self._on_threshold_change,
            length=120)
        self.slider_threshold.pack(side=tk.RIGHT, padx=(0, 4))

        # Frequency cutoff bar
        self.freq_bar = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        self.freq_bar.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(self.freq_bar, text="Freq Cutoff:").pack(side=tk.LEFT)

        self._freq_cutoff_pct = tk.DoubleVar(value=100.0)
        self.slider_freq_cutoff = ttk.Scale(
            self.freq_bar, from_=1, to=100, orient=tk.HORIZONTAL,
            variable=self._freq_cutoff_pct, command=self._on_freq_cutoff_change,
            length=260)
        self.slider_freq_cutoff.pack(side=tk.LEFT, padx=(4, 4))

        self.lbl_freq_cutoff = ttk.Label(self.freq_bar, text="100 %")
        self.lbl_freq_cutoff.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_freq_cutoff_hz = ttk.Label(self.freq_bar, text="", foreground="gray")
        self.lbl_freq_cutoff_hz.pack(side=tk.LEFT)

        # Matplotlib figure
        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, side=tk.BOTTOM)

        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.ax.callbacks.connect('xlim_changed', lambda _ax: self._refresh_peak_markers())

        self._draw_waiting()
        self._check_for_update()

    def _on_scroll(self, event):
        """Zoom X axis (plain scroll) or Y axis (Ctrl+scroll), anchored at cursor."""
        if event.inaxes != self.ax:
            return

        scale_factor = 0.8 if event.button == 'up' else 1.25

        if event.key == 'control':
            # Ctrl+scroll - zoom vertical axis
            y_min, y_max = self.ax.get_ylim()
            y_cursor = event.ydata
            new_min = y_cursor - (y_cursor - y_min) * scale_factor
            new_max = y_cursor + (y_max - y_cursor) * scale_factor
            self.ax.set_ylim(new_min, new_max)
        else:
            # Plain scroll - zoom horizontal axis
            x_min, x_max = self.ax.get_xlim()
            x_cursor = event.xdata
            new_min = x_cursor - (x_cursor - x_min) * scale_factor
            new_max = x_cursor + (x_max - x_cursor) * scale_factor
            self.ax.set_xlim(new_min, new_max)

        self.canvas.draw_idle()

    def _toggle_peaks(self):
        """Toggle peak markers on/off."""
        self._show_peaks = not self._show_peaks
        self.btn_peaks.config(text=f"Peaks: {'ON' if self._show_peaks else 'OFF'}")
        self._refresh_peak_markers()

    def _on_threshold_change(self, _value=None):
        """Update peak markers when threshold slider moves."""
        val = self._peak_threshold.get()
        self.lbl_threshold.config(text=f"Threshold: {val:.0f} dB")
        if self._show_peaks:
            self._refresh_peak_markers()

    def _on_freq_cutoff_change(self, _value=None):
        """Replot with updated frequency cutoff."""
        pct = self._freq_cutoff_pct.get()
        cutoff_hz = self._max_data_freq * pct / 100.0
        self.lbl_freq_cutoff.config(text=f"{pct:.0f} %")
        self.lbl_freq_cutoff_hz.config(text=self._format_freq(cutoff_hz))
        if self.fft_data is not None:
            self._update_plot()

    def _clear_peak_markers(self):
        for artist in self._peak_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._peak_artists.clear()

    def _refresh_peak_markers(self):
        if self._refreshing_peaks:
            return
        self._refreshing_peaks = True
        try:
            self._do_refresh_peaks()
        finally:
            self._refreshing_peaks = False

    def _do_refresh_peaks(self):
        self._clear_peak_markers()

        if not self._show_peaks or self._freqs is None:
            self.canvas.draw_idle()
            return

        x_min, x_max = self.ax.get_xlim()

        mask = (self._freqs >= x_min) & (self._freqs <= x_max)
        if not np.any(mask):
            self.canvas.draw_idle()
            return

        view_freqs = self._freqs[mask]
        view_db = self._freqs_db[mask]

        threshold = self._peak_threshold.get()
        peak_indices, properties = find_peaks(
            view_db,
            prominence=threshold,
            distance=max(1, len(view_db) // 50),
        )

        if len(peak_indices) == 0:
            self.canvas.draw_idle()
            return

        if len(peak_indices) > 15:
            top = np.argsort(properties['prominences'])[-15:]
            peak_indices = peak_indices[top]

        for idx in peak_indices:
            freq = view_freqs[idx]
            db = view_db[idx]
            marker, = self.ax.plot(freq, db, 'v', color='#ff7f0e',
                                   markersize=8, markeredgecolor='black',
                                   markeredgewidth=0.5)
            ann = self.ax.annotate(
                f"{self._format_freq(freq)}\n{db:.1f} dB",
                xy=(freq, db), xytext=(0, 10),
                textcoords='offset points', fontsize=7,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='#fff3cd', ec='#ff7f0e', alpha=0.85),
            )
            self._peak_artists.extend([marker, ann])

        self.canvas.draw_idle()

    def _check_for_update(self):
        """Poll for file changes"""
        try:
            if os.path.exists(self.data_path):
                mtime = os.path.getmtime(self.data_path)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    self._load_and_refresh()
        except Exception:
            pass

        self.root.after(POLL_INTERVAL_MS, self._check_for_update)

    def _load_and_refresh(self):
        try:
            with open(self.data_path, 'r') as f:
                self.fft_data = json.load(f)
            self._update_info_bar()
            self._update_plot()
            now = datetime.now().strftime('%H:%M:%S')
            self.lbl_status.config(text=f"Updated {now}", foreground="green")
        except (json.JSONDecodeError, KeyError):
            self.lbl_status.config(text="Waiting for valid data…", foreground="orange")

    def _draw_waiting(self):
        self.ax.clear()
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude (V)')
        self.ax.set_title('FFT Spectrum — Waiting for data…')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _update_info_bar(self):
        fundamental = self.fft_data.get('fundamental_frequency', 0)
        sample_rate = self.fft_data.get('sample_rate', 0)
        num_samples = self.fft_data.get('num_samples', 0)

        self.lbl_fundamental.config(
            text=f"Fundamental Frequency: {self._format_freq(fundamental)}")
        self.lbl_sample_rate.config(
            text=f"Sample Rate: {self._format_freq(sample_rate)}")
        self.lbl_samples.config(text=f"Samples: {num_samples:,}")

    def _update_plot(self):
        freqs = np.array(self.fft_data['frequencies'])
        mags = np.array(self.fft_data['magnitudes'])
        fundamental = self.fft_data.get('fundamental_frequency', 0)

        mags_db = 20.0 * np.log10(np.maximum(mags, 1e-12))

        # Apply frequency cutoff
        self._max_data_freq = freqs[-1] if len(freqs) > 0 else 1.0
        cutoff_hz = self._max_data_freq * self._freq_cutoff_pct.get() / 100.0
        mask = freqs <= cutoff_hz
        freqs = freqs[mask]
        mags_db = mags_db[mask]

        self._freqs = freqs
        self._freqs_db = mags_db

        self.lbl_freq_cutoff_hz.config(text=self._format_freq(cutoff_hz))

        self.ax.clear()
        self._peak_artists.clear()

        self.ax.plot(freqs, mags_db, color='#1f77b4', linewidth=0.8, label='FFT Magnitude')

        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_title('FFT Spectrum')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        if self._show_peaks:
            self._refresh_peak_markers()

        self.canvas.draw_idle()

    @staticmethod
    def _format_freq(freq: float) -> str:
        if freq >= 1e9:
            return f"{freq / 1e9:.3f} GHz"
        elif freq >= 1e6:
            return f"{freq / 1e6:.3f} MHz"
        elif freq >= 1e3:
            return f"{freq / 1e3:.3f} kHz"
        else:
            return f"{freq:.2f} Hz"


def main():
    data_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_DATA_FILE
    print(f"Watching: {data_path}")
    print(f"Polling every {POLL_INTERVAL_MS}ms — run a measurement in Logic 2 to see results.\n")

    root = tk.Tk()
    FFTPlotApp(root, data_path)
    root.mainloop()


if __name__ == '__main__':
    main()
