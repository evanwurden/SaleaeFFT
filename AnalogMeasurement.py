import numpy as np
import os
import sys
import json
import subprocess
import tempfile

from saleae.range_measurements import AnalogMeasurer

FUNDAMENTAL_FREQ = 'fundamentalFrequency'
FFT_DATA_FILE = os.path.join(tempfile.gettempdir(), 'saleae_fft_data.json')

# Path to the bundled FFT viewer EXE (sits next to this script)
_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
FFT_VIEWER_EXE = os.path.join(_EXT_DIR, 'SaleaeFFTViewer.exe')

_gui_process = None  # track the launched process so we only start it once

_FFT_VIEWER_EXE_NAME = 'SaleaeFFTViewer.exe'


def _is_viewer_running():
    """Return True if the FFT viewer EXE is already running system-wide."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'IMAGENAME eq {_FFT_VIEWER_EXE_NAME}', '/NH'],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return _FFT_VIEWER_EXE_NAME.lower() in result.stdout.lower()
    except Exception:
        return False


def _launch_fft_viewer():
    """Launch the FFT plot GUI EXE if it is not already running."""
    global _gui_process
    # Quick check: did *we* already launch it and is it still alive?
    if _gui_process is not None and _gui_process.poll() is None:
        return
    # System-wide check: is the viewer running from any source?
    if _is_viewer_running():
        return
    # Gracefully skip if the EXE doesn't exist
    if not os.path.isfile(FFT_VIEWER_EXE):
        return
    try:
        _gui_process = subprocess.Popen(
            [FFT_VIEWER_EXE],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except Exception:
        pass


class MyAnalogMeasurer(AnalogMeasurer):
    supported_measurements = [FUNDAMENTAL_FREQ]

    def __init__(self, requested_measurements):
        super().__init__(requested_measurements)
        self.samples = []
        self.sample_count = 0
        self._start_time = None
        self._end_time = None

    def process_data(self, data):
        """Accumulate analog sample batches."""
        self.samples.append(data.samples.copy())
        self.sample_count += data.sample_count

        if self._start_time is None:
            self._start_time = data.start_time
        self._end_time = data.end_time

    def measure(self):
        values = {}

        if self.sample_count == 0:
            values[FUNDAMENTAL_FREQ] = 0.0
            return values

        all_samples = np.concatenate(self.samples)
        n = len(all_samples)

        duration = float(self._end_time - self._start_time)
        if duration > 0 and n > 1:
            sample_rate = (n - 1) / duration
        else:
            values[FUNDAMENTAL_FREQ] = 0.0
            return values

        all_samples = all_samples - np.mean(all_samples)  # Remove DC offset

        window = np.hanning(n)
        windowed = all_samples * window

        fft_vals = np.fft.rfft(windowed)
        fft_magnitude = np.abs(fft_vals) * 2.0 / n
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        max_freq = sample_rate / 4.0
        freq_mask = fft_freqs <= max_freq
        fft_freqs = fft_freqs[freq_mask]
        fft_magnitude = fft_magnitude[freq_mask]

        if len(fft_magnitude) > 1:
            fundamental_idx = np.argmax(fft_magnitude[1:]) + 1
            fundamental_frequency = float(fft_freqs[fundamental_idx])
        else:
            fundamental_frequency = 0.0

        values[FUNDAMENTAL_FREQ] = fundamental_frequency

        self._save_fft_data(fft_freqs, fft_magnitude, fundamental_frequency, sample_rate)

        # Auto-launch the FFT viewer GUI (first measurement triggers it)
        _launch_fft_viewer()

        return values

    def _save_fft_data(self, freqs, magnitudes, fundamental_freq, sample_rate):
        """Write FFT results to the JSON file."""
        try:
            fft_data = {
                'frequencies': freqs.tolist(),
                'magnitudes': magnitudes.tolist(),
                'fundamental_frequency': fundamental_freq,
                'sample_rate': float(sample_rate),
                'num_samples': int(self.sample_count),
            }

            tmp_file = FFT_DATA_FILE + '.tmp'
            with open(tmp_file, 'w') as f:
                json.dump(fft_data, f)
            os.replace(tmp_file, FFT_DATA_FILE)
        except Exception:
            pass
