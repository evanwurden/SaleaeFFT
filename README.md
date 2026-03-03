# Saleae FFT

An analog measurement extension for Saleae Logic that computes the FFT of a selected analog signal range, reports the fundamental frequency as an in-app measurement, and sends selected data to an interactive FFT window.

## Features

- **Saleae Extension** — fundamental frequency is displayed in the Logic 2 measurements window.
- **FFT Plot Viewer** — a separate matplotlib/tkinter GUI window showing the full FFT plot.

## Requirements

- **Saleae Logic 2** (version 2.4.0+)
- **Python 3.8+**
- **matplotlib** (for the FFT plot viewer)
- **scipy** (for peak detection in the FFT plot viewer)
- **numpy**

Install the required packages:

```bash
pip install -r requirements.txt
```

## Install Extension

1. Open **Saleae Logic 2**.
2. Navigate to **Extensions** - **Load Existing Extension**
3. Select the folder containing this plugin.
4. The Saleae FFT plugin will appear in the measurement extensions list.

## Usage

1. Run fft_plot_gui.py or .exe
2. Capture an analog signal in Logic 2.
3. Open the Measurements sidebar and add the Saleae FFT Plugin measurement.
4. Select a region of analog data on your channel.
5. The fundamental frequency value appears in the measurement results.

### Note: Disable this extension while not using the GUI, as the resource-intensive FFT calculation runs on any analog waveform selection.