
# Efficiency Calculation Tool

## Overview

The Efficiency Calculation Tool is a Python-based application that analyses gamma spectra to calculate the efficiency of a Germanium detector.

- Load and analyse gamma spectra, calibrated with EuBa source.
- Fit peaks with Gaussian functions.
- Calculate detector efficiency based on specified sources and peaks.
- Calculate FWHM as a function of peak energies.
- Make a residuals plot.
- Save and load configuration settings.
- Plot results and fitted curves.

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - tkinter
  - sympy

Optional (recommended):
- `scienceplots` for enhanced plotting styles.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/JChads4/effProgram.git
   cd effProgram
   ```

2. **Install dependencies:**

   ```sh
   pip install pandas numpy matplotlib scipy sympy

   or relevant command for the users OS.
   ```

   If you want to use `scienceplots`:

   ```sh
   pip install scienceplots
   ```

## Usage

1. **Run the GUI:**

   ```sh
   python JuroCal.py
   ```

2. **Input Parameters:**

   - **JYFL Ba Source #:** Enter the source number for Ba-133.
   - **JYFL Eu Source #:** Enter the source number for Eu-152.
   - **Calibration Date:** Enter the calibration date in `dd.mm.yyyy` format.
   - **Length of Calibration (mins):** Enter the calibration length in minutes.
   - **Spectrum Filename:** Enter the filename of the spectrum data (e.g., `ExpCode.dat`).
   - **Peak Window:** Enter the window size for peak fitting.
   - **Background Type:** Select the background type (`Linear` or `Quadratic`).

3. **Save Settings:**

   - Enter a name for the settings and click **Save Settings** to save the current configuration.

4. **Load Settings:**

   - Select a previously saved settings file from the dropdown and click **Load Settings** to load the configuration.

5. **Calculate:**

   - Click **Calculate** to run the efficiency calculation with the specified parameters.

## Example

```sh
python gui.py
```

1. Input the following parameters:
   - JYFL Ba Source #: BARIUM SOURCE NUMBER
   - JYFL Eu Source #: EURPOIUM SOURCE NUMBER 
   - Calibration Date: DD.MM.YYYY
   - Length of Calibration (mins): CALIB LENGTH (INTEGER)
   - Spectrum Filename: JRxxx.dat
   - Peak Window: 20 (for example)
   - Background Type: Linear (for example)

2. Save the settings as `JRxx`, if you want to easily load them in again. Just the experiment code is fine here.

3. Click **Calculate** to process the data and generate the results.

## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, please contact Jamie Chadderton at: jamiechadderton8@gmail.com.
