
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

3. **Configure files**

	Update 'conf/sources.txt' to match your list of sources used to calibrate (dd.mm.yyy), currently the ones listed are purely examples.
	Load in experimental ascii spectrum in the 'data' directory, and name appropriately. 	

## Example Usage

```sh
python JuroCal.py
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
