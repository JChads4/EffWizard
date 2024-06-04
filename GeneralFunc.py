import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import sympy as sp
import numpy as np
from scipy.integrate import quad
from datetime import datetime
import math
from scipy.stats import linregress
from scipy.signal import find_peaks
import csv
from scipy.optimize import curve_fit
from scipy.integrate import simps
try:
    import scienceplots
    plt.style.use(['science'])
except ImportError:
    print("Error: Could not import SciencePlots.")

def find_peaks_relative_to_max(data, threshold_ratio, min_distance):
    max_intensity = np.max(data)
    threshold = max_intensity * threshold_ratio
    peaks, _ = find_peaks(data, height=threshold, distance=min_distance)
    return peaks



# Calculate supposed activity of source at a particular time
def calc_expected_activity(row, end_date):
    """
    Calculate expected activity at a particular time based on the information provided in the sources DataFrame.

    Parameters:
    - row: DataFrame row containing the source information, including 'Activity (Bq)', 'Ref', 'Half Life', and 'Half Life Units'
    - end_date: End date for calculating the time difference

    Returns:
    - expected_activity: Expected activity at the given time
    """
    # Extract necessary information from the DataFrame row
    a0 = row['Activity (Bq)']
    t_half = row['Half Life']
    t_half_units = row['Units']
    ref_date = datetime.strptime(row['Ref'], '%d.%m.%Y')

    # Calculate the time difference based on the end date provided
    end_date = datetime.strptime(end_date, '%d.%m.%Y')
    dt_years = (end_date - ref_date).days / 365.25   # Assuming 1 year = 365.25 days
    #print(f"datetime output: {dt_years}") # check datetime is correct

    # Calculate expected activity using the provided formula
    expected_activity = a0 * np.exp(-np.log(2) * dt_years / t_half)
    return expected_activity

def calc_activity(n, eff, intensity, time):
    return n/(eff*intensity*time)

def calc_eff_params(a, b, d, e, f, energy):
    x = np.log(energy/100)
    y = np.log(energy/1000)
    return np.exp( ( (a + b*x)**(-3) + (d + e*y + f*(y**2))**(-3) )**(-1/3) )

def calc_sage_eff_params(a, b, c, d, e, energy):
    x = np.log(energy)
    return np.exp( a + (b*x) + (c*x**2) + (d*x**3) + (e*x**4) )

def eff_function(energy, a, b, c, d, e):
    x = np.log(energy/350)
    #x = math.log10(energy/350)
    return np.exp( a + b*x + c*(x**2) + d*(x**3) + e*(x**4) )

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# normal distrbution
def normal(x, A, mu, sigma):
    return A * (1/(sigma*np.sqrt(2* math.pi))) *np.exp(-(x - mu)**2 / (2 * sigma**2))

def compute_error_bands(popt, pcov, background_polynomial, e_range, optimal_curve, num_curves=100):

    curve_samples = np.random.multivariate_normal(popt, pcov, num_curves)
    error_bands = np.array([background_polynomial(e_range, *params) for params in curve_samples])
    std_dev = np.sqrt(np.sum((error_bands - optimal_curve)**2, axis=0) / (num_curves - 1))
    return std_dev

def linear(x, a, b):
    return a * x + b

def fit_roi_peaks(x, y, initial_guess):
    popt, _ = curve_fit(gaussian, x, y, p0=initial_guess)
    return popt

def quad_background(x, a, b, c):
    return a * x ** 2 + b * x + c

def background(x, b, c):
    return b * x + c

def gaussian_plus_quad_background(x, amplitude, mean, stddev, a, b, c):
    return normal(x, amplitude, mean, stddev) + quad_background(x, a, b, c)

def gaussian_plus_background(x, amplitude, mean, stddev, b, c):
    return normal(x, amplitude, mean, stddev) + background(x, b, c)

def load_data(folder, filename):
    data_file = os.path.join(folder, filename)
    return np.loadtxt(data_file)

def analyse_data(data, expected_peak_energies, window, background_type, label='', xlabel='', ylabel='', title='', xlim=None, ylim=None, fontsize=20, color='b', figsize=(12, 6), save_path=None, log_scale=False):
    energy_bin_edges = data[:, 0]
    counts = data[:, 1]
    peak_data = []

    for expected_peak in expected_peak_energies:
        peak_index = np.argmin(np.abs(energy_bin_edges - expected_peak))
        peak_x = energy_bin_edges[max(0, peak_index - window):min(len(energy_bin_edges), peak_index + window + 1)]
        peak_y = counts[max(0, peak_index - window):min(len(counts), peak_index + window + 1)]
        amplitude_guess = peak_y.max() - peak_y.min()
        mean_guess = energy_bin_edges[peak_index]
        stddev_guess = window / 2

        if background_type == 'linear':
            background_guess = [0, peak_y.min()]
            popt, pcov = curve_fit(gaussian_plus_background, peak_x, peak_y, p0=[amplitude_guess, mean_guess, stddev_guess] + background_guess)
            x_fit = np.linspace(peak_x.min(), peak_x.max(), 100)
            y_fit = gaussian_plus_background(x_fit, *popt)
            def integrand(x):
                return gaussian_plus_background(x, *popt) - background(x, *popt[3:])
            background_func = background
        else:  # quadratic
            background_guess = [0, 0, peak_y.min()]
            popt, pcov = curve_fit(gaussian_plus_quad_background, peak_x, peak_y, p0=[amplitude_guess, mean_guess, stddev_guess] + background_guess)
            x_fit = np.linspace(peak_x.min(), peak_x.max(), 100)
            y_fit = gaussian_plus_quad_background(x_fit, *popt)
            def integrand(x):
                return gaussian_plus_quad_background(x, *popt) - quad_background(x, *popt[3:])
            background_func = quad_background

        shaded_area, _ = quad(integrand, x_fit[0], x_fit[-1])
        bin_width = energy_bin_edges[1] - energy_bin_edges[0]
        peak_counts = shaded_area / bin_width
        peak_counts = int(peak_counts)

        plt.plot(x_fit, y_fit, color='r', linestyle='--')
        plt.plot(x_fit, background_func(x_fit, *popt[3:]), color='b', linestyle='--')
        plt.fill_between(x_fit, background_func(x_fit, *popt[3:]), y_fit, color='green', alpha=0.1)
        plt.annotate(f"{popt[1]:.2f}, FWHM= {np.abs(popt[2]) * 2.355:.2f}", xy=(energy_bin_edges[peak_index], counts[peak_index]), xytext=(0, 20), textcoords='offset points', fontsize=12, rotation=90, ha='center', va='bottom', arrowprops=dict(facecolor='black', arrowstyle='->'))
        peak_data.append([peak_index, np.abs(popt[1]), np.abs(popt[2]), np.abs(popt[2]) * 2.355, shaded_area, peak_counts])

    peak_df = pd.DataFrame(peak_data, columns=['Peak Index', 'Energy', 'Sigma', 'FWHM', 'Area', 'Counts'])

    plt.step(energy_bin_edges, counts, where='mid', label=label, color=color, alpha=0.6)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(f'Counts/ {bin_width:.1f} keV', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize - 4)
    ax = plt.gca()
    ax.tick_params(direction='in', which='both', top=True, right=True, length=8, labelsize=16)
    plt.grid()
    if save_path:
        os.makedirs("Images", exist_ok=True)
        save_path = os.path.join("Images", save_path)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

    fs = 20
    plt.figure(figsize=(8, 6))
    plt.scatter(peak_df['Energy'], peak_df['FWHM'], color='r', label='Data')
    gradient, intercept, _, _, _ = linregress(peak_df['Energy'], peak_df['FWHM'])
    plt.plot(peak_df['Energy'], gradient * peak_df['Energy'] + intercept, color='b', linestyle='--', label='Linear Fit')
    plt.text(0.1, 0.8, f'FWHM = {gradient:.4f} * Energy + {intercept:.4f}', transform=plt.gca().transAxes, fontsize=12)
    plt.xlabel('Peak Energy', fontsize=fs-2)
    plt.ylabel('FWHM', fontsize=fs-2)
    plt.title('FWHM vs. Peak Energy', fontsize=fs)
    plt.legend(fontsize=fs-4)
    plt.show()

    print("############################################################################")
    print("######################### PEAK INFORMATION #################################")
    print("############################################################################")
    print(peak_df)
    print("")

    return peak_df

def load_source_data(calib_date, source_number):

    sources_path = 'conf/sources.txt'
    half_life_path = 'conf/half_lives.txt'
    
    # Load source data: activity, ref date
    sources_df = pd.read_csv(sources_path, skipinitialspace=True, comment='#')

    # Get half lives
    half_life_df = pd.read_csv(half_life_path, skipinitialspace=True, comment = '#')

    # Merge half lives with appropriate isotopes
    sources_df = pd.merge(sources_df, half_life_df, on='Type', how='left')

    # Calculate expected activities at calib time
    sources_df['Expected Activity'] = sources_df.apply(lambda row: calc_expected_activity(row, calib_date), axis=1)

    # Select the source based on source number
    selected_source = sources_df.loc[sources_df['Number'] == source_number].squeeze()
    expected_activity = selected_source['Expected Activity']

    # 
    isotope = selected_source['Type']
    # print(isotope)

    # Load in the relevant intensities
    intensities_path = f'conf/{isotope}_intensities_jyfl.txt'
    # intensities_path = f'data/{isotope}_intensities_adjusted.txt'
    # intensities_path = f'data/{isotope}_intensities_idaho.txt'
    # intensities_path = f'data/{isotope}_intensities.txt'
    # print(intensities_path)

    # Load source transition intensities and energies
    intensities_df = pd.read_csv(intensities_path, skipinitialspace=True, comment='#') 
    transition_intensities = intensities_df['Intensity'].tolist()
    transition_energies = intensities_df['Energy'].tolist()
    source_number = selected_source['Number']

    return selected_source, expected_activity, transition_intensities, transition_energies, source_number

def load_electron_source_data(calib_date, source_number):

    sources_path = 'conf/sources.txt'
    half_life_path = 'conf/half_lives.txt'
    
    # Load source data: activity, ref date
    sources_df = pd.read_csv(sources_path, skipinitialspace=True, comment ='#')

    # Get half lives
    half_life_df = pd.read_csv(half_life_path, skipinitialspace=True, comment='#')

    # Merge half lives with appropriate isotopes
    sources_df = pd.merge(sources_df, half_life_df, on='Type', how='left')

    # Calculate expected activities at calib time
    sources_df['Expected Activity'] = sources_df.apply(lambda row: calc_expected_activity(row, calib_date), axis=1)

    # Select the source based on source number
    selected_source = sources_df.loc[sources_df['Number'] == source_number].squeeze()
    expected_activity = selected_source['Expected Activity']

    # 
    isotope = selected_source['Type']
    # print(isotope)

    # Load in the relevant intensities
    intensities_path = f'conf/{isotope}_electrons.txt'
    # intensities_path = f'data/{isotope}_intensities_adjusted.txt'
    # intensities_path = f'data/{isotope}_intensities_idaho.txt'
    # intensities_path = f'data/{isotope}_intensities.txt'
    # print(intensities_path)

    # Load source transition intensities and energies
    intensities_df = pd.read_csv(intensities_path, skipinitialspace=True, comment='#') 
    transition_intensities = intensities_df['Intensity'].tolist()
    transition_energies = intensities_df['Energy'].tolist()
    source_number = selected_source['Number']

    return selected_source, expected_activity, transition_intensities, transition_energies, source_number


# Calculating absolute efficiencies
def calc_eff(n, intensity, time, activity):
    eff = n/(intensity*time*activity)
    return eff
    
def calc_eff_errs(eff, counts, err_counts):
	err = eff*err_counts/counts
	return err

def calculate_efficiencies(energies, transition_energies, intensities, counts, err_counts, time, activity):
    efficiencies = []
    err_efficiencies = []
    measured_energies = []
    print("")
    print("############################################################################")
    print("###################### EFFICIENCY INFORMATION ##############################")
    print("############################################################################")
    print("")

    for energy, count in zip(energies, counts):
        matched_intensity = None
        for t_energy, intensity in zip(transition_energies, intensities):
            if abs(energy - t_energy) <= 2:
                matched_intensity = intensity
                efficiency = calc_eff(count, matched_intensity, time, activity)
                err_effs = calc_eff_errs(efficiency, count, np.sqrt(count))
                err_efficiencies.append(err_effs)
                efficiencies.append(efficiency)
                measured_energies.append(energy)
                print(f'Energy = {energy} keV, Intensity = {matched_intensity}, Efficiency = {efficiency}') 
                break
        else:
            # print(f'Intensity not found within tolerance for energy peak at {energy} keV')
            pass

    return measured_energies, efficiencies, err_efficiencies, counts

def do_efficiencies(ba_sources, eu_sources, calib_date, dt, spectrum_name, peak_window, background_type, expected_peak_energies):
    # Load calibrated sum spectrum
    juro_energy = load_data('data/', f'{spectrum_name}')

    bin_width = juro_energy[1] - juro_energy[0]
    params = {
        'label': 'data',
        'log_scale': False,
        'window': peak_window,
        'background_type': background_type,
        'expected_peak_energies': expected_peak_energies,
        'xlabel': 'Energy (keV)',
        'title': r'EuBa Calibrated Jurogam',
        'fontsize': 20,
        'xlim': (0, 2000),
        'ylim': None,
        'color': 'k',
        'figsize': (10, 7),
    }
    juro_df = analyse_data(data=juro_energy, **params)
    juro_e, juro_sigE, juro_n = juro_df[['Energy', 'Sigma', 'Counts']].values.T.tolist()
    juro_n_err = np.sqrt(juro_n)

    # Plot settings
    fs = 16
    plt.figure(figsize=(8, 5))
    plt.ylabel('Absolute Efficiency', fontsize=fs)
    plt.xlabel('Transition Energy (keV)', fontsize=fs)

    ba_colors = ['red', 'blue', 'magenta', 'purple', 'orange']
    eu_colors = ['green', 'cyan', 'yellow', 'black', 'pink']
    markers = ['^', 's', 'o', 'D', 'v']

    energies = np.linspace(0, 1500, 100000)

    for i, (ba_source_number, ba_color, marker) in enumerate(zip(ba_sources, ba_colors, markers)):
        ba_source, ba_activity, ba_transition_intensities, ba_transition_energies, _ = load_source_data(calib_date, ba_source_number)
        ba_measured_energies, ba_measured_effs, ba_err_effs, ba_counts = calculate_efficiencies(
            energies=juro_e,
            intensities=ba_transition_intensities,
            transition_energies=ba_transition_energies,
            counts=juro_n,
            err_counts=juro_n_err,
            time=dt,
            activity=ba_activity
        )

        plt.errorbar(ba_measured_energies, ba_measured_effs, xerr=None, yerr=ba_err_effs, fmt=marker,
                     label=f'Ba-133: Source {ba_source_number}', capsize=3, color=ba_color)

    for i, (eu_source_number, eu_color, marker) in enumerate(zip(eu_sources, eu_colors, markers)):
        eu_source, eu_activity, eu_transition_intensities, eu_transition_energies, _ = load_source_data(calib_date, eu_source_number)
        eu_measured_energies, eu_measured_effs, eu_err_effs, eu_counts = calculate_efficiencies(
            energies=juro_e,
            intensities=eu_transition_intensities,
            transition_energies=eu_transition_energies,
            counts=juro_n,
            err_counts=juro_n_err,
            time=dt,
            activity=eu_activity
        )

        plt.errorbar(eu_measured_energies, eu_measured_effs, xerr=None, yerr=eu_err_effs, fmt=marker,
                     label=f'Eu-152: Source {eu_source_number}', capsize=3, color=eu_color)

    combined_energies = np.concatenate((ba_measured_energies, eu_measured_energies))
    combined_efficiencies = np.concatenate((ba_measured_effs, eu_measured_effs))
    combined_errs = np.concatenate((ba_err_effs, eu_err_effs))

    p0 = [1.85694, -1.2808, -0.165651, 1.58654, -2.19377]
    popt, pcov = curve_fit(eff_function, xdata=combined_energies[0:-1], ydata=combined_efficiencies[0:-1], sigma=combined_errs[0:-1], absolute_sigma=False, p0=p0)
    eff_errors = np.sqrt(np.diag(pcov))
    param_names = ['a', 'b', 'c', 'd', 'e']
    param_labels = [f'{param}: {value:.2f} ± {error:.2f}' for param, value, error in zip(param_names, popt, eff_errors)]
    legend_label = ', '.join(param_labels)

    energies = np.linspace(min(combined_energies), max(combined_energies), len(energies))
    optimal_curve = eff_function(energies, *popt)

    print("")
    print("############################################################################")
    print("##################### FITTED EFFICIENCY PARAMS #############################")
    print("############################################################################")
    print("")
    print("EQUATION USED -> ")
    print("")

    E, a, b, c, d, e = sp.symbols('E a b c d e')
    ln_E = sp.log(E / 350)
    equation = a + b * ln_E + c * ln_E**2 + d * ln_E**3 + e * ln_E**4
    equation_str = f"exp({equation})"
    print(equation_str)
    print("")
    for param, value, error in zip(param_names, popt, eff_errors):
        print(f"{param}: {value:.2f} ± {error:.2f}")
    print("")
    print("############################### END ########################################")

    num_curves = 100
    curve_samples = np.random.multivariate_normal(popt, pcov, num_curves)
    error_bands = np.array([eff_function(energies, *params) for params in curve_samples])
    std_dev = np.sqrt(np.sum((error_bands - optimal_curve) ** 2, axis=0) / (num_curves - 1))
    plt.fill_between(energies, optimal_curve - std_dev, optimal_curve + std_dev, color='b', alpha=0.1)

    plt.plot(energies, optimal_curve, color='b', linestyle='dashdot', label=f'Fitted Curve \n{legend_label}')
    plt.legend(fontsize=fs)
    plt.grid()
    plt.title('EuBa Calibrated Jurogam', fontsize=fs + 2)
    plt.show()
    plt.close()

    residuals = combined_efficiencies - eff_function(combined_energies, *popt)
    plt.figure(figsize=(8, 5))
    plt.plot(combined_energies, residuals, marker='o', linestyle='', color='r')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Transition Energy (keV)', fontsize=fs)
    plt.ylabel('Residuals', fontsize=fs)
    plt.title('Residuals Plot', fontsize=fs + 2)
    plt.grid(True)
    plt.ylim(-0.1, 0.1)
    plt.show()
    plt.close()