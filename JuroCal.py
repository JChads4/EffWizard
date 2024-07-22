import os
import tkinter as tk
from tkinter import messagebox, ttk
from GeneralFunc import do_efficiencies
import matplotlib.pyplot as plt

def calculate_efficiencies():
    try:

        expected_peak_energies = [81, 122, 244, 276, 303, 411, 444, 779, 867, 964, 1112, 1408]

        # Get Ba sources
        ba_sources_input = ba_entry.get()
        ba_sources = [int(x.strip()) for x in ba_sources_input.split(',')]

        # Get Eu sources
        eu_sources_input = eu_entry.get()
        eu_sources = [int(x.strip()) for x in eu_sources_input.split(',')]

        # Get calibration date
        calib_date = calib_entry.get()

        # Get dt (length of calibration in minutes)
        dt = int(dt_entry.get())
        dt_seconds = dt * 60  # Convert minutes to seconds

        # Get path of the data file
        data_path = data_entry.get()

        # Get window
        peak_window_input = peak_window_entry.get()
        peak_window = int(peak_window_input)

        # Get background type
        background_type = background_var.get()

        # Call the do_efficiencies function with user inputs
        do_efficiencies(ba_sources, eu_sources, calib_date, dt_seconds, data_path, peak_window, background_type, expected_peak_energies)
        
        # Display success message
        messagebox.showinfo("Success", "Efficiency calculation completed successfully! Check you out :D")
    except Exception as e:
        # Display error message if any exception occurs
        messagebox.showerror("Error", str(e))

def save_settings():
    try:
        settings_name = settings_name_entry.get()
        settings_dir = 'settings/'
        with open(os.path.join(settings_dir, f"{settings_name}.settings"), "w") as file:
            file.write(f"Ba Source #: {ba_entry.get()}\n")
            file.write(f"Eu Source #: {eu_entry.get()}\n")
            file.write(f"Calibration Date: {calib_entry.get()}\n")
            file.write(f"Length of Calibration (minutes): {dt_entry.get()}\n")
            file.write(f"Data File (with ext): {data_entry.get()}\n")
            file.write(f"Peak Window: {peak_window_entry.get()}\n")
            file.write(f"Background Type: {background_var.get()}\n")
        messagebox.showinfo("Success", "Settings saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def load_settings():
    try:
        selected_setting = settings_dropdown.get()
        settings_dir = 'settings/'
        with open(os.path.join(settings_dir, selected_setting), "r") as file:
            settings = file.readlines()
            ba_entry.delete(0, tk.END)
            ba_entry.insert(0, settings[0].split(": ")[1].strip())
            eu_entry.delete(0, tk.END)
            eu_entry.insert(0, settings[1].split(": ")[1].strip())
            calib_entry.delete(0, tk.END)
            calib_entry.insert(0, settings[2].split(": ")[1].strip())
            dt_entry.delete(0, tk.END)
            dt_entry.insert(0, settings[3].split(": ")[1].strip())
            data_entry.delete(0, tk.END)
            data_entry.insert(0, settings[4].split(": ")[1].strip())
            peak_window_entry.delete(0, tk.END)
            peak_window_entry.insert(0, settings[5].split(": ")[1].strip())
            background_var.set(settings[6].split(": ")[1].strip())
        messagebox.showinfo("Success", "Settings loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create main window
window = tk.Tk()
window.title("Efficiency Calculation Tool")

# Ba sources input
ba_label = tk.Label(window, text="JYFL Ba Source #")
ba_label.grid(row=0, column=0, padx=5, pady=5)
ba_entry = tk.Entry(window)
ba_entry.grid(row=0, column=1, padx=5, pady=5)

# Eu sources input
eu_label = tk.Label(window, text="JYFL Eu Source #")
eu_label.grid(row=1, column=0, padx=5, pady=5)
eu_entry = tk.Entry(window)
eu_entry.grid(row=1, column=1, padx=5, pady=5)

# Calibration date input
calib_label = tk.Label(window, text="Calibration Date (dd.mm.yyyy)")
calib_label.grid(row=2, column=0, padx=5, pady=5)
calib_entry = tk.Entry(window)
calib_entry.grid(row=2, column=1, padx=5, pady=5)

# Length of calibration input
dt_label = tk.Label(window, text="Length of Calibration (mins)")
dt_label.grid(row=3, column=0, padx=5, pady=5)
dt_entry = tk.Entry(window)
dt_entry.grid(row=3, column=1, padx=5, pady=5)

# Data path input
data_label = tk.Label(window, text="Spectrum Filename (ExpCode.dat)")
data_label.grid(row=4, column=0, padx=5, pady=5)
data_entry = tk.Entry(window)
data_entry.grid(row=4, column=1, padx=5, pady=5)

# Peak window input
peak_window_label = tk.Label(window, text="Peak Window")
peak_window_label.grid(row=5, column=0, padx=5, pady=5)
peak_window_entry = tk.Entry(window)
peak_window_entry.grid(row=5, column=1, padx=5, pady=5)

# Background type input
background_label = tk.Label(window, text="Background Type")
background_label.grid(row=6, column=0, padx=5, pady=5)
background_var = tk.StringVar(value="linear")
background_linear = tk.Radiobutton(window, text="Linear", variable=background_var, value="linear")
background_linear.grid(row=6, column=1, padx=5, pady=5)
background_quadratic = tk.Radiobutton(window, text="Quadratic", variable=background_var, value="quadratic")
background_quadratic.grid(row=6, column=2, padx=5, pady=5)

# Settings name input
settings_name_label = tk.Label(window, text="Settings Name")
settings_name_label.grid(row=7, column=0, padx=5, pady=5)
settings_name_entry = tk.Entry(window)
settings_name_entry.grid(row=7, column=1, padx=5, pady=5)

# Save settings button
save_button = tk.Button(window, text="Save Settings", command=save_settings)
save_button.grid(row=8, column=0, padx=5, pady=5, sticky="WE")

# Load settings dropdown
load_label = tk.Label(window, text="Load Settings")
load_label.grid(row=9, column=0, padx=5, pady=5)
settings_dir = 'settings/'
settings_files = [filename for filename in os.listdir(settings_dir) if filename.endswith(".settings")]
settings_dropdown = ttk.Combobox(window, values=settings_files)
settings_dropdown.grid(row=9, column=1, padx=5, pady=5)

# Load settings button
load_button = tk.Button(window, text="Load Settings", command=load_settings)
load_button.grid(row=9, column=2, columnspan=2, padx=5, pady=5, sticky="WE")

# Calculate button
calculate_button = tk.Button(window, text="Calculate", command=calculate_efficiencies)
calculate_button.grid(row=10, column=0, columnspan=3, padx=5, pady=10, sticky="WE")

# Contact info
contact_label = tk.Label(window, text="Created by Jamie Chadderton", font=("Helvetica", 10))
contact_label.grid(row=12, column=0, columnspan=3, pady=(20, 10))

# GitHub repository link
github_label = tk.Label(window, text="GitHub Repository: https://github.com/JChads4/EffWizard", font=("Helvetica", 10), fg="blue", cursor="hand2")
github_label.grid(row=13, column=0, columnspan=3, pady=(0, 10))
github_label.bind("<Button-1>", lambda e: os.system("xdg-open https://github.com/JChads4/EffWizard"))

# Run the GUI
window.mainloop()
