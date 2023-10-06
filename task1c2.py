import math
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft
from scipy.signal import convolve

# Load the provided NumPy arrays
action_potentials = np.load('action_potentials.npy')
firing_samples = np.load('firing_samples.npy', allow_pickle=True)

# Constants
duration = 20.0  # Signal duration in seconds
sampling_frequency = 10000  # Sampling frequency in Hz
num_samples = int(duration * sampling_frequency)

# Create a time vector
time_vector = np.arange(0, duration, 1/sampling_frequency)

# Select the specific action potential train you want to plot (e.g., index 0 for the first train)
selected_motor_unit_index = 0
motor_unit = action_potentials[selected_motor_unit_index]

# Initialize an array to store the EMG signal
emg_signal = np.zeros(num_samples)

# Access the firing times for the selected motor unit
firing_times = firing_samples[0][selected_motor_unit_index] / sampling_frequency

# Create a train of delta functions at firing times
delta_train = np.zeros(num_samples)
delta_indices = np.round(firing_times * sampling_frequency).astype(int)
delta_train[delta_indices] = 1

# Convolve action potentials with the delta train
emg_signal = convolve(delta_train, motor_unit, mode='full')[:num_samples]

# Define the time interval for zooming
zoom_start_time = 10.0  # Start time for zooming
zoom_end_time = 10.5  # End time for zooming
zoom_start_sample = int(zoom_start_time * sampling_frequency)
zoom_end_sample = int(zoom_end_time * sampling_frequency)

# Plot the EMG signal for the selected motor unit
plt.figure(figsize=(10, 6))
plt.plot(time_vector, emg_signal, label="EMG Signal")
plt.title("Generated EMG Signal (Convolution) for Motor Unit {}".format(selected_motor_unit_index))
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Zoom in on the time interval 10-10.5 seconds
plt.figure(figsize=(10, 6))
plt.plot(time_vector[zoom_start_sample:zoom_end_sample], emg_signal[zoom_start_sample:zoom_end_sample], label="Zoomed In")
plt.title("Zoomed In EMG Signal (Convolution) for 1 Action potential train".format(selected_motor_unit_index))
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True)
plt.show()