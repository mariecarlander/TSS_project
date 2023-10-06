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

# Initialize an array to store the EMG signal
emg_signal = np.zeros(num_samples)

# Create EMG signal by convolving action potentials with firing times
for i in range(len(action_potentials)):
    motor_unit = action_potentials[i]
    
    # Access the firing times for the current motor unit
    firing_times = firing_samples[0][i] / sampling_frequency
    
    # Create a train of delta functions at firing times
    delta_train = np.zeros(num_samples)
    delta_train[np.round(firing_times * sampling_frequency).astype(int)] = 1
    
    # Convolve action potentials with the delta train
    emg_signal += convolve(delta_train, motor_unit, mode='full')[:num_samples]

# Plot the EMG signal
plt.figure(figsize=(10, 6))
plt.plot(time_vector, emg_signal)
plt.title("Generated EMG Signal (Convolution)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


