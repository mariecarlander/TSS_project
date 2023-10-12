import math
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.fft import fft
from scipy.signal import convolve

# Load in the numpy arrays provided in the task 
action_potentials = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/action_potentials.npy', allow_pickle=True)
firing_samples = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/firing_samples.npy', allow_pickle=True)

# Initialize the constants 
duration = 20.0  # Signal duration in seconds
sampling_frequency = 10000  # Sampling frequency in Hz
num_samples = int(duration * sampling_frequency) # How many samples there is

# Create the time vector with an interval starting on 0 and stopping at duration and length of the interval
time_vector = np.arange(0, duration, 1/sampling_frequency)

# Create the array of the EMG signal and fill it with zeros
emg_signal = np.zeros(num_samples)

# Create EMG signal by convolving action potentials with firing times
for i in range(len(action_potentials)):
    motor_unit = action_potentials[i]
    
    # Access the firing times for the current motor unit
    firing_times = firing_samples[0][i] / sampling_frequency
    
    # Create a train of delta functions at firing times
    delta_train = np.zeros(num_samples)
    delta_train[np.around(firing_times * sampling_frequency).astype(int)] = 1
    
    # Convolve action potentials with the delta train and motor unit as inputs, and return the full
    # linear convolution
    emg_signal += convolve(delta_train, motor_unit, mode='full')[:num_samples]
    print(len(firing_times)*100)

# Plot the EMG signal
plt.figure(figsize=(10, 6))
plt.plot(time_vector, emg_signal)
plt.title("Generated EMG Signal (Convolution)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


