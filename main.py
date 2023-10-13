import math
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.fft import fft
from scipy.signal import convolve
from scipy.signal.windows import hann

# Load in the numpy arrays provided in the task 
action_potentials = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/action_potentials.npy', allow_pickle=True)
firing_samples = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/firing_samples.npy', allow_pickle=True)

# Initialize the constants 
duration = 20.0  # Signal duration in seconds
sampling_frequency = 10000  # Sampling frequency in Hz
num_samples = int(duration * sampling_frequency) # How many samples there is


# Task 1a
# Building the action potential trains, 0-7
train0 = np.zeros(num_samples)
train0[np.around(firing_samples[0][0]).astype(int)] = 1

train1 = np.zeros(num_samples)
train1[np.around(firing_samples[0][1]).astype(int)] = 1

train2 = np.zeros(num_samples)
train2[np.around(firing_samples[0][2]).astype(int)] = 1

train3 = np.zeros(num_samples)
train3[np.around(firing_samples[0][3]).astype(int)] = 1

train4 = np.zeros(num_samples)
train4[np.around(firing_samples[0][4]).astype(int)] = 1

train5 = np.zeros(num_samples)
train5[np.around(firing_samples[0][5]).astype(int)] = 1

train6 = np.zeros(num_samples)
train6[np.around(firing_samples[0][6]).astype(int)] = 1

train7 = np.zeros(num_samples)
train7[np.around(firing_samples[0][7]).astype(int)] = 1

trains = [train0, train1, train2, train3, train4, train5, train6, train7]


# Task 1c
# Create the time vector
time_vector = np.arange(0, duration, 1/sampling_frequency)

# Create the array for the EMG signal and fill it with zeros
emg_signal0 = np.zeros(num_samples)

# Building the emg signal for train 0
emg_signal0 += convolve(train0, action_potentials[0], mode='full')[:num_samples]

# Plot the EMG signal 0, Figure 1
plt.figure(figsize=(10, 6))
plt.plot(time_vector, emg_signal0)
plt.title("EMG Signal (Convolution) for action potential train 0")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True)

# Define the time interval for zooming
zoom_start_time = 10.0  # Start time for zooming
zoom_end_time = 10.5  # End time for zooming
zoom_start_sample = int(zoom_start_time * sampling_frequency)
zoom_end_sample = int(zoom_end_time * sampling_frequency)

# Zoom in on the time interval 10-10.5 seconds, Figure 2
plt.figure(figsize=(10, 6))
plt.plot(time_vector[zoom_start_sample:zoom_end_sample], emg_signal0[zoom_start_sample:zoom_end_sample], label="Zoomed In")
plt.title("Zoomed In EMG Signal (Convolution) for action potential train 0")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True)

# Task 1e
# Create the array for the EMG signal and fill it with zeros
emg_signal = np.zeros(num_samples)

# Create EMG signal by convolving action potentials with firing times
for i in range(len(action_potentials)):
    motor_unit = action_potentials[i]
    # Convolve action potentials with the delta train and motor unit as inputs, and return the full
    # Linear convolution
    emg_signal += convolve(trains[i], motor_unit)[:num_samples]


# Total EMG signal on interval 10-10.5 seconds, Figure 3
plt.figure(figsize=(10, 6))
plt.plot(time_vector[zoom_start_sample:zoom_end_sample], emg_signal[zoom_start_sample:zoom_end_sample], label="Zoomed In")
plt.title("Zoomed In EMG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True)

# Task 2a
# Initialize duration and length for the Hanning window
window_duration = 1.0 
window_length = int(sampling_frequency * window_duration)

hanning_window = hann(window_length) # Creating the Hanning window

# Initialize a list to store the filtered signals
filtered_signals = []
# Iterate through the 8 binary vectors and filter them
for i in range(len(action_potentials)):
    # Create a train of delta functions at firing times
    train = trains[i]
    
    # Apply the Hanning window as a filter
    filtered_signal = convolve(train, hanning_window, mode='same') / np.sum(hanning_window)
    
    # Append the filtered signal to the list
    filtered_signals.append(filtered_signal)

# Create a time vector in seconds
time_vector = np.arange(0, duration, 1 / sampling_frequency)

# Plot the 8 filtered signals on the same graph, figure 4
plt.figure(figsize=(10, 6))
for i, filtered_signal in enumerate(filtered_signals):
    plt.plot(time_vector, filtered_signal, label=f"Motor Unit {i+1}")

plt.title("Filtered Binary Signals for 8 Motor Units")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)


# Task 2c
# Select the fourth binary vector (corresponding to the fourth motor unit)
binary_vector = firing_samples[0][4]

# Apply the Hanning window as a filter to the selected binary vector
filtered_signal4 = convolve(train4, hanning_window, mode='same') / np.sum(hanning_window)


def binary_data(firing_samples):
    max_value = firing_samples.max()  # Find the maximum value in the array
    binary_array = np.zeros(max_value + 1)  # Create a binary array filled with zeros
    binary_array[firing_samples] = 1  # Set the values at firing_samples to 1
    return binary_array


bindata = binary_data(binary_vector)
xaxis = np.arange(0, binary_vector[-1] + 1)
yaxis = np.array(bindata)

# Plot the fourth binary vector and its corresponding filtered version, figure 5
plt.figure(figsize=(10, 6),linewidth=0.1)

plt.subplot(2, 1, 1)
plt.step(xaxis, yaxis)
plt.xlim(0, 200000)  # Set the x-axis limits to zoom in
plt.title("Binary Vector for the fourth motor unit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_vector, filtered_signal4, label="Filtered Version")
plt.title("Filtered Version for the Fourth Motor Unit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()


# Task 2d
# Select the seventh binary vector (corresponding to the fourth motor unit)
binary_vector = firing_samples[0][7]

# Apply the Hanning window as a filter to the selected binary vector
filtered_signal7 = convolve(train7, hanning_window, mode='same') / np.sum(hanning_window)

bindata = binary_data(binary_vector)
xaxis = np.arange(0, binary_vector[-1] + 1)
yaxis = np.array(bindata)

# Plot the seventh binary vector and its corresponding filtered version, figure 6
plt.figure(figsize=(10, 6),linewidth=0.1)

plt.subplot(2, 1, 1)
plt.step(xaxis, yaxis)
plt.xlim(0, 200000)  # Set the x-axis limits to zoom in
plt.title("Binary Vector for the seventh motor unit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_vector, filtered_signal7, label="Filtered Version")
plt.title("Filtered Version for the seventh Motor Unit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()


plt.show()