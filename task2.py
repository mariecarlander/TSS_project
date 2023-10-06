import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import hann

# Load the provided NumPy array for firing samples
firing_samples = np.load('firing_samples.npy', allow_pickle=True)

# Constants
sampling_frequency = 10000  # Sampling frequency in Hz
duration = 20.0  # Signal duration in seconds
num_samples = int(duration * sampling_frequency)
window_duration = 1.0  # Duration of the Hanning window in seconds
window_length = int(sampling_frequency * window_duration)

# Initialize a list to store the filtered signals
filtered_signals = []

# Create a Hanning window
hanning_window = hann(window_length)
# Iterate through the 8 binary vectors and filter them
for binary_vector in firing_samples[0]:
    # Create a train of delta functions at firing times
    delta_train = np.zeros(num_samples)
    firing_times = binary_vector / sampling_frequency
    delta_indices = np.round(firing_times * sampling_frequency).astype(int)
    delta_train[delta_indices] = 1
    
    # Apply the Hanning window as a filter
    filtered_signal = convolve(delta_train, hanning_window, mode='same') / np.sum(hanning_window)
    
    # Append the filtered signal to the list
    filtered_signals.append(filtered_signal)

# Create a time vector in seconds
time_vector = np.arange(0, duration, 1 / sampling_frequency)

# Plot the 8 filtered signals on the same graph
plt.figure(figsize=(10, 6))
for i, filtered_signal in enumerate(filtered_signals):
    plt.plot(time_vector, filtered_signal, label=f"Motor Unit {i+1}")

plt.title("Filtered Binary Signals for 8 Motor Units")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
