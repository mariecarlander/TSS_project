import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import hann

# Load in the numpy arrays for the firing samples provided in the task 
firing_samples = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/firing_samples.npy', allow_pickle=True)

# Initialize the constants 
sampling_frequency = 10000  # Sampling frequency in Hz
duration = 20.0  # Signal duration in seconds
num_samples = int(duration * sampling_frequency)
window_duration = 1.0  # Duration of the Hanning window in seconds
window_length = int(sampling_frequency * window_duration)

# Select the fourth binary vector (corresponding to the fourth motor unit)
binary_vector = firing_samples[0][3]
print(binary_vector)
# Create a Hanning window
hanning_window = hann(window_length)

# Create a train of delta functions at firing times for the selected binary vector
delta_train = np.zeros(num_samples)
firing_times = binary_vector / sampling_frequency
delta_indices = np.round(firing_times * sampling_frequency).astype(int)
delta_train[delta_indices] = 1

# Apply the Hanning window as a filter to the selected binary vector
filtered_signal = convolve(delta_train, hanning_window, mode='same') / np.sum(hanning_window)

# Create a time vector in seconds
time_vector = np.arange(0, duration, 1 / sampling_frequency)

# Plot the fourth binary vector and its corresponding filtered version
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_vector[:len(binary_vector)], binary_vector, label="Binary Vector")
plt.title("Binary Vector for the Fourth Motor Unit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_vector, filtered_signal, label="Filtered Version")
plt.title("Filtered Version for the Fourth Motor Unit")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
