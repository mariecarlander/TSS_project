import numpy as np
import matplotlib.pyplot as plt

# Load the provided NumPy array for the EMG signal
emg_signal = np.load('f.npy')

# Constants
sampling_frequency = 1024  # Sampling frequency in Hz
duration = len(emg_signal) / sampling_frequency  # Signal duration in seconds

# Simulate power line interference at 50 Hz with a peak-to-peak amplitude of 0.2
power_line_frequency = 50  # Frequency of power line interference in Hz
amplitude = 0.2
time_vector = np.arange(0, duration, 1 / sampling_frequency)
power_line_interference = amplitude * np.sin(2 * np.pi * power_line_frequency * time_vector)

# Combine the EMG signal with the power line interference
emg_with_interference = emg_signal + power_line_interference

# Perform the DFT on both signals
dft_emg_with_interference = np.abs(np.fft.fft(emg_with_interference))[:len(emg_with_interference)//2]
dft_emg_signal = np.abs(np.fft.fft(emg_signal))[:len(emg_signal)//2]

# Frequency axis for plotting
frequency_axis = np.fft.fftfreq(len(emg_with_interference), 1 / sampling_frequency)[:len(emg_with_interference)//2]

# Plot the absolute value of the DFT of the signal with interference and the interference-free signal
plt.figure(figsize=(10, 6))
plt.plot(frequency_axis, dft_emg_with_interference, label="Signal with Interference", color='blue')
plt.plot(frequency_axis, dft_emg_signal, label="Interference-Free Signal", color='red')
plt.title("DFT of EMG Signal with Power Line Interference")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (A.U.)")
plt.legend()
plt.grid(True)
plt.xlim(0, 100)  # Limit the x-axis to show frequencies up to 100 Hz
plt.show()
