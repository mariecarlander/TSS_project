import numpy as np
import matplotlib.pyplot as plt

emg_signal = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/f.npy')

frequency = 1024 
interference = 50
peak_amplitude = 0.2

duration = len(emg_signal) / frequency 
time_vector = np.linspace(0, duration, len(emg_signal), endpoint=False)

dft_emg_signal = np.fft.fft(emg_signal)


plt.figure(figsize=(10, 6),linewidth=0.1)
plt.plot(time_vector, dft_emg_signal)
plt.title("Generated EMG Signal (Convolution)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()