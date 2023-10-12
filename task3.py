import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

emg_signal = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/f.npy')
#example = np.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/fft_example.py')

frequency = 1024
interference = 50
amplitude = 0.2

duration = len(emg_signal) / frequency 
time_vector = np.linspace(0, duration, len(emg_signal), endpoint=False)

dft_emg_signal = np.fft.fft(emg_signal)


N = len(emg_signal)
frequency_axis = np.fft.fftfreq(N, 1 / frequency)[:N // 2]

print(emg_signal.shape)
print(emg_signal)


plt.figure(figsize=(12, 8))
plt.plot(frequency_axis, np.abs(dft_emg_signal[:N // 2]), color='red')
plt.title("DFT of Corrupted EMG Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True)
plt.show()
