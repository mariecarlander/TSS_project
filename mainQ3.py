import numpy 
from scipy.fft import fft
import matplotlib.pyplot as plt

# x_axis is a time vector
sample_frequency = 1024
x_axis = numpy.arange(0, 1, 1/sample_frequency)

#load in the provided emg signal
emg = numpy.load('/Users/conradolsson/Downloads/SSY081 project/TSS_project/f.npy')
emg = emg[0]

#build the interferance signal
frequency = 50
rad = 2 * numpy.pi * frequency
amplitude = 0.2
interferance = amplitude*numpy.sin(rad*x_axis)

#rebuild the signals using dft, with and without interferance
emg_dft = abs(fft(emg))
emg_interferance_dft = abs(fft(emg + interferance))

#plot the signals, green is without interferance and red is with interferance
plt.plot(emg_interferance_dft[:sample_frequency//2], color = 'red', label = 'with interferance')
plt.plot(emg_dft[:sample_frequency//2], color ='green', label = 'without interferance')         
plt.title("Emg signal with and without interferance")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True)
plt.show()