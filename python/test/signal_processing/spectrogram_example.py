#!/usr/bin/env python

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
def spectrogram_test():
	#--------------------------------------------------------------------
	# Use scipy.signal.

	# Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.
	fs = 10e3
	N = 1e5
	amp = 2 * np.sqrt(2)
	noise_power = 0.01 * fs / 2
	time = np.arange(N) / float(fs)
	mod = 500 * np.cos(2 * np.pi * 0.25 * time)
	carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
	noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
	noise *= np.exp(-time / 5)
	x = carrier + noise

	# Compute and plot the spectrogram.
	f, t, Sxx = signal.spectrogram(x, fs)

	plt.figure();
	plt.pcolormesh(t, f, Sxx)
	#plt.pcolormesh(t, f, 20 * np.log10(Sxx), cmap=plt.get_cmap('viridis'))
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

	#--------------------------------------------------------------------
	# Use matplotlib_plt_specgram_test().

	plt.figure();
	Sxx, freqs, times, axesImg = plt.specgram(x, Fs=fs)

def main():
	spectrogram_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
