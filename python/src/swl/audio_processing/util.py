import numpy as np
import scipy.io.wavfile  # For reading the .wav file.
import os, re

def load_wave_files(dir_path, file_suffix):
	wavs = []
	freqs = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.wav$', filename):
					filepath = os.path.join(root, filename)
					# fs: sampling frequency.
					# signal: the numpy 2D array where the data of the wav file is written.
					[fs, wav] = scipy.io.wavfile.read(filepath)
					wavs.append(wav)
					freqs.append(fs)
			break  # Do not include subdirectories.
	return wavs, freqs
