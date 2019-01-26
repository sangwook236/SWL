#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import swl

if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'
wav_dir_path = data_home_dir_path + '/failure_analysis/defect/knock_sound/Wave_75sample'

def main():
	wav_suffix = ''

	wavs, freqs = swl.audio_processing.util.load_wave_files(wav_dir_path, wav_suffix)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
