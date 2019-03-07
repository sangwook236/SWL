#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os
from swl.language_processing import synth90k_dataset

def main():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mjsynth/mnt/ramdisk/max/90kDICT32px'

	#lexicon, train_data, val_data, test_data = synth90k_dataset.load_synth90k_dataset(data_dir_path)  # Error: out-of-memory.

	base_save_dir_path = './synth90k_npy'  # base_save_dir_path/train, base_save_dir_path/val, base_save_dir_path/test.
	image_height, image_width, image_channels = 32, 128, 1
	num_files_loaded_at_a_time = 10000
	synth90k_dataset.save_synth90k_dataset_to_npy_files(data_dir_path, base_save_dir_path, image_height, image_width, image_channels, num_files_loaded_at_a_time)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
