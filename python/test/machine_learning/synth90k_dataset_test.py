#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os
from swl.machine_learning import synth90k_dataset

def main():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mjsynth/mnt/ramdisk/max/90kDICT32px'

	lexicon, train_data, val_data, test_data = synth90k_dataset.load_synth90k_dataset(data_dir_path)

	#base_save_dir_path = './synth90k_npy'  # base_save_dir_path/train, base_save_dir_path/val, base_save_dir_path/test.
	#synth90k_dataset.save_synth90k_to_npy_files(data_dir_path, base_save_dir_path)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
