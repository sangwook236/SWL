#!/usr/bin/env python

import os
import numpy as np
import swl.machine_vision.util as swl_cv_util

def generate_npy_files():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mjsynth/mnt/ramdisk/max/90kDICT32px'

	# filepath(filename: index_text_lexicon-idx) lexicon-idx.
	all_data_filepath = data_dir_path + '/annotation.txt'  # 8,919,273 files.
	train_data_filepath = data_dir_path + '/annotation_train.txt'  # 7,224,612 files.
	val_data_filepath = data_dir_path + '/annotation_val.txt'  # 802,734 files.
	test_data_filepath = data_dir_path + '/annotation_test.txt'  # 891,927 files.
	lexicon_filepath = data_dir_path + '/lexicon.txt'  # 88,172 words.

	print('Start loading lexicon...')
	with open(lexicon_filepath, 'r', encoding='UTF8') as fd:
		lexicon = [line.replace('\n', '') for line in fd.readlines()]
		print('\tLexicon size =', len(lexicon))
	print('End loading lexicon.')

	#--------------------
	image_height, image_width = 32, 128
	num_files_loaded_at_a_time = 100000

	input_filename_format = 'input_{}.npy'
	output_filename_format = 'output_{}.npy'
	npy_file_csv_filename = 'npy_file_info.csv'
	base_save_dir_path = './synth90k_npy'

	learning_info_list = [('train', train_data_filepath), ('val', val_data_filepath), ('test', test_data_filepath)]

	for learning_phase, data_filepath in learning_info_list:
		save_dir_path = os.path.join(base_save_dir_path, learning_phase)

		print('Start loading {} data...'.format(learning_phase))
		with open(data_filepath, 'r', encoding='UTF8') as fd:
			lines = [line.replace('\n', '').split(' ') for line in fd.readlines()]
			#file_label_dict = {os.path.join(data_dir_path, file): int(lbl) for (file, lbl) in lines}
			file_label_dict = {os.path.join(data_dir_path, file): lexicon[int(lbl)] for (file, lbl) in lines}
			print('\tDataset size =', len(file_label_dict))
		print('End loading {} data.'.format(learning_phase))

		print('Start saving {} data to npy files...'.format(learning_phase))
		swl_cv_util.save_images_to_npy_files(list(file_label_dict.keys()), list(file_label_dict.values()), image_height, image_width, num_files_loaded_at_a_time, save_dir_path, input_filename_format, output_filename_format, npy_file_csv_filename)
		print('End saving {} data to npy files.'.format(learning_phase))

def main():
	generate_npy_files()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
