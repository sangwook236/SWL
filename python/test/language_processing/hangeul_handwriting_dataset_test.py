#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, math, random, csv, time
import cv2
from swl.language_processing import hangeul_handwriting_dataset

# Usage:
#	1) Generate npy files from the PHD08 dataset.
#		python phd08_to_npy.py --data_dir ../phd08 --width 64 --height 64 --batch_size 1 > phd08_conversion_results.txt
#		python phd08_to_npy.py --data_dir ../phd08 --one_hot --width 64 --height 64 --batch_size 1 > phd08_conversion_results.txt
#			The size 32 x 32 is too small.
#			The size 64 x 64 is better, but still small.
#	2) Generate an info file for the npy files generated from the PHD08 dataset.
#		Use hangeul_handwriting_dataset.generate_phd08_dataset_info().
#	3) Load the npy files.
#		Refer to hangeul_handwriting_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath).
def generate_npy_dataset_from_phd08_conversion_result():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'

	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset'

	#phd08_conversion_result_filepath = data_dir_path + '/phd08_conversion_results_32x32.txt'
	phd08_conversion_result_filepath = data_dir_path + '/phd08_conversion_results_64x64.txt'
	#phd08_conversion_result_filepath = data_dir_path + '/phd08_conversion_results_100x100.txt'
	phd08_npy_dataset_info_filepath = data_dir_path + '/phd08_npy_dataset.csv'

	print('Start generating a PHD08 dataset info file...')
	start_time = time.time()
	hangeul_handwriting_dataset.generate_phd08_dataset_info(data_dir_path, phd08_conversion_result_filepath, phd08_npy_dataset_info_filepath)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('End generating a PHD08 dataset info file.')
	return

	print('Start loading PHD08 npy files...')
	start_time = time.time()
	letter_dict = hangeul_handwriting_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('End loading PHD08 npy files.')

	print('가:', len(letter_dict['가']), letter_dict['가'][0].shape)
	print('나:', len(letter_dict['나']), letter_dict['나'][0].shape)
	print('다:', len(letter_dict['다']), letter_dict['다'][0].shape)

def visualize_dataset(dataset_dict):
	for label, images in dataset_dict.items():
		for idx, img in enumerate(images):
			print('Label =', label)
			cv2.imshow('Image', img)
			cv2.waitKey(0)
			if idx >= 5:
				break

def load_phd08_dataset_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'

	#data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/phd08_npy_results_32x32'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/phd08_npy_results_64x64'
	#data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/phd08_npy_results_100x100'

	# Create the file, phd08_npy_dataset.csv.
	#	REF [function] >> generate_npy_dataset_from_phd08_conversion_result().
	phd08_npy_dataset_info_filepath = data_dir_path + '/phd08_npy_dataset.csv'

	print('Start loading PHD08 dataset...')
	start_time = time.time()
	phd08_dict = hangeul_handwriting_dataset.load_phd08_dataset(data_dir_path, phd08_npy_dataset_info_filepath)
	print('End loading PHD08 dataset: {} secs.'.format(time.time() - start_time))

	if True:
		visualize_dataset(phd08_dict)

def load_handb_dataset_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'

	train_data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/HanDB_train_npy'
	test_data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/HanDB_test_npy'

	train_dataset_info_filepath = train_data_dir_path + '/text_dataset.csv'
	test_dataset_info_filepath = test_data_dir_path + '/text_dataset.csv'

	print('Start loading HanDB train dataset...')
	start_time = time.time()
	handb_train_dict = hangeul_handwriting_dataset.load_handb_dataset(train_data_dir_path, train_dataset_info_filepath)
	print('End loading HanDB train dataset: {} secs.'.format(time.time() - start_time))

	print('Start loading HanDB test dataset...')
	start_time = time.time()
	handb_test_dict = hangeul_handwriting_dataset.load_handb_dataset(test_data_dir_path, test_dataset_info_filepath)
	print('End loading HanDB test dataset: {} secs.'.format(time.time() - start_time))

	if True:
		visualize_dataset(handb_train_dict)
		visualize_dataset(handb_test_dict)

def load_pe92_dataset_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'

	train_data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/PE92_train_npy'
	test_data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/PE92_test_npy'

	train_dataset_info_filepath = train_data_dir_path + '/text_dataset.csv'
	test_dataset_info_filepath = test_data_dir_path + '/text_dataset.csv'

	print('Start loading PE92 train dataset...')
	start_time = time.time()
	pe92_train_dict = hangeul_handwriting_dataset.load_handb_dataset(train_data_dir_path, train_dataset_info_filepath)
	print('End loading PE92 train dataset: {} secs.'.format(time.time() - start_time))

	print('Start loading PE92 test dataset...')
	start_time = time.time()
	pe92_test_dict = hangeul_handwriting_dataset.load_handb_dataset(test_data_dir_path, test_dataset_info_filepath)
	print('End loading PE92 test dataset: {} secs.'.format(time.time() - start_time))

	if True:
		visualize_dataset(pe92_train_dict)
		visualize_dataset(pe92_test_dict)

def load_seri_dataset_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'

	train_data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/SERI_Train_npy'
	test_data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/hangeul/0_download/dataset/SERI_Test_npy'

	train_dataset_info_filepath = train_data_dir_path + '/text_dataset.csv'
	test_dataset_info_filepath = test_data_dir_path + '/text_dataset.csv'

	print('Start loading SERI train dataset...')
	start_time = time.time()
	seri_train_dict = hangeul_handwriting_dataset.load_handb_dataset(train_data_dir_path, train_dataset_info_filepath)
	print('End loading SERI train dataset: {} secs.'.format(time.time() - start_time))

	print('Start loading SERI test dataset...')
	start_time = time.time()
	seri_test_dict = hangeul_handwriting_dataset.load_handb_dataset(test_data_dir_path, test_dataset_info_filepath)
	print('End loading SERI test dataset: {} secs.'.format(time.time() - start_time))

	if True:
		visualize_dataset(seri_train_dict)
		visualize_dataset(seri_test_dict)

def main():
	#generate_npy_dataset_from_phd08_conversion_result()

	#load_phd08_dataset_test()

	#load_handb_dataset_test()  # The dataset can not be loaded at a time due to memory error.
	#load_pe92_dataset_test()  # The dataset can not be loaded at a time due to memory error.
	load_seri_dataset_test()  # The dataset can not be loaded at a time due to memory error.

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
