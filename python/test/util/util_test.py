#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, math, time
import numpy as np
from swl.util import util as swl_util

def download_test():
	url = 'http://www.example.com/example.tar.zip'
	#url = 'http://www.example.com/example.tar.gz'
	#url = 'http://www.example.com/example.tar.bzip2'
	#url = 'http://www.example.com/example.tar.xz'
	output_dir_path = './uncompressed'

	print('Start downloading files...')
	start_time = time.time()
	download(url, output_dir_path)  # Not yet completely tested.
	print('End downloading files: {} secs.'.format(time.time() - start_time))

def generate_dataset(num_examples, is_output_augmented=False):
	if is_output_augmented:
		inputs = np.zeros((num_examples, 2, 2, 1))
		outputs = np.zeros((num_examples, 2, 2, 1))
	else:
		inputs = np.zeros((num_examples, 2, 2, 1))
		outputs = np.zeros((num_examples, 1))

	for idx in range(num_examples):
		inputs[idx] = idx
		outputs[idx] = idx
	return inputs, outputs

def generate_npy_file_dataset(dir_path, num_examples, is_output_augmented=False):
	inputs, outputs = generate_dataset(num_examples, is_output_augmented)

	swl_util.make_dir(dir_path)

	num_files = math.ceil(num_examples / 8500) #np.random.randint(1500, 2000)
	input_filename_format, output_filename_format = 'inputs_{}.npy', 'outputs_{}.npy'
	npy_file_csv_filename = 'npy_file_info.csv'
	input_filepaths, output_filepaths, _ = swl_util.save_data_to_npy_files(inputs, outputs, dir_path, num_files, input_filename_format, output_filename_format, npy_file_csv_filename, batch_axis=0, start_file_index=0, mode='w')

	return input_filepaths, output_filepaths

def generate_npz_file_dataset(dir_path, num_examples, is_output_augmented=False):
	inputs, outputs = generate_dataset(num_examples, is_output_augmented)

	swl_util.make_dir(dir_path)

	npz_filepaths = list()
	num_examples_in_a_file = 1000
	file_idx, batch_idx, start_idx = 0, 0, 0
	while True:
		npz_filepath = os.path.join(dir_path, 'dataset_{}.npz'.format(file_idx))
		end_idx = start_idx + 10000
		swl_util.save_data_to_npz_file(inputs[start_idx:end_idx], outputs[start_idx:end_idx], npz_filepath, num_examples_in_a_file, shuffle=False, batch_axis=0)

		npz_filepaths.append(npz_filepath)

		if end_idx >= num_examples:
			break;
		start_idx = end_idx
		file_idx += 1
	return npz_filepaths

def load_filepaths_from_npy_file_info_test():
	dir_path = './npy_dir'
	npy_file_csv_filename = 'npy_file_info.csv'
	num_examples = 115600
	is_output_augmented = False
	input_npy_filepaths, output_npy_filepaths = generate_npy_file_dataset(dir_path, num_examples, is_output_augmented)

	print('Generated npy input files =', input_npy_filepaths)
	print('Generated npy output files =', output_npy_filepaths)

	#--------------------
	loaded_input_npy_filepaths, loaded_output_npy_filepaths, loaded_example_counts = swl_util.load_filepaths_from_npy_file_info(os.path.join(dir_path, npy_file_csv_filename))

	print('Loaded npy input files =', loaded_input_npy_filepaths)
	print('Loaded npy output files =', loaded_output_npy_filepaths)
	print('Loaded example counts =', loaded_example_counts)

def load_data_from_npy_files_test():
	dir_path = './npy_dir'
	num_examples = 115600
	is_output_augmented = False
	input_npy_filepaths, output_npy_filepaths = generate_npy_file_dataset(dir_path, num_examples, is_output_augmented)

	print('Generated npy input files =', input_npy_filepaths)
	print('Generated npy output files =', output_npy_filepaths)

	#--------------------
	inputs, outputs = swl_util.load_data_from_npy_files(input_npy_filepaths, output_npy_filepaths, batch_axis=0)

	print('#loaded input exmaples =', len(inputs))
	print('#loaded output exmaples =', len(outputs))

def shuffle_data_in_npy_files_test():
	dir_path = './shuffle_data_test_dir'
	num_examples = 100000
	is_output_augmented = False
	input_filepaths, output_filepaths = generate_npy_file_dataset(dir_path, num_examples, is_output_augmented)
	input_filepaths, output_filepaths = np.array(input_filepaths), np.array(output_filepaths)

	#--------------------
	num_loaded_files = 9
	num_shuffles = 10
	tmp_dir_path_prefix = 'shuffle_tmp_dir_{}'
	shuffle_input_filename_format = 'shuffle_input_{}.npy'
	shuffle_output_filename_format = 'shuffle_output_{}.npy'
	shuffle_info_csv_filename = 'shuffle_info.csv'
	is_time_major = False

	swl_util.shuffle_data_in_npy_files(input_filepaths, output_filepaths, num_loaded_files, num_shuffles, tmp_dir_path_prefix, shuffle_input_filename_format, shuffle_output_filename_format, shuffle_info_csv_filename, is_time_major)

	#--------------------
	npy_file_csv_filepath = os.path.join(tmp_dir_path_prefix.format(num_shuffles - 1), shuffle_info_csv_filename)
	input_filepaths, output_filepaths, example_counts = swl_util.load_filepaths_from_npy_file_info(npy_file_csv_filepath)
	print('#examples = {}'.format(example_counts))

	#--------------------
	#inputs, outputs = swl_util.load_data_from_npy_files([input_filepaths[0]], [output_filepaths[0]], batch_axis=0)
	#print('Shapes = {}, {}'.format(inputs.shape, outputs.shape))
	##print('Data =\n{}\n{}'.format(inputs.T, outputs.T))

	inputs, outputs = swl_util.load_data_from_npy_files(input_filepaths, output_filepaths, batch_axis=0)
	print('Shapes = {}, {}'.format(inputs.shape, outputs.shape))
	#print('Data =\n{}\n{}'.format(inputs[:10].T, outputs[:10].T))
	print('Data =\n{}'.format(outputs[:100].T))

def load_data_from_npz_file_test():
	dir_path = './npz_dir'
	num_examples = 115600
	is_output_augmented = False
	npz_filepaths = generate_npz_file_dataset(dir_path, num_examples, is_output_augmented)

	print('Generated npz files =', npz_filepaths)

	#--------------------
	file_prefix = 'dataset_'
	file_suffix = ''
	loaded_npz_filepaths = swl_util.list_files_in_directory(dir_path, file_prefix, file_suffix, 'npz', is_recursive=False)

	print('Loaded npz files =', loaded_npz_filepaths)

	for npz_filepath in loaded_npz_filepaths:
		try:
			inputs, outputs = swl_util.load_data_from_npz_file(npz_filepath, batch_axis=0)
			print('{}: inputs = {}, outputs = {}.'.format(npz_filepath, inputs.shape, outputs.shape))
		except ValueError as ex:
			print('Failed to loaded an npz file:', npz_filepath)

def convert_currency_to_float_test():
	import locale

	try:
		locale.setlocale(locale.LC_ALL, 'ko_KR.UTF8')
		conv = locale.localeconv()
		currency_symbols = conv['currency_symbol']
		currency = '₩3,285,192'
		swl_util.convert_currency_to_float(currency, currency_symbols)
	except locale.Error as ex:
		print('Locale error, {}: {}.'.format(currency, ex))
	except ValueError as ex:
		print('ValueError, {}: {}.'.format(currency, ex))

	try:
		locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
		conv = locale.localeconv()
		currency_symbols = conv['currency_symbol']
		currency = '$6,150,593.22'
		swl_util.convert_currency_to_float(currency, currency_symbols)
	except locale.Error as ex:
		print('Locale error, {}: {}.'.format(currency, ex))
	except ValueError as ex:
		print('ValueError, {}: {}.'.format(currency, ex))

	try:
		locale.setlocale(locale.LC_ALL, 'fr_FR.UTF8')
		conv = locale.localeconv()
		currency_symbols = conv['currency_symbol']
		currency = '17,30 €'
		swl_util.convert_currency_to_float(currency, currency_symbols)
	except locale.Error as ex:
		print('Locale error, {}: {}.'.format(currency, ex))
	except ValueError as ex:
		print('ValueError, {}: {}.'.format(currency, ex))

	# Sets the locale for all categories to the user's default setting.
	locale.setlocale(locale.LC_ALL, '')

def main():
	#download_test()

	load_filepaths_from_npy_file_info_test()
	load_data_from_npy_files_test()
	#shuffle_data_in_npy_files_test()

	#load_data_from_npz_file_test()

	convert_currency_to_float_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
