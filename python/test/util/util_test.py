#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import os
import numpy as np
from swl.util import util as swl_util

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

def generate_file_dataset(dir_path, num_examples, is_output_augmented=False):
	inputs, outputs = generate_dataset(num_examples, is_output_augmented)

	swl_util.make_dir(dir_path)

	input_filepaths, output_filepaths = list(), list()
	idx, start_idx = 0, 0
	while True:
		end_idx = start_idx + 1700 #np.random.randint(1500, 2000)
		batch_inputs, batch_outputs = inputs[start_idx:end_idx], outputs[start_idx:end_idx]
		input_filepath, output_filepath = os.path.join(dir_path, 'inputs_{}.npy'.format(idx)), os.path.join(dir_path, 'outputs_{}.npy'.format(idx))
		np.save(input_filepath, batch_inputs)
		np.save(output_filepath, batch_outputs)
		input_filepaths.append(input_filepath)
		output_filepaths.append(output_filepath)
		if end_idx >= num_examples:
			break;
		start_idx = end_idx
		idx += 1
	return input_filepaths, output_filepaths

def shuffle_data_in_npy_files_test():
	dir_path = 'shuffle_data_test_dir'
	num_examples = 100000
	is_output_augmented = False
	input_filepaths, output_filepaths = generate_file_dataset(dir_path, num_examples, is_output_augmented)
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

def main():
	shuffle_data_in_npy_files_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
