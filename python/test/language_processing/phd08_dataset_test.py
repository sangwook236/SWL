#!/usr/bin/env python

import sys
sys.path.append('../../src')

import time
import swl.language_processing.phd08_dataset as phd08_dataset

# Usage:
#	1) Generate npy file from PHD08 dataset.
#		python phd08_to_npy.py --data_dir ./phd08 --width 32 --height 32 --batch_size 1 > phd08_conversion_results.txt
#		python phd08_to_npy.py --data_dir ./phd08 --one_hot --width 32 --height 32 --batch_size 1 > phd08_conversion_results.txt
#	2) python phd08_datset_test.py
#	3) Use phd08_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath)
def generate_npy_dataset_from_phd08_conversion_result():
	phd08_conversion_result_filepath = './phd08_conversion_results.txt'
	phd08_npy_dataset_info_filepath = './phd08_npy_dataset.csv'

	print('Start generating a PHD08 dataset info file...')
	start_time = time.time()
	phd08_dataset.generate_phd08_dataset_info(phd08_conversion_result_filepath, phd08_npy_dataset_info_filepath)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('End generating a PHD08 dataset info file.')

	print('Start loading PHD08 npy files...')
	start_time = time.time()
	letter_dict = phd08_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('End loading PHD08 npy files.')

	print('가:', len(letter_dict['가']), letter_dict['가'][0].shape)
	print('나:', len(letter_dict['나']), letter_dict['나'][0].shape)
	print('다:', len(letter_dict['다']), letter_dict['다'][0].shape)

def main():
	generate_npy_dataset_from_phd08_conversion_result()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
