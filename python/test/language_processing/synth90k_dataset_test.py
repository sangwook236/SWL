#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os
from swl.language_processing import synth90k_dataset

class Synth90kLabelConverter(object):
	def __init__(self):
		self._max_label_len = 23  # Max length of words in lexicon.

		# Label: 0~9 + a~z + A~Z.
		#label_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
		# Label: 0~9 + a~z.
		label_characters = '0123456789abcdefghijklmnopqrstuvwxyz'

		SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		EOS = '<EOS>'  # All strings will end with the End-Of-String token.
		#extended_label_list = [SOS] + list(label_characters) + [EOS]
		extended_label_list = list(label_characters) + [EOS]
		#extended_label_list = list(label_characters)

		#self._int2char = extended_label_list
		self._char2int = {c:i for i, c in enumerate(extended_label_list)}

		self._num_labels = len(extended_label_list)
		self._num_classes = self._num_labels + 1  # extended labels + blank label.
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._blank_label = self._num_classes - 1
		self._label_eos_token = self._char2int[EOS]

	def __call__(self, inputs, outputs, *args, **kwargs):
		"""
		Inputs:
			inputs (numpy.array): images of size (samples, height, width).
			outputs (numpy.array of strings): labels made up of digits (0~9) & alphabet (a~z) of size (samples,).
		Outputs:
			inputs (numpy.array): unchanged.
			outputs (numpy.array): labels of size (samples, max_label_length) and type int.
		"""

		if inputs is not None:
			#image_height, image_width, image_channel = 32, 128, 1

			# Preprocessing (normalization, standardization, etc.).
			#inputs = inputs.astype(np.float32) / 255.0
			#inputs = (inputs - np.mean(inputs, axis=axis)) / np.std(inputs, axis=axis)
			#inputs = np.reshape(inputs, inputs.shape + (1,))
			pass

		if outputs is not None:
			num_examples = len(outputs)
			#self._max_label_len = 0
			#for outp in outputs:
			#	if len(outp) > self._max_label_len:
			#		self._max_label_len = len(outp)

			outputs2 = np.full((num_examples, self._max_label_len), self._label_eos_token, dtype=np.int)
			for idx, outp in enumerate(outputs):
				outputs2[idx,:len(outp)] = [self._char2int[ch] for ch in outp]

			# One-hot encoding (num_examples, max_label_len) -> (num_examples, max_label_len, num_classes).
			#outputs2 = swl_ml_util.to_one_hot_encoding(outputs2, self._num_classes).astype(np.int)
			#outputs2 = swl_ml_util.to_one_hot_encoding(outputs2, self._num_labels).astype(np.int)
		else:
			outputs2 = outputs

		return inputs, outputs2

def main():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mjsynth/mnt/ramdisk/max/90kDICT32px'

	#lexicon, train_data, val_data, test_data = synth90k_dataset.load_synth90k_dataset(data_dir_path)  # Error: out-of-memory.

	#--------------------
	base_save_dir_path = './synth90k_npy'  # base_save_dir_path/train, base_save_dir_path/val, base_save_dir_path/test.
	image_height, image_width, image_channels = 32, 128, 1
	num_files_loaded_at_a_time = 10000
	input_filename_format = 'input_{}.npy'
	output_filename_format = 'output_{}.npy'
	npy_file_csv_filename = 'npy_file_info.csv'
	data_processing_functor = Synth90kLabelConverter()

	synth90k_dataset.save_synth90k_dataset_to_npy_files(data_dir_path, base_save_dir_path, image_height, image_width, image_channels, num_files_loaded_at_a_time, input_filename_format, output_filename_format, npy_file_csv_filename, data_processing_functor)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
