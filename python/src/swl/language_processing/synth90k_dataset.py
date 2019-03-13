import os, time
import numpy as np
import cv2
import swl.util.util as swl_util
import swl.machine_vision.util as swl_cv_util

def load_synth90k_dataset(data_dir_path, subset_ratio=None):
	"""
	Inputs:
		data_dir_path (string): The directory path of Synth90k dataset.
		subset_ratio (float or None): The ratio of subset of data. 0.0 < subset_ratio <= 1.0.
	"""

	if subset_ratio is not None and subset_ratio <= 0.0 or subset_ratio > 1.0:
		raise ValueError("Invalid parameter 'subset_ratio': 0 < subset_ratio <= 1.0.")

	# filepath (filename: index_text_lexicon-idx) lexicon-idx.
	data_filepath_list = data_dir_path + '/annotation.txt'  # 8,919,273 files.
	train_data_filepath_list = data_dir_path + '/annotation_train.txt'  # 7,224,612 files.
	val_data_filepath_list = data_dir_path + '/annotation_val.txt'  # 802,734 files.
	test_data_filepath_list = data_dir_path + '/annotation_test.txt'  # 891,927 files.
	lexicon_filepath_list = data_dir_path + '/lexicon.txt'  # 88,172 words.

	print('Start loading lexicon...')
	start_time = time.time()
	with open(lexicon_filepath_list, 'r', encoding='UTF8') as fd:
		lexicon = [line.replace('\n', '') for line in fd.readlines()]
	print('\tLexicon size =', len(lexicon))
	print('End loading lexicon: {} secs.'.format(time.time() - start_time))

	max_word_len_in_lexicon = 0
	for lex in lexicon:
		if len(lex) > max_word_len_in_lexicon:
			max_word_len_in_lexicon = len(lex)
	print('Max length of words in lexicon =', max_word_len_in_lexicon)  # Max label length.

	label_characters = ''.join(sorted(set(''.join(lexicon))))
	print('Label characeters in lexicon (count = {}) = {}.'.format(len(label_characters), label_characters))

	#--------------------
	def process_line(line, data_dir_path):
		fpath, idx = line
		img = cv2.imread(os.path.join(data_dir_path, fpath))
		if img is None:
			print('Failed to load an image:', os.path.join(data_dir_path, fpath))
		return img, int(idx)

	print('Start loading train data...')
	start_time = time.time()
	with open(train_data_filepath_list, 'r', encoding='UTF8') as fd:
		lines = [line.replace('\n', '').split(' ') for line in fd.readlines()]
		if subset_ratio is not None:
			lines = swl_util.extract_subset_of_data(np.array(lines), subset_ratio)
		train_data = [process_line(line, data_dir_path) for line in lines]
	print('\tTrain data size =', len(train_data))
	print('End loading train data: {} secs.'.format(time.time() - start_time))

	print('Start loading validation data...')
	start_time = time.time()
	with open(val_data_filepath_list, 'r', encoding='UTF8') as fd:
		lines = [line.replace('\n', '').split(' ') for line in fd.readlines()]
		if subset_ratio is not None:
			lines = swl_util.extract_subset_of_data(np.array(lines), subset_ratio)
		val_data = [process_line(line, data_dir_path) for line in lines]
	print('\tValidation data size =', len(val_data))
	print('End loading validation data: {} secs.'.format(time.time() - start_time))

	print('Start loading test data...')
	start_time = time.time()
	with open(test_data_filepath_list, 'r', encoding='UTF8') as fd:
		lines = [line.replace('\n', '').split(' ') for line in fd.readlines()]
		if subset_ratio is not None:
			lines = swl_util.extract_subset_of_data(np.array(lines), subset_ratio)
		test_data = [process_line(line, data_dir_path) for line in lines]
	print('\tTest data size =', len(test_data))
	print('End loading test data: {} secs.'.format(time.time() - start_time))

	return lexicon, train_data, val_data, test_data

def save_synth90k_dataset_to_npy_files(data_dir_path, base_save_dir_path, image_height, image_width, image_channels, num_files_loaded_at_a_time, input_filename_format, output_filename_format, npy_file_csv_filename, data_processing_functor, subset_ratio=None):
	"""
	Inputs:
		data_dir_path (string): The directory path of Synth90k dataset.
		subset_ratio (float or None): The ratio of subset of data. (0.0, 1.0].
	"""

	if subset_ratio is not None and subset_ratio <= 0.0 or subset_ratio > 1.0:
		raise ValueError("Invalid parameter 'subset_ratio': 0 < subset_ratio <= 1.0.")

	# filepath(filename: index_text_lexicon-idx) lexicon-idx.
	all_data_filepath = data_dir_path + '/annotation.txt'  # 8,919,273 files.
	train_data_filepath = data_dir_path + '/annotation_train.txt'  # 7,224,612 files.
	val_data_filepath = data_dir_path + '/annotation_val.txt'  # 802,734 files.
	test_data_filepath = data_dir_path + '/annotation_test.txt'  # 891,927 files.
	lexicon_filepath = data_dir_path + '/lexicon.txt'  # 88,172 words.

	print('Start loading lexicon...')
	start_time = time.time()
	with open(lexicon_filepath, 'r', encoding='UTF8') as fd:
		lexicon = [line.replace('\n', '') for line in fd.readlines()]
	print('\tLexicon size =', len(lexicon))
	print('End loading lexicon: {} secs.'.format(time.time() - start_time))

	max_word_len_in_lexicon = 0
	for lex in lexicon:
		if len(lex) > max_word_len_in_lexicon:
			max_word_len_in_lexicon = len(lex)
	print('Max length of words in lexicon =', max_word_len_in_lexicon)  # Max label length.

	label_characters = ''.join(sorted(set(''.join(lexicon))))
	print('Label characeters in lexicon (count = {}) = {}.'.format(len(label_characters), label_characters))

	#--------------------
	learning_info_list = [('train', train_data_filepath), ('val', val_data_filepath), ('test', test_data_filepath)]

	for learning_phase, data_filepath in learning_info_list:
		save_dir_path = os.path.join(base_save_dir_path, learning_phase)

		print('Start loading {} data...'.format(learning_phase))
		start_time = time.time()
		with open(data_filepath, 'r', encoding='UTF8') as fd:
			lines = [line.replace('\n', '').split(' ') for line in fd.readlines()]
			if subset_ratio is not None:
				lines = swl_util.extract_subset_of_data(np.array(lines), subset_ratio)
			#file_label_dict = {os.path.join(data_dir_path, file): int(lbl) for (file, lbl) in lines}
			file_label_dict = {os.path.join(data_dir_path, file): lexicon[int(lbl)] for (file, lbl) in lines}
		print('\tDataset size =', len(file_label_dict))
		print('End loading {} data: {} secs.'.format(learning_phase, time.time() - start_time))

		print('Start saving {} data to npy files...'.format(learning_phase))
		start_time = time.time()
		swl_cv_util.save_images_to_npy_files(list(file_label_dict.keys()), list(file_label_dict.values()), image_height, image_width, image_channels, num_files_loaded_at_a_time, save_dir_path, input_filename_format, output_filename_format, npy_file_csv_filename, data_processing_functor)
		print('End saving {} data to npy files: {} secs.'.format(learning_phase, time.time() - start_time))
