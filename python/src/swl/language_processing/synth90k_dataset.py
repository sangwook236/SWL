import os, time
import numpy as np
import cv2
import swl.machine_learning.util as swl_ml_util
import swl.machine_vision.util as swl_cv_util

def preprocess_synth90k_dataset(inputs, outputs, *args, **kwargs):
	"""
	Inputs:
		inputs (numpy.array): images of size (samples, height, width, channels).
		outputs (numpy.array of strings): labels made up of alphabet (a~z) & digits (0~9) of size (samples,).
	"""

	if inputs is not None:
		#image_height, image_width, image_channel = 32, 128, 1

		# Preprocessing (normalization, standardization, etc.).
		inputs = inputs.astype(np.float32) / 255.0
		#inputs = (inputs - np.mean(inputs, axis=axis)) / np.std(inputs, axis=axis)
		#inputs = np.reshape(inputs, inputs.shape + (1,))

	if outputs is not None:
		max_label_len = 23  # Max length in lexicon.

		# Label: 0~9 + a~z + A~Z.
		#label_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
		# Label: 0~9 + a~z.
		label_characters = '0123456789abcdefghijklmnopqrstuvwxyz'

		SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		EOS = '<EOS>'  # All strings will end with the End-Of-String token.
		#extended_label_list = [SOS] + list(label_characters) + [EOS]
		extended_label_list = list(label_characters) + [EOS]
		#extended_label_list = list(label_characters)

		#int2char = extended_label_list
		char2int = {c:i for i, c in enumerate(extended_label_list)}

		num_labels = len(extended_label_list)
		num_classes = num_labels + 1  # extended labels + blank label.
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		blank_label = num_classes - 1
		label_eos_token = char2int[EOS]

		num_examples = len(outputs)
		#max_label_len = 0
		#for outp in outputs:
		#	if len(outp) > max_label_len:
		#		max_label_len = len(outp)

		outputs2 = np.full((num_examples, max_label_len), label_eos_token, dtype=np.int)
		for idx, outp in enumerate(outputs):
			outputs2[idx,:len(outp)] = [char2int[ch] for ch in outp]

		# One-hot encoding (num_examples, max_label_len) -> (num_examples, max_label_len, num_classes).
		#outputs2 = swl_ml_util.to_one_hot_encoding(outputs2, num_classes).astype(np.uint8)
		outputs2 = swl_ml_util.to_one_hot_encoding(outputs2, num_labels).astype(np.uint8)
	else:
		outputs2 = outputs

	return inputs, outputs2

def load_synth90k_dataset(data_dir_path):
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

	def process_line(line, data_dir_path):
		fpath, idx = line.replace('\n', '').split(' ')
		img = cv2.imread(os.path.join(data_dir_path, fpath))
		if img is None:
			print('Failed to load an image:', os.path.join(data_dir_path, fpath))
		return img, int(idx)

	print('Start loading train data...')
	start_time = time.time()
	with open(train_data_filepath_list, 'r', encoding='UTF8') as fd:
		train_data = [process_line(line, data_dir_path) for line in fd.readlines()]
	print('\tTrain data size =', len(train_data))
	print('End loading train data: {} secs.'.format(time.time() - start_time))

	print('Start loading validation data...')
	start_time =  time.time()
	with open(val_data_filepath_list, 'r', encoding='UTF8') as fd:
		val_data = [process_line(line, data_dir_path) for line in fd.readlines()]
	print('\tValidation data size =', len(val_data))
	print('End loading validation data: {} secs.'.format(time.time() - start_time))

	print('Start loading test data...')
	start_time =  time.time()
	with open(test_data_filepath_list, 'r', encoding='UTF8') as fd:
		test_data = [process_line(line, data_dir_path) for line in fd.readlines()]
	print('\tTest data size =', len(test_data))
	print('End loading test data: {} secs.'.format(time.time() - start_time))

	return lexicon, train_data, val_data, test_data

def save_synth90k_dataset_to_npy_files(data_dir_path, base_save_dir_path, image_height, image_width, image_channels, num_files_loaded_at_a_time):
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

	max_lexicon_len = 0
	for lex in lexicon:
		if len(lex) > max_lexicon_len:
			max_lexicon_len = len(lex)
	print('Max lexicon length =', max_lexicon_len)  # Max label length.

	#--------------------
	input_filename_format = 'input_{}.npy'
	output_filename_format = 'output_{}.npy'
	npy_file_csv_filename = 'npy_file_info.csv'
	data_processing_proc = preprocess_synth90k_dataset

	learning_info_list = [('train', train_data_filepath), ('val', val_data_filepath), ('test', test_data_filepath)]

	for learning_phase, data_filepath in learning_info_list:
		save_dir_path = os.path.join(base_save_dir_path, learning_phase)

		print('Start loading {} data...'.format(learning_phase))
		start_time = time.time()
		with open(data_filepath, 'r', encoding='UTF8') as fd:
			lines = [line.replace('\n', '').split(' ') for line in fd.readlines()]
			#file_label_dict = {os.path.join(data_dir_path, file): int(lbl) for (file, lbl) in lines}
			file_label_dict = {os.path.join(data_dir_path, file): lexicon[int(lbl)] for (file, lbl) in lines}
		print('\tDataset size =', len(file_label_dict))
		print('End loading {} data: {} secs.'.format(learning_phase, time.time() - start_time))

		print('Start saving {} data to npy files...'.format(learning_phase))
		start_time = time.time()
		swl_cv_util.save_images_to_npy_files(list(file_label_dict.keys()), list(file_label_dict.values()), image_height, image_width, image_channels, num_files_loaded_at_a_time, save_dir_path, input_filename_format, output_filename_format, npy_file_csv_filename, data_processing_proc)
		print('End saving {} data to npy files: {} secs.'.format(learning_phase, time.time() - start_time))
