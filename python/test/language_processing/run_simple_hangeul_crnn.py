#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, time, datetime, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import cv2
import swl.machine_learning.util as swl_ml_util
import text_generation_util as tg_util

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count):
		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio')

		#--------------------
		hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
		with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
			#hangeul_charset = fd.readlines()  # A string.
			#hangeul_charset = fd.read().strip('\n')  # A list of strings.
			#hangeul_charset = fd.read().splitlines()  # A list of strings.
			hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.

		hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

		alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		digit_charset = '0123456789'
		symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		# There are words of Unicode Hangeul letters besides KS X 1001.
		labels_set = set(list(hangeul_charset + hangeul_jamo_charset))
		for f in os.listdir(data_dir_path):
			label_str = f.split('_')[0]
			labels_set = labels_set.union(list(label_str))
		self._labels = list(labels_set)
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		# Load data.
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		examples = self._load_data(data_dir_path, image_height, image_width, image_channel, max_char_count)
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

		num_examples = len(examples)
		test_offset = round(train_test_ratio * num_examples)
		self._train_data, self._test_data = examples[:test_offset], examples[test_offset:]

	@property
	def num_classes(self):
		return self._num_classes

	def create_train_batch_generator(self, batch_size, shuffle=True):
		return MyDataset._create_batch_generator(self._train_data, batch_size, shuffle)

	def create_test_batch_generator(self, batch_size, shuffle=False):
		return MyDataset._create_batch_generator(self._test_data, batch_size, shuffle)

	# String label -> integer label.
	def encode_label(self, label_str):
		try:
			return [self._labels.index(ch) for ch in label_str]
		except Exception as ex:
			print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
			raise

	# Integer label -> string label.
	def decode_label(self, label_int, default_value=-1):
		try:
			return ''.join([self._labels[id] for id in label_int if id != default_value])
		except Exception as ex:
			print('[SWL] Error: Failed to decode a label: {}.'.format(label_int))
			raise

	# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
	def _load_data(self, data_dir_path, image_height, image_width, image_channel, max_char_count):
		examples = list()
		for f in os.listdir(data_dir_path):
			label_str = f.split('_')[0]
			if len(label_str) > max_char_count:
				continue
			image = MyDataset._resize_image(os.path.join(data_dir_path, f), image_height, image_width)
			image, label_int = MyDataset._preprocess_data(image, self.encode_label(label_str))
			examples.append((image, label_str, label_int))

		return examples

	@staticmethod
	def _create_batch_generator(data, batch_size, shuffle):
		images, labels_str, labels_int = zip(*data)

		# (examples, height, width) -> (examples, width, height).
		images = np.swapaxes(np.array(images), 1, 2)
		images = np.reshape(images, images.shape + (1,))  # Image channel = 1.
		labels_str = np.reshape(np.array(labels_str), (-1))
		labels_int = np.reshape(np.array(labels_int), (-1))

		num_examples = len(images)
		if len(labels_str) != num_examples or len(labels_int) != num_examples:
			raise ValueError('[SWL] Error: Invalid data length: {} != {} != {}'.format(num_examples, len(labels_str), len(labels_int)))
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('[SWL] Error: Invalid batch size: {}'.format(batch_size))

		indices = np.arange(num_examples)
		if shuffle:
			np.random.shuffle(indices)

		start_idx = 0
		while True:
			end_idx = start_idx + batch_size
			batch_indices = indices[start_idx:end_idx]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				batch_data1, batch_data2, batch_data3 = images[batch_indices], labels_str[batch_indices], labels_int[batch_indices]
				batch_data3 = swl_ml_util.sequences_to_sparse(batch_data3, dtype=np.int32)  # Sparse tensor.
				if batch_data1.size > 0 and batch_data2.size > 0 and batch_data3[2][0] > 0:  # If batch_data1, batch_data2, and batch_data3 are non-empty.
					yield (batch_data1, batch_data2, batch_data3), batch_indices.size
				else:
					yield (None, None, None), 0
			else:
				yield (None, None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	@staticmethod
	def _preprocess_data(inputs, outputs):
		if inputs is not None:
			# Contrast limited adaptive histogram equalization (CLAHE).
			#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			#inputs = np.array([clahe.apply(inp) for inp in inputs])

			# TODO [check] >> Preprocessing has influence on recognition rate.

			# Normalization, standardization, etc.
			#inputs = inputs.astype(np.float32)

			if False:
				inputs = preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
			elif False:
				# NOTE [info] >> Not good.
				inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
			elif False:
				# NOTE [info] >> Not bad.
				in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
				out_min, out_max = 0, 1 #-1, 1
				inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
			elif False:
				inputs /= 255.0  # Normalization.

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs

	@staticmethod
	def _resize_image(image_filepath, image_height, image_width):
		img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
		r, c = img.shape
		if c >= image_width:
			return cv2.resize(img, (image_width, image_height))
		else:
			img_zeropadded = np.zeros((image_height, image_width))
			ratio = image_height / r
			img = cv2.resize(img, (int(c * ratio), image_height))
			width = min(image_width, img.shape[1])
			img_zeropadded[:, 0:width] = img[:, 0:width]
			return img_zeropadded

class MyOnlineSyntheticDataset(object):
	def __init__(self, image_height, image_width, image_channel, width_downsample_factor, num_classes, eos_token_label, blank_label):
		hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
		with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
			#hangeul_charset = fd.readlines()  # A string.
			#hangeul_charset = fd.read().strip('\n')  # A list of strings.
			#hangeul_charset = fd.read().splitlines()  # A list of strings.
			hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
		alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		digit_charset = '0123456789'
		symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		self._charset = hangeul_charset
		#self._charset = hangeul_charset + hangeul_jamo_charset
		#self._charset = hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset
		#self._charset = hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset + symbol_charset
		print('Charset =', len(self._charset), self._charset)

		#--------------------
		characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
		#characterTransformer = tg_util.IdentityTransformer()
		characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
		self._textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

		#--------------------
		self._num_char_repetitions_for_train, self._num_char_repetitions_for_test = 250, 5
		self._min_char_count, self._max_char_count = 2, 10
		self._min_font_size, self._max_font_size = 15, 30
		self._min_char_space_ratio, self._max_char_space_ratio = 0.8, 2

		#self._font_color = (255, 255, 255)
		#self._font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
		self._font_color = None  # Uses random font colors.
		#self._bg_color = (0, 0, 0)
		self._bg_color = None  # Uses random colors.

	def create_train_batch_generator(self, batch_size):
		return self._create_batch_generator(self._num_char_repetitions_for_train, batch_size)

	def create_test_batch_generator(self, batch_size):
		return self._create_batch_generator(self._num_char_repetitions_for_test, batch_size)

	def _create_batch_generator(self, num_char_repetitions, batch_size):
		word_set = tg_util.generate_repetitive_word_set(num_char_repetitions, self._charset, self._min_char_count, self._max_char_count)
		return tg_util.generate_text_lines(word_set, self._textGenerator, (self._in_font_size, self._max_font_size), (self._min_char_space_ratio, self._max_char_space_ratio), batch_size, self._font_color, self._bg_color)

		# FIXME [implement] >> Creates another generator from tg_util.generate_text_lines().

class MyFileBasedSyntheticDataset(object):
	def __init__(self, image_height, image_width, image_channel, width_downsample_factor, num_classes, eos_token_label, blank_label):
		self._eos_token_label = eos_token_label
		self._model_output_time_steps = image_width // width_downsample_factor

		print('Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, self._test_images, self._test_labels = MyFileBasedSyntheticDataset.load_data_from_json(image_height, image_width, image_channel, num_classes, eos_token_label, blank_label)
		print('End loading dataset: {} secs.'.format(time.time() - start_time))

		#max_label_len = max(self._train_labels.shape[-1], self._test_labels.shape[-1])
		#train_label_lengths, test_label_lengths = np.full((self._train_labels.shape[0],), self._train_labels.shape[-1]), np.full((self._test_labels.shape[0],), self._test_labels.shape[-1])

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_labels.shape, self._train_labels.dtype, np.min(self._train_labels), np.max(self._train_labels)))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_labels.shape, self._test_labels.dtype, np.min(self._test_labels), np.max(self._test_labels)))

	def create_train_batch_generator(self, batch_size, shuffle=True):
		# FIXME [improve] >> Stupid implementation.
		if True:
			model_output_lengths = np.full((self._train_images.shape[0],), self._model_output_time_steps)
		else:
			model_output_lengths = np.full((self._train_images.shape[0],), self._model_output_time_steps - 2)  # See MyTensorFlowModel.get_loss().

		return MyFileBasedSyntheticDataset._create_batch_generator(self._train_images, self._train_labels, model_output_lengths, batch_size, shuffle, self._eos_token_label)

	def create_test_batch_generator(self, batch_size, shuffle=False):
		# FIXME [improve] >> Stupid implementation.
		if True:
			model_output_lengths = np.full((self._test_images.shape[0],), self._model_output_time_steps)
		else:
			model_output_lengths = np.full((self._test_images.shape[0],), self._model_output_time_steps - 2)  # See MyTensorFlowModel.get_loss().

		return MyFileBasedSyntheticDataset._create_batch_generator(self._test_images, self._test_labels, model_output_lengths, batch_size, shuffle, self._eos_token_label)

	@staticmethod
	def _create_batch_generator(data1, data2, data3, batch_size, shuffle, eos_token_label):
		num_examples = len(data1)
		if len(data2) != num_examples or len(data3) != num_examples:
			raise ValueError('Invalid data length: {} != {} != {}'.format(num_examples, len(data2), len(data3)))
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		indices = np.arange(num_examples)
		if shuffle:
			np.random.shuffle(indices)

		start_idx = 0
		while True:
			end_idx = start_idx + batch_size
			batch_indices = indices[start_idx:end_idx]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				batch_data1, batch_data2, batch_data3 = data1[batch_indices], data2[batch_indices], data3[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0 and batch_data3.size > 0:  # If batch_data1, batch_data2, and batch_data3 are non-empty.
					batch_sparse_data2 = tf.SparseTensorValue(*swl_ml_util.dense_to_sparse(batch_data2, default_value=eos_token_label, dtype=np.int32))
					yield (batch_data1, batch_sparse_data2, batch_data3), batch_indices.size
				else:
					yield (None, None, None), 0
			else:
				yield (None, None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	@staticmethod
	def load_data_from_json(image_height, image_width, image_channel, num_classes, eos_token, blank_label):
		train_dataset_json_filepath = './text_train_dataset_tmp/text_dataset.json'
		test_dataset_json_filepath = './text_test_dataset_tmp/text_dataset.json'
	
		print('Start loading train dataset to numpy...')
		start_time = time.time()
		train_data, train_labels = MyFileBasedSyntheticDataset.text_dataset_to_numpy(train_dataset_json_filepath, image_height, image_width, image_channel, eos_token, blank_label)
		print('End loading train dataset: {} secs.'.format(time.time() - start_time))
		print('Start loading test dataset to numpy...')
		start_time = time.time()
		test_data, test_labels = MyFileBasedSyntheticDataset.text_dataset_to_numpy(test_dataset_json_filepath, image_height, image_width, image_channel, eos_token, blank_label)
		print('End loading test dataset: {} secs.'.format(time.time() - start_time))

		# Preprocessing.
		train_data = (train_data.astype(np.float32) / 255.0) * 2 - 1  # [-1, 1].
		#train_labels = tf.keras.utils.to_categorical(train_labels, num_classes, np.int16)
		train_labels = train_labels.astype(np.int16)
		test_data = (test_data.astype(np.float32) / 255.0) * 2 - 1  # [-1, 1].
		#test_labels = tf.keras.utils.to_categorical(test_labels, num_classes, np.int16)
		test_labels = test_labels.astype(np.int16)

		# (samples, height, width, channels) -> (samples, width, height, channels).
		train_data = train_data.transpose((0, 2, 1, 3))
		test_data = test_data.transpose((0, 2, 1, 3))

		return train_data, train_labels, test_data, test_labels

	@staticmethod
	def text_dataset_to_numpy(dataset_json_filepath, image_height, image_width, image_channel, eos_token, blank_label):
		with open(dataset_json_filepath, 'r', encoding='UTF8') as json_file:
			dataset = json.load(json_file)

		"""
		print(dataset['charset'])
		for datum in dataset['data']:
			print('file =', datum['file'])
			print('size =', datum['size'])
			print('text =', datum['text'])
			print('char IDs =', datum['char_id'])
		"""

		num_examples = len(dataset['data'])
		max_height, max_width, max_channel, max_label_len = 0, 0, 0, 0
		for datum in dataset['data']:
			sz = datum['size']
			if len(sz) != 3:
				print('[Warning] Invalid data size: {}.'.format(datum['file']))
				continue

			if sz[0] > max_height:
				max_height = sz[0]
			if sz[1] > max_width:
				max_width = sz[1]
			if sz[2] > max_channel:
				max_channel = sz[2]
			if len(datum['char_id']) > max_label_len:
				max_label_len = len(datum['char_id'])

		max_label_len += 1  # For EOS token.
		#max_label_len += 2  # For EOS token + blank label.

		if 0 == max_height or 0 == max_width or 0 == max_channel or 0 == max_label_len:
			raise ValueError('[Error] Invalid dataset size')

		charset = list(dataset['charset'].values())
		#charset = sorted(charset)

		#data = np.zeros((num_examples, max_height, max_width, max_channel))
		data = np.zeros((num_examples, image_height, image_width, image_channel))
		#labels = np.zeros((num_examples, max_label_len))
		labels = np.full((num_examples, max_label_len), blank_label)
		for idx, datum in enumerate(dataset['data']):
			img = cv2.imread(datum['file'], cv2.IMREAD_GRAYSCALE)
			sz = datum['size']
			if sz[0] != image_height or sz[1] != image_width:
				img = cv2.resize(img, (image_width, image_height))
			#data[idx,:sz[0],:sz[1],:sz[2]] = img.reshape(img.shape + (-1,))
			data[idx,:,:,0] = img
			if False:  # Char ID.
				#labels[idx,:len(datum['char_id'])] = datum['char_id']
				labels[idx,:(len(datum['char_id']) + 1)] = datum['char_id'] + [eos_token]
				#labels[idx,:(len(datum['char_id']) + 2)] = datum['char_id'] + [eos_token, blank_label]
			else:  # Unicode -> char ID.
				#labels[idx,:len(datum['char_id'])] = list(charset.index(chr(id)) for id in datum['char_id'])
				labels[idx,:(len(datum['char_id']) + 1)] = list(charset.index(chr(id)) for id in datum['char_id']) + [eos_token]
				#labels[idx,:(len(datum['char_id']) + 2)] = list(charset.index(chr(id)) for id in datum['char_id']) + [eos_token, blank_label]

		return data, labels

#--------------------------------------------------------------------

class MyKerasModel(object):
	def __init__(self):
		pass

	# REF [site] >> https://github.com/qjadud1994/CRNN-Keras
	def create_model(self, input_tensor, num_classes):
		#inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, width, height, 1).

		# Convolution layer (VGG).
		inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_tensor)  # (None, width, height, 64).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)
		inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None, width/2, height/2, 64).

		inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, width/2, height/2, 128).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)
		inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, width/4, height/4, 128).

		inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, width/4, height/4, 256).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)
		inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, width/4, height/4, 256).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)
		inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, width/4, height/8, 256).

		inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, width/4, height/8, 512).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)
		inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, width/4, height/8, 512).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)
		inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, width/4, height/16, 512).

		inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, width/4, height/16, 512).
		inner = BatchNormalization()(inner)
		inner = Activation('relu')(inner)

		# CNN to RNN.
		rnn_input_shape = inner.shape #inner.shape.as_list()
		inner = Reshape(target_shape=((rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3])), name='reshape')(inner)  # (None, width/4, height/16 * 512).
		if True:
			inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256).

			# RNN layer.
			lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
			lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
			lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 512).
			lstm1_merged = BatchNormalization()(lstm1_merged)
			lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
			lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
			lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 512).
			lstm2_merged = BatchNormalization()(lstm2_merged)
		elif False:
			inner = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256).

			# RNN layer.
			lstm_1 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
			lstm_1b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
			lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 1024).
			lstm1_merged = BatchNormalization()(lstm1_merged)
			lstm_2 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
			lstm_2b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
			lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 1024).
			lstm2_merged = BatchNormalization()(lstm2_merged)
		elif False:
			inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256).

			# RNN layer.
			lstm_1 = LSTM(1024, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
			lstm_1b = LSTM(1024, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
			lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 2048).
			lstm1_merged = BatchNormalization()(lstm1_merged)
			lstm_2 = LSTM(1024, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
			lstm_2b = LSTM(1024, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
			lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 2048).
			lstm2_merged = BatchNormalization()(lstm2_merged)  # NOTE [check] >> Different from the original implementation.

		# Transforms RNN output to character activations.
		inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, width/4, num_classes).
		y_pred = Activation('softmax', name='softmax')(inner)

		return y_pred

class MyTensorFlowModel(object):
	def __init__(self, image_width, image_height, image_channel):
		self._input_ph = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel], name='input_ph')
		self._output_ph = tf.sparse_placeholder(tf.int32, name='output_ph')
		self._model_output_len_ph = tf.placeholder(tf.int32, [None], name='model_output_len_ph')

	@property
	def placeholders(self):
		return self._input_ph, self._output_ph, self._model_output_len_ph

	def create_model(self, inputs, seq_len, num_classes, default_value=-1):
		with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
			cnn_output = MyTensorFlowModel.create_cnn(inputs)

		rnn_input_shape = cnn_output.shape #cnn_output.shape.as_list()

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# FIXME [decide] >> [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]] or [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]] ?
			#rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]])
			rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]])
			rnn_output = MyTensorFlowModel.create_bidirectionnal_rnn(rnn_input, seq_len)

		time_steps = rnn_input.shape.as_list()[1]  # Model output time-steps.
		print('***** Model output time-steps = {}.'.format(time_steps))

		with tf.variable_scope('transcription', reuse=tf.AUTO_REUSE):
			logits = tf.layers.dense(rnn_output, num_classes, activation=tf.nn.relu, name='dense')

		logits = tf.transpose(logits, (1, 0, 2))  # Time-major.

		# Decoding.
		with tf.variable_scope('decoding', reuse=tf.AUTO_REUSE):
			#decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=100, top_paths=1, merge_repeated=False)
			decoded, log_prob = tf.nn.ctc_beam_search_decoder_v2(logits, seq_len, beam_width=100, top_paths=1)
			sparse_decoded = decoded[0]
			dense_decoded = tf.sparse.to_dense(sparse_decoded, default_value=default_value)

		return {'logit': logits, 'sparse_label': sparse_decoded, 'dense_label': dense_decoded, 'time_step': time_steps}

	def get_loss(self, y, t_sparse, y_len):
		loss = tf.nn.ctc_loss(t_sparse, y, y_len)
		#loss = tf.nn.ctc_loss_v2(t_sparse, y, t_len, y_len)
		loss = tf.reduce_mean(loss)

		return loss

	def get_accuracy(self, y_sparse, t_sparse):
		# The error rate.
		acc = tf.reduce_mean(tf.edit_distance(tf.cast(y_sparse, tf.int32), t_sparse))

		return acc

	@staticmethod
	def create_unit_cell(num_units, name):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, name=name)

	@staticmethod
	def create_bidirectionnal_rnn(inputs, seq_len=None):
		with tf.variable_scope('birnn_1', reuse=tf.AUTO_REUSE):
			fw_cell_1, bw_cell_1 = MyTensorFlowModel.create_unit_cell(256, 'fw_cell'), MyTensorFlowModel.create_unit_cell(256, 'bw_cell')

			outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_1, bw_cell_1, inputs, seq_len, dtype=tf.float32)
			outputs_1 = tf.concat(outputs_1, 2)

		with tf.variable_scope('birnn_2', reuse=tf.AUTO_REUSE):
			fw_cell_2, bw_cell_2 = MyTensorFlowModel.create_unit_cell(256, 'fw_cell'), MyTensorFlowModel.create_unit_cell(256, 'bw_cell')

			outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_2, bw_cell_2, outputs_1, seq_len, dtype=tf.float32)
			outputs_2 = tf.concat(outputs_2, 2)

		return outputs_2

	@staticmethod
	def create_cnn(inputs):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), padding='same', name='conv')
			conv1 = tf.nn.relu(conv1, name='relu')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='maxpool')
			#conv1 = tf.nn.relu(conv1, name='relu')

			# (None, width/2, height/2, 64).

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), padding='same', name='conv')
			conv2 = tf.nn.relu(conv2, name='relu')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='maxpool')
			#conv2 = tf.nn.relu(conv2, name='relu')

			# (None, width/4, height/4, 128).

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), padding='same', name='conv1')
			conv3 = tf.nn.relu(conv3, name='relu1')
			conv3 = tf.layers.batch_normalization(conv3, name='batchnorm')
			#conv3 = tf.nn.relu(conv3, name='relu1')

			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), padding='same', name='conv2')
			conv3 = tf.nn.relu(conv3, name='relu2')
			#conv3 = tf.layers.batch_normalization(conv3, name='batchnorm2')
			#conv3 = tf.nn.relu(conv3, name='relu2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[1, 2], padding='same', name='maxpool')
			#conv3 = tf.layers.max_pooling2d(conv3, pool_size=[1, 2], strides=[1, 2], padding='same', name='maxpool')

			# (None, width/4, height/8, 256).

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv1')
			conv4 = tf.nn.relu(conv4, name='relu1')
			conv4 = tf.layers.batch_normalization(conv4, name='batchnorm')
			#conv4 = tf.nn.relu(conv4, name='relu1')

			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv2')
			conv4 = tf.nn.relu(conv4, name='relu2')
			#conv4 = tf.layers.batch_normalization(conv4, name='batchnorm2')
			#conv4 = tf.nn.relu(conv4, name='relu2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[1, 2], padding='same', name='maxpool')
			#conv4 = tf.layers.max_pooling2d(conv4, pool_size=[1, 2], strides=[1, 2], padding='same', name='maxpool')

			# (None, width/4, height/16, 512).

		with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
			# FIXME [decide] >>
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), padding='valid', name='conv')
			#conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), padding='same', name='conv')
			conv5 = tf.nn.relu(conv5, name='relu')
			#conv5 = tf.layers.batch_normalization(conv5, name='batchnorm')
			#conv5 = tf.nn.relu(conv5, name='relu')

			# (None, width/4, height/16, 512).

		return conv5

#--------------------------------------------------------------------

class MyOldRunner(object):
	def __init__(self):
		image_height, image_width, image_channel = 64, 320, 1
		num_labels = 2350
		width_downsample_factor = 4  # Depends on models.

		if False:
			self._num_classes = num_labels
		elif False:
			# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
			self._num_classes = num_labels + 1  # #labels + blank label.
			blank_label = self._num_classes - 1
		else:
			# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
			self._num_classes = num_labels + 1 + 1  # #labels + EOS + blank label.
			self._eos_token_label = self._num_classes - 2
			blank_label = self._num_classes - 1

		#--------------------
		# Create a dataset.

		self._dataset = MyFileBasedSyntheticDataset(image_height, image_width, image_channel, width_downsample_factor, self._num_classes, self._eos_token_label, blank_label)

		#--------------------
		# (samples, height, width, channels) -> (samples, width, height, channels).
		#self._input_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_ph')  # NOTE [caution] >> (?, image_height, image_width, ?)
		self._input_ph = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='input_ph')  # NOTE [caution] >> (?, image_width, image_height, ?)
		if False:
			self._output_ph = tf.placeholder(tf.float32, shape=[None, max_label_len], name='output_ph')
		else:
			self._output_ph = tf.sparse.placeholder(tf.int32, name='output_ph')
		self._output_length_ph = tf.placeholder(tf.int32, shape=[None], name='output_length_ph')
		self._model_output_length_ph = tf.placeholder(tf.int32, shape=[None], name='model_time_step_ph')

	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0):
		with tf.Session() as sess:
			# Create a model.
			#model = MyKerasModel()
			model = MyTensorFlowModel()
			model_output = model.create_model(self._input_ph, self._model_output_length_ph, self._num_classes)

			saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			# Create a trainer.
			loss = model.get_loss(model_output, self._output_ph, self._model_output_length_ph)
			accuracy = model.get_accuracy(model_output, self._output_ph, self._model_output_length_ph, default_value=self._eos_token_label)

			learning_rate = 0.001
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1.0e-7, use_locking=False)

			train_op = optimizer.minimize(loss)

			#--------------------
			print('Start training...')
			start_total_time = time.time()
			sess.run(tf.global_variables_initializer())
			for epoch in range(num_epochs):
				print('Epoch {}:'.format(epoch + 1))

				#--------------------
				start_time = time.time()
				train_loss, train_acc, num_examples = 0, 0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					# TODO [improve] >> CTC beam search decoding runs on CPU (too slow).
					#_, batch_loss, batch_acc = sess.run([train_op, loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					_, batch_loss = sess.run([train_op, loss], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					train_loss += batch_loss * num_batch_examples
					#train_acc += batch_acc * num_batch_examples
					num_examples += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				#train_acc /= num_examples
				print('\tTrain:      loss = {:.6f}: {} secs.'.format(train_loss, time.time() - start_time))
				#print('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

				#--------------------
				start_time = time.time()
				val_loss, val_acc, num_examples = 0, 0, 0
				for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
					# TODO [improve] >> CTC beam search decoding runs on CPU (too slow).
					#batch_loss, batch_acc = sess.run([loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					batch_loss = sess.run(loss, feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					val_loss += batch_loss * num_batch_examples
					#val_acc += batch_acc * num_batch_examples
					num_examples += num_batch_examples
				val_loss /= num_examples
				#val_acc /= num_examples
				print('\tValidation: loss = {:.6f}: {} secs.'.format(val_loss, time.time() - start_time))
				#print('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

				#--------------------
				print('Start saving a model...')
				start_time = time.time()
				saved_model_path = saver.save(sess, checkpoint_dir_path + '/model.ckpt', initial_epoch + epoch)
				print('End saving a model: {} secs.'.format(time.time() - start_time))
			print('End training: {} secs.'.format(time.time() - start_total_time))

	def infer(self, checkpoint_dir_path, batch_size=None, shuffle=False):
		with tf.Session() as sess:
			# Create a model.
			#model = MyKerasModel()
			model = MyTensorFlowModel()
			model_output = model.create_model(self._input_ph, self._model_output_length_ph, self._num_classes)

			# Load a model.
			print('Start loading a model...')
			start_time = time.time()
			saver = tf.train.Saver()
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			#saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('End loading a model: {} secs.'.format(time.time() - start_time))

			#--------------------
			print('Start inferring...')
			start_time = time.time()
			inferences, test_labels = list(), list()
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=shuffle):
				inferences.append(sess.run(model_output, feed_dict={self._input_ph: batch_data[0]}))
				test_labels.append(batch_data[1])
			print('End inferring: {} secs.'.format(time.time() - start_time))

			inferences, test_labels = np.vstack(inferences), np.vstack(test_labels)
			if inferences is not None:
				print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				inferences = np.argmax(inferences, -1)

				print('**********', inferences[:10])
				print('**********', test_labels[:10])
			else:
				print('[SWL] Warning: Invalid inference results.')

class MyRunner(object):
	def __init__(self):
		data_dir_path = './kr_samples_100000'
		#data_dir_path = './kr_samples_200000'

		self._image_height, self._image_width, self._image_channel = 32, 160, 1  # TODO [modify] >> image_channel is fixed.
		#self._image_height, self._image_width, self._image_channel = 64, 320, 1  # TODO [modify] >> image_channel is fixed.
		train_test_ratio = 0.8
		# TODO [modify] >> Depends on a model.
		#	model_output_time_steps = image_width / width_downsample_factor or image_width / width_downsample_factor - 1.
		#	REF [function] >> MyModel.create_model().
		#width_downsample_factor = 4
		model_output_time_steps = 39
		#model_output_time_steps = 79

		self._default_value = -1

		#--------------------
		# Create a dataset.

		self._dataset = MyDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count=model_output_time_steps)

	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyTensorFlowModel(self._image_height, self._image_width, self._image_channel)
			input_ph, output_ph, model_output_len_ph = model.placeholders

			model_output = model.create_model(input_ph, model_output_len_ph, self._dataset.num_classes, self._default_value)

			loss = model.get_loss(model_output['logit'], output_ph, model_output_len_ph)
			accuracy = model.get_accuracy(model_output['sparse_label'], output_ph)

			# Create a trainer.
			optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
			train_op = optimizer.minimize(loss)

			# Create a saver.
			saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with tf.Session(graph=graph).as_default() as sess:
			sess.run(initializer)

			# Restore a model.
			if is_training_resumed:
				print('[SWL] Info: Start restoring a model...')
				start_time = time.time()
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
				ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
				#ckpt_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
				if ckpt_filepath:
					initial_epoch = int(ckpt_filepath.split('-')[1])
					saver.restore(sess, ckpt_filepath)
				else:
					print('[SWL] Error: Failed to restore a model from {}.'.format(checkpoint_dir_path))
					return
				print('[SWL] Info: End restoring a model: {} secs.'.format(time.time() - start_time))

			#--------------------
			print('[SWL] Info: Start training...')
			start_total_time = time.time()
			for epoch in range(initial_epoch, num_epochs + initial_epoch):
				print('Epoch {}:'.format(epoch + 1))

				start_time = time.time()
				train_loss = 0
				correct_count, total_count = 0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					#batch_images, batch_labels_char, batch_sparse_labels_int = batch_data
					# TODO [improve] >> CTC beam search decoding runs on CPU (too slow).
					_, batch_loss, batch_dense_labels_int = sess.run(
						[train_op, loss, model_output['dense_label']],
						feed_dict={
							input_ph: batch_data[0],
							output_ph: batch_data[2],
							model_output_len_ph: [model_output['time_step']] * num_batch_examples
						}
					)

					train_loss += batch_loss
					correct_count += len(list(filter(lambda x: x[1] == self._dataset.decode_label(x[0], self._default_value), zip(batch_dense_labels_int, batch_data[1]))))
					total_count += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				print('\tTrain: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, correct_count / total_count, time.time() - start_time))

				#--------------------
				print('[SWL] Info: Start saving a model...')
				start_time = time.time()
				saved_model_path = saver.save(sess, os.path.join(checkpoint_dir_path, 'model.ckpt'), global_step=epoch)
				print('[SWL] Info: End saving a model: {} secs.'.format(time.time() - start_time))
			print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))
		return None

	def infer(self, checkpoint_dir_path, batch_size):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyTensorFlowModel(self._image_height, self._image_width, self._image_channel)
			input_ph, output_ph, model_output_len_ph = model.placeholders

			model_output = model.create_model(input_ph, model_output_len_ph, self._dataset.num_classes, self._default_value)

			# Create a saver.
			saver = tf.train.Saver()

		with tf.Session(graph=graph).as_default() as sess:
			# Load a model.
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
			#ckpt_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
			if ckpt_filepath:
				saver.restore(sess, ckpt_filepath)
			else:
				print('[SWL] Error: Failed to load a model from {}.'.format(checkpoint_dir_path))
				return
			print('[SWL] Info: End loading a model: {} secs.'.format(time.time() - start_time))

			#--------------------
			print('[SWL] Info: Start inferring...')
			start_time = time.time()
			inferences, ground_truths = list(), list()
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
				#batch_images, batch_labels_char, batch_sparse_labels_int = batch_data
				# TODO [improve] >> CTC beam search decoding runs on CPU (too slow).
				batch_dense_labels_int = sess.run(
					model_output['dense_label'],
					feed_dict={
						input_ph: batch_data[0],
						model_output_len_ph: [model_output['time_step']] * num_batch_examples
					}
				)
				inferences.append(batch_dense_labels_int)
				ground_truths.append(batch_data[1])
			print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

			if inferences and ground_truths:
				#print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0
				for pred, gt in zip(inferences, ground_truths):
					pred = np.array(list(map(lambda x: self._dataset.decode_label(x), pred)))

					correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(pred, gt))))
					total_word_count += len(gt)
					for ps, gs in zip(pred, gt):
						correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(ps, gs))))
						total_char_count += max(len(ps), len(gs))
				print('Inference: word accurary = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
				print('Inference: character accurary = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))
			else:
				print('[SWL] Warning: Invalid inference results.')
		return None

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	num_epochs, batch_size = 20, 32
	initial_epoch = 0
	is_training_resumed = False

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		output_dir_prefix = 'simple_hangeul_crnn'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		#output_dir_suffix = '20190724T231604'
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')

	#--------------------
	if True:
		runner = MyRunner()

		if True:
			if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
				os.makedirs(checkpoint_dir_path, exist_ok=True)

			runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch, is_training_resumed)

		if True:
			if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
				print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
				return

			runner.infer(checkpoint_dir_path, batch_size)
	else:
		runner = MyOldRunner()

		if True:
			if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
				os.makedirs(checkpoint_dir_path, exist_ok=True)

			runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch)

		if True:
			if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
				print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
				return

			runner.infer(checkpoint_dir_path)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
