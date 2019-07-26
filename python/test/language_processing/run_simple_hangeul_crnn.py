#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, datetime, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import cv2
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class MySyntheticDataset(object):
	def __init__(self, image_height, image_width, image_channel, width_downsample_factor, num_classes, eos_token_label, blank_label):
		self._eos_token_label = eos_token_label

class MyFileBasedDataset(object):
	def __init__(self, image_height, image_width, image_channel, width_downsample_factor, num_classes, eos_token_label, blank_label):
		self._eos_token_label = eos_token_label
		self._model_output_time_steps = image_width // width_downsample_factor

		print('Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, self._test_images, self._test_labels = MyFileBasedDataset.load_data_from_json(image_height, image_width, image_channel, num_classes, eos_token_label, blank_label)
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
			model_output_lengths = np.full((self._train_images.shape[0],), self._model_output_time_steps - 2)  # See MyModel.get_loss().

		return MyFileBasedDataset._create_batch_generator(self._train_images, self._train_labels, model_output_lengths, batch_size, shuffle, self._eos_token_label)

	def create_test_batch_generator(self, batch_size, shuffle=False):
		# FIXME [improve] >> Stupid implementation.
		if True:
			model_output_lengths = np.full((self._test_images.shape[0],), self._model_output_time_steps)
		else:
			model_output_lengths = np.full((self._test_images.shape[0],), self._model_output_time_steps - 2)  # See MyModel.get_loss().

		return MyFileBasedDataset._create_batch_generator(self._test_images, self._test_labels, model_output_lengths, batch_size, shuffle, self._eos_token_label)

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
		train_data, train_labels = MyFileBasedDataset.text_dataset_to_numpy(train_dataset_json_filepath, image_height, image_width, image_channel, eos_token, blank_label)
		print('End loading train dataset: {} secs.'.format(time.time() - start_time))
		print('Start loading test dataset to numpy...')
		start_time = time.time()
		test_data, test_labels = MyFileBasedDataset.text_dataset_to_numpy(test_dataset_json_filepath, image_height, image_width, image_channel, eos_token, blank_label)
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
	def __init__(self):
		pass

	# REF [site] >> https://github.com/Belval/CRNN
	def create_model(self, input_tensor, model_time_step_tensor, num_classes):
		cnn_output = MyTensorFlowModel.create_cnn(input_tensor)

		rnn_input_shape = cnn_output.shape #cnn_output.shape.as_list()
		# FIXME [decide] >> [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]] or [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]] ?
		#rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]])
		rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]])
		rnn_output = MyTensorFlowModel.create_bidirectionnal_rnn(rnn_input, model_time_step_tensor)

		#max_char_count = rnn_input.shape.as_list()[1]  #  Model time-steps.

		logits = tf.layers.dense(rnn_output, num_classes, activation=tf.nn.relu, name='dense')
		logits = tf.transpose(logits, (1, 0, 2))  # Time-major.
		
		return logits

	def get_loss(self, y, t, y_length):
		with tf.name_scope('loss'):
			# The 2 is critical here since the first couple outputs of the RNN tend to be garbage.
			#y = y[:, 2:, :]

			# Connectionist temporal classification (CTC) loss.
			# FIXME [decide] >> t_length or y_length?
			#loss = tf.nn.ctc_loss(t, y, t_length)
			loss = tf.nn.ctc_loss(t, y, y_length)
			loss = tf.reduce_mean(loss)

			return loss

	def get_accuracy(self, y, t, y_length, default_value):
		with tf.name_scope('accuracy'):
			# FIXME [decide] >> t_length or y_length?
			##decoded, log_prob = tf.nn.ctc_beam_search_decoder(y, t_length, beam_width=100, top_paths=1, merge_repeated=False)  # y: time-major.
			#decoded, log_prob = tf.nn.ctc_beam_search_decoder(y, y_length, beam_width=100, top_paths=1, merge_repeated=False)  # y: time-major.
			decoded, log_prob = tf.nn.ctc_beam_search_decoder_v2(y, y_length, beam_width=100, top_paths=1)  # y: time-major.
			#dense_decoded = tf.sparse.to_dense(decoded[0], default_value=default_value)

			# The error rate.
			acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), t))

			return acc

	@staticmethod
	def create_unit_cell(num_units, name):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, name=name)

	@staticmethod
	def create_bidirectionnal_rnn(inputs, seq_len=None):
		with tf.variable_scope(None, default_name='bidirectional-rnn-1'):
			# Forward & backward.
			lstm_fw_cell_1, lstm_bw_cell_1 = MyTensorFlowModel.create_unit_cell(256, 'lstm_fw_cell_1'), MyTensorFlowModel.create_unit_cell(256, 'lstm_bw_cell_1')

			inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)
			inter_output = tf.concat(inter_output, 2)

		with tf.variable_scope(None, default_name='bidirectional-rnn-2'):
			# Forward & backward.
			lstm_fw_cell_2, lstm_bw_cell_2 = MyTensorFlowModel.create_unit_cell(256, 'lstm_fw_cell_2'), MyTensorFlowModel.create_unit_cell(256, 'lstm_bw_cell_2')

			outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)
			outputs = tf.concat(outputs, 2)

		return outputs

	@staticmethod
	def create_cnn(inputs):
		# 64 / 3 x 3 / 1 / 1
		conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		# 2 x 2 / 1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

		# 128 / 3 x 3 / 1 / 1
		conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		# 2 x 2 / 1
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

		# 256 / 3 x 3 / 1 / 1
		conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		# Batch normalization layer
		bnorm3 = tf.layers.batch_normalization(conv3)

		# 256 / 3 x 3 / 1 / 1
		conv4 = tf.layers.conv2d(inputs=bnorm3, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		# 1 x 2 / 1
		pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding='same')

		# 512 / 3 x 3 / 1 / 1
		conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		# Batch normalization layer
		bnorm5 = tf.layers.batch_normalization(conv5)

		# 512 / 3 x 3 / 1 / 1
		conv6 = tf.layers.conv2d(inputs=bnorm5, filters=512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
		# 1 x 2 / 2
		pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding='same')

		# 512 / 2 x 2 / 1 / 0
		# FIXME [decide] >>
		#conv7 = tf.layers.conv2d(inputs=pool6, filters=512, kernel_size=(2, 2), padding='valid', activation=tf.nn.relu)
		conv7 = tf.layers.conv2d(inputs=pool6, filters=512, kernel_size=(2, 2), padding='same', activation=tf.nn.relu)

		return conv7

#--------------------------------------------------------------------

class MyRunner(object):
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

		self._dataset = MyFileBasedDataset(image_height, image_width, image_channel, width_downsample_factor, self._num_classes, self._eos_token_label, blank_label)

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

			# Create a trainer.
			loss = model.get_loss(model_output, self._output_ph, self._model_output_length_ph)
			accuracy = model.get_accuracy(model_output, self._output_ph, self._model_output_length_ph, default_value=self._eos_token_label)

			learning_rate = 0.001
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1.0e-7, use_locking=False)

			train_op = optimizer.minimize(loss)

			saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			#--------------------
			print('Start training...')
			start_total_time = time.time()
			sess.run(tf.global_variables_initializer())
			for epoch in range(num_epochs):
				print('Epoch {}:'.format(epoch + 1))

				#--------------------
				start_time = time.time()
				train_loss, train_accuracy, num_examples = 0, 0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					#_, batch_loss, batch_accuracy = sess.run([train_op, loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					_, batch_loss = sess.run([train_op, loss], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					train_loss += batch_loss * num_batch_examples
					#train_accuracy += batch_accuracy * num_batch_examples
					num_examples += num_batch_examples
					if (batch_step + 1) % 100 == 0:
						print('\tStep = {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				#train_accuracy /= num_examples
				print('\tTrain:      loss = {:.6f}: {} secs.'.format(train_loss, time.time() - start_time))
				#print('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_accuracy, time.time() - start_time))

				#--------------------
				start_time = time.time()
				val_loss, val_accuracy, num_examples = 0, 0, 0
				for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
					#batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					batch_loss = sess.run(loss, feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1], self._model_output_length_ph: batch_data[2]})
					val_loss += batch_loss * num_batch_examples
					#val_accuracy += batch_accuracy * num_batch_examples
					num_examples += num_batch_examples
				val_loss /= num_examples
				#val_accuracy /= num_examples
				print('\tValidation: loss = {:.6f}: {} secs.'.format(val_loss, time.time() - start_time))
				#print('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_accuracy, time.time() - start_time))

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

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	num_epochs, batch_size = 1000, 128
	initial_epoch = 0

	#--------------------
	output_dir_prefix = 'simple_hangeul_crnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20190724T231604'
	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	os.makedirs(checkpoint_dir_path, exist_ok=True)

	#--------------------
	runner = MyRunner()

	runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch)
	runner.infer(checkpoint_dir_path)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
