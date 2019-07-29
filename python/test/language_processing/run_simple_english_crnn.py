#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, datetime
import numpy as np
import tensorflow as tf
import cv2
#import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

def sparse_tuple_from(sequences, dtype=np.int32):
	"""
		Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
	"""

	indices = []
	values = []

	for n, seq in enumerate(sequences):
		indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

	return indices, values, shape

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count):
		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio')

		self._labels = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-\'.!?,&"'
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		# Load data.
		print('Start loading dataset...')
		start_time = time.time()
		examples = self._load_data(data_dir_path, image_height, image_width, image_channel, max_char_count)
		print('End loading dataset: {} secs.'.format(time.time() - start_time))

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
			print(label_str)
			raise

	# Integer label -> string label.
	def decode_label(self, label_int, default_value=-1):
		try:
			return ''.join([self._labels[id] for id in label_int if id != default_value])
		except Exception as ex:
			print(label_int)
			raise

	def _load_data(self, data_dir_path, image_height, image_width, image_channel, max_char_count):
		examples = []
		for f in os.listdir(data_dir_path):
			label_str = f.split('_')[0]
			if len(label_str) > max_char_count:
				continue
			image = MyDataset.resize_image(os.path.join(data_dir_path, f), image_height, image_width)
			image, label_int = MyDataset.preprocess_data(image, self.encode_label(label_str))
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
			raise ValueError('Invalid data length: {} != {} != {}'.format(num_examples, len(labels_str), len(labels_int)))
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
				batch_data1, batch_data2, batch_data3 = images[batch_indices], labels_str[batch_indices], labels_int[batch_indices]
				#batch_data3 = swl_ml_util.sequences_to_sparse(batch_data3, dtype=np.int32)  # Sparse tensor.
				batch_data3 = sparse_tuple_from(batch_data3, dtype=np.int32)  # Sparse tensor.
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
	def preprocess_data(inputs, outputs):
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
	def resize_image(image_filepath, input_height, input_width):
		img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
		r, c = img.shape
		if c >= input_width:
			return cv2.resize(img, (input_width, input_height))
		else:
			img_zeropadded = np.zeros((input_height, input_width))
			ratio = input_height / r
			img = cv2.resize(img, (int(c * ratio), input_height))
			width = min(input_width, img.shape[1])
			img_zeropadded[:, 0:width] = img[:, 0:width]
			return img_zeropadded

#--------------------------------------------------------------------

class MyModel(object):
	def __init__(self):
		pass

	def create_model(self, inputs, seq_len, num_classes, default_value=-1):
		cnn_output = MyModel.create_cnn(inputs)

		rnn_input_shape = cnn_output.shape #cnn_output.shape.as_list()
		# FIXME [decide] >> [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]] or [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]] ?
		#rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]])
		rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]])
		rnn_output = MyModel.create_bidirectionnal_rnn(rnn_input, seq_len)

		max_char_count = rnn_input.shape.as_list()[1]  # Model output time-steps.

		logits = tf.layers.dense(rnn_output, num_classes, activation=tf.nn.relu, name='dense')
		logits = tf.transpose(logits, (1, 0, 2))  # Time-major.

		# Decoding.
		decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
		sparse_decoded = decoded[0]
		dense_decoded = tf.sparse_tensor_to_dense(sparse_decoded, default_value=default_value)

		return {'logits': logits, 'sparse_decoded': sparse_decoded, 'dense_decoded': dense_decoded, 'max_char_count': max_char_count}

	def get_loss(self, y, t, seq_len):
		# Loss and cost calculation
		loss = tf.nn.ctc_loss(t, y, seq_len)
		loss = tf.reduce_mean(loss)

		return loss

	def get_accuracy(self, y_sparse_decoded, t):
		# The error rate
		acc = tf.reduce_mean(tf.edit_distance(tf.cast(y_sparse_decoded, tf.int32), t))

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
			lstm_fw_cell_1, lstm_bw_cell_1 = MyModel.create_unit_cell(256, 'lstm_fw_cell_1'), MyModel.create_unit_cell(256, 'lstm_bw_cell_1')

			inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)
			inter_output = tf.concat(inter_output, 2)

		with tf.variable_scope(None, default_name='bidirectional-rnn-2'):
			# Forward & backward.
			lstm_fw_cell_2, lstm_bw_cell_2 = MyModel.create_unit_cell(256, 'lstm_fw_cell_2'), MyModel.create_unit_cell(256, 'lstm_bw_cell_2')

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
	def __init__(self, input_height, image_width, image_channel):
		self._inputs = tf.placeholder(tf.float32, [None, image_width, input_height, image_channel], name='inputs')
		self._outputs = tf.sparse_placeholder(tf.int32, name='outputs')
		self._seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

	def train(self, session, dataset, model_path, num_epochs, batch_size, default_value, restore):
		with session.as_default():
			print('Training')

			model = MyModel()
			model_output = model.create_model(self._inputs, self._seq_len, dataset.num_classes, default_value)

			loss = model.get_loss(model_output['logits'], self._outputs, self._seq_len)
			acc = model.get_accuracy(model_output['sparse_decoded'], self._outputs)

			# Training step
			optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
			train_op = optimizer.minimize(loss)

			step = 0
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

			# Loading last save if needed
			if restore:
				print('Restoring')
				ckpt = tf.train.latest_checkpoint(model_path)
				if ckpt:
					print('Checkpoint is valid')
					step = int(ckpt.split('-')[1])
					saver.restore(session, ckpt)

			init = tf.global_variables_initializer()
			init.run()

			for i in range(step, num_epochs + step):
				iter_loss = 0
				correct_num, total_num, batch_step = 0, 0, 0
				for (batch_images, batch_labels_char, batch_labels_int), num_batch_examples in dataset.create_train_batch_generator(batch_size, shuffle=True):
					op, decoded, loss_value = session.run(
						[train_op, model_output['dense_decoded'], loss],
						feed_dict={
							self._inputs: batch_images,
							self._seq_len: [model_output['max_char_count']] * num_batch_examples,
							self._outputs: batch_labels_int
						}
					)

					#if i % 10 == 0:
					#    for j in range(2):
					#        print(batch_labels_char[j], dataset.decode_label(decoded[j], default_value))
					
					for gt, pred in zip(batch_labels_char, decoded):
						if gt == dataset.decode_label(pred, default_value):
							correct_num += 1
					total_num += len(batch_labels_char)
					if (batch_step + 1) % 100 == 0:
						print('Batch step =', batch_step + 1)
					batch_step += 1

					iter_loss += loss_value
				print('Epoch {}: accuracy = {}.'.format(i, correct_num / total_num))

				save_path = os.path.join(model_path, 'ckp')
				saver.save(
					session,
					save_path,
					global_step=step
				)

				print('[{}] Iteration loss: {}'.format(step, iter_loss))

				step += 1
		return None

	def test(self, session, dataset, model_path, batch_size, default_value):
		with session.as_default():
			print('Testing')

			model = MyModel()
			model_output = model.create_model(self._inputs, self._seq_len, batch_size, default_value)

			print('Loading')
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
			ckpt = tf.train.latest_checkpoint(model_path)
			if ckpt:
				print('Checkpoint is valid')
				saver.restore(session, ckpt)

			correct_num, total_num = 0, 0
			for (batch_images, batch_labels_char, _), num_batch_examples in dataset.create_train_batch_generator(batch_size, shuffle=Fase):
				decoded = session.run(
					model_output['dense_decoded'],
					feed_dict={
						self._inputs: batch_images,
						self._seq_len: [model_output['max_char_count']] * num_batch_examples
					}
				)

				#for i, y in enumerate(batch_labels_char):
				#    print(batch_labels_char[i], dataset.decode_label(decoded[i]))

				for gt, pred in zip(batch_labels_char, decoded):
					if gt == dataset.decode_label(pred):
						correct_num += 1
				total_num += len(batch_labels_char)
			print('Accuracy = {}.'.format(i, correct_num / total_num))
		return None

#--------------------------------------------------------------------

def main():
	batch_size = 64
	num_epochs = 10
	model_path = './saved_model'
	data_dir_path = './en_samples_100000'
	#data_dir_path = './en_samples_200000'
	image_height, image_width, image_channel = 32, 100, 1  # TODO [modify] >> 32 is hard-coded.
	train_test_ratio = 0.8
	restore = False
	default_value = -1
	max_char_count = 24  # TODO [modify] >>

	# Creating data_manager
	dataset = MyDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)

	runner = MyRunner(image_height, image_width, image_channel)

	if True:
		session = tf.Session()
		runner.train(session, dataset, model_path, num_epochs, batch_size, default_value, restore)

	if True:
		session2 = tf.Session()
		runner.test(session2, dataset, model_path, batch_size, default_value)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
