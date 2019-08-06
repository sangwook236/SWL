#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, datetime, csv
import numpy as np
import tensorflow as tf
import cv2
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count):
		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio')

		self._labels = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-\'.!?,&"'

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		# Load data.
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		examples = self._load_data(data_dir_path, image_height, image_width, image_channel, max_char_count)
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

		np.random.shuffle(examples)
		num_examples = len(examples)
		test_offset = round(train_test_ratio * num_examples)
		self._train_data, self._test_data = examples[:test_offset], examples[test_offset:]

	@property
	def num_classes(self):
		return self._num_classes

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

	def create_train_batch_generator(self, batch_size, shuffle=True):
		return MyDataset._create_batch_generator(self._train_data, batch_size, shuffle)

	def create_test_batch_generator(self, batch_size, shuffle=False):
		return MyDataset._create_batch_generator(self._test_data, batch_size, shuffle)

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

#--------------------------------------------------------------------

class MyModel(object):
	def __init__(self, image_height, image_width, image_channel):
		self._input_ph = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel], name='input_ph')
		self._output_ph = tf.sparse_placeholder(tf.int32, name='output_ph')
		self._model_output_len_ph = tf.placeholder(tf.int32, [None], name='model_output_len_ph')

	@property
	def placeholders(self):
		return self._input_ph, self._output_ph, self._model_output_len_ph

	def create_model(self, inputs, seq_len, num_classes, default_value=-1):
		kernel_initializer = None
		#kernel_initializer = tf.initializers.he_normal()
		#kernel_initializer = tf.initializers.he_uniform()
		#kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
		#kernel_initializer = tf.initializers.uniform_unit_scaling(factor=1.0)
		#kernel_initializer = tf.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
		#kernel_initializer = tf.initializers.glorot_normal()  # Xavier normal initialization.
		#kernel_initializer = tf.initializers.glorot_uniform()  # Xavier uniform initialization.

		if True:
			create_cnn_functor = MyModel.create_cnn_without_batch_normalization
		else:
			create_cnn_functor = MyModel.create_cnn_with_batch_normalization

		#--------------------
		with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
			cnn_output = create_cnn_functor(inputs, kernel_initializer)

		rnn_input_shape = cnn_output.shape #cnn_output.shape.as_list()

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# FIXME [decide] >> [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]] or [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]] ?
			#rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1] * rnn_input_shape[2], rnn_input_shape[3]])
			rnn_input = tf.reshape(cnn_output, [-1, rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3]])
			rnn_output = MyModel.create_bidirectionnal_rnn(rnn_input, seq_len, kernel_initializer)

		time_steps = rnn_input.shape.as_list()[1]  # Model output time-steps.
		#print('***** Model output time-steps = {}.'.format(time_steps))

		with tf.variable_scope('transcription', reuse=tf.AUTO_REUSE):
			logits = tf.layers.dense(rnn_output, num_classes, activation=tf.nn.relu, kernel_initializer=kernel_initializer, name='dense')

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
	def create_cnn_without_batch_normalization(inputs, kernel_initializer=None):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv1 = tf.nn.relu(conv1, name='relu')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='maxpool')

			# (None, width/2, height/2, 64).

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv2 = tf.nn.relu(conv2, name='relu')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='maxpool')

			# (None, width/4, height/4, 128).

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv1')
			conv3 = tf.nn.relu(conv3, name='relu1')
			conv3 = tf.layers.batch_normalization(conv3, name='batchnorm')

			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv2')
			conv3 = tf.nn.relu(conv3, name='relu2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[1, 2], padding='same', name='maxpool')

			# (None, width/4, height/8, 256).

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv1')
			conv4 = tf.nn.relu(conv4, name='relu1')
			conv4 = tf.layers.batch_normalization(conv4, name='batchnorm')

			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv2')
			conv4 = tf.nn.relu(conv4, name='relu2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[1, 2], padding='same', name='maxpool')

			# (None, width/4, height/16, 512).

		with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
			# FIXME [decide] >>
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), padding='valid', kernel_initializer=kernel_initializer, name='conv')
			#conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv5 = tf.nn.relu(conv5, name='relu')

			# (None, width/4, height/16, 512).

		return conv5

	@staticmethod
	def create_cnn_with_batch_normalization(inputs, kernel_initializer=None):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv1 = tf.layers.batch_normalization(conv1, name='batchnorm')
			conv1 = tf.nn.relu(conv1, name='relu')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='maxpool')

			# (None, width/2, height/2, 64).

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv2 = tf.layers.batch_normalization(conv2, name='batchnorm')
			conv2 = tf.nn.relu(conv2, name='relu')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='maxpool')

			# (None, width/4, height/4, 128).

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv1')
			conv3 = tf.layers.batch_normalization(conv3, name='batchnorm')
			conv3 = tf.nn.relu(conv3, name='relu1')

			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv2')
			conv3 = tf.layers.batch_normalization(conv3, name='batchnorm2')
			conv3 = tf.nn.relu(conv3, name='relu2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=[1, 2], strides=[1, 2], padding='same', name='maxpool')

			# (None, width/4, height/8, 256).

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv1')
			conv4 = tf.layers.batch_normalization(conv4, name='batchnorm')
			conv4 = tf.nn.relu(conv4, name='relu1')

			# FIXME [decide] >>
			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=None, name='conv2')
			#conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv2')
			conv4 = tf.layers.batch_normalization(conv4, name='batchnorm2')
			conv4 = tf.nn.relu(conv4, name='relu2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=[1, 2], strides=[1, 2], padding='same', name='maxpool')

			# (None, width/4, height/16, 512).

		with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
			# FIXME [decide] >>
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), padding='same', kernel_initializer=kernel_initializer, name='conv')
			#conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), padding='valid', kernel_initializer=kernel_initializer, name='conv')
			conv5 = tf.layers.batch_normalization(conv5, name='batchnorm')
			conv5 = tf.nn.relu(conv5, name='relu')

			# (None, width/4, height/16, 512).

		return conv5

	@staticmethod
	def create_bidirectionnal_rnn(inputs, seq_len=None, kernel_initializer=None):
		with tf.variable_scope('birnn_1', reuse=tf.AUTO_REUSE):
			fw_cell_1, bw_cell_1 = MyModel.create_unit_cell(256, kernel_initializer, 'fw_cell'), MyModel.create_unit_cell(256, kernel_initializer, 'bw_cell')

			outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_1, bw_cell_1, inputs, seq_len, dtype=tf.float32)
			outputs_1 = tf.concat(outputs_1, 2)

		with tf.variable_scope('birnn_2', reuse=tf.AUTO_REUSE):
			fw_cell_2, bw_cell_2 = MyModel.create_unit_cell(256, kernel_initializer, 'fw_cell'), MyModel.create_unit_cell(256, kernel_initializer, 'bw_cell')

			outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_2, bw_cell_2, outputs_1, seq_len, dtype=tf.float32)
			outputs_2 = tf.concat(outputs_2, 2)

		return outputs_2

	@staticmethod
	def create_unit_cell(num_units, kernel_initializer=None, name=None):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, initializer=kernel_initializer, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=kernel_initializer, name=name)

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, data_dir_path, train_test_ratio):
		# Set parameters.
		self._image_height, self._image_width, self._image_channel = 32, 100, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
		# TODO [modify] >> Depends on a model.
		#	model_output_time_steps = image_width / width_downsample_factor or image_width / width_downsample_factor - 1.
		#	REF [function] >> MyModel.create_model().
		#width_downsample_factor = 4
		model_output_time_steps = 24

		self._default_value = -1

		#--------------------
		# Create a dataset.

		self._dataset = MyDataset(data_dir_path, self._image_height, self._image_width, self._image_channel, train_test_ratio, max_char_count=model_output_time_steps)

	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(self._image_height, self._image_width, self._image_channel)
			input_ph, output_ph, model_output_len_ph = model.placeholders

			model_output = model.create_model(input_ph, model_output_len_ph, self._dataset.num_classes, self._default_value)

			loss = model.get_loss(model_output['logit'], output_ph, model_output_len_ph)
			accuracy = model.get_accuracy(model_output['sparse_label'], output_ph)

			# Create a trainer.
			learning_rate = 0.0001
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
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

			history = {
				'acc': list(),
				'loss': list(),
				'val_acc': list(),
				'val_loss': list()
			}

			#--------------------
			if is_training_resumed:
				print('[SWL] Info: Resume training...')
			else:
				print('[SWL] Info: Start training...')
			start_total_time = time.time()
			final_epoch = num_epochs + initial_epoch
			for epoch in range(initial_epoch + 1, final_epoch + 1):
				print('Epoch {}/{}:'.format(epoch, final_epoch))

				start_time = time.time()
				"""
				train_loss, train_acc, num_examples = 0.0, 0.0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					#batch_images, batch_labels_char, batch_sparse_labels_int = batch_data
					# TODO [improve] >> CTC beam search decoding is too slow. It seems to run on CPU.
					#	If the number of classes increases, its computation time becomes much slower.
					#_, batch_loss, batch_acc = sess.run(
					#	[train_op, loss, accuracy],
					_, batch_loss, batch_dense_labels_int = sess.run(
						[train_op, loss, model_output['dense_label']],
						feed_dict={
							input_ph: batch_data[0],
							output_ph: batch_data[2],
							model_output_len_ph: [model_output['time_step']] * num_batch_examples
						}
					)

					train_loss += batch_loss * num_batch_examples
					#train_acc += batch_acc * num_batch_examples
					train_acc += len(list(filter(lambda x: x[1] == self._dataset.decode_label(x[0], self._default_value), zip(batch_dense_labels_int, batch_data[1]))))
					num_examples += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				train_acc /= num_examples
				print('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)
				"""
				train_loss, train_acc, num_examples = 0.0, None, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					#batch_images, batch_labels_char, batch_sparse_labels_int = batch_data
					_, batch_loss = sess.run(
						[train_op, loss],
						feed_dict={
							input_ph: batch_data[0],
							output_ph: batch_data[2],
							model_output_len_ph: [model_output['time_step']] * num_batch_examples
						}
					)

					train_loss += batch_loss * num_batch_examples
					num_examples += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				print('\tTrain:      loss = {:.6f}, accuracy = {}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

				history['loss'].append(train_loss)
				#history['acc'].append(train_acc)

				#--------------------
				if epoch % 10 == 0:
					start_time = time.time()
					val_loss, val_acc, num_examples = 0.0, 0.0, 0
					for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_test_batch_generator(batch_size, shuffle=False)):
						#batch_images, batch_labels_char, batch_sparse_labels_int = batch_data
						# TODO [improve] >> CTC beam search decoding is too slow. It seems to run on CPU.
						#	If the number of classes increases, its computation time becomes much slower.
						#batch_loss, batch_acc = sess.run(
						#	[loss, accuracy],
						batch_loss, batch_dense_labels_int = sess.run(
							[loss, model_output['dense_label']],
							feed_dict={
								input_ph: batch_data[0],
								output_ph: batch_data[2],
								model_output_len_ph: [model_output['time_step']] * num_batch_examples
							}
						)

						val_loss += batch_loss * num_batch_examples
						#val_acc += batch_acc * num_batch_examples
						val_acc += len(list(filter(lambda x: x[1] == self._dataset.decode_label(x[0], self._default_value), zip(batch_dense_labels_int, batch_data[1]))))
						num_examples += num_batch_examples

						# Show some results.
						if 0 == batch_step:
							preds, gts = list(), list()
							for count, (pred, gt) in enumerate(zip(batch_dense_labels_int, batch_data[1])):
								pred = self._dataset.decode_label(pred, self._default_value)
								preds.append(pred)
								gts.append(gt)
								if (count + 1) >= 10:
									break
							print('\tValidation: G/T         = {}.'.format(gts))	
							print('\tValidation: predictions = {}.'.format(preds))	
					val_loss /= num_examples
					val_acc /= num_examples
					print('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

					history['val_loss'].append(val_loss)
					history['val_acc'].append(val_acc)
				else:
					start_time = time.time()
					val_loss, val_acc, num_examples = 0.0, None, 0
					for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_test_batch_generator(batch_size, shuffle=False)):
						#batch_images, batch_labels_char, batch_sparse_labels_int = batch_data
						batch_loss = sess.run(
							loss,
							feed_dict={
								input_ph: batch_data[0],
								output_ph: batch_data[2],
								model_output_len_ph: [model_output['time_step']] * num_batch_examples
							}
						)

						val_loss += batch_loss * num_batch_examples
						num_examples += num_batch_examples
					val_loss /= num_examples
					print('\tValidation: loss = {:.6f}, accuracy = {}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

					history['val_loss'].append(val_loss)
					#history['val_acc'].append(val_acc)

				#--------------------
				print('[SWL] Info: Start saving a model...')
				start_time = time.time()
				saved_model_path = saver.save(sess, os.path.join(checkpoint_dir_path, 'model.ckpt'), global_step=epoch - 1)
				print('[SWL] Info: End saving a model: {} secs.'.format(time.time() - start_time))

				sys.stdout.flush()
				time.sleep(0)
			print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))

			return history

	def infer(self, checkpoint_dir_path, inference_dir_path, batch_size):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(self._image_height, self._image_width, self._image_channel)
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
				# TODO [improve] >> CTC beam search decoding is too slow. It seems to run on CPU.
				#	If the number of classes increases, its computation time becomes much slower.
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
					pred = np.array(list(map(lambda x: self._dataset.decode_label(x, self._default_value), pred)))

					correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(pred, gt))))
					total_word_count += len(gt)
					for ps, gs in zip(pred, gt):
						correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(ps, gs))))
						total_char_count += max(len(ps), len(gs))
					#correct_char_count += functools.reduce(lambda l, pgs: l + len(list(filter(lambda pg: pg[0] == pg[1], zip(pgs[0], pgs[1])))), zip(pred, gt), 0)
					#total_char_count += functools.reduce(lambda l, pg: l + max(len(pg[0]), len(pg[1])), zip(pred, gt), 0)
				print('Inference: word accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
				print('Inference: character accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

				# Output to a file.
				csv_filepath = os.path.join(inference_dir_path, 'inference_results.csv')
				with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
					writer = csv.writer(csvfile, delimiter=',')

					for pred, gt in zip(inferences, ground_truths):
						pred = np.array(list(map(lambda x: self._dataset.decode_label(x, self._default_value), pred)))
						writer.writerow([gt, pred])
			else:
				print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	num_epochs, batch_size = 100, 64  # batch_size affects training.
	initial_epoch = 0
	is_training_resumed = False

	train_test_ratio = 0.8

	#data_dir_path = './en_samples_100000'
	data_dir_path = './en_samples_200000'

	#--------------------
	output_dir_path = None
	if not output_dir_path:
		output_dir_prefix = 'simple_english_crnn'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		#output_dir_suffix = '20190724T231604'
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = None
	if not inference_dir_path:
		inference_dir_path = os.path.join(output_dir_path, 'inference')

	#--------------------
	runner = MyRunner(data_dir_path, train_test_ratio)

	if True:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		history = runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch, is_training_resumed)

		#print('History =', history)
		#swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if True:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return
		if inference_dir_path and inference_dir_path.strip() and not os.path.exists(inference_dir_path):
			os.makedirs(inference_dir_path, exist_ok=True)

		runner.infer(checkpoint_dir_path, inference_dir_path, batch_size)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
