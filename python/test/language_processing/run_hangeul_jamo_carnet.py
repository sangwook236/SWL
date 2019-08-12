#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, datetime, functools, glob, csv
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import cv2
import swl.machine_learning.util as swl_ml_util
import hangeul_util as hg_util
import text_line_data
from TextRecognitionDataGenerator_data import HangeulJamoTextRecognitionDataGeneratorTextLineDataset as TextLineDataset

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
		#kernel_initializer = None
		kernel_initializer = tf.initializers.he_normal()
		#kernel_initializer = tf.initializers.he_uniform()
		#kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
		#kernel_initializer = tf.initializers.uniform_unit_scaling(factor=1.0)
		#kernel_initializer = tf.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
		#kernel_initializer = tf.initializers.glorot_normal()  # Xavier normal initialization.
		#kernel_initializer = tf.initializers.glorot_uniform()  # Xavier uniform initialization.

		if False:
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
			rnn_output = MyModel.create_bidirectionnal_rnn(rnn_input, seq_len)

		time_steps = rnn_input.shape.as_list()[1]  # Model output time-steps.
		print('***** Model output time-steps = {}.'.format(time_steps))

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
	def __init__(self, is_dataset_generated_at_runtime, data_dir_path=None, train_test_ratio=0.8):
		# Set parameters.
		# TODO [modify] >> Depends on a model.
		#	model_output_time_steps = image_width / width_downsample_factor or image_width / width_downsample_factor - 1.
		#	REF [function] >> MyModel.create_model().
		#width_downsample_factor = 4
		if False:
			image_height, image_width, image_channel = 32, 160, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
			model_output_time_steps = 39
		else:
			image_height, image_width, image_channel = 64, 320, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
			model_output_time_steps = 79

		#--------------------
		# Create a dataset.

		if is_dataset_generated_at_runtime:
			print('[SWL] Info: Start loading a Korean dictionary...')
			start_time = time.time()
			korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
			with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
				#korean_words = fd.readlines()
				#korean_words = fd.read().strip('\n')
				korean_words = fd.read().splitlines()
			print('[SWL] Info: End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

			print('[SWL] Info: Start creating a Korean dataset...')
			start_time = time.time()
			dataset = text_line_data.HangeulJamoRunTimeTextLineDataset(korean_word_set, image_height, image_width, image_channel)
			print('[SWL] Info: End creating a Korean dataset: {} secs.'.format(time.time() - start_time))
		else:
			# When using TextRecognitionDataGenerator_data.HangeulJamoTextRecognitionDataGeneratorTextLineDataset.
			self._dataset = TextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count=model_output_time_steps)

	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape)
			input_ph, output_ph, model_output_len_ph = model.placeholders

			model_output = model.create_model(input_ph, model_output_len_ph, self._dataset.num_classes, self._dataset.default_value)

			loss = model.get_loss(model_output['logit'], output_ph, model_output_len_ph)
			accuracy = model.get_accuracy(model_output['sparse_label'], output_ph)

			# Create a trainer.
			#optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
			##optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
			#optimizer = tf.keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-07)
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)
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
				print('[SWL] Info: End restoring a model from {}: {} secs.'.format(ckpt_filepath, time.time() - start_time))

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
					#batch_images, batch_labels_str, batch_sparse_labels_int = batch_data
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
					train_acc += len(list(filter(lambda x: x[1] == self._dataset.decode_label(x[0]), zip(batch_dense_labels_int, batch_data[1]))))
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
					#batch_images, batch_labels_str, batch_sparse_labels_int = batch_data
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
				# TODO [check] >> Accuracy is computed.
				if epoch % 10 == 0:
				#if False:
					start_time = time.time()
					val_loss, val_acc, num_examples = 0.0, 0.0, 0
					for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_test_batch_generator(batch_size, shuffle=False)):
						#batch_images, batch_labels_str, batch_sparse_labels_int = batch_data
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
						val_acc += len(list(filter(lambda x: x[1] == self._dataset.decode_label(x[0]), zip(batch_dense_labels_int, batch_data[1]))))
						num_examples += num_batch_examples

						# Show some results.
						if 0 == batch_step:
							preds, gts = list(), list()
							for count, (pred, gt) in enumerate(zip(batch_dense_labels_int, batch_data[1])):
								pred = self._dataset.decode_label(pred)
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
						#batch_images, batch_labels_str, batch_sparse_labels_int = batch_data
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
				print('[SWL] Info: End saving a model to {}: {} secs.'.format(saved_model_path, time.time() - start_time))

				sys.stdout.flush()
				time.sleep(0)
			print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))

			return history

	def test(self, checkpoint_dir_path, test_dir_path, batch_size):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape)
			input_ph, output_ph, model_output_len_ph = model.placeholders

			model_output = model.create_model(input_ph, model_output_len_ph, self._dataset.num_classes, self._dataset.default_value)

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
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(ckpt_filepath, time.time() - start_time))

			#--------------------
			print('[SWL] Info: Start testing...')
			start_time = time.time()
			inferences, ground_truths = list(), list()
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
				#batch_images, batch_labels_str, batch_sparse_labels_int = batch_data
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
			print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

			if inferences and ground_truths:
				#print('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0
				for pred, gt in zip(inferences, ground_truths):
					pred = np.array(list(map(lambda x: self._dataset.decode_label(x), pred)))

					correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(pred, gt))))
					total_word_count += len(gt)
					for ps, gs in zip(pred, gt):
						correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(ps, gs))))
						total_char_count += max(len(ps), len(gs))
					#correct_char_count += functools.reduce(lambda l, pgs: l + len(list(filter(lambda pg: pg[0] == pg[1], zip(pgs[0], pgs[1])))), zip(pred, gt), 0)
					#total_char_count += functools.reduce(lambda l, pg: l + max(len(pg[0]), len(pg[1])), zip(pred, gt), 0)
				print('Test: word accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
				print('Test: character accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

				# Output to a file.
				csv_filepath = os.path.join(test_dir_path, 'test_results.csv')
				with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
					writer = csv.writer(csvfile, delimiter=',')

					for pred, gt in zip(inferences, ground_truths):
						pred = np.array(list(map(lambda x: self._dataset.decode_label(x), pred)))
						writer.writerow([gt, pred])
			else:
				print('[SWL] Warning: Invalid test results.')

	def infer(self, checkpoint_dir_path, image_filepaths, inference_dir_path, batch_size):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape)
			input_ph, output_ph, model_output_len_ph = model.placeholders

			model_output = model.create_model(input_ph, model_output_len_ph, self._dataset.num_classes, self._dataset.default_value)

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
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(ckpt_filepath, time.time() - start_time))

			#--------------------
			print('[SWL] Info: Start loading images...')
			inf_images = self._dataset.load_images_from_files(image_filepaths, self._image_height, self._image_width, self._image_channel)
			print('[SWL] Info: End loading images: {} secs.'.format(time.time() - start_time))

			num_examples = len(inf_images)
			if batch_size is None:
				batch_size = num_examples
			if batch_size <= 0:
				raise ValueError('Invalid batch size: {}'.format(batch_size))

			indices = np.arange(num_examples)

			#--------------------
			print('[SWL] Info: Start inferring...')
			start_time = time.time()
			inferences = list()
			start_idx = 0
			while True:
				end_idx = start_idx + batch_size
				batch_indices = indices[start_idx:end_idx]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					batch_images = inf_images[batch_indices]
					if batch_images.size > 0:  # If batch_images is non-empty.
						# TODO [improve] >> CTC beam search decoding is too slow. It seems to run on CPU.
						#	If the number of classes increases, its computation time becomes much slower.
						batch_dense_labels_int = sess.run(
							model_output['dense_label'],
							feed_dict={
								input_ph: batch_images,
								model_output_len_ph: [model_output['time_step']] * len(batch_images)
							}
						)
						inferences.append(batch_dense_labels_int)

				if end_idx >= num_examples:
					break
				start_idx = end_idx
			print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

			if inferences:
				#print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				inferences_str = list()
				for inf in inferences:
					inferences_str.extend(map(lambda x: self._dataset.decode_label(x), inf))

				# Output to a file.
				csv_filepath = os.path.join(inference_dir_path, 'inference_results.csv')
				with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
					writer = csv.writer(csvfile, delimiter=',')

					for fpath, inf in zip(image_filepaths, inferences_str):
						writer.writerow([fpath, inf])
			else:
				print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#--------------------
	num_epochs, batch_size = 100, 128
	initial_epoch = 0
	is_trained, is_tested, is_inferred = True, True, False
	is_training_resumed = False

	is_dataset_generated_at_runtime = False
	if is_trained or is_tested:
		#data_dir_path = './kr_samples_100000'
		data_dir_path = './kr_samples_200000'
	else:
		data_dir_path = None
	train_test_ratio = 0.8

	#--------------------
	output_dir_path = None
	if not output_dir_path:
		output_dir_prefix = 'hangeul_jamo_carnet'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	test_dir_path = None
	if not test_dir_path:
		test_dir_path = os.path.join(output_dir_path, 'test')
	inference_dir_path = None
	if not inference_dir_path:
		inference_dir_path = os.path.join(output_dir_path, 'inference')

	#--------------------
	runner = MyRunner(is_dataset_generated_at_runtime, data_dir_path, train_test_ratio)

	if is_trained:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		history = runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch, is_training_resumed)

		#print('History =', history)
		#swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if is_tested:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return
		if test_dir_path and test_dir_path.strip() and not os.path.exists(test_dir_path):
			os.makedirs(test_dir_path, exist_ok=True)

		runner.test(checkpoint_dir_path, test_dir_path, batch_size)

	if is_inferred:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return
		if inference_dir_path and inference_dir_path.strip() and not os.path.exists(inference_dir_path):
			os.makedirs(inference_dir_path, exist_ok=True)

		image_filepaths = glob.glob('./images/*.jpg')  # TODO [modify] >>
		runner.test(checkpoint_dir_path, image_filepaths, inference_dir_path, batch_size)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
