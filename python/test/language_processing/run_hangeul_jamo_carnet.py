#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, datetime, random
import numpy as np
import tensorflow as tf
import cv2

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, image_height, image_width, image_channel, num_classes, eos_token_label, blank_label):
		print('Start loading dataset...')
		load_data(image_height, image_width, num_classes, BATCH_SIZE)
		start_time = time.time()
		train_images, train_labels, test_images, test_labels = 
		print('End loading dataset: {} secs.'.format(time.time() - start_time))

		#max_label_len = max(train_labels.shape[-1], test_labels.shape[-1])
		#train_label_lengths, test_label_lengths = np.full((train_labels.shape[0], 1), train_labels.shape[-1]), np.full((test_labels.shape[0], 1), test_labels.shape[-1])

		# FIXME [improve] >> Stupid implementation.
		if False:
			train_model_output_lengths = np.full((train_images.shape[0], 1), image_width // width_downsample_factor)
			test_model_output_lengths = np.full((test_images.shape[0], 1), image_width // width_downsample_factor)
		else:
			train_model_output_lengths = np.full((train_images.shape[0], 1), image_width // width_downsample_factor - 2)  # See get_loss().
			test_model_output_lengths = np.full((test_images.shape[0], 1), image_width // width_downsample_factor - 2)  # See get_loss().

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_images.shape, train_images.dtype, np.min(train_images), np.max(train_images)))
		print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_labels.shape, train_labels.dtype, np.min(train_labels), np.max(train_labels)))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_images.shape, test_images.dtype, np.min(test_images), np.max(test_images)))
		print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_labels.shape, test_labels.dtype, np.min(test_labels), np.max(test_labels)))

#--------------------------------------------------------------------

class MyModel(object):
	def __init__(self):
		pass

	def create_model(self, input_tensor, num_classes):
		dropout_rate = 0.5

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(input_tensor, filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.layers.conv2d(conv1, filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

			#conv1 = tf.layers.dropout(conv1, rate=dropout_rate, training=is_training, name='dropout')
			conv1 = tf.layers.batch_normalization(conv1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.layers.conv2d(conv2, filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			#conv2 = tf.layers.dropout(conv2, rate=dropout_rate, training=is_training, name='dropout')
			conv2 = tf.layers.batch_normalization(conv2, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool')

		with tf.variable_scope('fc3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense')
			conv3 = tf.nn.relu(conv3, name='relu')
			# TODO [check] >> Which is better, dropout or batch normalization?
			#conv3 = tf.layers.dropout(conv3, rate=dropout_rate, training=is_training, name='dropout')
			conv3 = tf.layers.batch_normalization(conv3, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')

		with tf.variable_scope('fc4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense')
			#conv4 = tf.nn.relu(conv4, name='relu')

		with tf.variable_scope('atrous_conv5', reuse=tf.AUTO_REUSE):
			if True:
				#conv5 = tf.nn.atrous_conv2d(conv4, filters=(3, 3, 512, 512), rate=2, padding='valid', name='atrous_conv1')
				conv5 = tf.layers.conv2d(conv4, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
				conv5 = tf.nn.relu(conv5, name='relu1')

				#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, 512, 512), rate=2, padding='valid', name='atrous_conv2')
				conv5 = tf.layers.conv2d(conv5, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv2')
				conv5 = tf.nn.relu(conv5, name='relu2')

				#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, 512, 512), rate=2, padding='valid', name='atrous_conv3')
				conv5 = tf.layers.conv2d(conv5, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv3')
				conv5 = tf.nn.relu(conv5, name='relu3')
			else:
				conv5 = tf.pad(conv2, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

				#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, num_classes, num_classes), rate=2, padding='valid', name='atrous_conv1')
				conv5 = tf.layers.conv2d(conv5, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
				conv5 = tf.nn.relu(conv5, name='relu1')

				conv5 = tf.pad(conv5, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

				#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, num_classes, num_classes), rate=4, padding='valid', name='atrous_conv2')
				conv5 = tf.layers.conv2d(conv5, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
				conv5 = tf.nn.relu(conv5, name='relu2')

				conv5 = tf.pad(conv5, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', constant_values=0, name='pad3')

				#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, num_classes, num_classes), rate=8, padding='valid', name='atrous_conv3')
				conv5 = tf.layers.conv2d(conv5, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
				conv5 = tf.nn.relu(conv5, name='relu3')

			#conv5 = tf.layers.dropout(conv5, rate=dropout_rate, training=is_training, name='dropout')
			conv5 = tf.layers.batch_normalization(conv5, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')

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

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self):
		image_height, image_width, image_channel = 64, 320, 1
		num_labels =
		width_downsample_factor = 4  # Fixed.

		if False:
			self._num_classes = num_labels
		elif False:
			# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
			self._num_classes = num_labels + 1  # #labels + blank label.
			self._blank_label = self._num_classes - 1
		else:
			# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
			self._num_classes = num_labels + 1 + 1  # #labels + EOS + blank label.
			self._eos_token_label = self._num_classes - 2
			self._blank_label = self._num_classes - 1

		#--------------------
		# Create a dataset.

		self._dataset = MyDataset()

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
			model = MyModel()
			model_output = model.create_model(self._input_ph, self._num_classes)

			# Create a trainer.
			loss = model.get_loss(model_output, self._output_ph, self._model_output_length_ph)
			accuracy = model.get_accuracy(model_output, self._output_ph, self._model_output_length_ph, default_value=self._eos_token_label)

			learning_rate = 0.001
			#optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1.0e-6, momentum=0.0, nesterov=True)
			#optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon=1.0e-7, centered=False)
			#optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1.0e-7, amsgrad=False)
			#optimizer = tf.keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1.0e-7)
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1.0e-7, use_locking=False)

			train_op = optimizer.minimize(loss)

			saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			#--------------------
			print('Start training...')
			start_total_time = time.time()
			sess.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				print('Epoch {}:'.format(epoch + 1))

				#--------------------
				start_time = time.time()
				train_loss, train_accuracy, num_examples = 0, 0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					#_, batch_loss, batch_accuracy = sess.run([train_op, loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1]})
					_, batch_loss = sess.run([train_op, loss], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1]})
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

	def infer(self, checkpoint_dir_path, batch_size=None):
		with tf.Session() as sess:
			# Create a model.
			model = MyModel()
			model_output = model.create_model(self._input_ph, self._num_classes)

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
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
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
	output_dir_prefix = 'hangeul_jamo_carnet'
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
