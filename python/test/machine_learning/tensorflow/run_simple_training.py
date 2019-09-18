#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, argparse, logging, time, datetime
import numpy as np
import tensorflow as tf
#from sklearn import preprocessing
import cv2
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes

		#--------------------
		# Load data.
		print('Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, self._test_images, self._test_labels = MyDataset._load_data(self._image_height, self._image_width, self._image_channel, self._num_classes)
		print('End loading dataset: {} secs.'.format(time.time() - start_time))

		self._num_train_examples = len(self._train_images)
		if len(self._train_labels) != self._num_train_examples:
			raise ValueError('Invalid train data length: {} != {}'.format(self._num_train_examples, len(self._train_labels)))
		self._num_test_examples = len(self._test_images)
		if len(self._test_labels) != self._num_test_examples:
			raise ValueError('Invalid test data length: {} != {}'.format(self._num_test_examples, len(self._test_labels)))

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_labels.shape, self._train_labels.dtype, np.min(self._train_labels), np.max(self._train_labels)))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_labels.shape, self._test_labels.dtype, np.min(self._test_labels), np.max(self._test_labels)))

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def test_data(self):
		return self._test_images, self._test_labels

	def create_train_batch_generator(self, batch_size, shuffle=True):
		return MyDataset._create_batch_generator(self._train_images, self._train_labels, batch_size, shuffle)

	def create_test_batch_generator(self, batch_size, shuffle=False):
		return MyDataset._create_batch_generator(self._test_images, self._test_labels, batch_size, shuffle)

	@staticmethod
	def _create_batch_generator(data1, data2, batch_size, shuffle):
		num_examples = len(data1)
		if len(data2) != num_examples:
			raise ValueError('Invalid data length: {} != {}'.format(num_examples, len(data2)))
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
				batch_data1, batch_data2 = data1[batch_indices], data2[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0:  # If batch_data1 and batch_data2 are non-empty.
					yield (batch_data1, batch_data2), batch_indices.size
				else:
					yield (None, None), 0
			else:
				yield (None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	@staticmethod
	def _preprocess(inputs, outputs, image_height, image_width, image_channel, num_classes):
		if inputs is not None:
			# Contrast limited adaptive histogram equalization (CLAHE).
			#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			#inputs = np.array([clahe.apply(inp) for inp in inputs])

			# Normalization, standardization, etc.
			inputs = inputs.astype(np.float32)

			if False:
				inputs = preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
			elif True:
				inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
			elif False:
				in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
				out_min, out_max = 0, 1 #-1, 1
				inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
			elif False:
				inputs /= 255.0  # Normalization.

			# Reshape.
			inputs = np.reshape(inputs, (-1, image_height, image_width, image_channel))

		if outputs is not None:
			# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
			#outputs = swl_ml_util.to_one_hot_encoding(outputs, num_classes).astype(np.uint8)
			outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)

		return inputs, outputs

	@staticmethod
	def _load_data(image_height, image_width, image_channel, num_classes):
		# Pixel value: [0, 255].
		(train_inputs, train_outputs), (test_inputs, test_outputs) = tf.keras.datasets.mnist.load_data()

		# Preprocess.
		train_inputs, train_outputs = MyDataset._preprocess(train_inputs, train_outputs, image_height, image_width, image_channel, num_classes)
		test_inputs, test_outputs = MyDataset._preprocess(test_inputs, test_outputs, image_height, image_width, image_channel, num_classes)

		return train_inputs, train_outputs, test_inputs, test_outputs

#--------------------------------------------------------------------

class MyModel(object):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		self._num_classes = num_classes
		self._input_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_ph')
		self._output_ph = tf.placeholder(tf.float32, shape=[None, self._num_classes], name='output_ph')

	@property
	def placeholders(self):
		return self._input_ph, self._output_ph

	def create_model(self, input_tensor):
		# Preprocess.
		with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
			input_tensor = tf.nn.local_response_normalization(input_tensor, depth_radius=5, bias=1, alpha=1, beta=0.5, name='lrn')

		#--------------------
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv')
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv')
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(conv2, name='flatten')

			fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='dense')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			if 1 == self._num_classes:
				fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, name='dense')
				#fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			elif self._num_classes >= 2:
				fc2 = tf.layers.dense(fc1, self._num_classes, activation=tf.nn.softmax, name='dense')
				#fc2 = tf.layers.dense(fc1, self._num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			else:
				assert self._num_classes > 0, 'Invalid number of classes.'

			return fc2

	def get_loss(self, y, t):
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
			return loss

	def get_accuracy(self, y, t):
		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			return accuracy

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self):
		# Create a dataset.
		image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
		num_classes = 10
		self._dataset = MyDataset(image_height, image_width, image_channel, num_classes)

	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape, self._dataset.num_classes)
			input_ph, output_ph = model.placeholders

			model_output = model.create_model(input_ph)

			loss = model.get_loss(model_output, output_ph)
			accuracy = model.get_accuracy(model_output, output_ph)

			# Create a trainer.
			learning_rate = 0.001
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

			train_op = optimizer.minimize(loss)

			# Create a saver.
			saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with tf.Session(graph=graph) as sess:
			sess.run(initializer)

			if is_training_resumed:
				# Restore a model.
				print('[SWL] Info: Start restoring a model...')
				start_time = time.time()
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
				ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
				#ckpt_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
				if ckpt_filepath:
					initial_epoch = int(ckpt_filepath.split('-')[1])
					saver.restore(session, ckpt_filepath)
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

				#--------------------
				start_time = time.time()
				train_loss, train_acc, num_examples = 0.0, 0.0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, shuffle=True)):
					_, batch_loss, batch_accuracy = sess.run([train_op, loss, accuracy], feed_dict={input_ph: batch_data[0], output_ph: batch_data[1]})
					train_loss += batch_loss * num_batch_examples
					train_acc += batch_accuracy * num_batch_examples
					num_examples += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				train_acc /= num_examples
				print('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)

				#--------------------
				start_time = time.time()
				val_loss, val_acc, num_examples = 0.0, 0.0, 0
				for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
					batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={input_ph: batch_data[0], output_ph: batch_data[1]})
					val_loss += batch_loss * num_batch_examples
					val_acc += batch_accuracy * num_batch_examples
					num_examples += num_batch_examples
				val_loss /= num_examples
				val_acc /= num_examples
				print('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

				history['val_loss'].append(val_loss)
				history['val_acc'].append(val_acc)

				sys.stdout.flush()
				time.sleep(0)
			print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))

			#--------------------
			print('[SWL] Info: Start saving a model...')
			start_time = time.time()
			saved_model_path = saver.save(sess, os.path.join(checkpoint_dir_path, 'model.ckpt'))
			print('[SWL] Info: End saving a model to {}: {} secs.'.format(saved_model_path, time.time() - start_time))

			return history

	def test(self, checkpoint_dir_path, batch_size, shuffle=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape, self._dataset.num_classes)
			input_ph, output_ph = model.placeholders

			model_output = model.create_model(input_ph)

			# Create a saver.
			saver = tf.train.Saver()

		with tf.Session(graph=graph) as sess:
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
			inferences, test_labels = list(), list()
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=shuffle):
				inferences.append(sess.run(model_output, feed_dict={input_ph: batch_data[0]}))
				test_labels.append(batch_data[1])
			print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

			inferences, test_labels = np.vstack(inferences), np.vstack(test_labels)
			if inferences is not None and test_labels is not None:
				print('\tTest: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				if self._dataset.num_classes > 2:
					inferences = np.argmax(inferences, -1)
					ground_truths = np.argmax(test_labels, -1)
				elif 2 == self._dataset.num_classes:
					inferences = np.around(inferences)
					ground_truths = test_labels
				else:
					raise ValueError('Invalid number of classes')

				correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
				print('\tTest: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
			else:
				print('[SWL] Warning: Invalid test results.')

	def infer(self, checkpoint_dir_path, batch_size=None, shuffle=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape, self._dataset.num_classes)
			input_ph, output_ph = model.placeholders

			model_output = model.create_model(input_ph)

			# Create a saver.
			saver = tf.train.Saver()

		with tf.Session(graph=graph) as sess:
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
			inf_images, _ = self._dataset.test_data

			num_examples = len(inf_images)
			if batch_size is None:
				batch_size = num_examples
			if batch_size <= 0:
				raise ValueError('Invalid batch size: {}'.format(batch_size))

			indices = np.arange(num_examples)
			if shuffle:
				np.random.shuffle(indices)

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
					batch_data = inf_images[batch_indices]
					if batch_data.size > 0:  # If batch_data is non-empty.
						inferences.append(sess.run(model_output, feed_dict={input_ph: batch_data}))

				if end_idx >= num_examples:
					break
				start_idx = end_idx
			print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

			inferences = np.vstack(inferences)
			if inferences is not None:
				print('\tInference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				if self._dataset.num_classes > 2:
					inferences = np.argmax(inferences, -1)
				elif 2 == self._dataset.num_classes:
					inferences = np.around(inferences)
				else:
					raise ValueError('Invalid number of classes')

				print('\tInference results: index,inference')
				for idx, inf in enumerate(inferences):
					print('{},{}'.format(idx, inf))
					if (idx + 1) >= 10:
						break
			else:
				print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

def parse_command_line_options():
	parser = argparse.ArgumentParser(description='Train, test, or infer a CNN model for MNIST dataset.')

	parser.add_argument(
		'--train',
		action='store_true',
		help='Specify whether to train a model'
	)
	parser.add_argument(
		'--test',
		action='store_true',
		help='Specify whether to test a trained model'
	)
	parser.add_argument(
		'--infer',
		action='store_true',
		help='Specify whether to infer by a trained model'
	)
	parser.add_argument(
		'-r',
		'--resume',
		action='store_true',
		help='Specify whether to resume training'
	)
	parser.add_argument(
		'-m',
		'--model_dir',
		type=str,
		#nargs='?',
		help='The model directory path where a trained model is saved or a pretrained model is loaded',
		#required=True,
		default=None
	)
	parser.add_argument(
		'-tr',
		'--train_data_dir',
		type=str,
		#nargs='?',
		help='The directory path of training data',
		default='./train_data'
	)
	parser.add_argument(
		'-te',
		'--test_data_dir',
		type=str,
		#nargs='?',
		help='The directory path of test data',
		default='./test_data'
	)
	parser.add_argument(
		'-e',
		'--epoch',
		type=int,
		help='Number of epochs',
		default=30
	)
	parser.add_argument(
		'-b',
		'--batch_size',
		type=int,
		help='Batch size',
		default=512
	)
	parser.add_argument(
		'-g',
		'--gpu',
		type=str,
		help='Specify GPU to use',
		default='0'
	)
	parser.add_argument(
		'-l',
		'--log_level',
		type=int,
		help='Log level, [0, 50]',  # {NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL}.
		default=None
	)

	return parser.parse_args()

def set_logger(log_level):
	"""
	# When log_level is string.
	if log_level is not None:
		log_level = getattr(logging, log_level.upper(), None)
		if not isinstance(log_level, int):
			raise ValueError('Invalid log level: {}'.format(log_level))
	else:
		log_level = logging.WARNING
	"""
	print('[SWL] Info: Log level = {}.'.format(log_level))

	handler = logging.handlers.RotatingFileHandler('./simple_training.log', maxBytes=5000, backupCount=10)
	formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	handler.setFormatter(formatter)

	#logger = logging.getLogger(__name__)
	logger = logging.getLogger('simple_training_logger')
	logger.addHandler(handler) 
	logger.setLevel(log_level)

	return logger

def main():
	args = parse_command_line_options()

	if not args.train and not args.test:
		print('[SWL] Error: At least one of command line options "--train", "--test", and "--infer" has to be specified.')
		return

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.log_level:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#logger = set_logger(args.log_level)

	#--------------------
	num_epochs, batch_size = args.epoch, args.batch_size
	initial_epoch = 0
	is_training_resumed = False

	#--------------------
	output_dir_path = None
	if not output_dir_path:
		output_dir_prefix = 'simple_training'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = args.model_dir
	if not checkpoint_dir_path:
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')

	#--------------------
	runner = MyRunner()

	if args.train:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		history = runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch, is_training_resumed)

		#print('History =', history)
		swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if args.test:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.test(checkpoint_dir_path, batch_size)

	if args.infer:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.infer(checkpoint_dir_path)

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 30

if '__main__' == __name__:
	main()
