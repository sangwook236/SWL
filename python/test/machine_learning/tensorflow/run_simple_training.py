#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, argparse, time, datetime
import numpy as np
import tensorflow as tf
#from sklearn import preprocessing
import cv2

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		# Load data.
		print('Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, self._test_images, self._test_labels = MyDataset.load_data(image_height, image_width, image_channel, num_classes)
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
	def preprocess_data(inputs, outputs, image_height, image_width, image_channel, num_classes):
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

			# Reshaping.
			inputs = np.reshape(inputs, (-1, image_height, image_width, image_channel))

		if outputs is not None:
			# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
			#outputs = swl_ml_util.to_one_hot_encoding(outputs, num_classes).astype(np.uint8)
			outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)

		return inputs, outputs

	@staticmethod
	def load_data(image_height, image_width, image_channel, num_classes):
		# Pixel value: [0, 255].
		(train_inputs, train_outputs), (test_inputs, test_outputs) = tf.keras.datasets.mnist.load_data()

		# Preprocessing.
		train_inputs, train_outputs = MyDataset.preprocess_data(train_inputs, train_outputs, image_height, image_width, image_channel, num_classes)
		test_inputs, test_outputs = MyDataset.preprocess_data(test_inputs, test_outputs, image_height, image_width, image_channel, num_classes)

		return train_inputs, train_outputs, test_inputs, test_outputs

#--------------------------------------------------------------------

class MyModel(object):
	def __init__(self):
		pass

	def create_model(self, input_tensor, num_classes):
		# Preprocessing.
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
			if 1 == num_classes:
				fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, name='dense')
				#fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			elif num_classes >= 2:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='dense')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

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
		image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
		self._num_classes = 10

		#--------------------
		# Create a dataset.

		self._dataset = MyDataset(image_height, image_width, image_channel, self._num_classes)

		#--------------------
		self._input_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_ph')
		self._output_ph = tf.placeholder(tf.float32, shape=[None, self._num_classes], name='output_ph')

	# Train and evaluate.
	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0):
		with tf.Session() as sess:
			# Create a model.
			model = MyModel()
			model_output = model.create_model(self._input_ph, self._num_classes)

			# Create a trainer.
			loss = model.get_loss(model_output, self._output_ph)
			accuracy = model.get_accuracy(model_output, self._output_ph)

			learning_rate = 0.001
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

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
				for batch_data, num_batch_examples in self._dataset.create_train_batch_generator(batch_size, shuffle=True):
					_, batch_loss, batch_accuracy = sess.run([train_op, loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1]})
					train_loss += batch_loss * num_batch_examples
					train_accuracy += batch_accuracy * num_batch_examples
					num_examples += num_batch_examples
				train_loss /= num_examples
				train_accuracy /= num_examples
				print('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_accuracy, time.time() - start_time))

				#--------------------
				start_time = time.time()
				val_loss, val_accuracy, num_examples = 0, 0, 0
				for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=False):
					batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={self._input_ph: batch_data[0], self._output_ph: batch_data[1]})
					val_loss += batch_loss * num_batch_examples
					val_accuracy += batch_accuracy * num_batch_examples
					num_examples += num_batch_examples
				val_loss /= num_examples
				val_accuracy /= num_examples
				print('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_accuracy, time.time() - start_time))
			print('End training: {} secs.'.format(time.time() - start_total_time))

			#--------------------
			print('Start saving a model...')
			start_time = time.time()
			saved_model_path = saver.save(sess, checkpoint_dir_path + '/model.ckpt')
			print('End saving a model: {} secs.'.format(time.time() - start_time))

	def infer(self, checkpoint_dir_path, batch_size=None, shuffle=False):
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
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, shuffle=shuffle):
				inferences.append(sess.run(model_output, feed_dict={self._input_ph: batch_data[0]}))
				test_labels.append(batch_data[1])
			print('End inferring: {} secs.'.format(time.time() - start_time))

			inferences, test_labels = np.vstack(inferences), np.vstack(test_labels)
			if inferences is not None:
				print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				if self._num_classes > 2:
					inferences = np.argmax(inferences, -1)
					ground_truths = np.argmax(test_labels, -1)
				elif 2 == self._num_classes:
					inferences = np.around(inferences)
					ground_truths = test_labels
				else:
					raise ValueError('Invalid number of classes')
				correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
				print('Inference: accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
			else:
				print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

def parse_command_line_options():
	parser = argparse.ArgumentParser(description='Train and test a CNN model for MNIST dataset.')

	parser.add_argument(
		'--train',
		action='store_true',
		help='Specify whether to train a model'
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
		help='Log level',
		default=None
	)

	return parser.parse_args()

def main():
	args = parse_command_line_options()

	if not args.train and not args.infer:
		print('[SWL] Error: At least one of command line options "--train" and "--infer" has to be specified.')

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.log_level:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.log_level)

	#--------------------
	num_epochs, batch_size = args.epoch, args.batch_size
	initial_epoch = 0

	checkpoint_dir_path = args.model_dir
	if not checkpoint_dir_path:
		output_dir_prefix = 'simple_training'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		#output_dir_suffix = '20190724T231604'
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')

	#--------------------
	runner = MyRunner()

	if args.train:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch)

	if args.infer:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.infer(checkpoint_dir_path)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
