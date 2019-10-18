#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, math, argparse, logging, time, datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#import sklearn
import cv2
import matplotlib.pyplot as plt
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

# REF [class] >> MyDataset in ${SWL_PYTHON_HOME}/test/machine_learning/tensorflow/run_simple_training.py.
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
	def train_data_length(self):
		return self._num_train_examples

	@property
	def test_data_length(self):
		return self._num_test_examples

	@property
	def train_data(self):
		return self._train_images, self._train_labels

	@property
	def test_data(self):
		return self._test_images, self._test_labels

	def create_train_batch_generator(self, batch_size, shuffle=True):
		return MyDataset.create_batch_generator(self._train_images, self._train_labels, batch_size, shuffle)

	def create_test_batch_generator(self, batch_size, shuffle=False):
		return MyDataset.create_batch_generator(self._test_images, self._test_labels, batch_size, shuffle)

	@staticmethod
	def create_batch_generator(data1, data2, batch_size, shuffle):
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

		if data2 is None:
			start_idx = 0
			while True:
				end_idx = start_idx + batch_size
				batch_indices = indices[start_idx:end_idx]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					batch_data1 = data1[batch_indices]
					if batch_data1.size > 0:  # If batch_data1 is non-empty.
						yield (batch_data1, None), batch_indices.size
					else:
						yield (None, None), 0
				else:
					yield (None, None), 0

				if end_idx >= num_examples:
					break
				start_idx = end_idx
		else:
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
				inputs = sklearn.preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = sklearn.preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = sklearn.preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = sklearn.preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
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

class MyDataSequence(tf.keras.utils.Sequence):
	def __init__(self, inputs, outputs, batch_size=None, shuffle=False):
		self.inputs, self.outputs = inputs, outputs
		self.batch_size = batch_size

		self.num_examples = len(self.inputs)
		if self.outputs is not None and len(self.outputs) != self.num_examples:
			raise ValueError('Invalid data size: {} != {}'.format(self.num_examples, len(self.outputs)))
		if self.batch_size is None:
			self.batch_size = self.num_examples
		if self.batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(self.batch_size))

		self.indices = np.arange(self.num_examples)
		if shuffle:
			np.random.shuffle(self.indices)

	def __len__(self):
		return math.ceil(self.num_examples / self.batch_size)

	def __getitem__(self, idx):
		start_idx = idx * self.batch_size
		end_idx = start_idx + self.batch_size
		batch_indices = self.indices[start_idx:end_idx]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			batch_input, batch_output = self.inputs[batch_indices], None if self.outputs is None else self.outputs[batch_indices]
			if batch_input.size > 0 and (batch_output is None or batch_output.size > 0):  # If batch_input and batch_output are non-empty.
				return (batch_input, batch_output)
		return (None, None)

#--------------------------------------------------------------------

class MyModel(object):
	@classmethod
	def create_model(cls, input_shape, num_classes):
		model = tf.keras.models.Sequential()

		# Layer 1.
		model.add(Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=2, strides=2))
		# Layer 2.
		model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
		model.add(MaxPooling2D(pool_size=2, strides=2))
		model.add(Flatten())
		# Layer 3.
		model.add(Dense(units=1024, activation='relu'))
		# Layer 4.
		model.add(Dense(units=num_classes, activation='softmax'))

		return model

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self):
		# Set parameters.
		self._use_keras_data_sequence, self._use_generator = True, False

		self._max_queue_size, self._num_workers = 10, 8
		self._use_multiprocessing = True

		#self._sess = tf.Session(config=config)
		#tf.keras.backend.set_session(self._sess)
		#tf.keras.backend.set_learning_phase(0)  # Sets the learning phase to 'test'.
		#tf.keras.backend.set_learning_phase(1)  # Sets the learning phase to 'train'.

		#--------------------
		# Create a dataset.
		image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
		num_classes = 10
		self._dataset = MyDataset(image_height, image_width, image_channel, num_classes)

	def train(self, model_filepath, model_checkpoint_filepath, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
		if is_training_resumed:
			# Restore a model.
			try:
				print('[SWL] Info: Start restoring a model...')
				start_time = time.time()
				"""
				# Load only the architecture of a model.
				model = tf.keras.models.model_from_json(json_string)
				#model = tf.keras.models.model_from_yaml(yaml_string)
				# Load only the weights of a model.
				model.load_weights(model_weight_filepath)
				"""
				# Load a model.
				model = tf.keras.models.load_model(model_filepath)
				print('[SWL] Info: End restoring a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except (ImportError, IOError):
				print('[SWL] Error: Failed to restore a model from {}.'.format(model_filepath))
				return
		else:
			# Create a model.
			model = MyModel.create_model(self._dataset.shape, self._dataset.num_classes)
			#print('Model summary =', model.summary())

		# Create a trainer.
		loss = tf.keras.losses.categorical_crossentropy
		optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)

		model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
		def lr_schedule(epoch, learning_rate):
			return learning_rate
		lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule)
		lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
		csv_logger_callback = tf.keras.callbacks.CSVLogger('./train_log.csv')  # epoch, acc, loss, lr, val_acc, val_loss.
		#callbacks = [model_checkpoint_callback, early_stopping_callback, lr_schedule_callback, lr_reduce_callback, csv_logger_callback]
		callbacks = [model_checkpoint_callback, early_stopping_callback, csv_logger_callback]

		#--------------------
		if is_training_resumed:
			print('[SWL] Info: Resume training...')
		else:
			print('[SWL] Info: Start training...')
		start_time = time.time()
		if self._use_keras_data_sequence:
			# Use Keras sequences.
			train_images, train_labels = self._dataset.train_data
			train_sequence = MyDataSequence(train_images, train_labels, batch_size=batch_size, shuffle=True)
			val_images, val_labels = self._dataset.test_data
			val_sequence = MyDataSequence(val_images, val_labels, batch_size=batch_size, shuffle=False)
			history = model.fit_generator(train_sequence, epochs=num_epochs, steps_per_epoch=None if batch_size is None else math.ceil(self._dataset.train_data_length / batch_size), validation_data=val_sequence, validation_steps=math.ceil(self._dataset.test_data_length / batch_size), shuffle=True, initial_epoch=initial_epoch, class_weight=None, max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing, callbacks=callbacks)
		elif self._use_generator:
			# Use generators.
			train_generator = self._dataset.create_train_batch_generator(batch_size, shuffle=True)
			val_generator = self._dataset.create_test_batch_generator(batch_size, shuffle=False)
			history = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=None if batch_size is None else math.ceil(self._dataset.train_data_length / batch_size), validation_data=val_generator, validation_steps=math.ceil(self._dataset.test_data_length / batch_size), shuffle=True, initial_epoch=initial_epoch, class_weight=None, max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing, callbacks=callbacks)
		else:
			train_images, train_labels = self._dataset.train_data
			history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs, validation_split=0.2, shuffle=True, initial_epoch=initial_epoch, class_weight=None, sample_weight=None, callbacks=callbacks)
		print('[SWL] Info: End training: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('[SWL] Info: Start evaluating...')
		start_time = time.time()
		if self._use_keras_data_sequence:
			# Use a Keras sequence.
			val_images, val_labels = self._dataset.test_data
			val_sequence = MyDataSequence(val_images, val_labels, batch_size=batch_size, shuffle=False)
			score = model.evaluate_generator(val_sequence, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		elif self._use_generator:
			# Use a generator.
			val_generator = self._dataset.create_test_batch_generator(batch_size, shuffle=False)
			score = model.evaluate_generator(val_generator, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		else:
			val_images, val_labels = self._dataset.test_data
			score = model.evaluate(val_images, val_labels, batch_size=batch_size, sample_weight=None)
		print('\tValidation: loss = {:.6f}, accuracy = {:.6f}.'.format(*score))
		print('[SWL] Info: End evaluating: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('[SWL] Info: Start saving a model...')
		start_time = time.time()
		"""
		# Save only the architecture of a model.
		json_string = model.to_json()
		#yaml_string = model.to_yaml()
		# Save only the weights of a model.
		model.save_weights(model_weight_filepath)
		"""
		# Save a model.
		model.save(model_filepath)
		print('[SWL] Info: End saving a model to {}: {} secs.'.format(model_filepath, time.time() - start_time))

		return history.history

	def test(self, model_filepath, batch_size=None, shuffle=False):
		# Load a model.
		try:
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			"""
			# Load only the architecture of a model.
			model = tf.keras.models.model_from_json(json_string)
			#model = tf.keras.models.model_from_yaml(yaml_string)
			# Load only the weights of a model.
			model.load_weights(model_weight_filepath)
			"""
			# Load a model.
			model = tf.keras.models.load_model(model_filepath)
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except (ImportError, IOError):
			print('[SWL] Error: Failed to load a model from {}.'.format(model_filepath))
			return

		#--------------------
		print('[SWL] Info: Start testing...')
		start_time = time.time()
		if self._use_keras_data_sequence:
			# Use a Keras sequence.
			test_images, test_labels = self._dataset.test_data
			test_sequence = MyDataSequence(test_images, test_labels, batch_size=batch_size, shuffle=shuffle)
			inferences = model.predict_generator(test_sequence, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		elif self._use_generator:
			# Use a generator.
			test_generator = self._dataset.create_test_batch_generator(batch_size, shuffle=shuffle)
			inferences = model.predict_generator(test_generator, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
			# TODO [implement] >> self._test_labels have to be generated.
			test_labels = self._dataset.test_data[1]
		else:
			test_images, test_labels = self._dataset.test_data
			inferences = model.predict(test_images, batch_size=batch_size)
		print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

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

	def infer(self, model_filepath, batch_size=None, shuffle=False):
		# Load a model.
		try:
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			"""
			# Load only the architecture of a model.
			model = tf.keras.models.model_from_json(json_string)
			#model = tf.keras.models.model_from_yaml(yaml_string)
			# Load only the weights of a model.
			model.load_weights(model_weight_filepath)
			"""
			# Load a model.
			model = tf.keras.models.load_model(model_filepath)
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except (ImportError, IOError):
			print('[SWL] Error: Failed to load a model from {}.'.format(model_filepath))
			return

		#--------------------
		inf_images, _ = self._dataset.test_data

		#--------------------
		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		if self._use_keras_data_sequence:
			# Use a Keras sequence.
			test_sequence = MyDataSequence(inf_images, None, batch_size=batch_size, shuffle=shuffle)
			inferences = model.predict_generator(test_sequence, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		elif self._use_generator:
			# Use a generator.
			test_generator = self._dataset.create_batch_generator(inf_images, None, batch_size, shuffle=shuffle)
			inferences = model.predict_generator(test_generator, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		else:
			if shuffle:
				np.random.shuffle(inf_images)
			inferences = model.predict(inf_images, batch_size=batch_size)
		print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

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
			print('[SWL] Warning: Invalid test results.')

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
		'--model_file',
		type=str,
		#nargs='?',
		help='The model file path where a trained model is saved or a pretrained model is loaded',
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
		default=128
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

	model_filepath = args.model_file
	if model_filepath:
		output_dir_path = os.path.dirname(model_filepath)
	else:
		output_dir_prefix = 'simple_training'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		model_filepath = os.path.join(output_dir_path, 'model.hdf5')
		#model_weight_filepath = os.path.join(output_dir_path, 'model_weights.hdf5')

	#--------------------
	runner = MyRunner()

	if args.train:
		model_checkpoint_filepath = os.path.join(output_dir_path, 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		history = runner.train(model_filepath, model_checkpoint_filepath, num_epochs, batch_size, initial_epoch, is_training_resumed)

		#print('History =', history)
		swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if args.test:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return

		runner.test(model_filepath)

	if args.infer:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return

		runner.infer(model_filepath)

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 30

if '__main__' == __name__:
	main()
