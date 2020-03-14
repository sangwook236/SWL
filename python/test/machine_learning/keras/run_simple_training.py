#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, math, shutil, argparse, logging, logging.handlers, time, datetime
import numpy as np
import tensorflow as tf
#import sklearn
#import cv2
#import matplotlib.pyplot as plt
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

def swish(x, beta=1):
	return (x * tf.keras.backend.sigmoid(beta * x))
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})
#tf.keras.layers.Dense(256, activation='swish')

#--------------------------------------------------------------------

# REF [class] >> MyDataset in ${SWL_PYTHON_HOME}/test/machine_learning/tensorflow/run_simple_training.py.
class MyDataset(object):
	def __init__(self, image_height, image_width, image_channel, num_classes, logger):
		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes

		#--------------------
		# Load data.
		logger.info('Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, self._test_images, self._test_labels = MyDataset._load_data(self._image_height, self._image_width, self._image_channel, self._num_classes)
		logger.info('End loading dataset: {} secs.'.format(time.time() - start_time))

		self._num_train_examples = len(self._train_images)
		if len(self._train_labels) != self._num_train_examples:
			raise ValueError('Invalid train data length: {} != {}'.format(self._num_train_examples, len(self._train_labels)))
		self._num_test_examples = len(self._test_images)
		if len(self._test_labels) != self._num_test_examples:
			raise ValueError('Invalid test data length: {} != {}'.format(self._num_test_examples, len(self._test_labels)))

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

	def show_data_info(self, logger):
		logger.info('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		logger.info('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_labels.shape, self._train_labels.dtype, np.min(self._train_labels), np.max(self._train_labels)))
		logger.info('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		logger.info('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_labels.shape, self._test_labels.dtype, np.min(self._test_labels), np.max(self._test_labels)))

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
		model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', input_shape=input_shape))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
		# Layer 2.
		model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
		model.add(tf.keras.layers.Flatten())
		# Layer 3.
		model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
		# Layer 4.
		model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

		return model

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, logger):
		# Set parameters.
		self._logger = logger
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
		self._dataset = MyDataset(image_height, image_width, image_channel, num_classes, self._logger)
		self._dataset.show_data_info(self._logger)

	def train(self, model_filepath, model_checkpoint_filepath, output_dir_path, batch_size, final_epoch, initial_epoch=0, is_training_resumed=False):
		if is_training_resumed:
			# Restore a model.
			try:
				self._logger.info('Start restoring a model...')
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
				self._logger.info('End restoring a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except (ImportError, IOError):
				self._logger.error('Failed to restore a model from {}.'.format(model_filepath))
				return
		else:
			# Create a model.
			model = MyModel.create_model(self._dataset.shape, self._dataset.num_classes)
			#model.summary()

		# Create a trainer.
		loss = tf.keras.losses.categorical_crossentropy
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
		#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.9, epsilon=1.0e-7, centered=False)
		#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)  # Not good.

		model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

		def schedule_learning_rate(epoch, learning_rate):
			if epoch < 10:
				return 1.0e-2
			elif epoch < 20:
				return 1.0e-3
			elif epoch < 30:
				return 1.0e-4
			else:
				return 1.0e-4 * tf.math.exp(0.1 * (30 - epoch))
		lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule_learning_rate, verbose=0)
		lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
		if True:
			timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			csv_log_filepath = os.path.join(output_dir_path, 'train_log_{}.csv'.format(timestamp))
			file_logger_callback = tf.keras.callbacks.CSVLogger(csv_log_filepath, separator=',', append=False)  # epoch, acc, loss, lr, val_acc, val_loss.
		else:
			import json
			timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			json_log_filepath = os.path.join(output_dir_path, 'train_log_{}.json'.format(timestamp))
			json_log = open(json_log_filepath, mode='wt', encoding='UTF8', buffering=1)
			file_logger_callback = tf.keras.callbacks.LambdaCallback(
				on_epoch_end=lambda epoch, logs: json_log.write(json.dumps({'epoch': epoch, 'acc': logs['acc'], 'loss': logs['loss'], 'lr': logs['lr'], 'val_acc': logs['val_acc'], 'val_loss': logs['val_loss']}) + '\n'),
				on_train_end=lambda logs: json_log.close()
			)
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
		#callbacks = [lr_schedule_callback, lr_reduce_callback, early_stopping_callback, file_logger_callback, model_checkpoint_callback]
		callbacks = [early_stopping_callback, file_logger_callback, model_checkpoint_callback]

		num_epochs = final_epoch - initial_epoch

		#--------------------
		if is_training_resumed:
			self._logger.info('Resume training...')
		else:
			self._logger.info('Start training...')
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
		self._logger.info('End training: {} secs.'.format(time.time() - start_time))

		#--------------------
		self._logger.info('Start evaluating...')
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
		self._logger.info('\tValidation: loss = {:.6f}, accuracy = {:.6f}.'.format(*score))
		self._logger.info('End evaluating: {} secs.'.format(time.time() - start_time))

		#--------------------
		self._logger.info('Start saving a model...')
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
		self._logger.info('End saving a model to {}: {} secs.'.format(model_filepath, time.time() - start_time))

		return history.history

	def test(self, model, batch_size=None, shuffle=False):
		self._logger.info('Start testing...')
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
		self._logger.info('End testing: {} secs.'.format(time.time() - start_time))

		if inferences is not None and test_labels is not None:
			self._logger.info('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			if self._dataset.num_classes > 2:
				inferences = np.argmax(inferences, -1)
				ground_truths = np.argmax(test_labels, -1)
			elif 2 == self._dataset.num_classes:
				inferences = np.around(inferences)
				ground_truths = test_labels
			else:
				raise ValueError('Invalid number of classes')

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			self._logger.info('Test: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			self._logger.warning('Invalid test results.')

	def infer(self, model, batch_size=None, shuffle=False):
		inf_images, _ = self._dataset.test_data

		#--------------------
		self._logger.info('Start inferring...')
		start_time = time.time()
		if self._use_keras_data_sequence:
			# Use a Keras sequence.
			test_sequence = MyDataSequence(inf_images, None, batch_size=batch_size, shuffle=shuffle)
			inferences = model.predict_generator(test_sequence, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		elif self._use_generator:
			# Use a generator.
			test_generator = MyDataset.create_batch_generator(inf_images, None, batch_size, shuffle=shuffle)
			inferences = model.predict_generator(test_generator, steps=None if batch_size is None else math.ceil(self._dataset.test_data_length / batch_size), max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		else:
			if shuffle:
				np.random.shuffle(inf_images)
			inferences = model.predict(inf_images, batch_size=batch_size)
		self._logger.info('End inferring: {} secs.'.format(time.time() - start_time))

		if inferences is not None:
			self._logger.info('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			if self._dataset.num_classes > 2:
				inferences = np.argmax(inferences, -1)
			elif 2 == self._dataset.num_classes:
				inferences = np.around(inferences)
			else:
				raise ValueError('Invalid number of classes')

			results = {idx: inf for idx, inf in enumerate(inferences) if idx < 100}
			self._logger.info('Inference results (index: inference): {}.'.format(results))
		else:
			self._logger.info('Invalid inference results.')

	def load_evaluation_model(self, model_filepath):
		try:
			self._logger.info('Start loading a model...')
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
			self._logger.info('End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))

			return model
		except (ImportError, IOError):
			self._logger.error('Failed to load a model from {}.'.format(model_filepath))
			return None

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
		'-o',
		'--out_dir',
		type=str,
		#nargs='?',
		help='The output directory path to save results such as images and log',
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
		help='Final epoch',
		default=30
	)
	parser.add_argument(
		'-b',
		'--batch_size',
		type=int,
		help='Batch size',
		default=32
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
		help='Log level, [0, 50]',  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
		default=None
	)

	return parser.parse_args()

def get_logger(name, log_level=None, log_dir_path=None, is_rotating=True):
	if not log_level: log_level = logging.INFO
	if not log_dir_path: log_dir_path = './log'
	if not os.path.isdir(log_dir_path):
		os.mkdir(log_dir_path)

	log_filepath = os.path.join(log_dir_path, (name if name else 'swl') + '.log')
	if is_rotating:
		file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10000000, backupCount=10)
	else:
		file_handler = logging.FileHandler(log_filepath)
	stream_handler = logging.StreamHandler()

	formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)s][%(asctime)s] [SWL] %(message)s')
	#formatter = logging.Formatter('[%(levelname)s][%(asctime)s] [SWL] %(message)s')
	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def main():
	args = parse_command_line_options()

	logger = get_logger(os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, './log', is_rotating=True)
	logger.info('----------------------------------------------------------------------')
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Python version: {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('TensorFlow version: {}.'.format(tf.__version__))

	if not args.train and not args.test and not args.infer:
		logger.error('At least one of command line options "--train", "--test", and "--infer" has to be specified.')
		return

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.log_level:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#--------------------
	is_training_resumed = args.resume
	initial_epoch, final_epoch, batch_size = 0, args.epoch, args.batch_size

	model_filepath, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	if model_filepath:
		if not output_dir_path:
			output_dir_path = os.path.dirname(model_filepath)
	else:
		if not output_dir_path:
			output_dir_prefix = 'simple_training'
			output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		model_filepath = os.path.join(output_dir_path, 'model.hdf5')
		#model_weight_filepath = os.path.join(output_dir_path, 'model_weights.hdf5')

	#--------------------
	runner = MyRunner(logger)

	if args.train:
		model_checkpoint_filepath = os.path.join(output_dir_path, 'model_ckpt.{epoch:04d}-{val_loss:.5f}.hdf5')
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		# Copy the model file to the output directory.
		new_model_filepath = os.path.join(output_dir_path, os.path.basename(model_filepath))
		if os.path.exists(model_filepath) and not os.path.samefile(model_filepath, new_model_filepath):
			try:
				shutil.copyfile(model_filepath, new_model_filepath)
			except (FileNotFoundError, PermissionError) as ex:
				logger.error('Failed to copy a model, {}: {}.'.format(model_filepath, ex))
				return
		model_filepath = new_model_filepath

		history = runner.train(model_filepath, model_checkpoint_filepath, output_dir_path, batch_size, final_epoch, initial_epoch, is_training_resumed)

		#logger.info('Train history = {}.'.format(history))
		swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if args.test or args.infer:
		if model_filepath and os.path.exists(model_filepath):
			model = runner.load_evaluation_model(model_filepath)

			if args.test and model:
				runner.test(model)

			if args.infer and model:
				runner.infer(model)
		else:
			logger.error('Model file, {} does not exist.'.format(model_filepath))

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 20 --gpu 0

if '__main__' == __name__:
	main()
