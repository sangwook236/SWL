#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, argparse, logging, time, datetime, functools
import numpy as np
import tensorflow as tf
#from sklearn import preprocessing
import cv2
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, batch_size):
		self._num_classes = 10

		#--------------------
		# Load data.
		print('Start loading dataset...')
		start_time = time.time()
		self._train_ds, self._test_ds, self._shape = MyDataset._load_data(self._num_classes, batch_size)
		print('End loading dataset: {} secs.'.format(time.time() - start_time))

	@property
	def shape(self):
		return self._shape  # (image height, image width, image channel).

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def train_data(self):
		return self._train_ds

	@property
	def test_data(self):
		return self._test_ds

	@staticmethod
	def _preprocess(inputs, outputs, num_classes):
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
			elif False:
				inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
			elif False:
				in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
				out_min, out_max = 0, 1 #-1, 1
				inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
			elif True:
				inputs /= 255.0  # Normalization.

			# Reshape.
			inputs = np.expand_dims(inputs, axis=-1)

		if outputs is not None:
			# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
			#outputs = swl_ml_util.to_one_hot_encoding(outputs, num_classes).astype(np.uint8)
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs

	@staticmethod
	def _load_data(num_classes, batch_size):
		# Pixel value: [0, 255].
		(train_inputs, train_outputs), (test_inputs, test_outputs) = tf.keras.datasets.mnist.load_data()

		# Preprocess.
		train_inputs, train_outputs = MyDataset._preprocess(train_inputs, train_outputs, num_classes)
		test_inputs, test_outputs = MyDataset._preprocess(test_inputs, test_outputs, num_classes)
		assert train_inputs.shape[1:] == test_inputs.shape[1:]

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_inputs.shape, train_inputs.dtype, np.min(train_inputs), np.max(train_inputs)))
		print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_outputs.shape, train_outputs.dtype, np.min(train_outputs), np.max(train_outputs)))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_inputs.shape, test_inputs.dtype, np.min(test_inputs), np.max(test_inputs)))
		print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_outputs.shape, test_outputs.dtype, np.min(test_outputs), np.max(test_outputs)))

		#--------------------
		train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs)).shuffle(batch_size).batch(batch_size)
		test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs)).batch(batch_size)

		return train_ds, test_ds, train_inputs.shape[1:]

#--------------------------------------------------------------------

class MyModel(tf.keras.Model):
	def __init__(self, num_classes):
		super(MyModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(128, activation='relu')
		self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

	def call(self, x):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.dense1(x)
		return self.dense2(x)

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, batch_size):
		# Create a dataset.
		self._dataset = MyDataset(batch_size)

		self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
		self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

		self._train_loss = tf.keras.metrics.Mean(name='train_loss')
		self._train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

		self._test_loss = tf.keras.metrics.Mean(name='test_loss')
		self._test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

	def train(self, checkpoint_dir_path, output_dir_path, batch_size, final_epoch, initial_epoch=0, is_training_resumed=False):
		@tf.function
		def train_step(model, inputs, outputs):
			with tf.GradientTape() as tape:
				predictions = model(inputs)
				loss = self._loss_object(outputs, predictions)
			variables = model.trainable_variables
			gradients = tape.gradient(loss, variables)
			"""
			# Gradient clipping.
			max_gradient_norm = 5
			gradients = list(map(lambda grad: (tf.clip_by_norm(grad, clip_norm=max_gradient_norm)), gradients))
			#gradients = list(map(lambda grad: (tf.clip_by_value(grad, clip_value_min=min_clip_val, clip_value_max=max_clip_val)), gradients))
			"""
			self._optimizer.apply_gradients(zip(gradients, variables))

			self._train_loss(loss)
			self._train_accuracy(outputs, predictions)

		@tf.function
		def test_step(model, inputs, outputs):
			predictions = model(inputs)
			loss = self._loss_object(outputs, predictions)

			self._test_loss(loss)
			self._test_accuracy(outputs, predictions)

		# Create a model.
		model = MyModel(self._dataset.num_classes)

		# Create checkpoint objects.
		ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, net=model)
		ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_path, max_to_keep=5, keep_checkpoint_every_n_hours=2)

		if is_training_resumed:
			# Restore a model.
			print('[SWL] Info: Start restoring a model...')
			start_time = time.time()
			ckpt.restore(ckpt_manager.latest_checkpoint)
			if ckpt_manager.latest_checkpoint:
				print('[SWL] Info: End restoring a model from {}: {} secs.'.format(ckpt_manager.latest_checkpoint, time.time() - start_time))
			else:
				print('[SWL] Error: Failed to restore a model from {}.'.format(checkpoint_dir_path))
				return

		history = {
			'acc': list(),
			'loss': list(),
			'val_acc': list(),
			'val_loss': list()
		}

		# Create writers to write all the summaries out to a directory.
		train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
		val_summary_dir_path = os.path.join(output_dir_path, 'val_log')
		train_summary_writer = tf.summary.create_file_writer(train_summary_dir_path)
		val_summary_writer = tf.summary.create_file_writer(val_summary_dir_path)

		#--------------------
		if is_training_resumed:
			print('[SWL] Info: Resume training...')
		else:
			print('[SWL] Info: Start training...')
		best_performance_measure = 0
		start_total_time = time.time()
		for epoch in range(initial_epoch, final_epoch):
			print('Epoch {}/{}:'.format(epoch, final_epoch - 1))

			#--------------------
			start_time = time.time()
			for inputs, outputs in self._dataset.train_data:
				train_step(model, inputs, outputs)
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', self._train_loss.result(), step=epoch)
				tf.summary.scalar('accuracy', self._train_accuracy.result(), step=epoch)
			train_loss, train_acc = self._train_loss.result().numpy(), self._train_accuracy.result().numpy()
			print('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

			history['loss'].append(train_loss)
			history['acc'].append(train_acc)

			#--------------------
			start_time = time.time()
			for inputs, outputs in self._dataset.test_data:
				test_step(model, inputs, outputs)
			with val_summary_writer.as_default():
				tf.summary.scalar('loss', self._test_loss.result(), step=epoch)
				tf.summary.scalar('accuracy', self._test_accuracy.result(), step=epoch)
			val_loss, val_acc = self._test_loss.result().numpy(), self._test_accuracy.result().numpy()
			print('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)

			ckpt.step.assign_add(1)
			if val_acc > best_performance_measure:
				print('[SWL] Info: Start saving a model...')
				start_time = time.time()
				saved_model_path = ckpt_manager.save()
				print('[SWL] Info: End saving a model to {} for step {}: {} secs.'.format(saved_model_path, int(ckpt.step), time.time() - start_time))
				best_performance_measure = val_acc
			"""
			if val_acc > best_performance_measure or int(ckpt.step) % 100 == 0:
				print('[SWL] Info: Start saving a model...')
				start_time = time.time()
				saved_model_path = ckpt_manager.save()
				print('[SWL] Info: End saving a model to {} for step {}: {} secs.'.format(saved_model_path, int(ckpt.step), time.time() - start_time))
				if val_acc > best_performance_measure:
					best_performance_measure = val_acc
			"""

			# Reset metrics every epoch.
			self._train_loss.reset_states()
			self._train_accuracy.reset_states()
			self._test_loss.reset_states()
			self._test_accuracy.reset_states()

			sys.stdout.flush()
			time.sleep(0)
		print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))

		return history

	def test(self, checkpoint_dir_path, batch_size, shuffle=False):
		# Create a model.
		model = MyModel(self._dataset.num_classes)

		# Create checkpoint objects.
		#ckpt = tf.train.Checkpoint(net=model)  # Not good.
		ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, net=model)
		ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_path, max_to_keep=None)

		# Load a model.
		print('[SWL] Info: Start loading a model...')
		start_time = time.time()
		ckpt.restore(ckpt_manager.latest_checkpoint)
		if ckpt_manager.latest_checkpoint:
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(ckpt_manager.latest_checkpoint, time.time() - start_time))
		else:
			print('[SWL] Error: Failed to load a model from {}.'.format(checkpoint_dir_path))
			return

		#--------------------
		print('[SWL] Info: Start testing...')
		inferences, ground_truths = list(), list()
		start_time = time.time()
		for inputs, outputs in self._dataset.test_data:
			inferences.append(model(inputs).numpy())
			ground_truths.append(outputs.numpy())
		print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.vstack(inferences), np.concatenate(ground_truths)
		if inferences is not None and ground_truths is not None:
			print('\tTest: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			inferences = np.argmax(inferences, -1)
			#ground_truths = np.argmax(ground_truths, -1)

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			print('\tTest: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			print('[SWL] Warning: Invalid test results.')

	def infer(self, checkpoint_dir_path, batch_size=None, shuffle=False):
		# Create a model.
		model = MyModel(self._dataset.num_classes)

		# Create checkpoint objects.
		#ckpt = tf.train.Checkpoint(net=model)  # Not good.
		ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, net=model)
		ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_path, max_to_keep=None)

		# Load a model.
		print('[SWL] Info: Start loading a model...')
		start_time = time.time()
		ckpt.restore(ckpt_manager.latest_checkpoint)
		if ckpt_manager.latest_checkpoint:
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(ckpt_manager.latest_checkpoint, time.time() - start_time))
		else:
			print('[SWL] Error: Failed to load a model from {}.'.format(checkpoint_dir_path))
			return

		#--------------------
		print('[SWL] Info: Start inferring...')
		inferences = list()
		start_time = time.time()
		for inputs, _ in self._dataset.test_data:
			inferences.append(model(inputs).numpy())
		print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

		inferences = np.vstack(inferences)
		if inferences is not None:
			print('\tInference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			inferences = np.argmax(inferences, -1)

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

	if not args.train and not args.test and not args.infer:
		print('[SWL] Error: At least one of command line options "--train", "--test", and "--infer" has to be specified.')
		return

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	if args.log_level:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#logger = set_logger(args.log_level)

	#--------------------
	is_training_resumed = args.resume
	initial_epoch, final_epoch, batch_size = 0, args.epoch, args.batch_size

	checkpoint_dir_path, output_dir_path = os.path.normpath(args.model_dir), os.path.normpath(args.out_dir)
	if checkpoint_dir_path:
		if not output_dir_path:
			output_dir_path = os.path.dirname(checkpoint_dir_path)
	else:
		if not output_dir_path:
			output_dir_prefix = 'simple_training'
			output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')

	#--------------------
	runner = MyRunner(batch_size)

	if args.train:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		# TODO [check] >> Make sure whether the checkpoint directory ('tf_checkpoint') is copied to 'output_dir_path'.

		history = runner.train(checkpoint_dir_path, output_dir_path, batch_size, final_epoch, initial_epoch, is_training_resumed)

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
#	python run_simple_training.py --train --test --infer --epoch 30 --gpu 0

if '__main__' == __name__:
	main()
