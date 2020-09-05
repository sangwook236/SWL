#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, argparse, logging, logging.handlers, time, datetime, functools
import numpy as np
import tensorflow as tf
#from sklearn import preprocessing
#import cv2
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self, logger):
		self._num_classes = 10

		# Load data.
		logger.info('Start loading dataset...')
		start_time = time.time()
		self._train_inputs, self._train_outputs, self._test_inputs, self._test_outputs, self._shape = MyDataset._load_data(self._num_classes, logger)
		logger.info('End loading dataset: {} secs.'.format(time.time() - start_time))

	@property
	def shape(self):
		return self._shape  # (image height, image width, image channel).

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def train_data(self):
		return self._train_inputs, self._train_outputs

	@property
	def test_data(self):
		return self._test_inputs, self._test_outputs

	def show_data_info(self, logger, visualize=True):
		logger.info('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_inputs.shape, self._train_inputs.dtype, np.min(self._train_inputs), np.max(self._train_inputs)))
		logger.info('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_outputs.shape, self._train_outputs.dtype, np.min(self._train_outputs), np.max(self._train_outputs)))
		logger.info('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_inputs.shape, self._test_inputs.dtype, np.min(self._test_inputs), np.max(self._test_inputs)))
		logger.info('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_outputs.shape, self._test_outputs.dtype, np.min(self._test_outputs), np.max(self._test_outputs)))

		if visualize:
			import cv2
			def show_images(inputs, outputs):
				inputs = inputs.squeeze(axis=-1)
				for idx, (img, lbl) in enumerate(zip(inputs, outputs)):
					print('Label #{} = {}.'.format(idx, lbl))
					cv2.imshow('Image', img)
					cv2.waitKey()
					if idx >= 9: break
			show_images(self._train_inputs, self._train_outputs)
			show_images(self._test_inputs, self._test_outputs)
			cv2.destroyAllWindows()

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
	def _load_data(num_classes, logger):
		# Pixel value: [0, 255].
		(train_inputs, train_outputs), (test_inputs, test_outputs) = tf.keras.datasets.mnist.load_data()

		# Preprocess.
		train_inputs, train_outputs = MyDataset._preprocess(train_inputs, train_outputs, num_classes)
		test_inputs, test_outputs = MyDataset._preprocess(test_inputs, test_outputs, num_classes)
		assert train_inputs.shape[1:] == test_inputs.shape[1:]

		return train_inputs, train_outputs, test_inputs, test_outputs, train_inputs.shape[1:]

#--------------------------------------------------------------------

def load_model(checkpoint_dir_path, optimizer, num_classes, logger, is_train=False, is_loaded=False):
	# Build a model.
	model = MyModel(num_classes)

	# Create checkpoint objects.
	#ckpt = tf.train.Checkpoint(net=model)  # Not good.
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
	if is_train:
		ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_path, max_to_keep=5, keep_checkpoint_every_n_hours=2)
	else:
		ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir_path, max_to_keep=None)

	if is_loaded:
		# Load a model.
		if logger: logger.info('Start loading a model...')
		start_time = time.time()
		ckpt.restore(ckpt_manager.latest_checkpoint)
		if ckpt_manager.latest_checkpoint:
			if logger: logger.info('End loading a model from {}: {} secs.'.format(ckpt_manager.latest_checkpoint, time.time() - start_time))
		else:
			if logger: logger.error('Failed to load a model from {}.'.format(checkpoint_dir_path))
			return None, None, None

	return model, ckpt, ckpt_manager

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
	def __init__(self):
		# Create losses and accuracies.
		self._train_loss = tf.keras.metrics.Mean(name='train_loss')
		self._train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

		self._test_loss = tf.keras.metrics.Mean(name='test_loss')
		self._test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

	def train(self, model, criterion, optimizer, train_dataset, test_dataset, output_dir_path, ckpt, ckpt_manager, batch_size, final_epoch, initial_epoch=0, logger=None):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		@tf.function
		def train_step(model, inputs, outputs):
			with tf.GradientTape() as tape:
				predictions = model(inputs)
				loss = criterion(outputs, predictions)
			variables = model.trainable_variables
			gradients = tape.gradient(loss, variables)
			"""
			# Gradient clipping.
			max_gradient_norm = 5
			gradients = list(map(lambda grad: (tf.clip_by_norm(grad, clip_norm=max_gradient_norm)), gradients))
			#gradients = list(map(lambda grad: (tf.clip_by_value(grad, clip_value_min=min_clip_val, clip_value_max=max_clip_val)), gradients))
			"""
			optimizer.apply_gradients(zip(gradients, variables))

			self._train_loss(loss)
			self._train_accuracy(outputs, predictions)

		@tf.function
		def test_step(model, inputs, outputs):
			predictions = model(inputs)
			loss = criterion(outputs, predictions)

			self._test_loss(loss)
			self._test_accuracy(outputs, predictions)

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
		if logger: logger.info('Start training...')
		best_performance_measure = 0
		start_total_time = time.time()
		for epoch in range(initial_epoch, final_epoch):
			if logger: logger.info('Epoch {}/{}:'.format(epoch, final_epoch - 1))

			#--------------------
			start_time = time.time()
			for inputs, outputs in train_dataset:
				train_step(model, inputs, outputs)
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', self._train_loss.result(), step=epoch)
				tf.summary.scalar('accuracy', self._train_accuracy.result(), step=epoch)
			train_loss, train_acc = self._train_loss.result().numpy(), self._train_accuracy.result().numpy()
			if logger: logger.info('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

			history['loss'].append(train_loss)
			history['acc'].append(train_acc)

			#--------------------
			start_time = time.time()
			for inputs, outputs in test_dataset:
				test_step(model, inputs, outputs)
			with val_summary_writer.as_default():
				tf.summary.scalar('loss', self._test_loss.result(), step=epoch)
				tf.summary.scalar('accuracy', self._test_accuracy.result(), step=epoch)
			val_loss, val_acc = self._test_loss.result().numpy(), self._test_accuracy.result().numpy()
			if logger: logger.info('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)

			ckpt.step.assign_add(1)
			if val_acc > best_performance_measure:
				if logger: logger.info('Start saving a model...')
				start_time = time.time()
				saved_model_path = ckpt_manager.save()
				if logger: logger.info('End saving a model to {} for step {}: {} secs.'.format(saved_model_path, int(ckpt.step), time.time() - start_time))
				best_performance_measure = val_acc
			"""
			if val_acc > best_performance_measure or int(ckpt.step) % 100 == 0:
				if logger: logger.info('Start saving a model...')
				start_time = time.time()
				saved_model_path = ckpt_manager.save()
				if logger: logger.info('End saving a model to {} for step {}: {} secs.'.format(saved_model_path, int(ckpt.step), time.time() - start_time))
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
		if logger: logger.info('End training: {} secs.'.format(time.time() - start_total_time))

		return history

	def test(self, model, dataset, batch_size, shuffle=False, logger=None):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		if logger: logger.info('Start testing...')
		inferences, ground_truths = list(), list()
		start_time = time.time()
		for inputs, outputs in dataset:
			inferences.append(model(inputs).numpy())
			ground_truths.append(outputs.numpy())
		if logger: logger.info('End testing: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.vstack(inferences), np.concatenate(ground_truths)
		if inferences is not None and ground_truths is not None:
			if logger: logger.info('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			inferences = np.argmax(inferences, -1)
			#ground_truths = np.argmax(ground_truths, -1)

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			if logger: logger.info('Test: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			if logger: logger.warning('Invalid test results.')

	def infer(self, model, inputs, logger=None):
		if logger: logger.info('Start inferring...')
		start_time = time.time()
		inferences = model(inputs)
		if logger: logger.info('End inferring: {} secs.'.format(time.time() - start_time))
		return inferences.numpy()

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
		'--batch',
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
		'--log',
		type=str,
		help='The name of logger and log files',
		default=None
	)
	parser.add_argument(
		'-ll',
		'--log_level',
		type=int,
		help='Log level, [0, 50]',  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
		default=None
	)
	parser.add_argument(
		'-ld',
		'--log_dir',
		type=str,
		help='The directory path to log',
		default=None
	)

	return parser.parse_args()

def get_logger(name, log_level=None, log_dir_path=None, is_rotating=True):
	if not log_level: log_level = logging.INFO
	if not log_dir_path: log_dir_path = './log'
	if not os.path.exists(log_dir_path):
		os.makedirs(log_dir_path, exist_ok=True)

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

	logger = get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else args.out_dir, is_rotating=True)
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
	initial_epoch, final_epoch, batch_size = 0, args.epoch, args.batch
	is_resumed = args.model_dir is not None

	checkpoint_dir_path, output_dir_path = os.path.normpath(args.model_dir) if args.model_dir else None, os.path.normpath(args.out_dir) if args.out_dir else None
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
	# Create datasets.
	logger.info('Start creating datasets...')
	start_time = time.time()
	dataset = MyDataset(logger)

	train_dataset = tf.data.Dataset.from_tensor_slices(dataset.train_data).shuffle(batch_size).batch(batch_size, drop_remainder=False)
	test_dataset = tf.data.Dataset.from_tensor_slices(dataset.test_data).batch(batch_size, drop_remainder=False)
	logger.info('End creating datasets: {} secs.'.format(time.time() - start_time))

	dataset.show_data_info(logger, visualize=False)

	#--------------------
	runner = MyRunner()

	# Create a trainer.
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

	if args.train:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		# Load a model.
		model, ckpt, ckpt_manager = load_model(checkpoint_dir_path, optimizer, dataset.num_classes, logger, is_train=True, is_loaded=is_resumed)
		#if model: print('Model summary:\n{}.'.format(model))

		if model:
			# Create a criterion.
			criterion = tf.keras.losses.SparseCategoricalCrossentropy()

			history = runner.train(model, criterion, optimizer, train_dataset, test_dataset, output_dir_path, ckpt, ckpt_manager, batch_size, final_epoch, initial_epoch, logger)

			if history:
				#logger.info('Train history = {}.'.format(history))
				swl_ml_util.display_train_history(history)
				if os.path.exists(output_dir_path):
					swl_ml_util.save_train_history(history, output_dir_path)

	if args.test or args.infer:
		if checkpoint_dir_path and os.path.exists(checkpoint_dir_path):
			model, _, _ = load_model(checkpoint_dir_path, optimizer, dataset.num_classes, logger, is_train=False, is_loaded=True)

			#if model:
			#	# A new probability model which does not need to be trained because it has no trainable parameter.
			#	#model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

			if args.test and model:
				runner.test(model, test_dataset, batch_size, logger=logger)

			if args.infer and model:
				inferences = list()
				for inputs, _ in test_dataset:
					inferences.append(runner.infer(model, inputs, logger=logger))

				inferences = np.vstack(inferences)
				if inferences is not None:
					logger.info('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

					inferences = np.argmax(inferences, -1)

					results = {idx: inf for idx, inf in enumerate(inferences) if idx < 100}
					logger.info('Inference results (index: inference): {}.'.format(results))
				else:
					logger.warning('Invalid inference results.')
		else:
			logger.error('Model directory, {} does not exist.'.format(checkpoint_dir_path))

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 20 --gpu 0

if '__main__' == __name__:
	main()
