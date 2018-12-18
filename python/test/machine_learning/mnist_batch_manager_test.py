#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import os, time, datetime
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.augmentation_batch_manager import AugmentationBatchManager, AugmentationFileBatchManager
from swl.machine_learning.imgaug_batch_manager import ImgaugBatchManager, ImgaugFileBatchManager
from swl.util.directory_queue_manager import DirectoryQueueManager
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util

class MnistCnn(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		num_classes = output_shape[-1]
		with tf.variable_scope('mnist_cnn_using', reuse=tf.AUTO_REUSE):
			return self._create_model(input_tensor, is_training, num_classes)

	def _create_model(self, input_tensor, is_training, num_classes):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv')
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv')
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(conv2, name='flatten')

			fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='dense')
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			fc1 = tf.layers.dropout(fc1, rate=0.75, training=is_training, name='dropout')

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

def create_mnist_cnn(input_shape, output_shape):
	return MnistCnn(input_shape, output_shape)

def load_data(image_shape):
	# Pixel value: [0, 255].
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images / 255.0
	train_images = np.reshape(train_images, (-1,) + image_shape)
	train_labels = tf.keras.utils.to_categorical(train_labels).astype(np.uint8)
	test_images = test_images / 255.0
	test_images = np.reshape(test_images, (-1,) + image_shape)
	test_labels = tf.keras.utils.to_categorical(test_labels).astype(np.uint8)

	return train_images, train_labels, test_images, test_labels

def get_imgaug_augmenter(image_height, image_width):
	return iaa.Sequential([
		iaa.SomeOf(1, [
			#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
			iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))), # Crop images by 0-10% of their height/width.
			iaa.Sometimes(0.5, iaa.Affine(
				scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
				rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
				shear=(-16, 16),  # Shear by -16 to +16 degrees.
				#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			))
			#iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
		])
	])

class ImgaugAugmenter(object):
	def __init__(self, image_height, image_width):
		self._augmenter = get_imgaug_augmenter(image_height, image_width)

	def augment(self, images, labels, is_label_augmented=False):
		images = self._augmenter.augment_images(images)
		return images, labels

def mnist_augmentation_batch_manager_example():
	#--------------------
	# Parameters.

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181211T172200'

	initial_epoch = 0

	image_height, image_width = 28, 28
	num_classes = 10
	input_shape = (None, image_height, image_width, 1)
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True
	is_label_augmented = False
	is_time_major = False
	is_sparse_output = False

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	train_graph = tf.Graph()

	with train_graph.as_default():
		#K.set_learning_phase(1)  # Set the learning phase to 'train'. (Required)

		# Create a model.
		modelForTraining = create_mnist_cnn(input_shape, output_shape)
		modelForTraining.create_training_model()

		# Create a trainer.
		nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch)

		# Create a saver.
		#	Save a model every 2 hours and maximum 5 latest models are saved.
		train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		initializer = tf.global_variables_initializer()

	# Create sessions.
	train_session = tf.Session(graph=train_graph, config=sess_config)

	# Initialize.
	train_session.run(initializer)

	#--------------------
	# Train.

	augmenter = ImgaugAugmenter(image_height, image_width)

	with mp.Pool() as pool:
		batchMgr = AugmentationBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major, pool)

		total_elapsed_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				print('[SWL] Info: Start training...')

				# Create writers to write all the summaries out to a directory.
				train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, sess.graph) if train_summary_dir_path is not None else None

				history = {
					'acc': [],
					'loss': [],
					'val_acc': None,
					'val_loss': None
				}
				
				batch_axis = 1 if is_time_major else 0
				for epoch in range(1, num_epochs + 1):
					print('Epoch {}/{}'.format(epoch, num_epochs))
					start_time = time.time()

					batches = batchMgr.getBatches()  # Generates and augments batches.
					train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
					for batch_images, batch_labels in batches:
						# Train with each batch (images & labels).
						batch_acc, batch_loss = nnTrainer.train_by_batch(sess, batch_images, batch_labels, train_summary_writer, is_time_major, is_sparse_output)

						# TODO [check] >> Are these calculations correct?
						batch_size = train_images.shape[batch_axis]
						train_acc += batch_acc * batch_size
						train_loss += batch_loss * batch_size
						num_train_examples += batch_size

					train_acc /= num_train_examples
					train_loss /= num_train_examples

					history['acc'].append(train_acc)
					history['loss'].append(train_loss)

					print('\tTraining time = {}'.format(time.time() - start_time))
					print('\tTraining: loss = {}, accuracy = {}'.format(train_loss, train_acc))

				# Close writers.
				if train_summary_writer is not None:
					train_summary_writer.close()

				#--------------------
				# Display results.
				#swl_ml_util.display_train_history(history)
				if output_dir_path is not None:
					swl_ml_util.save_train_history(history, output_dir_path)
				print('[SWL] Info: End training.')
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

	#--------------------
	# Close sessions.

	train_session.close()
	del train_session

def mnist_augmentation_file_batch_manager_example():
	#--------------------
	# Parameters.

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181211T172200'

	initial_epoch = 0

	image_height, image_width = 28, 28
	num_classes = 10
	input_shape = (None, image_height, image_width, 1)
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True
	is_label_augmented = False
	is_time_major = False
	is_sparse_output = False

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	train_graph = tf.Graph()

	with train_graph.as_default():
		#K.set_learning_phase(1)  # Set the learning phase to 'train'. (Required)

		# Create a model.
		modelForTraining = create_mnist_cnn(input_shape, output_shape)
		modelForTraining.create_training_model()

		# Create a trainer.
		nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch)

		# Create a saver.
		#	Save a model every 2 hours and maximum 5 latest models are saved.
		train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		initializer = tf.global_variables_initializer()

	# Create sessions.
	train_session = tf.Session(graph=train_graph, config=sess_config)

	# Initialize.
	train_session.run(initializer)

	#--------------------
	# Train.

	augmenter = ImgaugAugmenter(image_height, image_width)

	with mp.Pool() as pool:
		total_elapsed_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				print('[SWL] Info: Start training...')

				# Create writers to write all the summaries out to a directory.
				train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, sess.graph) if train_summary_dir_path is not None else None

				history = {
					'acc': [],
					'loss': [],
					'val_acc': None,
					'val_loss': None
				}
				
				batch_axis = 1 if is_time_major else 0
				for epoch in range(1, num_epochs + 1):
					print('Epoch {}/{}'.format(epoch, num_epochs))
					start_time = time.time()

					dir_path = dirQueueMgr.getAvailableDirectory()
					if dir_path is None:
						break

					print('\tBatch directory: {}.'.format(dir_path))

					batchMgr = AugmentationFileBatchManager(augmenter, train_images, train_labels, dir_path, batch_size, shuffle, is_label_augmented, is_time_major, pool)
					batchMgr.putBatches()  # Generates, augments, and saves batches.

					batches = batchMgr.getBatches()  # Loads batches.
					for batch_images, batch_labels in batches:
						# Train with each batch (images & labels).
						batch_acc, batch_loss = nnTrainer.train_by_batch(sess, batch_images, batch_labels, train_summary_writer, is_time_major, is_sparse_output)

						# TODO [check] >> Are these calculations correct?
						batch_size = train_images.shape[batch_axis]
						train_acc += batch_acc * batch_size
						train_loss += batch_loss * batch_size
						num_train_examples += batch_size

					dirQueueMgr.returnDirectory(dir_path)				

					train_acc /= num_train_examples
					train_loss /= num_train_examples

					history['acc'].append(train_acc)
					history['loss'].append(train_loss)

					print('\tTraining time = {}'.format(time.time() - start_time))
					print('\tTraining: loss = {}, accuracy = {}'.format(train_loss, train_acc))

				# Close writers.
				if train_summary_writer is not None:
					train_summary_writer.close()

				#--------------------
				# Display results.
				#swl_ml_util.display_train_history(history)
				if output_dir_path is not None:
					swl_ml_util.save_train_history(history, output_dir_path)
				print('[SWL] Info: End training.')
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

	#--------------------
	# Close sessions.

	train_session.close()
	del train_session

def mnist_imgaug_batch_manager_example():
	image_height, image_width = 28, 28
	num_classes = 10
	input_shape = (None, image_height, image_width, 1)
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True
	is_time_major = False

	#--------------------
	train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])

	augmenter = get_imgaug_augmenter(image_height, image_width)

	#--------------------
	batchMgr = ImgaugBatchManager(augmenter, images, labels, batch_size, shuffle, is_time_major)
	for epoch in range(num_epoches):
		print('>>>>> Epoch #{}.'.format(epoch))

		batches = batchMgr.getBatches()  # Generates and augments batches.
		for idx, batch in enumerate(batches):
			# Train with batch (images & labels).
			#print('{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def mnist_imgaug_file_batch_manager_example():
	image_height, image_width = 28, 28
	num_classes = 10
	input_shape = (None, image_height, image_width, 1)
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True
	is_time_major = False

	base_dir_path = './batch_dir'
	num_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_dir_path, num_dirs)

	#--------------------
	train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])

	augmenter = get_imgaug_augmenter(image_height, image_width)

	#--------------------
	for epoch in range(num_epoches):
		print('>>>>> Epoch #{}.'.format(epoch))

		dir_path = dirQueueMgr.getAvailableDirectory()
		if dir_path is None:
			break

		print('\t>>>>> Directory: {}.'.format(dir_path))

		batchMgr = ImgaugFileBatchManager(augmenter, images, labels, dir_path, batch_size, shuffle, is_label_augmented, is_time_major)
		batchMgr.putBatches()  # Generates, augments, and saves batches.

		batches = batchMgr.getBatches()  # Loads batches.
		for idx, batch in enumerate(batches):
			# Train with batch (images & labels).
			#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirQueueMgr.returnDirectory(dir_path)				

def main():
	mnist_augmentation_batch_manager_example()
	#mnist_augmentation_file_batch_manager_example()

	#mnist_imgaug_batch_manager_example()
	#mnist_imgaug_file_batch_manager_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
