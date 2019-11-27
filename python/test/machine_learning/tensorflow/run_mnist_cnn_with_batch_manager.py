#!/usr/bin/env python

import sys
sys.path.append('../../../src')

#--------------------
import os, time, datetime
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
from swl.machine_learning.batch_manager import SimpleBatchManager, SimpleFileBatchManager
from swl.machine_learning.augmentation_batch_manager import AugmentationBatchManager, AugmentationFileBatchManager
from swl.machine_learning.imgaug_batch_manager import ImgaugBatchManager, ImgaugFileBatchManager
from swl.util.working_directory_manager import WorkingDirectoryManager
import swl.util.util as swl_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from mnist_cnn_tf import MnistCnnUsingTF

#--------------------------------------------------------------------

def create_mnist_cnn(input_shape, output_shape):
	model_type = 0  # {0, 1}.
	return MnistCnnUsingTF(input_shape, output_shape, model_type)

#--------------------------------------------------------------------

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

#--------------------------------------------------------------------

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

class IdentityAugmenter(object):
	def augment(self, images, labels, is_label_augmented=False):
		return images, labels

class ImgaugAugmenter(object):
	def __init__(self, image_height, image_width):
		self._augmenter = get_imgaug_augmenter(image_height, image_width)

	def augment(self, images, labels, is_label_augmented=False):
		images = self._augmenter.augment_images(images)
		return images, labels

#--------------------------------------------------------------------

def mnist_batch_manager(method=0):
	#np.random.seed(7)

	#--------------------
	# Sets parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181211T172200'

	initial_epoch = 0

	image_height, image_width = 28, 28
	num_classes = 10
	input_shape = (None, image_height, image_width, 1)
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 30  # Number of times to iterate over training data.
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
	# Prepares directories.

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
	# Prepares data.

	train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])

	#--------------------
	# Creates models, sessions, and graphs.

	# Creates graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Creates a model.
			modelForTraining = create_mnist_cnn(input_shape, output_shape)
			modelForTraining.create_training_model()

			# Creates a trainer.
			nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch)

			# Creates a saver.
			#	Saves a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			# Creates a model.
			modelForEvaluation = create_mnist_cnn(input_shape, output_shape)
			modelForEvaluation.create_evaluation_model()

			# Creates an evaluator.
			nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

			# Creates a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		# Creates a model.
		modelForInference = create_mnist_cnn(input_shape, output_shape)
		modelForInference.create_inference_model()

		# Creates an inferrer.
		nnInferrer = NeuralNetInferrer(modelForInference)

		# Creates a saver.
		infer_saver = tf.train.Saver()

	# Creates sessions.
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initializes.
	if does_need_training:
		train_session.run(initializer)

	#--------------------------------------------------------------------
	# Trains and evaluates.

	if does_need_training:
		# Method #0: AugmentationBatchManager without process pool.
		if 0 == method:
			#augmenter = IdentityAugmenter()
			augmenter = ImgaugAugmenter(image_height, image_width)
			trainBatchMgr = AugmentationBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major, None)
			valBatchMgr = SimpleBatchManager(test_images, test_labels, batch_size, False, is_time_major)

			start_time = time.time()
			with train_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.train_neural_net_by_batch_manager(sess, nnTrainer, trainBatchMgr, valBatchMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
			print('\tTotal training time = {}'.format(time.time() - start_time))
		# Method #1: AugmentationBatchManager with process pool.
		elif 1 == method:
			with mp.Pool() as pool:
				#augmenter = IdentityAugmenter()
				augmenter = ImgaugAugmenter(image_height, image_width)
				trainBatchMgr = AugmentationBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major, pool)
				valBatchMgr = SimpleBatchManager(test_images, test_labels, batch_size, False, is_time_major)

				start_time = time.time()
				with train_session.as_default() as sess:
					with sess.graph.as_default():
						swl_tf_util.train_neural_net_by_batch_manager(sess, nnTrainer, trainBatchMgr, valBatchMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
				print('\tTotal training time = {}'.format(time.time() - start_time))
		# Method #2: AugmentationFileBatchManager without process pool.
		elif 2 == method:
			batch_dir_path_prefix = './batch_dir'
			num_batch_dirs = 5
			dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			#augmenter = IdentityAugmenter()
			augmenter = ImgaugAugmenter(image_height, image_width)
			trainFileBatchMgr = AugmentationFileBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major, None, image_file_format='train_batch_images_{}.npy', label_file_format='train_batch_labels_{}.npy')
			valFileBatchMgr = SimpleFileBatchManager(test_images, test_labels, batch_size, False, is_time_major, image_file_format='val_batch_images_{}.npy', label_file_format='val_batch_labels_{}.npy')

			start_time = time.time()
			with train_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.train_neural_net_by_file_batch_manager(sess, nnTrainer, trainFileBatchMgr, valFileBatchMgr, dirMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
			print('\tTotal training time = {}'.format(time.time() - start_time))
		# Method #3: AugmentationFileBatchManager with process pool.
		elif 3 == method:
			batch_dir_path_prefix = './batch_dir'
			num_batch_dirs = 5
			dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			with mp.Pool() as pool:
				#augmenter = IdentityAugmenter()
				augmenter = ImgaugAugmenter(image_height, image_width)
				trainFileBatchMgr = AugmentationFileBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major, pool, image_file_format='train_batch_images_{}.npy', label_file_format='train_batch_labels_{}.npy')
				valFileBatchMgr = SimpleFileBatchManager(test_images, test_labels, batch_size, False, is_time_major, image_file_format='val_batch_images_{}.npy', label_file_format='val_batch_labels_{}.npy')

				start_time = time.time()
				with train_session.as_default() as sess:
					with sess.graph.as_default():
						swl_tf_util.train_neural_net_by_file_batch_manager(sess, nnTrainer, trainFileBatchMgr, valFileBatchMgr, dirMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
				print('\tTotal training time = {}'.format(time.time() - start_time))
		# Method #4: ImgaugBatchManager with background processes.
		elif 4 == method:
			augmenter = get_imgaug_augmenter(image_height, image_width)
			trainBatchMgr = ImgaugBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major)
			valBatchMgr = SimpleBatchManager(test_images, test_labels, batch_size, False, is_time_major)

			start_time = time.time()
			with train_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.train_neural_net_by_batch_manager(sess, nnTrainer, trainBatchMgr, valBatchMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
			print('\tTotal training time = {}'.format(time.time() - start_time))
		# Method #5: ImgaugFileBatchManager without background processes.
		elif 5 == method:
			batch_dir_path_prefix = './batch_dir'
			num_batch_dirs = 5
			dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			augmenter = get_imgaug_augmenter(image_height, image_width)
			trainFileBatchMgr = ImgaugFileBatchManager(augmenter, train_images, train_labels, batch_size, shuffle, is_label_augmented, is_time_major, image_file_format='train_batch_images_{}.npy', label_file_format='train_batch_labels_{}.npy')
			valFileBatchMgr = SimpleFileBatchManager(test_images, test_labels, batch_size, False, is_time_major, image_file_format='val_batch_images_{}.npy', label_file_format='val_batch_labels_{}.npy')

			start_time = time.time()
			with train_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.train_neural_net_by_file_batch_manager(sess, nnTrainer, trainFileBatchMgr, valFileBatchMgr, dirMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
			print('\tTotal training time = {}'.format(time.time() - start_time))
		else:
			raise ValueError('Invalid batch manager method: {}'.format(method))

		#--------------------
		if method in (0, 1, 4):
			valBatchMgr = SimpleBatchManager(test_images, test_labels, batch_size, False, is_time_major)

			start_time = time.time()
			with eval_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.evaluate_neural_net_by_batch_manager(sess, nnEvaluator, valBatchMgr, eval_saver, checkpoint_dir_path, is_time_major, is_sparse_output)
			print('\tTotal evaluation time = {}'.format(time.time() - start_time))
		elif method in (2, 3, 5):
			batch_dir_path_prefix = './batch_dir'
			num_batch_dirs = 5
			dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			valFileBatchMgr = SimpleFileBatchManager(test_images, test_labels, batch_size, False, is_time_major, image_file_format='val_batch_images_{}.npy', label_file_format='val_batch_labels_{}.npy')

			start_time = time.time()
			with eval_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.evaluate_neural_net_by_file_batch_manager(sess, nnEvaluator, valFileBatchMgr, dirMgr, eval_saver, checkpoint_dir_path, is_time_major, is_sparse_output)
			print('\tTotal evaluation time = {}'.format(time.time() - start_time))
		else:
			raise ValueError('Invalid batch manager method: {}'.format(method))

	#--------------------------------------------------------------------
	# Infers.

	if method in (0, 1, 4):
		testBatchMgr = SimpleBatchManager(test_images, test_labels, batch_size, False, is_time_major)

		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences = swl_tf_util.infer_by_neural_net_and_batch_manager(sess, nnInferrer, testBatchMgr, infer_saver, checkpoint_dir_path, is_time_major)
		print('\tTotal inference time = {}'.format(time.time() - start_time))
	elif method in (2, 3, 5):
		batch_dir_path_prefix = './batch_dir'
		num_batch_dirs = 5
		dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

		testFileBatchMgr = SimpleFileBatchManager(test_images, test_labels, batch_size, False, is_time_major, image_file_format='val_batch_images_{}.npy', label_file_format='val_batch_labels_{}.npy')

		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences = swl_tf_util.infer_by_neural_net_and_file_batch_manager(sess, nnInferrer, testFileBatchMgr, dirMgr, infer_saver, checkpoint_dir_path, is_time_major)
		print('\tTotal inference time = {}'.format(time.time() - start_time))
	else:
		raise ValueError('Invalid batch manager method: {}'.format(method))

	if inferences is not None:
		inferences = np.vstack(inferences)
		if num_classes >= 2:
			inferences = np.argmax(inferences, -1)
			groundtruths = np.argmax(test_labels, -1)
		else:
			inferences = np.around(inferences)
			groundtruths = test_labels
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
	else:
		print('[SWL] Warning: Invalid inference results.')

	#--------------------
	# Closes sessions.

	if does_need_training:
		train_session.close()
		del train_session
		eval_session.close()
		del eval_session
	infer_session.close()
	del infer_session

def main():
	# NOTE [info] >> Too slow when using process pool.
	# Method #0: AugmentationBatchManager without process pool.
	# Method #1: AugmentationBatchManager with process pool.
	# Method #2: AugmentationFileBatchManager without process pool.
	# Method #3: AugmentationFileBatchManager with process pool.
	# Method #4: ImgaugBatchManager with background processes.
	# Method #5: ImgaugFileBatchManager without background processes.
	mnist_batch_manager(method=4)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
