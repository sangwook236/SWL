#!/usr/bin/env python

import sys
sys.path.append('../../../src')

#--------------------
import os, time, datetime
from functools import partial
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import threading
import numpy as np
import tensorflow as tf
#import imgaug as ia
from imgaug import augmenters as iaa
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from swl.machine_learning.batch_generator import SimpleBatchGenerator, NpyFileBatchGenerator
from swl.machine_learning.batch_loader import NpyFileBatchLoader
import swl.util.util as swl_util
from swl.util.working_directory_manager import WorkingDirectoryManager, TwoStepWorkingDirectoryManager
from mnist_cnn_tf import MnistCnnUsingTF

#%%------------------------------------------------------------------

def create_mnist_cnn(input_shape, output_shape):
	model_type = 0  # {0, 1}.
	return MnistCnnUsingTF(input_shape, output_shape, model_type)

#%%------------------------------------------------------------------

def create_imgaug_augmenter():
	return iaa.Sequential([
		iaa.SomeOf(1, [
			#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
			iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))), # Crop images by 0-10% of their height/width.
			iaa.Fliplr(0.1),  # Horizontally flip 10% of the images.
			iaa.Flipud(0.1),  # Vertically flip 10% of the images.
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
			)),
			iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
		]),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

class ImgaugAugmenter(object):
	def __init__(self):
		self._augmenter = create_imgaug_augmenter()

	def __call__(self, inputs, outputs, is_output_augmented=False):
		# Augments here.
		if is_output_augmented:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)
		else:
			return self._augmenter.augment_images(inputs), outputs

def preprocess_data(data, labels, num_classes, axis=0):
	if data is not None:
		# Preprocessing (normalization, standardization, etc.).
		#data = data.astype(np.float32)
		#data /= 255.0
		#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
		#data = np.reshape(data, data.shape + (1,))
		pass

	if labels is not None:
		# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
		#labels = to_one_hot_encoding(labels, num_classes).astype(np.uint8)
		pass

	return data, labels

def load_data(image_shape):
	# Pixel value: [0, 255].
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images / 255.0
	train_images = np.reshape(train_images, (-1,) + image_shape)
	train_labels = tf.keras.utils.to_categorical(train_labels).astype(np.uint8)
	test_images = test_images / 255.0
	test_images = np.reshape(test_images, (-1,) + image_shape)
	test_labels = tf.keras.utils.to_categorical(test_labels).astype(np.uint8)

	# Pre-process.
	#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
	#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

	return train_images, train_labels, test_images, test_labels

#%%------------------------------------------------------------------

def initialize_lock(lock):
	global global_lock
	global_lock = lock

# REF [function] >> training_worker_proc() in ${SWL_PYTHON_HOME}/python/test/machine_learning/batch_generator_and_loader_test.py.
#def training_worker_proc(train_session, nnTrainer, trainDirMgr, valDirMgr, trainFileBatchLoader, valFileBatchLoader, num_epochs):
def training_worker_proc(train_session, nnTrainer, trainDirMgr, valDirMgr, batch_info_csv_filename, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	print('\t{}: Start training worker process.'.format(os.getpid()))

	trainFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename)
	valFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename)

	#--------------------
	start_time = time.time()
	with train_session.as_default() as sess:
		with sess.graph.as_default():
			swl_tf_util.train_neural_net_by_file_batch_loader(sess, nnTrainer, trainFileBatchLoader, valFileBatchLoader, trainDirMgr, valDirMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
	print('\tTotal training time = {}'.format(time.time() - start_time))

	print('\t{}: End training worker process.'.format(os.getpid()))

# REF [function] >> augmentation_worker_proc() in ${SWL_PYTHON_HOME}/python/test/machine_learning/batch_generator_and_loader_test.py.
#def augmentation_worker_proc(augmenter, is_output_augmented, dirMgr, fileBatchGenerator, epoch):
def augmentation_worker_proc(augmenter, is_output_augmented, dirMgr, inputs, outputs, batch_size, shuffle, is_time_major, epoch):
	print('\t{}: Start augmentation worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request a preparatory train directory.'.format(os.getpid()))
	while True:
		with global_lock:
			dir_path = dirMgr.requestDirectory(is_workable=False)

		if dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\t{}: Got a preparatory train directory: {}.'.format(os.getpid(), dir_path))

	#--------------------
	fileBatchGenerator = NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, False, augmenter=augmenter, is_output_augmented=is_output_augmented)
	fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.

	#--------------------
	with global_lock:
		dirMgr.returnDirectory(dir_path)
	print('\t{}: Returned a directory: {}.'.format(os.getpid(), dir_path))
	print('\t{}: End augmentation worker process.'.format(os.getpid()))

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Sets parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20190127T001424'

	initial_epoch = 0

	num_classes = 10
	input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True

	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	use_multiprocessing = True
	use_file_batch_loader = True

	num_processes = 5
	#num_batch_dirs = 5
	#batch_dir_path_prefix = './batch_dir'
	batch_info_csv_filename = 'batch_info.csv'

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepares multiprocessing.

	if use_multiprocessing:
		# set_start_method() should not be used more than once in the program.
		#mp.set_start_method('spawn')

		BaseManager.register('WorkingDirectoryManager', WorkingDirectoryManager)
		BaseManager.register('TwoStepWorkingDirectoryManager', TwoStepWorkingDirectoryManager)
		BaseManager.register('NpyFileBatchGenerator', NpyFileBatchGenerator)
		#BaseManager.register('NpyFileBatchLoader', NpyFileBatchLoader)
		manager = BaseManager()
		manager.start()

		lock = mp.Lock()
		#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

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

	#%%------------------------------------------------------------------
	# Trains and evaluates.

	if does_need_training:
		if use_multiprocessing:
			#--------------------
			batch_dir_path_prefix = './val_batch_dir'
			num_batch_dirs = 1
			valDirMgr = manager.WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			while True:
				val_dir_path = valDirMgr.requestDirectory()
				if val_dir_path is not None:
					break
				else:
					time.sleep(0.1)
			print('\tGot a validation batch directory: {}.'.format(val_dir_path))

			valFileBatchGenerator = NpyFileBatchGenerator(test_images, test_labels, batch_size, False, False, batch_info_csv_filename=batch_info_csv_filename)
			valFileBatchGenerator.saveBatches(val_dir_path)  # Generates and saves batches.

			valDirMgr.returnDirectory(val_dir_path)				

			#valFileBatchLoader = manager.NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)

			#--------------------
			batch_dir_path_prefix = './train_batch_dir'
			num_batch_dirs = 5
			trainDirMgr = manager.TwoStepWorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			#trainFileBatchGenerator = manager.NpyFileBatchGenerator(train_images, train_labels, batch_size, shuffle, False, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
			#trainFileBatchLoader = manager.NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)

			#--------------------
			if False:
				# Multiprocessing only.

				# FIXME [fix] >> This code does not work.
				#	TensorFlow session and saver cannot be passed to a worker procedure in using multiprocessing.pool.apply_async().

				#timeout = 10
				timeout = None
				with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
					training_results = pool.apply_async(training_worker_proc, args=(train_session, nnTrainer, trainDirMgr, valDirMgr, batch_info_csv_filename, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, False, False))
					data_augmentation_results = pool.map_async(partial(augmentation_worker_proc, augmenter, is_output_augmented, trainDirMgr, train_images, train_labels, batch_size, shuffle, False), [epoch for epoch in range(num_epochs)])

					training_results.get(timeout)
					data_augmentation_results.get(timeout)
			else:
				# Multiprocessing (augmentation) + multithreading (training).				

				training_worker_thread = threading.Thread(target=training_worker_proc, args=(train_session, nnTrainer, trainDirMgr, valDirMgr, batch_info_csv_filename, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, False, False))
				training_worker_thread.start()

				#timeout = 10
				timeout = None
				with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
					data_augmentation_results = pool.map_async(partial(augmentation_worker_proc, augmenter, is_output_augmented, trainDirMgr, train_images, train_labels, batch_size, shuffle, False), [epoch for epoch in range(num_epochs)])

					data_augmentation_results.get(timeout)

				training_worker_thread.join()
		elif use_file_batch_loader:
			batch_dir_path_prefix = './train_batch_dir'
			num_batch_dirs = num_epochs
			trainDirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			# TODO [improve] >> Not-so-good implementation.
			#	Usaually training is performed for much more epochs, so too many batches have to be generated before training.
			for _ in range(num_batch_dirs):
				while True:
					train_dir_path = trainDirMgr.requestDirectory()
					if train_dir_path is not None:
						break
					else:
						time.sleep(0.1)
				print('\tGot a train batch directory: {}.'.format(train_dir_path))

				trainFileBatchGenerator = NpyFileBatchGenerator(train_images, train_labels, batch_size, shuffle, False, batch_info_csv_filename=batch_info_csv_filename)
				trainFileBatchGenerator.saveBatches(train_dir_path)  # Generates and saves batches.

				trainDirMgr.returnDirectory(train_dir_path)				

			batch_dir_path_prefix = './val_batch_dir'
			num_batch_dirs = 1
			valDirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			while True:
				val_dir_path = valDirMgr.requestDirectory()
				if val_dir_path is not None:
					break
				else:
					time.sleep(0.1)
			print('\tGot a validation batch directory: {}.'.format(val_dir_path))

			valFileBatchGenerator = NpyFileBatchGenerator(test_images, test_labels, batch_size, False, False, batch_info_csv_filename=batch_info_csv_filename)
			valFileBatchGenerator.saveBatches(val_dir_path)  # Generates and saves batches.

			valDirMgr.returnDirectory(val_dir_path)				

			#--------------------
			trainFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename)
			valFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename)

			start_time = time.time()
			with train_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.train_neural_net_by_file_batch_loader(sess, nnTrainer, trainFileBatchLoader, valFileBatchLoader, trainDirMgr, valDirMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, False, False)
			print('\tTotal training time = {}'.format(time.time() - start_time))
		else:
			trainBatchGenerator = SimpleBatchGenerator(train_images, train_labels, batch_size, shuffle, False, augmenter, is_output_augmented)
			valBatchGenerator = SimpleBatchGenerator(test_images, test_labels, batch_size, False, False)

			start_time = time.time()
			with train_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.train_neural_net_by_batch_generator(sess, nnTrainer, trainBatchGenerator, valBatchGenerator, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, False, False)
			print('\tTotal training time = {}'.format(time.time() - start_time))

		#--------------------
		if use_file_batch_loader:
			batch_dir_path_prefix = './val_batch_dir'
			num_batch_dirs = 1
			valDirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

			#--------------------
			while True:
				val_dir_path = valDirMgr.requestDirectory()
				if val_dir_path is not None:
					break
				else:
					time.sleep(0.1)
			print('\tGot a validation batch directory: {}.'.format(val_dir_path))

			valFileBatchGenerator = NpyFileBatchGenerator(test_images, test_labels, batch_size, False, False, batch_info_csv_filename=batch_info_csv_filename)
			valFileBatchGenerator.saveBatches(val_dir_path)  # Generates and saves batches.

			valDirMgr.returnDirectory(val_dir_path)				

			#--------------------
			valFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename)

			start_time = time.time()
			with eval_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.evaluate_neural_net_by_file_batch_loader(sess, nnEvaluator, valFileBatchLoader, valDirMgr, eval_saver, checkpoint_dir_path, False, False)
			print('\tTotal evaluation time = {}'.format(time.time() - start_time))
		else:
			valBatchGenerator = SimpleBatchGenerator(test_images, test_labels, batch_size, False, False)

			start_time = time.time()
			with eval_session.as_default() as sess:
				with sess.graph.as_default():
					swl_tf_util.evaluate_neural_net_by_batch_generator(sess, nnEvaluator, valBatchGenerator, eval_saver, checkpoint_dir_path, False, False)
			print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infers.

	if use_file_batch_loader:
		batch_dir_path_prefix = './test_batch_dir'
		num_batch_dirs = 1
		testDirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

		#--------------------
		while True:
			test_dir_path = testDirMgr.requestDirectory()
			if test_dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\tGot a test batch directory: {}.'.format(test_dir_path))

		testFileBatchGenerator = NpyFileBatchGenerator(test_images, test_labels, batch_size, False, False, batch_info_csv_filename=batch_info_csv_filename)
		testFileBatchGenerator.saveBatches(test_dir_path)  # Generates and saves batches.

		testDirMgr.returnDirectory(test_dir_path)				

		#--------------------
		testFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename)

		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences = swl_tf_util.infer_by_neural_net_and_file_batch_loader(sess, nnInferrer, testFileBatchLoader, testDirMgr, infer_saver, checkpoint_dir_path, False)
		print('\tTotal inference time = {}'.format(time.time() - start_time))
	else:
		testBatchGenerator = SimpleBatchGenerator(test_images, test_labels, batch_size, False, False)

		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences = swl_tf_util.infer_by_neural_net_and_batch_generator(sess, nnInferrer, testBatchGenerator, infer_saver, checkpoint_dir_path, False)
		print('\tTotal inference time = {}'.format(time.time() - start_time))

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

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
