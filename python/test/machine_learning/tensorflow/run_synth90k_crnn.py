#!/usr/bin/env python

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
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
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from swl.machine_learning.batch_generator import SimpleBatchGenerator, NpyFileBatchGeneratorWithFileInput
from swl.machine_learning.batch_loader import NpyFileBatchLoader
from swl.util.working_directory_manager import WorkingDirectoryManager, TwoStepWorkingDirectoryManager
from synth90k_crnn import Synth90kCrnnWithCrossEntropyLoss, Synth90kCrnnWithCtcLoss

#%%------------------------------------------------------------------

def create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output):
	if is_sparse_output:
		return Synth90kCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes, label_eos_token)
	else:
		return Synth90kCrnnWithCrossEntropyLoss(image_height, image_width, image_channel, num_classes)

#%%------------------------------------------------------------------

class SimpleCrnnTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 1.0
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08)

		super().__init__(neuralNet, optimizer, global_step)

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

#%%------------------------------------------------------------------

class Synth90kPreprocessor(object):
	def __init__(self, is_sparse_output):
		self._is_sparse_output = is_sparse_output

		#self._num_labels = 36  # 0~9 + a~z.
		self._num_labels = 37  # 0~9 + a~z + <EOS>.
		#self._num_labels = 38  # <SOS> + 0~9 + a~z + <EOS>.
		self._num_classes = self._num_labels + 1  # blank label.
		self._label_eos_token = self._num_classes - 2

	def __call__(self, inputs, outputs, *args, **kwargs):
		"""
		Inputs:
			inputs (numpy.array): images of size (samples, height, width) and type uint8.
			outputs (numpy.array): labels of size (samples, max_label_length) and type uint8.
		Outputs:
			inputs (numpy.array): images of size (samples, height, width, 1) and type float32.
			outputs (numpy.array or a tuple): labels of size (samples, max_label_length, num_labels) and type uint8 when is_sparse_output = False. A tuple with (indices, values, shape) for a sparse tensor when is_sparse_output = True.
		"""

		if inputs is not None:
			# inputs' shape = (32, 128) -> (32, 128, 1).

			# Preprocessing (normalization, standardization, etc.).
			inputs = np.reshape(inputs.astype(np.float32) / 255.0, inputs.shape + (1,))
			#inputs = (inputs - np.mean(inputs, axis=axis)) / np.std(inputs, axis=axis)

		if outputs is not None:
			if self._is_sparse_output:
				# Sparse tensor: (num_examples, max_label_len) -> A tuple with (indices, values, shape) for a sparse tensor.
				outputs = swl_ml_util.generate_sparse_tuple_from_numpy_array(outputs, self._label_eos_token, np.uint8)
			else:
				# One-hot encoding: (num_examples, max_label_len) -> (num_examples, max_label_len, num_classes).
				outputs = swl_ml_util.to_one_hot_encoding(outputs, self._num_classes).astype(np.uint8)
				#outputs = swl_ml_util.to_one_hot_encoding(outputs, self._num_classes).astype(np.uint8)  # Error.

		return inputs, outputs

def load_data(synth90k_base_dir_path):
	train_npy_file_csv_filepath = synth90k_base_dir_path + '/train/npy_file_info.csv'
	val_npy_file_csv_filepath = synth90k_base_dir_path + '/val/npy_file_info.csv'
	test_npy_file_csv_filepath = synth90k_base_dir_path + '/test/npy_file_info.csv'

	train_input_filepaths, train_output_filepaths, train_example_counts = swl_util.load_filepaths_from_npy_file_info(train_npy_file_csv_filepath)
	val_input_filepaths, val_output_filepaths, val_example_counts = swl_util.load_filepaths_from_npy_file_info(val_npy_file_csv_filepath)
	test_input_filepaths, test_output_filepaths, test_example_counts = swl_util.load_filepaths_from_npy_file_info(test_npy_file_csv_filepath)

	return train_input_filepaths, train_output_filepaths, val_input_filepaths, val_output_filepaths, test_input_filepaths, test_output_filepaths

#%%------------------------------------------------------------------

def initialize_lock(lock):
	global global_lock
	global_lock = lock

# REF [function] >> training_worker_proc() in ${SWL_PYTHON_HOME}/python/test/machine_learning/batch_generator_and_loader_test.py.
#def training_worker_proc(train_session, nnTrainer, trainDirMgr, valDirMgr, trainFileBatchLoader, valFileBatchLoader, num_epochs):
def training_worker_proc(train_session, nnTrainer, trainDirMgr, valDirMgr, batch_info_csv_filename, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output):
	print('\t{}: Start training worker process.'.format(os.getpid()))

	trainFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename, data_processing_functor=Synth90kPreprocessor(is_sparse_output))
	valFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename, data_processing_functor=Synth90kPreprocessor(is_sparse_output))

	#--------------------
	start_time = time.time()
	with train_session.as_default() as sess:
		with sess.graph.as_default():
			swl_tf_util.train_neural_net_by_file_batch_loader(sess, nnTrainer, trainFileBatchLoader, valFileBatchLoader, trainDirMgr, valDirMgr, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_output)
	print('\tTotal training time = {}'.format(time.time() - start_time))

	print('\t{}: End training worker process.'.format(os.getpid()))

# REF [function] >> augmentation_worker_proc() in ${SWL_PYTHON_HOME}/python/test/machine_learning/batch_generator_and_loader_test.py.
#def augmentation_worker_proc(augmenter, is_output_augmented, batch_info_csv_filename, dirMgr, fileBatchGenerator, epoch):
def augmentation_worker_proc(augmenter, is_output_augmented, batch_info_csv_filename, dirMgr, input_filepaths, output_filepaths, num_loaded_files_at_a_time, batch_size, shuffle, is_time_major, epoch):
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
	fileBatchGenerator = NpyFileBatchGeneratorWithFileInput(input_filepaths, output_filepaths, num_loaded_files_at_a_time, batch_size, shuffle, False, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
	num_saved_examples = fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.
	print('\t{}: #saved examples = {}.'.format(os.getpid(), num_saved_examples))

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

	output_dir_prefix = 'synth90k_crnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	initial_epoch = 0

	# When outputs are not sparse, CRNN model's output shape = (samples, 32, num_classes) and dataset's output shape = (samples, 23, num_classes).
	is_sparse_output = True  # Fixed.
	#is_time_major = False  # Fixed.

	# NOTE [info] >> Places with the same parameters.
	#	class Synth90kLabelConverter in ${SWL_PYTHON_HOME}/test/language_processing/synth90k_dataset_test.py.
	#	class Synth90kPreprocessor.

	image_height, image_width, image_channel = 32, 128, 1
	max_label_len = 23  # Max length of words in lexicon.

	# Label: 0~9 + a~z + A~Z.
	#label_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	# Label: 0~9 + a~z.
	label_characters = '0123456789abcdefghijklmnopqrstuvwxyz'

	SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
	EOS = '<EOS>'  # All strings will end with the End-Of-String token.
	#extended_label_list = [SOS] + list(label_characters) + [EOS]
	extended_label_list = list(label_characters) + [EOS]
	#extended_label_list = list(label_characters)

	int2char = extended_label_list
	char2int = {c:i for i, c in enumerate(extended_label_list)}

	num_labels = len(extended_label_list)
	num_classes = num_labels + 1  # extended labels + blank label.
	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	blank_label = num_classes - 1
	label_eos_token = char2int[EOS]
	#label_eos_token = blank_label

	batch_size = 256  # Number of samples per gradient update.
	num_epochs = 100  # Number of times to iterate over training data.
	shuffle = True

	augmenter = ImgaugAugmenter()
	#augmenter = create_imgaug_augmenter()  # If imgaug augmenter is used, data are augmented in background augmentation processes. (faster)
	is_output_augmented = False

	#use_multiprocessing = True  # Fixed. Batch generators & loaders are used in case of multiprocessing.
	#use_file_batch_loader = True  # Fixed. It is not related to multiprocessing.
	num_loaded_files_at_a_time = 5

	num_processes = 5
	train_batch_dir_path_prefix = './train_batch_dir'
	train_num_batch_dirs = 10
	val_batch_dir_path_prefix = './val_batch_dir'
	val_num_batch_dirs = 1
	test_batch_dir_path_prefix = './test_batch_dir'
	test_num_batch_dirs = 1
	batch_info_csv_filename = 'batch_info.csv'

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepares multiprocessing.

	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	BaseManager.register('WorkingDirectoryManager', WorkingDirectoryManager)
	BaseManager.register('TwoStepWorkingDirectoryManager', TwoStepWorkingDirectoryManager)
	BaseManager.register('NpyFileBatchGeneratorWithFileInput', NpyFileBatchGeneratorWithFileInput)
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

	# NOTE [info] >> Generate synth90k dataset using swl.language_processing.synth90k_dataset.save_synth90k_dataset_to_npy_files().
	#	Refer to ${SWL_PYTHON_HOME}/test/language_processing/synth90k_dataset_test.py.

	synth90k_base_dir_path = './synth90k_npy'
	train_input_filepaths, train_output_filepaths, val_input_filepaths, val_output_filepaths, test_input_filepaths, test_output_filepaths = load_data(synth90k_base_dir_path)

	#--------------------
	# Creates models, sessions, and graphs.

	# Creates graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			#K.set_learning_phase(1)  # Sets the learning phase to 'train'. (Required)

			# Creates a model.
			modelForTraining = create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output)
			modelForTraining.create_training_model()

			# Creates a trainer.
			nnTrainer = SimpleCrnnTrainer(modelForTraining, initial_epoch)

			# Creates a saver.
			#	Saves a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			#K.set_learning_phase(0)  # Sets the learning phase to 'test'. (Required)

			# Creates a model.
			modelForEvaluation = create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output)
			modelForEvaluation.create_evaluation_model()

			# Creates an evaluator.
			nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

			# Creates a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		#K.set_learning_phase(0)  # Sets the learning phase to 'test'. (Required)

		# Creates a model.
		modelForInference = create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output)
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
		valDirMgr = WorkingDirectoryManager(val_batch_dir_path_prefix, val_num_batch_dirs)

		print('\tWaiting for a validation batch directory...')
		while True:
			val_dir_path = valDirMgr.requestDirectory()
			if val_dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\tGot a validation batch directory: {}.'.format(val_dir_path))

		valFileBatchGenerator = NpyFileBatchGeneratorWithFileInput(val_input_filepaths, val_output_filepaths, num_loaded_files_at_a_time, batch_size, False, False, batch_info_csv_filename=batch_info_csv_filename)
		num_saved_examples  = valFileBatchGenerator.saveBatches(val_dir_path)  # Generates and saves batches.
		print('\t#saved examples = {}.'.format(num_saved_examples))

		valDirMgr.returnDirectory(val_dir_path)				

		#--------------------
		# Multiprocessing (augmentation) + multithreading (training).				

		trainDirMgr_mp = manager.TwoStepWorkingDirectoryManager(train_batch_dir_path_prefix, train_num_batch_dirs)
		valDirMgr_mp = manager.WorkingDirectoryManager(val_batch_dir_path_prefix, val_num_batch_dirs)

		#trainFileBatchGenerator_mp = manager.NpyFileBatchGeneratorWithFileInput(train_input_filepaths, train_output_filepaths, num_loaded_files_at_a_time, batch_size, shuffle, False, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
		#trainFileBatchLoader_mp = manager.NpyFileBatchLoader(batch_info_csv_filename, data_processing_functor=Synth90kPreprocessor(is_sparse_output))
		#valFileBatchLoader_mp = manager.NpyFileBatchLoader(batch_info_csv_filename, data_processing_functor=Synth90kPreprocessor(is_sparse_output))

		training_worker_thread = threading.Thread(target=training_worker_proc, args=(train_session, nnTrainer, trainDirMgr_mp, valDirMgr_mp, batch_info_csv_filename, num_epochs, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, False, is_sparse_output))
		training_worker_thread.start()

		#timeout = 10
		timeout = None
		with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
			data_augmentation_results = pool.map_async(partial(augmentation_worker_proc, augmenter, is_output_augmented, batch_info_csv_filename, trainDirMgr_mp, train_input_filepaths, train_output_filepaths, num_loaded_files_at_a_time, batch_size, shuffle, False), [epoch for epoch in range(num_epochs)])

			data_augmentation_results.get(timeout)

		training_worker_thread.join()

		#--------------------
		valFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename, data_processing_functor=Synth90kPreprocessor(is_sparse_output))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				swl_tf_util.evaluate_neural_net_by_file_batch_loader(sess, nnEvaluator, valFileBatchLoader, valDirMgr, eval_saver, checkpoint_dir_path, False, False)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infers.

	testDirMgr = WorkingDirectoryManager(test_batch_dir_path_prefix, test_num_batch_dirs)

	#--------------------
	print('\tWaiting for a test batch directory...')
	while True:
		test_dir_path = testDirMgr.requestDirectory()
		if test_dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\tGot a test batch directory: {}.'.format(test_dir_path))

	testFileBatchGenerator = NpyFileBatchGeneratorWithFileInput(test_input_filepaths, test_output_filepaths, num_loaded_files_at_a_time, batch_size, False, False, batch_info_csv_filename=batch_info_csv_filename)
	num_saved_examples = testFileBatchGenerator.saveBatches(test_dir_path)  # Generates and saves batches.
	print('\t#saved examples = {}.'.format(num_saved_examples))

	testDirMgr.returnDirectory(test_dir_path)				

	#--------------------
	testFileBatchLoader = NpyFileBatchLoader(batch_info_csv_filename, data_processing_functor=Synth90kPreprocessor(is_sparse_output))

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			inferences = swl_tf_util.infer_by_neural_net_and_file_batch_loader(sess, nnInferrer, testFileBatchLoader, testDirMgr, infer_saver, checkpoint_dir_path, False)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	#--------------------
	if inferences is not None:
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
