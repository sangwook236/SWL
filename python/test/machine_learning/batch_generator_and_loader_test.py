#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import os, time
from functools import partial
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import numpy as np
import cv2 as cv
from imgaug import augmenters as iaa
from swl.machine_learning.batch_generator import SimpleBatchGenerator, NpzFileBatchGenerator, NpzFileBatchGeneratorFromNpyFiles, NpzFileBatchGeneratorFromImageFiles
from swl.machine_learning.batch_loader import NpzFileBatchLoader
from swl.util.working_directory_manager import WorkingDirectoryManager, TwoStepWorkingDirectoryManager
import swl.util.util as swl_util

def augment_identically(inputs, outputs, is_output_augmented=False):
	# Augments here.
	return inputs, outputs

class IdentityAugmenter(object):
	def __call__(self, inputs, outputs, is_output_augmented=False):
		# Augments here.
		return inputs, outputs

class ImgaugAugmenter(object):
	def __init__(self):
		self._augmenter = iaa.Sequential([
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

	def __call__(self, inputs, outputs, is_output_augmented=False):
		# Augments here.
		if is_output_augmented:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)
		else:
			return self._augmenter.augment_images(inputs), outputs

def generate_numpy_dataset(num_examples, is_output_augmented=False):
	if is_output_augmented:
		inputs = np.zeros((num_examples, 2, 2, 1))
		outputs = np.zeros((num_examples, 2, 2, 1))
	else:
		inputs = np.zeros((num_examples, 2, 2, 1))
		outputs = np.zeros((num_examples, 1))

	for idx in range(num_examples):
		inputs[idx] = idx
		outputs[idx] = idx
	return inputs, outputs

def generate_image_dataset(num_examples, is_output_augmented=False):
	if is_output_augmented:
		#inputs = np.random.randint(0, 256, size=(num_examples, 200, 300, 3))
		inputs = np.zeros((num_examples, 200, 300, 3), dtype=np.uint)
		outputs = np.zeros((num_examples, 200, 300, 1), dtype=np.int32)
	else:
		#inputs = np.random.randint(0, 256, size=(num_examples, 200, 300, 3))
		inputs = np.zeros((num_examples, 200, 300, 3), dtype=np.uint8)
		outputs = np.zeros((num_examples, 1), dtype=np.int32)

	for idx in range(num_examples):
		inputs[idx] = idx
		outputs[idx] = idx
	return inputs, outputs

def generate_npy_file_dataset(dir_path, num_examples, is_output_augmented=False):
	inputs, outputs = generate_numpy_dataset(num_examples, is_output_augmented)

	swl_util.make_dir(dir_path)

	input_filepaths, output_filepaths = list(), list()
	idx, start_idx = 0, 0
	while True:
		end_idx = start_idx + np.random.randint(30, 50)
		batch_inputs, batch_outputs = inputs[start_idx:end_idx], outputs[start_idx:end_idx]
		input_filepath, output_filepath = os.path.join(dir_path, 'inputs_{}.npy'.format(idx)), os.path.join(dir_path, 'outputs_{}.npy'.format(idx))
		np.save(input_filepath, batch_inputs)
		np.save(output_filepath, batch_outputs)
		input_filepaths.append(input_filepath)
		output_filepaths.append(output_filepath)
		if end_idx >= num_examples:
			break;
		start_idx = end_idx
		idx += 1
	return input_filepaths, output_filepaths

def generate_image_file_dataset(dir_path, num_examples, is_output_augmented=False):
	inputs, outputs = generate_image_dataset(num_examples, is_output_augmented)

	swl_util.make_dir(dir_path)

	input_filepaths = list()
	for idx, inp in enumerate(inputs):
		input_filepath = os.path.join(dir_path, 'inputs_{}.png'.format(idx))
		cv.imwrite(input_filepath, inp)
		input_filepaths.append(input_filepath)
	return input_filepaths, outputs.tolist()

def simple_batch_generator_example():
	num_examples = 100
	inputs, outputs = generate_numpy_dataset(num_examples)

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	batchGenerator = SimpleBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major)
	#batchGenerator = SimpleBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented)
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		batches = batchGenerator.generateBatches()  # Generates batches.
		for idx, (batch_data, num_batch_examples) in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (inputs & outputs).
			# Train with each batch (inputs & outputs).
			#print('{}: {}, {}'.format(idx, batch_data[0].shape, batch_data[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch_data[0].shape, np.max(np.reshape(batch_data[0], (batch_data[0].shape[0], -1)), axis=-1), batch_data[1].shape, np.max(np.reshape(batch_data[1], (batch_data[1].shape[0], -1)), axis=-1)))

def simple_npz_file_batch_generator_and_loader_example():
	num_examples = 3000
	inputs, outputs = generate_numpy_dataset(num_examples)

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

	batch_info_csv_filename = 'batch_info.csv'
	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		while True:
			dir_path = dirMgr.requestDirectory()
			if dir_path is not None:
				break
			else:
				time.sleep(0.1)

		print('\t>>>>> Directory: {}.'.format(dir_path))

		#fileBatchGenerator = NpzFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major)
		fileBatchGenerator = NpzFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
		num_saved_examples = fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.

		fileBatchLoader = NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = fileBatchLoader.loadBatches(dir_path)  # Loads batches.

		#dirMgr.returnDirectory(dir_path)  # If dir_path is returned before completing a job, dir_path can be used in a different job.

		num_loaded_examples = 0
		for idx, (batch_data, num_batch_examples) in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (inputs & outputs).
			# Train with each batch (inputs & outputs).
			#print('\t{}: {}, {}, {}'.format(idx, num_batch_examples, batch_data[0].shape, batch_data[1].shape))
			print('\t{}: {}, {}-{}, {}-{}'.format(idx, num_batch_examples, batch_data[0].shape, np.max(np.reshape(batch_data[0], (batch_data[0].shape[0], -1)), axis=-1), batch_data[1].shape, np.max(np.reshape(batch_data[1], (batch_data[1].shape[0], -1)), axis=-1)))
			num_loaded_examples += num_batch_examples

		print('#saved examples =', num_saved_examples)
		print('#loaded examples =', num_loaded_examples)

		dirMgr.returnDirectory(dir_path)

def simple_npz_file_batch_generator_from_npy_files_and_loader_example():
	num_examples = 3000
	npy_input_filepaths, npy_output_filepaths = generate_npy_file_dataset('./npy_files', num_examples)
	npy_input_filepaths, npy_output_filepaths = np.array(npy_input_filepaths), np.array(npy_output_filepaths)
	num_loaded_files = 3

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

	batch_info_csv_filename = 'batch_info.csv'
	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		while True:
			dir_path = dirMgr.requestDirectory()
			if dir_path is not None:
				break
			else:
				time.sleep(0.1)

		print('\t>>>>> Directory: {}.'.format(dir_path))

		#fileBatchGenerator = NpzFileBatchGeneratorFromNpyFiles(npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major)
		fileBatchGenerator = NpzFileBatchGeneratorFromNpyFiles(npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
		num_saved_examples = fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.

		fileBatchLoader = NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = fileBatchLoader.loadBatches(dir_path)  # Loads batches.

		#dirMgr.returnDirectory(dir_path)  # If dir_path is returned before completing a job, dir_path can be used in a different job.

		num_loaded_examples = 0
		for idx, (batch_data, num_batch_examples) in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (inputs & outputs).
			# Train with each batch (inputs & outputs).
			#print('\t{}: {}, {}, {}'.format(idx, num_batch_examples, batch_data[0].shape, batch_data[1].shape))
			print('\t{}: {}, {}-{}, {}-{}'.format(idx, num_batch_examples, batch_data[0].shape, np.max(np.reshape(batch_data[0], (batch_data[0].shape[0], -1)), axis=-1), batch_data[1].shape, np.max(np.reshape(batch_data[1], (batch_data[1].shape[0], -1)), axis=-1)))
			num_loaded_examples += num_batch_examples

		print('#saved examples =', num_saved_examples)
		print('#loaded examples =', num_loaded_examples)

		dirMgr.returnDirectory(dir_path)

def simple_npz_file_batch_generator_from_image_files_and_loader_example():
	num_examples = 256
	npy_input_filepaths, output_seqs = generate_image_file_dataset('./image_files', num_examples)
	num_loaded_files = 57

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

	batch_info_csv_filename = 'batch_info.csv'
	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		while True:
			dir_path = dirMgr.requestDirectory()
			if dir_path is not None:
				break
			else:
				time.sleep(0.1)

		print('\t>>>>> Directory: {}.'.format(dir_path))

		#fileBatchGenerator = NpzFileBatchGeneratorFromImageFiles(npy_input_filepaths, output_seqs, num_loaded_files, batch_size, shuffle, is_time_major)
		fileBatchGenerator = NpzFileBatchGeneratorFromImageFiles(npy_input_filepaths, output_seqs, num_loaded_files, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
		num_saved_examples = fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.

		fileBatchLoader = NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = fileBatchLoader.loadBatches(dir_path)  # Loads batches.

		#dirMgr.returnDirectory(dir_path)  # If dir_path is returned before completing a job, dir_path can be used in a different job.

		num_loaded_examples = 0
		for idx, (batch_data, num_batch_examples) in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (inputs & outputs).
			# Train with each batch (inputs & outputs).
			#print('\t{}: {}, {}, {}'.format(idx, num_batch_examples, batch_data[0].shape, batch_data[1].shape))
			print('\t{}: {}, {}-{}, {}-{}'.format(idx, num_batch_examples, batch_data[0].shape, np.max(np.reshape(batch_data[0], (batch_data[0].shape[0], -1)), axis=-1), batch_data[1].shape, np.max(np.reshape(batch_data[1], (batch_data[1].shape[0], -1)), axis=-1)))
			num_loaded_examples += num_batch_examples

		print('#saved examples =', num_saved_examples)
		print('#loaded examples =', num_loaded_examples)

		dirMgr.returnDirectory(dir_path)

def initialize_lock(lock):
	global global_lock
	global_lock = lock

#def training_worker_proc(dirMgr, fileBatchLoader, num_epochs):
def training_worker_proc(dirMgr, batch_info_csv_filename, num_epochs):
	print('\t{}: Start training worker process.'.format(os.getpid()))

	for epoch in range(num_epochs):
		print('\t{}: Request a working directory: epoch {}.'.format(os.getpid(), epoch))
		while True:
			"""
			global_lock.acquire()
			try:
				dir_path = dirMgr.requestDirectory()
			finally:
				global_lock.release()
			"""
			with global_lock:
				dir_path = dirMgr.requestDirectory()

			if dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\t{}: Got a working directory: {}.'.format(os.getpid(), dir_path))

		#--------------------
		fileBatchLoader = NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = fileBatchLoader.loadBatches(dir_path)  # Loads batches.
		for idx, (batch_data, num_batch_examples) in enumerate(batches):
			# Train with each batch (inputs & outputs).
			#print('\t{}: {}, {}, {}'.format(idx, num_batch_examples, batch_data[0].shape, batch_data[1].shape))
			print('\t{}: {}, {}-{}, {}-{}'.format(idx, num_batch_examples, batch_data[0].shape, np.max(np.reshape(batch_data[0], (batch_data[0].shape[0], -1)), axis=-1), batch_data[1].shape, np.max(np.reshape(batch_data[1], (batch_data[1].shape[0], -1)), axis=-1)))

		#--------------------
		"""
		global_lock.acquire()
		try:
			dirMgr.returnDirectory(dir_path)
		finally:
			global_lock.release()
		"""
		with global_lock:
			dirMgr.returnDirectory(dir_path)
		print('\t{}: Returned a directory: {}.'.format(os.getpid(), dir_path))

	print('\t{}: End training worker process.'.format(os.getpid()))

#def augmentation_worker_proc(dirMgr, fileBatchGenerator, epoch):
def augmentation_worker_proc(dirMgr, inputs, outputs, batch_size, shuffle, is_time_major, epoch):
	print('\t{}: Start augmentation worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request a preparatory directory.'.format(os.getpid()))
	while True:
		with global_lock:
			dir_path = dirMgr.requestDirectory(is_workable=False)

		if dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\t{}: Got a preparatory directory: {}.'.format(os.getpid(), dir_path))

	#--------------------
	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#fileBatchGenerator = NpzFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major)
	fileBatchGenerator = NpzFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented)
	fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.

	#--------------------
	with global_lock:
		dirMgr.returnDirectory(dir_path)
	print('\t{}: Returned a directory: {}.'.format(os.getpid(), dir_path))
	print('\t{}: End augmentation worker process.'.format(os.getpid()))

#def augmentation_with_file_input_worker_proc(dirMgr, fileBatchGenerator, epoch):
def augmentation_with_file_input_worker_proc(dirMgr, npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major, epoch):
	print('\t{}: Start augmentation worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request a preparatory directory.'.format(os.getpid()))
	while True:
		with global_lock:
			dir_path = dirMgr.requestDirectory(is_workable=False)

		if dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\t{}: Got a preparatory directory: {}.'.format(os.getpid(), dir_path))

	#--------------------
	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#fileBatchGenerator = NpzFileBatchGeneratorFromNpyFiles(npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major)
	fileBatchGenerator = NpzFileBatchGeneratorFromNpyFiles(npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented)
	fileBatchGenerator.saveBatches(dir_path)  # Generates and saves batches.

	#--------------------
	with global_lock:
		dirMgr.returnDirectory(dir_path)
	print('\t{}: Returned a directory: {}.'.format(os.getpid(), dir_path))
	print('\t{}: End augmentation worker process.'.format(os.getpid()))

def multiprocessing_npz_file_batch_generator_and_loader_example():
	num_examples = 100
	inputs, outputs = generate_numpy_dataset(num_examples)

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#--------------------
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	num_processes = 5
	num_batch_dirs = 5
	batch_dir_path_prefix = './batch_dir'
	batch_info_csv_filename = 'batch_info.csv'

	BaseManager.register('TwoStepWorkingDirectoryManager', TwoStepWorkingDirectoryManager)
	BaseManager.register('NpzFileBatchGenerator', NpzFileBatchGenerator)
	#BaseManager.register('NpzFileBatchLoader', NpzFileBatchLoader)
	manager = BaseManager()
	manager.start()

	lock = mp.Lock()
	#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	dirMgr = manager.TwoStepWorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)
	#fileBatchGenerator = manager.NpzFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
	#fileBatchLoader = manager.NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)

	#--------------------
	#timeout = 10
	timeout = None
	#with mp.Pool(processes=num_processes) as pool:  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
		#training_results = pool.apply_async(training_worker_proc, args=(dirMgr, manager.NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename), num_epochs))  # Error.
		#training_results = pool.apply_async(training_worker_proc, args=(dirMgr, fileBatchLoader, num_epochs))  # TypeError: can't pickle generator objects.
		training_results = pool.apply_async(training_worker_proc, args=(dirMgr, batch_info_csv_filename, num_epochs))
		#data_augmentation_results = pool.map_async(partial(augmentation_worker_proc, dirMgr, manager.NpzFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)), [epoch for epoch in range(num_epochs)])  # Error.
		#data_augmentation_results = pool.map_async(partial(augmentation_worker_proc, dirMgr, fileBatchGenerator), [epoch for epoch in range(num_epochs)])  # Ok.
		data_augmentation_results = pool.map_async(partial(augmentation_worker_proc, dirMgr, inputs, outputs, batch_size, shuffle, is_time_major), [epoch for epoch in range(num_epochs)])

		training_results.get(timeout)
		data_augmentation_results.get(timeout)

def multiprocessing_npz_file_batch_generator_from_npy_files_and_loader_example():
	num_examples = 300
	npy_input_filepaths, npy_output_filepaths = generate_npy_file_dataset('./npy_files', num_examples)
	npy_input_filepaths, npy_output_filepaths = np.array(npy_input_filepaths), np.array(npy_output_filepaths)
	num_loaded_files = 3

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	#augmenter = augment_identically
	#augmenter = IdentityAugmenter()
	augmenter = ImgaugAugmenter()
	is_output_augmented = False

	#--------------------
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	num_processes = 5
	num_batch_dirs = 5
	batch_dir_path_prefix = './batch_dir'
	batch_info_csv_filename = 'batch_info.csv'

	BaseManager.register('TwoStepWorkingDirectoryManager', TwoStepWorkingDirectoryManager)
	BaseManager.register('NpzFileBatchGeneratorFromNpyFiles', NpzFileBatchGeneratorFromNpyFiles)
	#BaseManager.register('NpzFileBatchLoader', NpzFileBatchLoader)
	manager = BaseManager()
	manager.start()

	lock = mp.Lock()
	#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	dirMgr = manager.TwoStepWorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)
	#fileBatchGenerator = manager.NpzFileBatchGeneratorFromNpyFiles(npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)
	#fileBatchLoader = manager.NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)

	#--------------------
	#timeout = 10
	timeout = None
	#with mp.Pool(processes=num_processes) as pool:  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
		#training_results = pool.apply_async(training_worker_proc, args=(dirMgr, manager.NpzFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename), num_epochs))  # Error.
		#training_results = pool.apply_async(training_worker_proc, args=(dirMgr, fileBatchLoader, num_epochs))  # TypeError: can't pickle generator objects.
		training_results = pool.apply_async(training_worker_proc, args=(dirMgr, batch_info_csv_filename, num_epochs))
		#data_augmentation_results = pool.map_async(partial(augmentation_with_file_input_worker_proc, dirMgr, manager.NpzFileBatchGeneratorFromNpyFiles(npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major, augmenter=augmenter, is_output_augmented=is_output_augmented, batch_info_csv_filename=batch_info_csv_filename)), [epoch for epoch in range(num_epochs)])  # Error.
		#data_augmentation_results = pool.map_async(partial(augmentation_with_file_input_worker_proc, dirMgr, fileBatchGenerator), [epoch for epoch in range(num_epochs)])  # Ok.
		data_augmentation_results = pool.map_async(partial(augmentation_with_file_input_worker_proc, dirMgr, npy_input_filepaths, npy_output_filepaths, num_loaded_files, batch_size, shuffle, is_time_major), [epoch for epoch in range(num_epochs)])

		training_results.get(timeout)
		data_augmentation_results.get(timeout)

def main():
	# Batch generator.
	#simple_batch_generator_example()

	# Batch generator and loader.
	#simple_npz_file_batch_generator_and_loader_example()
	#simple_npz_file_batch_generator_from_npy_files_and_loader_example()
	simple_npz_file_batch_generator_from_image_files_and_loader_example()

	#multiprocessing_npz_file_batch_generator_and_loader_example()
	#multiprocessing_npz_file_batch_generator_from_npy_files_and_loader_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
