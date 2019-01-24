#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import os, time
from functools import partial
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import numpy as np
from swl.machine_learning.batch_generator import SimpleBatchGenerator, NpyFileBatchGenerator, NpyFileBatchLoader
from swl.util.working_directory_manager import SimpleWorkingDirectoryManager, WorkingDirectoryManager
import swl.util.util as swl_util

def generate_dataset(num_examples, is_output_augmented=False):
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

def generate_file_dataset(dir_path, num_examples, is_output_augmented=False):
	inputs, outputs = generate_dataset(num_examples, is_output_augmented)

	swl_util.make_dir(dir_path)

	idx, start_idx = 0, 0
	while True:
		end_idx = start_idx + np.random.randint(30, 50)
		batch_inputs = inputs[start_idx:end_idx]
		batch_outputs = outputs[start_idx:end_idx]
		np.save(os.path.join(dir_path, 'inputs_{}.npy'.format(idx)), batch_inputs)
		np.save(os.path.join(dir_path, 'outputs_{}.npy'.format(idx)), batch_outputs)
		if end_idx >= num_examples:
			break;
		start_idx = end_idx
		idx += 1
	return idx + 1  # The number of files.

def augment_identically(inputs, outputs):
	# Augments here.
	return inputs, outputs

def simple_batch_generator_example():
	num_examples = 100
	inputs, outputs = generate_dataset(num_examples)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	batchMgr = SimpleBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major)
	#batchMgr = SimpleBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, functor=augment_identically)
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		batches = batchMgr.generateBatches()  # Generates batches.
		for idx, batch in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (inputs & outputs).
			# Train with each batch (inputs & outputs).
			#print('{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def simple_npy_file_batch_generator_and_loader_example():
	num_examples = 100
	inputs, outputs = generate_dataset(num_examples)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = SimpleWorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

	batch_info_csv_filename = 'batch_info.csv'

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		while True:
			dir_path = dirMgr.requestAvailableDirectory()
			if dir_path is not None:
				break
			else:
				time.sleep(0.1)

		print('\t>>>>> Directory: {}.'.format(dir_path))

		#batchGenerator = NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major)
		batchGenerator = NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, functor=augment_identically, batch_info_csv_filename=batch_info_csv_filename)
		batchGenerator.saveBatches(dir_path)  # Generates and saves batches.

		batchLoader = NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = batchLoader.loadBatches(dir_path)  # Loads batches.
		for idx, batch in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (inputs & outputs).
			# Train with each batch (inputs & outputs).
			#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirMgr.returnDirectory(dir_path)				

def init(lock):
	global global_lock
	global_lock = lock

#def data_augmentation_worker(dirMgr, batchGenerator, epoch):
def data_augmentation_worker(dirMgr, inputs, outputs, batch_size, shuffle, is_time_major, epoch):
	print('\t{}: Start data augmentation worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request an available directory.'.format(os.getpid()))
	while True:
		"""
		global_lock.acquire()
		try:
			dir_path = dirMgr.requestAvailableDirectory()
		finally:
			global_lock.release()
		"""
		with global_lock:
			dir_path = dirMgr.requestAvailableDirectory()

		if dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\t{}: Got an available directory: {}.'.format(os.getpid(), dir_path))

	#--------------------
	#batchGenerator = NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major)
	batchGenerator = NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, functor=augment_identically)
	batchGenerator.saveBatches(dir_path)  # Generates and saves batches.

	#--------------------
	"""
	global_lock.acquire()
	try:
		dirMgr.returnDirectoryAsReady(dir_path)
	finally:
		global_lock.release()
	"""
	with global_lock:
		dirMgr.returnDirectoryAsReady(dir_path)
	print('\t{}: Returned a directory as ready: {}.'.format(os.getpid(), dir_path))
	print('\t{}: End data augmentation worker process.'.format(os.getpid()))

#def training_worker(dirMgr, batchLoader, num_epochs):
def training_worker(dirMgr, batch_info_csv_filename, num_epochs):
	print('\t{}: Start training worker process.'.format(os.getpid()))

	for epoch in range(num_epochs):
		print('\t{}: Request a ready directory: epoch {}.'.format(os.getpid(), epoch))
		while True:
			"""
			global_lock.acquire()
			try:
				dir_path = dirMgr.requestReadyDirectory()
			finally:
				global_lock.release()
			"""
			with global_lock:
				dir_path = dirMgr.requestReadyDirectory()

			if dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\t{}: Got a ready directory: {}.'.format(os.getpid(), dir_path))

		#--------------------
		batchLoader = NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = batchLoader.loadBatches(dir_path)  # Loads batches.
		for idx, batch in enumerate(batches):
			# Train with each batch (inputs & outputs).
			#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		#--------------------
		"""
		global_lock.acquire()
		try:
			dirMgr.returnDirectoryAsAvailable(dir_path)
		finally:
			global_lock.release()
		"""
		with global_lock:
			dirMgr.returnDirectoryAsAvailable(dir_path)
		print('\t{}: Returned a directory as available: {}.'.format(os.getpid(), dir_path))

	print('\t{}: End training worker process.'.format(os.getpid()))

def multiprocessing_npy_file_batch_generator_and_loader_example():
	num_examples = 100
	inputs, outputs = generate_dataset(num_examples)

	num_epochs = 7
	batch_size = 12
	shuffle = True
	is_time_major = False

	#--------------------
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	BaseManager.register('WorkingDirectoryManager', WorkingDirectoryManager)
	BaseManager.register('NpyFileBatchGenerator', NpyFileBatchGenerator)
	#BaseManager.register('NpyFileBatchLoader', NpyFileBatchLoader)
	manager = BaseManager()
	manager.start()

	num_processes = 5
	lock = mp.Lock()
	#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = manager.WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)
	
	batch_info_csv_filename = 'batch_info.csv'
	batchGenerator = manager.NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, functor=augment_identically, batch_info_csv_filename=batch_info_csv_filename)
	#batchLoader = manager.NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)

	#--------------------
	#timeout = 10
	timeout = None
	#with mp.Pool(processes=num_processes) as pool:  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	with mp.Pool(processes=num_processes, initializer=init, initargs=(lock,)) as pool:
		#training_results = pool.apply_async(training_worker, args=(dirMgr, manager.NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename), num_epochs))  # Error.
		#training_results = pool.apply_async(training_worker, args=(dirMgr, batchLoader, num_epochs))  # TypeError: can't pickle generator objects.
		training_results = pool.apply_async(training_worker, args=(dirMgr, batch_info_csv_filename, num_epochs))
		#data_augmentation_results = pool.map_async(partial(data_augmentation_worker, dirMgr, manager.NpyFileBatchGenerator(inputs, outputs, batch_size, shuffle, is_time_major, functor=augment_identically, batch_info_csv_filename=batch_info_csv_filename)), [epoch for epoch in range(num_epochs)])  # Error.
		#data_augmentation_results = pool.map_async(partial(data_augmentation_worker, dirMgr, batchGenerator), [epoch for epoch in range(num_epochs)])  # Ok.
		data_augmentation_results = pool.map_async(partial(data_augmentation_worker, dirMgr, inputs, outputs, batch_size, shuffle, is_time_major), [epoch for epoch in range(num_epochs)])

		training_results.get(timeout)
		data_augmentation_results.get(timeout)

def main():
	#simple_batch_generator_example()

	#simple_npy_file_batch_generator_and_loader_example()
	multiprocessing_npy_file_batch_generator_and_loader_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
