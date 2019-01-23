#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import os, time
from functools import partial
import multiprocessing as mp
import numpy as np
from swl.machine_learning.batch_generator import SimpleBatchGenerator, NpyFileBatchGenerator, NpyFileBatchLoader
from swl.util.working_directory_manager import SimpleWorkingDirectoryManager, WorkingDirectoryManager
import swl.util.util as swl_util

def generate_dataset(num_examples, is_label_augmented=False):
	if is_label_augmented:
		images = np.zeros((num_examples, 2, 2, 1))
		labels = np.zeros((num_examples, 2, 2, 1))
	else:
		images = np.zeros((num_examples, 2, 2, 1))
		labels = np.zeros((num_examples, 1))

	for idx in range(num_examples):
		images[idx] = idx
		labels[idx] = idx
	return images, labels

def generate_file_dataset(dir_path, num_examples, is_label_augmented=False):
	images, labels = generate_dataset(num_examples, is_label_augmented)

	swl_util.make_dir(dir_path)

	idx, start_idx = 0, 0
	while True:
		end_idx = start_idx + np.random.randint(30, 50)
		batch_images = images[start_idx:end_idx]
		batch_labels = labels[start_idx:end_idx]
		np.save(os.path.join(dir_path, 'images_{}.npy'.format(idx)), batch_images)
		np.save(os.path.join(dir_path, 'labels_{}.npy'.format(idx)), batch_labels)
		if end_idx >= num_examples:
			break;
		start_idx = end_idx
		idx += 1
	return idx + 1  # The number of files.

def augment_identically(images, labels):
	# Augments here.
	return images, labels

def simple_batch_generator_example():
	num_examples = 100
	images, labels = generate_dataset(num_examples)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	batchMgr = SimpleBatchGenerator(images, labels, batch_size, shuffle, is_time_major)
	#batchMgr = SimpleBatchGenerator(images, labels, batch_size, shuffle, is_time_major, functor=augment_identically)
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		batches = batchMgr.generateBatches()  # Generates batches.
		for idx, batch in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (images & labels).
			# Train with each batch (images & labels).
			#print('{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def simple_npy_file_batch_generator_example():
	num_examples = 100
	images, labels = generate_dataset(num_examples)

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

		#batchGenerator = NpyFileBatchGenerator(images, labels, batch_size, shuffle, is_time_major)
		batchGenerator = NpyFileBatchGenerator(images, labels, batch_size, shuffle, is_time_major, functor=augment_identically, batch_info_csv_filename=batch_info_csv_filename)
		batchGenerator.saveBatches(dir_path)  # Generates and saves batches.

		batchLoader = NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)
		batches = batchLoader.loadBatches(dir_path)  # Loads batches.
		for idx, batch in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (images & labels).
			# Train with each batch (images & labels).
			#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirMgr.returnDirectory(dir_path)				

import logging
from logging.handlers import RotatingFileHandler

def init(lock):
	global global_lock
	global_lock = lock

def data_augmentation_worker(dirMgr, batchGenerator, epoch):
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
	#batchGenerator = NpyFileBatchGenerator(images, labels, batch_size, shuffle, is_time_major)
	#batchGenerator = NpyFileBatchGenerator(images, labels, batch_size, shuffle, is_time_major, functor=augment_identically)
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

def training_worker(dirMgr, batchLoader, epoch):
	print('\t{}: Start training worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request a ready directory.'.format(os.getpid()))
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
	batches = batchLoader.loadBatches(dir_path)  # Loads batches.
	for idx, batch in enumerate(batches):
		# Train with each batch (images & labels).
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

def data_augmentation_worker0(ii, ff, ss, epoch):
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.info('\t{}: Start data augmentation worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Start data augmentation worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request an available directory.'.format(os.getpid()))
	while True:
		with global_lock:
			#dir_path = dirMgr.requestAvailableDirectory()
			dir_path = 'batch_dir_{}'.format(epoch)

		if dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\t{}: Got an available directory: {}.'.format(os.getpid(), dir_path))

	#--------------------
	# Do something.
	print('\t{}: Prepare something...'.format(os.getpid()))
	print('\t{}: {}, {}, {}'.format(os.getpid(), ii, ff, ss))
	time.sleep(np.random.randint(3, 6))

	#--------------------
	with global_lock:
		#dirMgr.returnDirectoryAsReady(dir_path)
		pass
	print('\t{}: Returned a directory as ready: {}.'.format(os.getpid(), dir_path))
	print('\t{}: End data augmentation worker process.'.format(os.getpid()))

def training_worker0(ii, ff, ss, epoch):
	#logger = logging.getLogger('python_mp_logging_test')
	#logger.exception('\t{}: Start training worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Start training worker process: epoch #{}.'.format(os.getpid(), epoch))
	print('\t{}: Request a ready directory.'.format(os.getpid()))
	while True:
		with global_lock:
			#dir_path = dirMgr.requestReadyDirectory()
			dir_path = 'batch_dir_{}'.format(epoch)

		if dir_path is not None:
			break
		else:
			time.sleep(0.1)
	print('\t{}: Got a ready directory: {}.'.format(os.getpid(), dir_path))

	#--------------------
	# Do something.
	print('\t{}: Train something...'.format(os.getpid()))
	print('\t{}: {}, {}, {}'.format(os.getpid(), ii, ff, ss))
	time.sleep(np.random.randint(3, 6))

	#--------------------
	with global_lock:
		#dirMgr.returnDirectoryAsAvailable(dir_path)
		pass
	print('\t{}: Returned a directory as available: {}.'.format(os.getpid(), dir_path))
	print('\t{}: End training worker process.'.format(os.getpid()))

def training_worker1(lock, ii, ff, ss, dirMgr, num_epochs):
	for epoch in range(num_epochs):
		#logger = logging.getLogger('python_mp_logging_test')
		#logger.exception('\t{}: Start training worker process: epoch #{}.'.format(os.getpid(), epoch))
		print('\t{}: Start training worker process: epoch #{}.'.format(os.getpid(), epoch))
		print('\t{}: Request a ready directory.'.format(os.getpid()))
		while True:
			with lock:
				#dir_path = dirMgr.requestReadyDirectory()
				dir_path = 'batch_dir_{}'.format(epoch)

			if dir_path is not None:
				break
			else:
				time.sleep(0.1)
		print('\t{}: Got a ready directory: {}.'.format(os.getpid(), dir_path))

		#--------------------
		# Do something.
		print('\t{}: Train something...'.format(os.getpid()))
		print('\t{}: {}, {}, {}'.format(os.getpid(), ii, ff, ss))
		time.sleep(np.random.randint(3, 6))

		#--------------------
		with lock:
			#dirMgr.returnDirectoryAsAvailable(dir_path)
			pass
		print('\t{}: Returned a directory as available: {}.'.format(os.getpid(), dir_path))
		print('\t{}: End training worker process.'.format(os.getpid()))

def run_data_preparation(lock, dirMgr, num_processes, num_epochs, images, labels, batch_size, shuffle, is_time_major, batch_info_csv_filename):
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')
	with mp.Pool(processes=num_processes, initializer=init, initargs=(lock,)) as pool:
		#pool.map(partial(data_augmentation_worker, dirMgr=dirMgr, batchGenerator=NpyFileBatchGenerator(images, labels, batch_size, shuffle, is_time_major, functor=augment_identically, batch_info_csv_filename=batch_info_csv_filename)), [epoch for epoch in range(num_epochs)])
		pool.map(partial(data_augmentation_worker0, 1, 2.0, 'abc'), [epoch for epoch in range(num_epochs)])
		# Async.
		#multiple_results = [pool.apply_async(data_augmentation_worker, args=(dirMgr, NpyFileBatchGenerator(images, labels, batch_size, shuffle, is_time_major, functor=augment_identically, batch_info_csv_filename=batch_info_csv_filename), epoch)) for epoch in range(num_epochs)]
		#multiple_results = [pool.apply_async(data_augmentation_worker0, args=(1, 2.0, 'abc', epoch)) for epoch in range(num_epochs)]
		#[res.get() for res in multiple_results]
		#for epoch in range(num_epochs):
		#	pool.apply_async(data_augmentation_worker0, args=(1, 2.0, 'abc', epoch))

def run_training(lock, dirMgr, num_processes, num_epochs, batch_info_csv_filename):
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')
	with mp.Pool(processes=num_processes, initializer=init, initargs=(lock,)) as pool:
		#pool.map(partial(training_worker, dirMgr=dirMgr, batchLoader=NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename)), [epoch for epoch in range(num_epochs)])
		pool.map(partial(training_worker0, 3, 4.0, 'def'), [epoch for epoch in range(num_epochs)])
		# Async.
		#multiple_results = [pool.apply_async(training_worker, args=(dirMgr, NpyFileBatchLoader(batch_info_csv_filename=batch_info_csv_filename), epoch)) for epoch in range(num_epochs)]
		#[res.get() for res in multiple_results]

def run_worker_processes_1(lock, dirMgr, num_processes, num_epochs, images, labels, batch_size, shuffle, is_time_major, batch_info_csv_filename):
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	train_worker_process = mp.Process(target=training_worker1, args=(lock, 3, 4.0, 'def', dirMgr, num_epochs))
	train_worker_process.start()

	with mp.Pool(processes=num_processes, initializer=init, initargs=(lock,)) as pool:
		data_augmentation_results = pool.map_async(partial(data_augmentation_worker0, 1, 2.0, 'abc'), [epoch for epoch in range(num_epochs)])

		#data_augmentation_results.get(timeout=20)
		data_augmentation_results.get(timeout=None)

	train_worker_process.join()

def run_worker_processes_2(lock, dirMgr, num_processes, num_epochs, images, labels, batch_size, shuffle, is_time_major, batch_info_csv_filename):
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')

	with mp.Pool(processes=num_processes, initializer=init, initargs=(lock,)) as pool:
		data_augmentation_results = pool.map_async(partial(data_augmentation_worker0, 1, 2.0, 'abc'), [epoch for epoch in range(num_epochs)])
		training_results = pool.map_async(partial(training_worker0, 3, 4.0, 'def'), [epoch for epoch in range(num_epochs)])

		#data_augmentation_results.get(timeout=20)
		#training_results.get(timeout=20)
		data_augmentation_results.get(timeout=None)
		training_results.get(timeout=None)

def multiprocessing_npy_file_batch_generator_example():
	num_examples = 100
	images, labels = generate_dataset(num_examples)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

	batch_info_csv_filename = 'batch_info.csv'

	num_processes = 5
	lock = mp.Lock()  # RuntimeError: Lock objects should only be shared between processes through inheritance.
	#lock= mp.Manager().Lock()  # TypeError: can't pickle _thread.lock objects.

	"""
	#--------------------
	run_data_preparation(lock, dirMgr, num_processes, num_epochs, images, labels, batch_size, shuffle, is_time_major, batch_info_csv_filename)

	print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

	#--------------------
	run_training(lock, dirMgr, num_processes, num_epochs, batch_info_csv_filename)
	"""

	run_worker_processes_1(lock, dirMgr, num_processes, num_epochs, images, labels, batch_size, shuffle, is_time_major, batch_info_csv_filename)
	#run_worker_processes_2(lock, dirMgr, num_processes, num_epochs, images, labels, batch_size, shuffle, is_time_major, batch_info_csv_filename)

def main():
	handler = RotatingFileHandler('./python_logging.log', maxBytes=5000, backupCount=10)
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	handler.setFormatter(formatter)

	logger = logging.getLogger('python_mp_logging_test')
	logger.addHandler(handler) 
	logger.setLevel(logging.INFO)

	#simple_batch_generator_example()

	#simple_npy_file_batch_generator_example()
	multiprocessing_npy_file_batch_generator_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
