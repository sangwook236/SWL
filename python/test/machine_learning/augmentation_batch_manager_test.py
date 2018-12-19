#!/usr/bin/env python

import sys
sys.path.append('../../src')

#--------------------
import os, time, random
import multiprocessing as mp
import numpy as np
from swl.machine_learning.augmentation_batch_manager import AugmentationBatchManager, AugmentationBatchManagerWithFileInput, AugmentationFileBatchManager, AugmentationFileBatchManagerWithFileInput
from swl.util.directory_queue_manager import DirectoryQueueManager
import swl.util.util as swl_util

class IdentityAugmenter(object):
	def augment(self, images, labels, is_label_augmented=False):
		# Augments here.
		return images, labels

class AsyncMultiprocessAugmenter(object):
	def __init__(self, pool):
		self._pool = pool

	def augment(self, images, labels, is_label_augmented=False):
		self._pool.apply_async(AsyncMultiprocessAugmenter._augment, args=(images, labels, is_label_augmented), callback=AsyncMultiprocessAugmenter._async_callback)
		return None, None

	@staticmethod
	def _augment(images, labels, is_label_augmented=False):
		print('============== Sleep: ', os.getpid())
		time.sleep(random.randint(1, 3))
		print('============== Wake up and augment: ', os.getpid())
		# Augments here.
		return images, labels

	@staticmethod
	def _async_callback(result):
		# This is called whenever _augment() returns a result.
		# Train with each batch (images & labels).
		#print('{}, {}'.format(result[0].shape, result[1].shape))
		print('{}-{}, {}-{}'.format(result[0].shape, np.max(np.reshape(result[0], (result[0].shape[0], -1)), axis=-1), result[1].shape, np.max(np.reshape(result[1], (result[1].shape[0], -1)), axis=-1)))

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

def augmentation_batch_manager_example():
	num_examples = 100
	is_label_augmented = False
	images, labels = generate_dataset(num_examples, is_label_augmented)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	augmenter = IdentityAugmenter()

	batchMgr = AugmentationBatchManager(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major)
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		batches = batchMgr.getBatches()  # Generates and augments batches.
		for idx, batch in enumerate(batches):
			# Train with each batch (images & labels).
			#print('{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def augmentation_batch_manager_with_file_input_example():
	num_examples = 300
	is_label_augmented = False
	num_files = generate_file_dataset('./batches', num_examples, is_label_augmented)
	npy_filepath_pairs = list()
	for idx in range(num_files):
		npy_filepath_pairs.append(('./batches/images_{}.npy'.format(idx), './batches/labels_{}.npy'.format(idx)))
	npy_filepath_pairs = np.array(npy_filepath_pairs)
	num_file_pairs = 3
	num_file_pair_steps = ((num_files - 1) // num_file_pairs + 1) if num_files > 0 else 0

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	augmenter = IdentityAugmenter()

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))
		
		indices = np.arange(num_files)
		if shuffle:
			np.random.shuffle(indices)

		for step in range(num_file_pair_steps):
			print('\t>>>>> File pairs #{}.'.format(step))
			
			start = step * num_file_pairs
			end = start + num_file_pairs
			file_pair_indices = indices[start:end]
			if file_pair_indices.size > 0:  # If file_pair_indices is non-empty.
				sub_filepath_pairs = npy_filepath_pairs[file_pair_indices]
				if sub_filepath_pairs.size > 0:  # If sub_filepath_pairs is non-empty.
					# Can run in an individual thread or process.
					batchMgr = AugmentationBatchManagerWithFileInput(augmenter, sub_filepath_pairs, batch_size, shuffle, is_label_augmented, is_time_major)

					batches = batchMgr.getBatches()  # Generates and augments batches.
					for idx, batch in enumerate(batches):
						# Train with each batch (images & labels).
						#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
						print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def augmentation_file_batch_manager_example():
	num_examples = 100
	is_label_augmented = False
	images, labels = generate_dataset(num_examples, is_label_augmented)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	augmenter = IdentityAugmenter()

	base_batch_dir_path = './batch_dir'
	num_batch_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_batch_dir_path, num_batch_dirs)

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		dir_path = dirQueueMgr.getAvailableDirectory()
		if dir_path is None:
			break

		print('\t>>>>> Directory: {}.'.format(dir_path))

		batchMgr = AugmentationFileBatchManager(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major)
		batchMgr.putBatches(dir_path)  # Generates, augments, and saves batches.

		batches = batchMgr.getBatches(dir_path)  # Loads batches.
		for idx, batch in enumerate(batches):
			# Train with each batch (images & labels).
			#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirQueueMgr.returnDirectory(dir_path)				

def augmentation_file_batch_manager_with_file_input_example():
	num_examples = 300
	is_label_augmented = False
	num_files = generate_file_dataset('./batches', num_examples, is_label_augmented)
	npy_filepath_pairs = list()
	for idx in range(num_files):
		npy_filepath_pairs.append(('./batches/images_{}.npy'.format(idx), './batches/labels_{}.npy'.format(idx)))
	npy_filepath_pairs = np.array(npy_filepath_pairs)
	num_file_pairs = 3
	num_file_pair_steps = ((num_files - 1) // num_file_pairs + 1) if num_files > 0 else 0

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	augmenter = IdentityAugmenter()

	base_batch_dir_path = './batch_dir'
	num_batch_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_batch_dir_path, num_batch_dirs)

	#--------------------
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		dir_path = dirQueueMgr.getAvailableDirectory()
		if dir_path is None:
			break

		print('\t>>>>> Directory: {}.'.format(dir_path))
		
		indices = np.arange(num_files)
		if shuffle:
			np.random.shuffle(indices)

		for step in range(num_file_pair_steps):
			print('\t\t>>>>> File pairs #{}.'.format(step))
			
			start = step * num_file_pairs
			end = start + num_file_pairs
			file_pair_indices = indices[start:end]
			if file_pair_indices.size > 0:  # If file_pair_indices is non-empty.
				sub_filepath_pairs = npy_filepath_pairs[file_pair_indices]
				if sub_filepath_pairs.size > 0:  # If sub_filepath_pairs is non-empty.
					# Can run in an individual thread or process.
					batchMgr = AugmentationFileBatchManagerWithFileInput(augmenter, sub_filepath_pairs, batch_size, shuffle, is_label_augmented, is_time_major)
					batchMgr.putBatches(dir_path)  # Generates, augments, and saves batches.

					batches = batchMgr.getBatches(dir_path)  # Loads batches.
					for idx, batch in enumerate(batches):
						# Train with each batch (images & labels).
						#print('\t\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
						print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirQueueMgr.returnDirectory(dir_path)

def sync_multiprocess_augmentation_batch_manager_example():
	num_examples = 100
	is_label_augmented = False
	images, labels = generate_dataset(num_examples, is_label_augmented)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	augmenter = IdentityAugmenter()
	with mp.Pool() as pool:
		batchMgr = AugmentationBatchManager(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major, pool)
		for epoch in range(num_epochs):
			print('>>>>> Epoch #{}.'.format(epoch))

			batches = batchMgr.getBatches()  # Generates and augments batches.
			for idx, batch in enumerate(batches):
				# Train with each batch (images & labels).
				#print('{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
				print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def sync_multiprocess_augmentation_file_batch_manager_example():
	num_examples = 100
	is_label_augmented = False
	images, labels = generate_dataset(num_examples, is_label_augmented)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	augmenter = IdentityAugmenter()

	base_batch_dir_path = './batch_dir'
	num_batch_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_batch_dir_path, num_batch_dirs)

	#--------------------
	with mp.Pool() as pool:
		for epoch in range(num_epochs):
			print('>>>>> Epoch #{}.'.format(epoch))

			dir_path = dirQueueMgr.getAvailableDirectory()
			if dir_path is None:
				break

			print('\t>>>>> Directory: {}.'.format(dir_path))

			batchMgr = AugmentationFileBatchManager(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major, pool)
			batchMgr.putBatches(dir_path)  # Generates, augments, and saves batches.

			batches = batchMgr.getBatches(dir_path)  # Loads batches.
			for idx, batch in enumerate(batches):
				# Train with each batch (images & labels).
				#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
				print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

			dirQueueMgr.returnDirectory(dir_path)				

def async_multiprocess_augmentation_batch_manager_example():
	num_examples = 100
	is_label_augmented = True
	images, labels = generate_dataset(num_examples, is_label_augmented)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	with mp.Pool() as pool:
		augmenter = AsyncMultiprocessAugmenter(pool)

		batchMgr = AugmentationBatchManager(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major, None)
		for epoch in range(num_epochs):
			print('>>>>> Epoch #{}.'.format(epoch))

			batches = batchMgr.getBatches()  # Generates and augments batches.
			for _ in batches:
				pass

		pool.close()
		pool.join()

def main():
	augmentation_batch_manager_example()
	#augmentation_batch_manager_with_file_input_example()

	#augmentation_file_batch_manager_example()
	#augmentation_file_batch_manager_with_file_input_example()

	# Multiprocess.
	# set_start_method() should not be used more than once in the program.
	#mp.set_start_method('spawn')
	#sync_multiprocess_augmentation_batch_manager_example()
	#sync_multiprocess_augmentation_file_batch_manager_example()

	# NOTE [warning] >> May not maintain batches of images and labels in each epoch.
	#async_multiprocess_augmentation_batch_manager_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
