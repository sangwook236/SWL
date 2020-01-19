#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

#--------------------
import os, time
import numpy as np
from swl.machine_learning.batch_manager import SimpleBatchManager, SimpleBatchManagerWithFileInput, SimpleFileBatchManager, SimpleFileBatchManagerWithFileInput
from swl.machine_learning.batch_generator import SimpleBatchGenerator, NpzFileBatchGenerator, NpzFileBatchLoader
from swl.util.working_directory_manager import WorkingDirectoryManager
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

def simple_batch_manager_example():
	num_examples = 100
	images, labels = generate_dataset(num_examples)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	batchMgr = SimpleBatchManager(images, labels, batch_size, shuffle, is_time_major)
	for epoch in range(num_epochs):
		print('>>>>> Epoch #{}.'.format(epoch))

		batches = batchMgr.getBatches()  # Generates batches.
		for idx, batch in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (images & labels).
			# Train with each batch (images & labels).
			#print('{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def simple_batch_manager_with_file_input_example():
	num_examples = 300
	num_files = generate_file_dataset('./batches', num_examples)
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
					batchMgr = SimpleBatchManagerWithFileInput(sub_filepath_pairs, batch_size, shuffle, is_time_major)

					batches = batchMgr.getBatches()  # Generates batches.
					for idx, batch in enumerate(batches):
						# Augment each batch (images & labels).
						# Train with each batch (images & labels).
						#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
						print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

def simple_file_batch_manager_example():
	num_examples = 100
	images, labels = generate_dataset(num_examples)

	batch_size = 12
	num_epochs = 7
	shuffle = True
	is_time_major = False

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

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

		batchMgr = SimpleFileBatchManager(images, labels, batch_size, shuffle, is_time_major)
		batchMgr.putBatches(dir_path)  # Generates and saves batches.

		batches = batchMgr.getBatches(dir_path)  # Loads batches.
		for idx, batch in enumerate(batches):
			# Can run in an individual thread or process.
			# Augment each batch (images & labels).
			# Train with each batch (images & labels).
			#print('\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
			print('\t{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirMgr.returnDirectory(dir_path)				

def simple_file_batch_manager_with_file_input_example():
	num_examples = 300
	num_files = generate_file_dataset('./batches', num_examples)
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

	batch_dir_path_prefix = './batch_dir'
	num_batch_dirs = 5
	dirMgr = WorkingDirectoryManager(batch_dir_path_prefix, num_batch_dirs)

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
					batchMgr = SimpleFileBatchManagerWithFileInput(sub_filepath_pairs, batch_size, shuffle, is_time_major)
					batchMgr.putBatches(dir_path)  # Generates and saves batches.

					batches = batchMgr.getBatches(dir_path)  # Loads batches.
					for idx, batch in enumerate(batches):
						# Augment each batch (images & labels).
						# Train with each batch (images & labels).
						#print('\t\t{}: {}, {}'.format(idx, batch[0].shape, batch[1].shape))
						print('{}: {}-{}, {}-{}'.format(idx, batch[0].shape, np.max(np.reshape(batch[0], (batch[0].shape[0], -1)), axis=-1), batch[1].shape, np.max(np.reshape(batch[1], (batch[1].shape[0], -1)), axis=-1)))

		dirMgr.returnDirectory(dir_path)

def main():
	# REF [info] >> Use batch generators and loaders.

	simple_batch_manager_example()
	#simple_batch_manager_with_file_input_example()

	#simple_file_batch_manager_example()
	#simple_file_batch_manager_with_file_input_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
