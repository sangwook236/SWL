#!/usr/bin/env python

import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append('../../src')

#--------------------
import numpy as np
from imgaug import augmenters as iaa
from swl.machine_learning.imgaug_batch_manager import ImgaugBatchManager, ImgaugBatchManagerWithFileInput, ImgaugFileBatchManager, ImgaugFileBatchManagerWithFileInput
from swl.util.directory_queue_manager import DirectoryQueueManager

def imgaug_batch_manager_example():
	num_examples = 100
	images = np.random.rand(num_examples, 64, 64, 1)
	labels = np.random.randint(2, size=(num_examples, 5))
	batch_size = 15
	num_epoches = 2
	shuffle = True
	is_time_major = False

	augmenter = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.CoarseDropout(p=0.1, size_percent=0.1)
	])

	batchMgr = ImgaugBatchManager(augmenter, images, labels, batch_size, shuffle, is_time_major)
	for epoch in range(num_epoches):
		batches = batchMgr.getBatches()
		for idx, batch in enumerate(batches):
			print(idx, batch[0].shape, batch[1].shape)

			# Train with batch (images & labels).
			#print('Trained.')

def imgaug_file_batch_manager_example():
	num_examples = 100
	batch_size = 15
	num_epoches = 2
	shuffle = True
	is_label_augmented = False
	is_time_major = False

	if is_label_augmented:
		images = np.random.rand(num_examples, 64, 64, 1)
		labels = np.random.rand(num_examples, 64, 64, 1)
	else:
		images = np.random.rand(num_examples, 64, 64, 1)
		labels = np.random.randint(2, size=(num_examples, 5))

	augmenter = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.CoarseDropout(p=0.1, size_percent=0.1)
	])

	base_dir_path = './batch_dir'
	num_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_dir_path, num_dirs)

	#--------------------
	while True:
		# Run in each thread or process.
		dir_path = dirQueueMgr.getAvailableDirectory()
		if dir_path is None:
			break

		batchMgr = ImgaugFileBatchManager(augmenter, images, labels, dir_path, batch_size, shuffle, is_label_augmented, is_time_major)
		batchMgr.putBatches()

		for epoch in range(num_epoches):
			for idx, batch in enumerate(batchMgr.getBatches()):
				print(idx, batch[0].shape, batch[1].shape)

				# Train with batch (images & labels).
				#print('Trained.')

		# TODO [uncomment] >> Commented for test.
		#dirQueueMgr.returnDirectory(dir_path)				

def imgaug_file_batch_manager_with_file_input_example():
	num_examples = 100
	batch_size = 15
	num_epoches = 2
	shuffle = True
	is_label_augmented = False
	is_time_major = False

	npy_filepath_pairs = [
		('./batches/images_0.npy', './batches/labels_0.npy'),
		('./batches/images_1.npy', './batches/labels_1.npy'),
		('./batches/images_2.npy', './batches/labels_2.npy'),
	]

	augmenter = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.CoarseDropout(p=0.1, size_percent=0.1)
	])

	base_dir_path = './batch_dir'
	num_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_dir_path, num_dirs)

	#--------------------
	while True:
		# Run in each thread or process.
		dir_path = dirQueueMgr.getAvailableDirectory()
		if dir_path is None:
			break

		batchMgr = ImgaugFileBatchManagerWithFileInput(augmenter, npy_filepath_pairs, dir_path, batch_size, shuffle, is_label_augmented, is_time_major)
		batchMgr.putBatches()

		for epoch in range(num_epoches):
			for idx, batch in enumerate(batchMgr.getBatches()):
				print(idx, batch[0].shape, batch[1].shape)

				# Train with batch (images & labels).
				#print('Trained.')

		# TODO [uncomment] >> Commented for test.
		#dirQueueMgr.returnDirectory(dir_path)				

def main():
	#imgaug_batch_manager_example()
	#imgaug_file_batch_manager_example()
	imgaug_file_batch_manager_with_file_input_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
