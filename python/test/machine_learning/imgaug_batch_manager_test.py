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
from swl.machine_learning.imgaug_batch_manager import ImgaugBatchManager, ImgaugFileBatchManager
from swl.util.directory_queue_manager import DirectoryQueueManager

def imgaug_batch_manager_example():
	num_examples = 100
	images = np.random.rand(num_examples, 64, 64, 1)
	labels = np.random.randint(2, size=(num_examples, 5))
	batch_size = 10
	num_epoches = 2
	shuffle = True
	is_time_major = False

	augseq = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.CoarseDropout(p=0.1, size_percent=0.1)
	])

	batchMgr = ImgaugBatchManager(images, labels, augseq, batch_size, shuffle, is_time_major)
	batches = batchMgr.getBatches(num_epoches)
	for idx, batch in enumerate(batches):
		print(idx, batch[0].shape, batch[1].shape)

def imgaug_file_batch_manager_example():
	num_examples = 100
	is_label_image = True

	batch_size = 15
	num_epoches = 2
	shuffle = True
	is_time_major = False

	num_files = ((num_examples - 1) // batch_size + 1) if num_examples > 0 else 0

	if is_label_image:
		images = np.random.rand(num_examples, 64, 64, 1)
		labels = np.random.rand(num_examples, 64, 64, 1)
	else:
		images = np.random.rand(num_examples, 64, 64, 1)
		labels = np.random.randint(2, size=(num_examples, 5))

	filename_pairs = list()
	for idx in range(num_files):
		img_filename = 'batch_images_{}.npy'.format(idx)
		lbl_filename = 'batch_labels_{}.npy'.format(idx)
		filename_pairs.append((img_filename, lbl_filename))

	augseq = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.CoarseDropout(p=0.1, size_percent=0.1)
	])

	batchMgr = ImgaugFileBatchManager(images, labels, augseq, is_label_image, is_time_major)

	base_dir_path = './batch_dir'
	num_dirs = 5
	dirQueueMgr = DirectoryQueueManager(base_dir_path, num_dirs)

	#--------------------
	while True:
		# Run in each thread or process.
		dir_path = dirQueueMgr.getAvailableDirectory()
		if dir_path is not None:
			batchMgr.putBatches(dir_path, filename_pairs, shuffle)

			for idx, batch in enumerate(batchMgr.getBatches(dir_path, filename_pairs, num_epoches)):
				print(idx, batch[0].shape, batch[1].shape)
				# Train with batch (images & labels).

			# TODO [uncomment] >> Commented for test.
			#dirQueueMgr.returnDirectory(dir_path)				
		else:
			break

def main():
	#imgaug_batch_manager_example()
	imgaug_file_batch_manager_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
