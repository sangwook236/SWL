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
from swl.machine_learning.imgaug_batch_loader import ImgaugBatchLoader

def main():
	augseq = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.CoarseDropout(p=0.1, size_percent=0.1)
	])

	images = np.random.rand(100, 64, 64, 1)
	labels = np.random.randint(10, size=(100, 10))
	batch_size = 10
	num_epoches = 2
	shuffle = True
	is_time_major = False

	batchLoader = ImgaugBatchLoader(augseq, images, labels, batch_size, shuffle, is_time_major)
	batches = batchLoader.getBatch(num_epoches)
	for idx, batch in enumerate(batches):
		print(idx, batch[0].shape, batch[1].shape)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
