#!/usr/bin/env python

# REF [site] >> https://github.com/MichalBusta/E2E-MLT

import os, math, time, glob, csv
import numpy as np
import cv2

def bounding_box_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	e2e_mlt_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/e2e_mlt/0_download/Korean'

	print('Loading file list...')
	start_time = time.time()
	img_filepaths = glob.glob(os.path.join(e2e_mlt_dir_path, '*.jpg'), recursive=False)
	gt_filepaths = glob.glob(os.path.join(e2e_mlt_dir_path, '*.txt'), recursive=False)

	class FilenameExtracter:
		def __init__(self, base_dir_path):
			self.base_dir_path = base_dir_path

		def __call__(self, filepath):
			idx = filepath.rfind(self.base_dir_path) + len(self.base_dir_path) + 1
			return filepath[idx:]

	img_filepaths = list(map(FilenameExtracter(e2e_mlt_dir_path), img_filepaths))
	gt_filepaths = list(map(FilenameExtracter(e2e_mlt_dir_path), gt_filepaths))

	img_filepaths.sort(key=lambda filepath: os.path.basename(filepath))
	gt_filepaths.sort(key=lambda filepath: os.path.basename(filepath))
	print('\tElapsed time = {}'.format(time.time() - start_time))

	if len(img_filepaths) != len(gt_filepaths):
		print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
		return

	#--------------------
	print('Loading data...')
	start_time = time.time()
	gt_boxes, gt_texts = list(), list()
	for img_filepath, gt_filepath in zip(img_filepaths, gt_filepaths):
		img = cv2.imread(os.path.join(e2e_mlt_dir_path, img_filepath))
		height, width = img.shape[:2]
		max_len = max(height, width)

		# ?, center, size, angle, text.
		boxes, texts = list(), list()
		with open(os.path.join(e2e_mlt_dir_path, gt_filepath), newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=' ', quotechar=None)
			for row in reader:
				if 7 != len(row):
					print('Different row length in {}: {}.'.format(gt_filepath, row))
				#boxes.append(row[:6])
				boxes.append(list(float(rr) for rr in row[:6]))
				texts.append(row[6])
				# TODO [check] >> Spaces which follow comma can be removed.
				#texts.append(' '.join(row[6:]) if len(row[6:]) > 1 else row[6])

		boxes = np.array(boxes, np.float).reshape(-1, 6)
		#boxes[:,0] = ?
		boxes[:,1] *= width
		boxes[:,2] *= height
		# FIXME [check] >> Correct? Still too short.
		#boxes[:,3] *= width
		#boxes[:,4] *= height
		boxes[:,3] *= max_len
		boxes[:,4] *= max_len

		#gt_boxes.append(list(cv2.boxPoints((box[1:3], box[3:5], box[5] * 180 / math.pi)) for box in boxes))
		gt_boxes.append(np.array(list(cv2.boxPoints((box[1:3], box[3:5], box[5] * 180 / math.pi)) for box in boxes), np.float).reshape(-1, 4, 2))
		gt_texts.append(texts)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	for img_filepath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
		img = cv2.imread(os.path.join(e2e_mlt_dir_path, img_filepath))

		print('GT texts =', texts)
		rgb = img.copy()
		for box in boxes:
			#box = box.reshape((-1, 2))
			box = box.astype(np.int)
			cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)

		cv2.imshow('Bounding Box', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	bounding_box_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
