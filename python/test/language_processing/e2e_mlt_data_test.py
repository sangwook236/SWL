#!/usr/bin/env python

# REF [site] >> https://github.com/MichalBusta/E2E-MLT

import os, math, time, glob, csv, pickle
import numpy as np
import cv2

def e2e_mlt_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	e2e_mlt_dir_path = data_base_dir_path + '/text/e2e_mlt/Korean'
	#e2e_mlt_dir_path = data_base_dir_path + '/text/e2e_mlt/Latin'

	print('Start loading file list...')
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
	print('End loading file list: {} secs.'.format(time.time() - start_time))

	if len(img_filepaths) != len(gt_filepaths):
		print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
		return

	#--------------------
	print('Start loading data...')
	start_time = time.time()
	gt_boxes, gt_texts = list(), list()
	for img_filepath, gt_filepath in zip(img_filepaths, gt_filepaths):
		img_filepath = os.path.join(e2e_mlt_dir_path, img_filepath)
		img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(img_filepath))
			continue

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
	print('End loading data: {} secs.'.format(time.time() - start_time))

	#--------------------
	if False:
		for img_fpath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
			img_fpath = os.path.join(e2e_mlt_dir_path, img_fpath)
			img = cv2.imread(img_fpath, cv2.IMREAD_UNCHANGED)
			if img is None:
				print('Failed to load an image, {}.'.format(img_fpath))
				continue

			print('GT texts =', texts)
			rgb = img.copy()
			for box in boxes:
				#box = box.reshape((-1, 2))
				box = box.astype(np.int)
				if False:
					cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)
				else:
					cv2.line(rgb, tuple(box[0,:]), tuple(box[1,:]), (0, 0, 255), 2, cv2.LINE_8)
					cv2.line(rgb, tuple(box[1,:]), tuple(box[2,:]), (0, 255, 0), 2, cv2.LINE_8)
					cv2.line(rgb, tuple(box[2,:]), tuple(box[3,:]), (255, 0, 0), 2, cv2.LINE_8)
					cv2.line(rgb, tuple(box[3,:]), tuple(box[0,:]), (255, 0, 255), 2, cv2.LINE_8)

			cv2.imshow('E2E-MLT', rgb)
			cv2.waitKey(0)

		cv2.destroyAllWindows()

	#--------------------
	if True:
		pkl_filepath = './e2e_mlt.pkl'

		print('Start preparing data...')
		start_time = time.time()
		image_box_text_triples = []
		for img_fpath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
			img_fpath = os.path.join(e2e_mlt_dir_path, img_fpath)
			img = cv2.imread(img_fpath, cv2.IMREAD_UNCHANGED)
			if img is None:
				print('Failed to load an image, {}.'.format(img_fpath))
				continue
			image_box_text_triples.append([img, boxes, texts])
		print('End preparing data: {} secs.'.format(time.time() - start_time))

		print('Start saving data to {}...'.format(pkl_filepath))
		start_time = time.time()
		try:
			with open(pkl_filepath, 'wb') as fd:
				pickle.dump(image_box_text_triples, fd)
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(pkl_filepath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(pkl_filepath))
		print('End saving data: {} secs.'.format(time.time() - start_time))
		del image_box_text_triples

		print('Start loading data from {}...'.format(pkl_filepath))
		start_time = time.time()
		try:
			with open(pkl_filepath, 'rb') as fd:
				loaded_image_box_text_triples = pickle.load(fd)
				print('#loaded pairs of image, boxes, and texts =', len(loaded_image_box_text_triples))
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(pkl_filepath))
			loaded_image_box_text_triples = None
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(pkl_filepath))
			loaded_image_box_text_triples = None
		print('End loading data: {} secs.'.format(time.time() - start_time))

		for idx, (img, boxes, texts) in enumerate(loaded_image_box_text_triples):
			print('Texts =', texts)
			for box in boxes:
				#box = box.reshape((-1, 2))
				box = box.astype(np.int)
				if False:
					cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
				else:
					cv2.line(img, tuple(box[0,:]), tuple(box[1,:]), (0, 0, 255), 2, cv2.LINE_8)
					cv2.line(img, tuple(box[1,:]), tuple(box[2,:]), (0, 255, 0), 2, cv2.LINE_8)
					cv2.line(img, tuple(box[2,:]), tuple(box[3,:]), (255, 0, 0), 2, cv2.LINE_8)
					cv2.line(img, tuple(box[3,:]), tuple(box[0,:]), (255, 0, 255), 2, cv2.LINE_8)

			cv2.imshow('Image', img)
			cv2.waitKey(0)

			if idx >= 9:
				break

		cv2.destroyAllWindows()

def main():
	e2e_mlt_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
