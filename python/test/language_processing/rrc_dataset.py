#!/usr/bin/env python

import os, math, time, glob, csv
import numpy as np
import cv2

# REF [site] >> https://rrc.cvc.uab.es/?ch=8
def rrc_mlt_2017_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	rrc_mlt_2017_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/rrc_icdar/0_download/mlt_2017'

	print('Loading file list...')
	start_time = time.time()
	img_filepaths = glob.glob(os.path.join(rrc_mlt_2017_dir_path, 'ch8_training_images_?/img_*.*'), recursive=False)
	gt_filepaths = glob.glob(os.path.join(rrc_mlt_2017_dir_path, 'ch8_training_localization_transcription_gt_v2/gt_img_*.txt'), recursive=False)

	class FilenameExtracter:
		def __init__(self, base_dir_path):
			self.base_dir_path = base_dir_path

		def __call__(self, filepath):
			idx = filepath.rfind(self.base_dir_path) + len(self.base_dir_path) + 1
			return filepath[idx:]

	img_filepaths = list(map(FilenameExtracter(rrc_mlt_2017_dir_path), img_filepaths))
	gt_filepaths = list(map(FilenameExtracter(rrc_mlt_2017_dir_path), gt_filepaths))

	if True:
		class FileFilter:
			def __init__(self, start_idx, end_idx):
				self.start_idx = start_idx
				self.end_idx = end_idx

			def __call__(self, filepath):
				si, ei = filepath.rfind('img_'), filepath.rfind('.')
				file_id = int(filepath[si+4:ei])
				return self.start_idx <= file_id <= self.end_idx

		# Korean: 4001 ~ 4800. (?)
		img_filepaths = list(filter(FileFilter(4001, 4800), img_filepaths))
		gt_filepaths = list(filter(FileFilter(4001, 4800), gt_filepaths))

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
		# REF [site] >> https://rrc.cvc.uab.es/?ch=8&com=tasks
		#	x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
		#	Valid scripts are: "Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols", "Mixed", "None".
		boxes, texts = list(), list()
		with open(os.path.join(rrc_mlt_2017_dir_path, gt_filepath), newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				if 10 != len(row):
					print('Different row length in {}: {}.'.format(gt_filepath, row))
				#boxes.append(row[:8])
				boxes.append(list(int(rr) for rr in row[:8]))
				texts.append(row[8:])
				# TODO [check] >> Spaces which follow comma can be removed.
				#texts.append(','.join(row[9:]) if len(row[9:]) > 1 else row[9])

		#gt_boxes.append(np.array(boxes, np.float).reshape(-1, 8))
		gt_boxes.append(np.array(boxes, np.float).reshape(-1, 4, 2))
		gt_texts.append(texts)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	for img_filepath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
		img = cv2.imread(os.path.join(rrc_mlt_2017_dir_path, img_filepath))

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

		cv2.imshow('RRC MLT 2017', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

# REF [site] >> https://rrc.cvc.uab.es/?ch=15
def rrc_mlt_2019_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	rrc_mlt_2019_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/rrc_icdar/0_download/mlt_2019'

	print('Loading file list...')
	start_time = time.time()
	img_filepaths = glob.glob(os.path.join(rrc_mlt_2019_dir_path, 'ImagesPart?/tr_img_*.*'), recursive=False)
	gt_filepaths = glob.glob(os.path.join(rrc_mlt_2019_dir_path, 'train_gt_t13/tr_img_*.txt'), recursive=False)

	class FilenameExtracter:
		def __init__(self, base_dir_path):
			self.base_dir_path = base_dir_path

		def __call__(self, filepath):
			idx = filepath.rfind(self.base_dir_path) + len(self.base_dir_path) + 1
			return filepath[idx:]

	img_filepaths = list(map(FilenameExtracter(rrc_mlt_2019_dir_path), img_filepaths))
	gt_filepaths = list(map(FilenameExtracter(rrc_mlt_2019_dir_path), gt_filepaths))

	if True:
		class FileFilter:
			def __init__(self, start_idx, end_idx):
				self.start_idx = start_idx
				self.end_idx = end_idx

			def __call__(self, filepath):
				si, ei = filepath.rfind('tr_img_'), filepath.rfind('.')
				file_id = int(filepath[si+7:ei])
				return self.start_idx <= file_id <= self.end_idx

		# Korean: 05001 ~ 06000.
		img_filepaths = list(filter(FileFilter(5001, 6000), img_filepaths))
		gt_filepaths = list(filter(FileFilter(5001, 6000), gt_filepaths))

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
		# REF [site] >> https://rrc.cvc.uab.es/?ch=8&com=tasks
		#	x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
		#	Valid scripts are: "Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols", "Mixed", "None".
		boxes, texts = list(), list()
		with open(os.path.join(rrc_mlt_2019_dir_path, gt_filepath), newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				if 10 != len(row):
					print('Different row length in {}: {}.'.format(gt_filepath, row))
				#boxes.append(row[:8])
				boxes.append(list(int(rr) for rr in row[:8]))
				texts.append(row[8:])
				# TODO [check] >> Spaces which follow comma can be removed.
				#texts.append(','.join(row[9:]) if len(row[9:]) > 1 else row[9])

		#gt_boxes.append(np.array(boxes, np.float).reshape(-1, 8))
		gt_boxes.append(np.array(boxes, np.float).reshape(-1, 4, 2))
		gt_texts.append(texts)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	for img_filepath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
		img = cv2.imread(os.path.join(rrc_mlt_2019_dir_path, img_filepath))

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

		cv2.imshow('RRC MLT 2019', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

# REF [site] >> https://rrc.cvc.uab.es/?ch=13
def rrc_sroie_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	rrc_sroie_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/rrc_icdar/0_download/sroie'

	print('Loading file list...')
	start_time = time.time()
	if True:
		img_filepaths = glob.glob(os.path.join(rrc_sroie_dir_path, '0325updated.task1train(626p)/*.jpg'), recursive=False)
		gt_filepaths = glob.glob(os.path.join(rrc_sroie_dir_path, '0325updated.task1train(626p)/*.txt'), recursive=False)
	else:
		img_filepaths = glob.glob(os.path.join(rrc_sroie_dir_path, '0325updated.task2train(626p)/*.jpg'), recursive=False)
		gt_filepaths = glob.glob(os.path.join(rrc_sroie_dir_path, '0325updated.task2train(626p)/*.txt'), recursive=False)

	if True:
		class FileFilter:
			def __call__(self, filepath):
				filename = os.path.basename(filepath)
				op, cp = filename.find('('), filename.find(')')
				return -1 == op and -1 == cp

		img_filepaths = list(filter(FileFilter(), img_filepaths))
		gt_filepaths = list(filter(FileFilter(), gt_filepaths))

	if len(img_filepaths) != len(gt_filepaths):
		print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
		return

	#--------------------
	print('Loading data...')
	start_time = time.time()
	gt_boxes, gt_texts = list(), list()
	for img_filepath, gt_filepath in zip(img_filepaths, gt_filepaths):
		# REF [site] >> https://rrc.cvc.uab.es/?ch=13&com=tasks
		#	x1,y1,x2,y2,x3,y3,x4,y4,transcription
		boxes, texts = list(), list()
		with open(os.path.join(rrc_sroie_dir_path, gt_filepath), newline='', encoding='UTF-8') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				#if 9 != len(row):
				#	print('Different row length in {}: {}.'.format(gt_filepath, row))
				#boxes.append(row[:8])
				boxes.append(list(int(rr) for rr in row[:8]))
				# TODO [check] >> Spaces which follow comma can be removed.
				texts.append(','.join(row[8:]) if len(row[8:]) > 1 else row[8])
				#if 9 != len(row):
				#	print('Different row length in {}: {}.'.format(gt_filepath, ','.join(row[8:]) if len(row[8:]) > 1 else row[8]))

		#gt_boxes.append(np.array(boxes, np.float).reshape(-1, 8))
		gt_boxes.append(np.array(boxes, np.float).reshape(-1, 4, 2))
		gt_texts.append(texts)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	for img_filepath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
		img = cv2.imread(os.path.join(rrc_sroie_dir_path, img_filepath))

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

		cv2.imshow('RRC SROIE', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	# NOTE [info] >> RRC MLT 2019 contains RRC MLT 2017.
	#rrc_mlt_2017_test()
	#rrc_mlt_2019_test()

	rrc_sroie_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
