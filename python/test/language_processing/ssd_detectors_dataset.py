#!/usr/bin/env python

# REF [site] >> https://github.com/mvoelk/ssd_detectors

import sys, os
if 'posix' == os.name:
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append(lib_home_dir_path + '/ssd_detectors_github')

import math, time, glob, csv, json, pickle
import numpy as np
import cv2
from ssd_data import BaseGTUtility

# REF [function] >> load_scene_text_dataset() in text_generator_test.py
def load_scene_text_dataset(dir_path, json_filename):
	json_filepath = os.path.join(dir_path, json_filename)
	with open(json_filepath, 'r', encoding='UTF-8') as json_file:
		json_data = json.load(json_file)

	image_filepaths, mask_filepaths, gt_texts, gt_boxes = list(), list(), list(), list()
	for dat in json_data:
		image_filepaths.append(dat['image'])
		mask_filepaths.append(dat['mask'])
		gt_texts.append(dat['texts'])
		#gt_boxes.append(dat['bboxes'])
		gt_boxes.append(np.array(dat['bboxes']))

	return image_filepaths, mask_filepaths, gt_texts, gt_boxes

# REF [file] >> ${ssd_detectors_HOME}/data_synthtext.py
# REF [site] >> https://github.com/mvoelk/ssd_detectors
class GTUtility(BaseGTUtility):
	"""Utility for my scene text dataset.

	# Arguments
		data_dir_path: Path to ground truth and image data.
		max_slope: Maximum slope of text lines. Boxes with slope lower 
			then max_slope degrees are rejected.
		polygon: Return oriented boxes defined by their four corner points.
			Required by SegLink...
	"""
	def __init__(self, use_my_scene_text_dataset, use_rrc_mlt_2019_dataset, use_e2e_mlt_dataset, max_slope=None, polygon=False):
		# Data directory structure:
		#	scene_text_dataset
		#		e2e_mlt
		#			Korean
		#				*.jpg
		#				*.txt
		#		mask
		#			*.png
		#		rrc_icdar
		#			mlt_2019
		#				ImagesPart2
		#					*.jpg / *.png / *.gif
		#				train_gt_t13
		#					*.txt
		#		scene
		#			*.png
		#		scene_text_dataset.json
		data_dir_path = './scene_text_dataset'

		self.classes = ['Background', 'Text']
		self.data_path = data_dir_path
		self.gt_path = gt_path = os.path.join(data_dir_path, 'gt.mat')  # Inexact.
		self.image_path = image_path = data_dir_path

		img_filepaths, gt_texts, gt_boxes = list(), list(), list()
		if use_my_scene_text_dataset:
			img_filepaths0, mask_filepaths0, gt_texts0, gt_boxes0 = self.load_my_scene_text_dataset(data_dir_path)
			img_filepaths.extend(img_filepaths0)
			gt_texts.extend(gt_texts0)
			gt_boxes.extend(gt_boxes0)
		if use_rrc_mlt_2019_dataset:
			img_filepaths0, mask_filepaths0, gt_texts0, gt_boxes0 = self.load_rrc_mlt_2019_dataset(data_dir_path)
			img_filepaths.extend(img_filepaths0)
			gt_texts.extend(gt_texts0)
			gt_boxes.extend(gt_boxes0)
		if use_e2e_mlt_dataset:
			img_filepaths0, mask_filepaths0, gt_texts0, gt_boxes0 = self.load_e2e_mlt_dataset(data_dir_path)
			img_filepaths.extend(img_filepaths0)
			gt_texts.extend(gt_texts0)
			gt_boxes.extend(gt_boxes0)
		del img_filepaths0, mask_filepaths0, gt_texts0, gt_boxes0

		#--------------------
		print('Processing samples...')
		start_time = time.time()
		self.image_names, self.data, self.text = list(), list(), list()
		for img_filepath, text, boxes in zip(img_filepaths, gt_texts, gt_boxes):
			img = cv2.imread(os.path.join(data_dir_path, img_filepath))
			if img is None:
				print('Failed to load an image: {}.'.format(os.path.join(data_dir_path, img_filepath)))
				continue
			img_height, img_width = img.shape[:2]

			#boxes = np.array(boxes)
			boxes[:,:,0] /= img_width
			boxes[:,:,1] /= img_height
			boxes = boxes.reshape(boxes.shape[0], -1)

			# Fix some bugs in the SynthText dataset.
			eps = 1e-3
			p1, p2, p3, p4 = boxes[:,0:2], boxes[:,2:4], boxes[:,4:6],boxes[:,6:8]
			# Fix twisted boxes (897 boxes, 0.012344 %).
			if True:
				mask = np.linalg.norm(p1 + p2 - p3 - p4, axis=1) < eps
				boxes[mask] = np.concatenate([p1[mask], p3[mask], p2[mask], p4[mask]], axis=1)
			# Filter out bad boxes (528 boxes, 0.007266 %).
			if True:
				mask = np.ones(len(boxes), dtype=np.bool)
				# Filter boxes with zero width (173 boxes, 0.002381 %).
				boxes_w = np.linalg.norm(p1-p2, axis=1)
				boxes_h = np.linalg.norm(p2-p3, axis=1)
				mask = np.logical_and(mask, boxes_w > eps)
				mask = np.logical_and(mask, boxes_h > eps)
				# Filter boxes that are too large (62 boxes, 0.000853 %).
				mask = np.logical_and(mask, np.all(boxes > -1, axis=1))
				mask = np.logical_and(mask, np.all(boxes < 2, axis=1))
				# Filter boxes with all vertices outside the image (232 boxes, 0.003196 %).
				boxes_x = boxes[:,0::2]
				boxes_y = boxes[:,1::2]
				mask = np.logical_and(mask, 
						np.sum(np.logical_or(np.logical_or(boxes_x < 0, boxes_x > 1), 
								np.logical_or(boxes_y < 0, boxes_y > 1)), axis=1) < 4)
				# Filter boxes with center outside the image (336 boxes, 0.004624 %).
				boxes_x_mean = np.mean(boxes[:,0::2], axis=1)
				boxes_y_mean = np.mean(boxes[:,1::2], axis=1)
				mask = np.logical_and(mask, np.logical_and(boxes_x_mean > 0, boxes_x_mean < 1))
				mask = np.logical_and(mask, np.logical_and(boxes_y_mean > 0, boxes_y_mean < 1))
				boxes = boxes[mask]
				text = np.asarray(text)[mask]

			# Only boxes with slope below max_slope.
			if not max_slope == None:
				angles = np.arctan(np.divide(boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]))
				angles[angles < 0] += np.pi
				angles = angles/np.pi*180-90
				boxes = boxes[np.abs(angles) < max_slope]

			# Only images with boxes.
			if len(boxes) == 0:
				continue

			if not polygon:
				xmax = np.max(boxes[:,0::2], axis=1)
				xmin = np.min(boxes[:,0::2], axis=1)
				ymax = np.max(boxes[:,1::2], axis=1)
				ymin = np.min(boxes[:,1::2], axis=1)
				boxes = np.array([xmin, ymin, xmax, ymax]).T

			# Append classes.
			boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
			self.image_names.append(img_filepath)
			self.data.append(boxes)
			self.text.append(text)
		print('\tGenerated dataset: #images = {}, #boxes = {}, #texts = {}.'.format(len(self.image_names), len(self.data), len(self.text)))
		print('\tElapsed time = {}'.format(time.time() - start_time))

		self.init()

	def load_my_scene_text_dataset(self, data_dir_path):
		print('Loading data (My scene text dataset)...')
		start_time = time.time()
		json_filename = 'scene_text_dataset.json'
		img_filepaths, mask_filepaths, gt_texts, gt_boxes = load_scene_text_dataset(data_dir_path, json_filename)
		print('\tElapsed time = {}'.format(time.time() - start_time))

		return img_filepaths, mask_filepaths, gt_texts, gt_boxes

	# REF [function] >> rrc_mlt_2019_bounding_box_test() in rrc_dataset.py
	def load_rrc_mlt_2019_dataset(self, data_dir_path):
		print('Loading file list (RRC MLT 2019 dataset)...')
		start_time = time.time()
		img_filepaths = glob.glob(os.path.join(data_dir_path, 'rrc_icdar/mlt_2019/ImagesPart?/tr_img_*.*'), recursive=False)
		gt_filepaths = glob.glob(os.path.join(data_dir_path, 'rrc_icdar/mlt_2019/train_gt_t13/tr_img_*.txt'), recursive=False)

		class FilenameExtracter:
			def __init__(self, base_dir_path):
				self.base_dir_path = base_dir_path

			def __call__(self, filepath):
				idx = filepath.rfind(self.base_dir_path) + len(self.base_dir_path) + 1
				return filepath[idx:]

		img_filepaths = list(map(FilenameExtracter(data_dir_path), img_filepaths))
		gt_filepaths = list(map(FilenameExtracter(data_dir_path), gt_filepaths))

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
		print('Loading data (RRC MLT 2019 dataset)...')
		start_time = time.time()
		gt_boxes, gt_texts = list(), list()
		for img_filepath, gt_filepath in zip(img_filepaths, gt_filepaths):
			img = cv2.imread(os.path.join(data_dir_path, img_filepath))
			#height, width = img.shape[:2]

			# REF [site] >> https://rrc.cvc.uab.es/?ch=8&com=tasks
			#	x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
			#	Valid scripts are: "Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols", "Mixed", "None".
			boxes, texts = list(), list()
			with open(os.path.join(data_dir_path, gt_filepath), newline='', encoding='UTF-8') as csvfile:
				reader = csv.reader(csvfile, delimiter=',')
				for row in reader:
					#if 10 != len(row):
					#	print('Different row length in {}: {}.'.format(gt_filepath, row))
					boxes.append(list(int(rr) for rr in row[:8]))
					#texts.append(row[8:])
					# TODO [check] >> Spaces which follow comma can be removed.
					texts.append(','.join(row[9:]) if len(row[9:]) > 1 else row[9])

			#gt_boxes.append(np.array(boxes, np.float).reshape(-1, 8))
			gt_boxes.append(np.array(boxes, np.float).reshape(-1, 4, 2))
			gt_texts.append(texts)
		print('\tElapsed time = {}'.format(time.time() - start_time))

		return img_filepaths, None, gt_texts, gt_boxes

	# REF [function] >> bounding_box_test() in e2e_mlt_dataset.py
	def load_e2e_mlt_dataset(self, data_dir_path):
		print('Loading file list (E2E-MLT dataset)...')
		start_time = time.time()
		img_filepaths = glob.glob(os.path.join(data_dir_path, 'e2e_mlt/Korean/*.jpg'), recursive=False)
		gt_filepaths = glob.glob(os.path.join(data_dir_path, 'e2e_mlt/Korean/*.txt'), recursive=False)

		class FilenameExtracter:
			def __init__(self, base_dir_path):
				self.base_dir_path = base_dir_path

			def __call__(self, filepath):
				idx = filepath.rfind(self.base_dir_path) + len(self.base_dir_path) + 1
				return filepath[idx:]

		img_filepaths = list(map(FilenameExtracter(data_dir_path), img_filepaths))
		gt_filepaths = list(map(FilenameExtracter(data_dir_path), gt_filepaths))

		img_filepaths.sort(key=lambda filepath: os.path.basename(filepath))
		gt_filepaths.sort(key=lambda filepath: os.path.basename(filepath))
		print('\tElapsed time = {}'.format(time.time() - start_time))

		if len(img_filepaths) != len(gt_filepaths):
			print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
			return

		#--------------------
		print('Loading data (E2E-MLT dataset)...')
		start_time = time.time()
		gt_boxes, gt_texts = list(), list()
		for img_filepath, gt_filepath in zip(img_filepaths, gt_filepaths):
			img = cv2.imread(os.path.join(data_dir_path, img_filepath))
			height, width = img.shape[:2]
			max_len = max(height, width)

			# ?, center, size, angle, text.
			boxes, texts = list(), list()
			with open(os.path.join(data_dir_path, gt_filepath), newline='', encoding='UTF-8') as csvfile:
				reader = csv.reader(csvfile, delimiter=' ', quotechar=None)
				for row in reader:
					if 7 != len(row):
						print('Different row length in {}: {}.'.format(gt_filepath, row))
					#boxes.append(row[:6])
					boxes.append(list(float(rr) for rr in row[:6]))
					#texts.append(row[6])
					# TODO [check] >> Spaces which follow comma can be removed.
					texts.append(' '.join(row[6:]) if len(row[6:]) > 1 else row[6])

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

		return img_filepaths, None, gt_texts, gt_boxes

def convert_scene_text_dataset_to_ssd_detectors():
	use_my_scene_text_dataset, use_rrc_mlt_2019_dataset, use_e2e_mlt_dataset = True, True, True
	polygon = True
	gt_util = GTUtility(use_my_scene_text_dataset, use_rrc_mlt_2019_dataset, use_e2e_mlt_dataset, polygon=polygon)

	file_name = 'gt_util_scene_text_seglink.pkl' if polygon else 'gt_util_scene_text.pkl'
	print('Save to {}...'.format(file_name))
	pickle.dump(gt_util, open(file_name, 'wb'))
	print('Done.')

def main():
	convert_scene_text_dataset_to_ssd_detectors()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
