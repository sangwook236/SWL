#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, time, glob, csv, re, pickle
import numpy as np
import cv2

def draw_bboxes(bboxes, rgb):
	for box in bboxes:
		#box = box.reshape((-1, 2))
		box = box.astype(np.int)
		if False:
			cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)
		else:
			cv2.line(rgb, tuple(box[0,:]), tuple(box[1,:]), (0, 0, 255), 2, cv2.LINE_8)
			cv2.line(rgb, tuple(box[1,:]), tuple(box[2,:]), (0, 255, 0), 2, cv2.LINE_8)
			cv2.line(rgb, tuple(box[2,:]), tuple(box[3,:]), (255, 0, 0), 2, cv2.LINE_8)
			cv2.line(rgb, tuple(box[3,:]), tuple(box[0,:]), (255, 0, 255), 2, cv2.LINE_8)
	cv2.imshow('BBoxes', rgb)

def visualize_data_using_image_file(data_dir_path, img_filepaths, bboxes_lst, texts_lst, num_images_to_show=10):
	for idx, (img_fpath, boxes, texts) in enumerate(zip(img_filepaths, bboxes_lst, texts_lst)):
		fpath = os.path.join(data_dir_path, img_fpath)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue

		print('Texts =', texts)
		draw_bboxes(boxes, img.copy())
		cv2.waitKey(0)

		if idx >= (num_images_to_show - 1):
			break

	cv2.destroyAllWindows()

def visualize_data_using_image(images, bboxes_lst, texts_lst, num_images_to_show=10):
	for idx, (img, boxes, texts) in enumerate(zip(images, bboxes_lst, texts_lst)):
		print('Texts =', texts)
		draw_bboxes(boxes, img)
		cv2.waitKey(0)

		if idx >= (num_images_to_show - 1):
			break

	cv2.destroyAllWindows()

def prepare_and_save_and_load_data_using_image_file(data_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath):
	print('Start preparing data...')
	start_time = time.time()
	imagefile_box_text_triples = []
	for img_fpath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
		fpath = os.path.join(data_dir_path, img_fpath)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue
		imagefile_box_text_triples.append([img_fpath, boxes, texts])
	print('End preparing data: {} secs.'.format(time.time() - start_time))

	print('Start saving data to {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'wb') as fd:
			pickle.dump(imagefile_box_text_triples, fd)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End saving data: {} secs.'.format(time.time() - start_time))
	#del imagefile_box_text_triples

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'rb') as fd:
			loaded_imagefile_box_text_triples = pickle.load(fd)
			print('#loaded triples of image file, boxes, and texts =', len(loaded_imagefile_box_text_triples))
			del loaded_imagefile_box_text_triples
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	return imagefile_box_text_triples

def prepare_and_save_and_load_data_using_image(data_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath):
	print('Start preparing data...')
	start_time = time.time()
	image_box_text_triples = []
	for img_fpath, boxes, texts in zip(img_filepaths, gt_boxes, gt_texts):
		fpath = os.path.join(data_dir_path, img_fpath)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
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
	#del image_box_text_triples

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'rb') as fd:
			loaded_image_box_text_triples = pickle.load(fd)
			print('#loaded triples of image, boxes, and texts =', len(loaded_image_box_text_triples))
			del loaded_image_box_text_triples
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	return image_box_text_triples

# REF [site] >> https://rrc.cvc.uab.es/?ch=8
def rrc_mlt_2017_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	rrc_mlt_2017_dir_path = data_base_dir_path + '/text/icdar_mlt_2017'

	if False:
		# Arabic.
		start_data_index, end_data_index = 0, 0
		lang_tag = 'ar'
	elif False:
		# Latin.
		start_data_index, end_data_index = 0, 0
		lang_tag = 'en'
	elif False:
		# Chinese.
		start_data_index, end_data_index = 0, 0
		lang_tag = 'ch'
	elif False:
		# Japanese.
		start_data_index, end_data_index = 0, 0
		lang_tag = 'jp'
	elif True:
		# Korean: 4001 ~ 4800. (?)
		start_data_index, end_data_index = 4001, 4800
		lang_tag = 'kr'
	else:
		start_data_index, end_data_index = 1, 4800
		lang_tag = 'all'
	pkl_filepath = rrc_mlt_2017_dir_path + '/icdar_mlt_2017_{}.pkl'.format(lang_tag)

	print('Start loading file list...')
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

		img_filepaths = list(filter(FileFilter(start_data_index, end_data_index), img_filepaths))
		gt_filepaths = list(filter(FileFilter(start_data_index, end_data_index), gt_filepaths))

	img_filepaths.sort(key=lambda filepath: os.path.basename(filepath))
	gt_filepaths.sort(key=lambda filepath: os.path.basename(filepath))

	for img_fpath, gt_fpath in zip(img_filepaths, gt_filepaths):
		assert os.path.splitext(os.path.basename(img_fpath))[0] == os.path.splitext(os.path.basename(gt_fpath))[0][3:]
	if len(img_filepaths) != len(gt_filepaths):
		print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
		return
	print('End loading file list: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start loading data...')
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
	print('End loading data: {} secs.'.format(time.time() - start_time))

	#visualize_data_using_image_file(rrc_mlt_2017_dir_path, img_filepaths, gt_boxes, gt_texts, num_images_to_show=10)

	#--------------------
	# Triples of (image filepath, bboxes, texts).
	if True:
		imagefile_box_text_triples = prepare_and_save_and_load_data_using_image_file(rrc_mlt_2017_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath) 

		#visualize_data_using_image_file(rrc_mlt_2017_dir_path, *list(zip(*imagefile_box_text_triples)), num_images_to_show=10)

	# Triples of (image, bboxes, texts).
	# NOTE [info] >> Cannot save triples of (image, bboxes, texts) to a pickle file.
	if False:
		image_box_text_triples = prepare_and_save_and_load_data_using_image(rrc_mlt_2017_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath)

		#visualize_data_using_image(*list(zip(*image_box_text_triples)), num_images_to_show=10)

# REF [site] >> https://rrc.cvc.uab.es/?ch=15
def rrc_mlt_2019_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	rrc_mlt_2019_dir_path = data_base_dir_path + '/text/scene_text/icdar_mlt_2019'

	if False:
		# Arabic: 00001 ~ 01000.
		start_data_index, end_data_index = 1, 1000
		lang_tag = 'ar'
	elif False:
		# English: 01001 ~ 02000.
		start_data_index, end_data_index = 1001, 2000
		lang_tag = 'en'
	elif False:
		# French: 02001 ~ 03000.
		start_data_index, end_data_index = 2001, 3000
		lang_tag = 'fr'
	elif False:
		# Chinese: 03001 ~ 04000.
		start_data_index, end_data_index = 3001, 4000
		lang_tag = 'ch'
	elif False:
		# German: 04001 ~ 05000.
		start_data_index, end_data_index = 4001, 5000
		lang_tag = 'de'
	elif True:
		# Korean: 05001 ~ 06000.
		start_data_index, end_data_index = 5001, 6000
		lang_tag = 'kr'
	elif False:
		# Japanese: 06001 ~ 07000.
		start_data_index, end_data_index = 6001, 7000
		lang_tag = 'jp'
	elif False:
		# Italian: 07001 ~ 08000.
		start_data_index, end_data_index = 7001, 8000
		lang_tag = 'it'
	else:
		start_data_index, end_data_index = 1, 8000
		lang_tag = 'all'
	pkl_filepath = rrc_mlt_2019_dir_path + '/icdar_mlt_2019_{}.pkl'.format(lang_tag)

	print('Start loading file list...')
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

		img_filepaths = list(filter(FileFilter(start_data_index, end_data_index), img_filepaths))
		gt_filepaths = list(filter(FileFilter(start_data_index, end_data_index), gt_filepaths))

	img_filepaths.sort(key=lambda filepath: os.path.basename(filepath))
	gt_filepaths.sort(key=lambda filepath: os.path.basename(filepath))

	for img_fpath, gt_fpath in zip(img_filepaths, gt_filepaths):
		assert os.path.splitext(os.path.basename(img_fpath))[0] == os.path.splitext(os.path.basename(gt_fpath))[0]
	if len(img_filepaths) != len(gt_filepaths):
		print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
		return
	print('End loading file list: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start loading data...')
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
	print('End loading data: {} secs.'.format(time.time() - start_time))

	#visualize_data_using_image_file(rrc_mlt_2019_dir_path, img_filepaths, gt_boxes, gt_texts, num_images_to_show=10)

	#--------------------
	# Triples of (image filepath, bboxes, texts).
	if True:
		imagefile_box_text_triples = prepare_and_save_and_load_data_using_image_file(rrc_mlt_2019_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath) 

		#visualize_data_using_image_file(rrc_mlt_2019_dir_path, *list(zip(*imagefile_box_text_triples)), num_images_to_show=10)

	# Triples of (image, bboxes, texts).
	# NOTE [info] >> Cannot save triples of (image, bboxes, texts) to a pickle file.
	if False:
		image_box_text_triples = prepare_and_save_and_load_data_using_image(rrc_mlt_2019_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath)

		#visualize_data_using_image(*list(zip(*image_box_text_triples)), num_images_to_show=10)

# REF [site] >> https://rrc.cvc.uab.es/?ch=13
def rrc_sroie_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	rrc_sroie_dir_path = data_base_dir_path + '/text/receipt/icdar2019_sroie'
	pkl_filepath = rrc_sroie_dir_path + '/icdar2019_sroie.pkl'

	print('Start loading file list...')
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

	for img_fpath, gt_fpath in zip(img_filepaths, gt_filepaths):
		assert os.path.splitext(img_fpath)[0] == os.path.splitext(gt_fpath)[0]
	if len(img_filepaths) != len(gt_filepaths):
		print('The numbers of image and ground-truth files have to be the same: {} != {}.'.format(len(img_filepaths), len(gt_filepaths)))
		return
	print('End loading file list: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start loading data...')
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
	print('End loading data: {} secs.'.format(time.time() - start_time))

	#visualize_data_using_image_file(rrc_sroie_dir_path, img_filepaths, gt_boxes, gt_texts, num_images_to_show=10)

	#--------------------
	# Triples of (image filepath, bboxes, texts).
	if True:
		imagefile_box_text_triples = prepare_and_save_and_load_data_using_image_file(rrc_sroie_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath) 

		#visualize_data_using_image_file(rrc_sroie_dir_path, *list(zip(*imagefile_box_text_triples)), num_images_to_show=10)

	# Triples of (image, bboxes, texts).
	# NOTE [info] >> Cannot save triples of (image, bboxes, texts) to a pickle file.
	if False:
		image_box_text_triples = prepare_and_save_and_load_data_using_image(rrc_sroie_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath)

		#visualize_data_using_image(*list(zip(*image_box_text_triples)), num_images_to_show=10)

def generate_chars_from_rrc_mlt_2017_data():
	raise NotImplementedError

def generate_words_from_rrc_mlt_2017_data():
	raise NotImplementedError

def generate_chars_from_rrc_mlt_2019_data():
	import craft.test_utils as test_utils
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	rrc_mlt_2019_dir_path = data_base_dir_path + '/text/scene_text/icdar_mlt_2019'

	#acceptable_types = ['Korean', 'Latin', 'Chinese', 'Japanese', 'Symbols', 'Mixed', 'None']
	acceptable_types = ['Korean', 'Latin', 'Symbols', 'Mixed', 'None']
	if True:
		lang_tag = 'kr'
	else:
		lang_tag = 'en'
	pkl_filepath = rrc_mlt_2019_dir_path + '/icdar_mlt_2019_{}.pkl'.format(lang_tag)
	char_image_label_filpath = rrc_mlt_2019_dir_path + '/char_images_{}.txt'.format(lang_tag)
	char_image_dir_path = rrc_mlt_2019_dir_path + '/char_images_{}'.format(lang_tag)

	os.makedirs(char_image_dir_path, exist_ok=False)

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	imagefile_box_text_triples = None
	try:
		with open(pkl_filepath, 'rb') as fd:
			imagefile_box_text_triples = pickle.load(fd)
			print('#loaded triples of image file, boxes, and texts =', len(imagefile_box_text_triples))
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	print('Start loading CRAFT...')
	start_time = time.time()
	trained_model = './craft/craft_mlt_25k.pth'
	refiner_model = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.
	refine = False  # Enable link refiner.
	cuda = True  # Use cuda for inference.
	net, refine_net = test_utils.load_craft(trained_model, refiner_model, refine, cuda)
	print('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	print('Start generating chars...')
	start_time = time.time()
	ch_bbox_id = 0
	for idx, (imgfile, bboxes_gt, texts_gt) in enumerate(imagefile_box_text_triples):
		fpath = os.path.join(rrc_mlt_2019_dir_path, imgfile)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue

		print('Start running CRAFT...')
		start_time = time.time()
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB order.
		bboxes_craft, ch_bboxes_lst_craft, score_text_craft = test_utils.run_craft(rgb, net, refine_net, cuda)
		print('End running CRAFT: {} secs.'.format(time.time() - start_time))

		matched_bbox_count = 0
		selected_bboxes, selected_ch_bboxes, selected_ch_bbox_text_pairs = [], [], []
		for bbox_gt, txt_gt in zip(bboxes_gt, texts_gt):
			if (txt_gt[0] not in acceptable_types) or (txt_gt[1] == '###'): continue

			poly_gt = Polygon(bbox_gt)
			for bbox_craft, ch_bboxes_craft in zip(bboxes_craft, ch_bboxes_lst_craft):
				poly_craft = Polygon(bbox_craft)

				#if poly_gt.intersects(poly_craft):
				if len(txt_gt[1]) == len(ch_bboxes_craft) and poly_gt.intersects(poly_craft):
					#area_int = poly_gt.intersection(poly_craft).area
					#if area_int / poly_gt.area >= 0.75 and area_int / poly_craft.area >= 0.75:
					iou = poly_gt.intersection(poly_craft).area / poly_gt.union(poly_craft).area
					if iou >= 0.75:
						matched = True
						for ch_bbox in ch_bboxes_craft:
							if not poly_gt.contains(Polygon(ch_bbox).centroid):
								matched = False
								break
						if matched:
							matched_bbox_count += 1
							#selected_bboxes.append(bbox_craft)
							#selected_ch_bboxes.extend(ch_bboxes_craft)
							selected_ch_bbox_text_pairs.append([ch_bboxes_craft, txt_gt[1]])

		if False:
			print('#matched bboxes =', matched_bbox_count, len(bboxes_gt))
			#draw_bboxes(selected_bboxes, img.copy())
			draw_bboxes(selected_ch_bboxes, img.copy())
			cv2.waitKey(0)

		try:
			with open(char_image_label_filpath, 'w' if idx == 0 else 'a', encoding='UTF8') as fd:
				for (ch_bboxes, txt) in selected_ch_bbox_text_pairs:
					for (ch_bbox, ch) in zip(ch_bboxes, txt):
						(x1, y1), (x2, y2) = np.floor(np.min(ch_bbox, axis=0)).astype(np.int32), np.ceil(np.max(ch_bbox, axis=0)).astype(np.int32)
						patch = img[y1:y2+1,x1:x2+1]
						if patch is None or patch.size == 0: continue

						fpath = os.path.join(char_image_dir_path, 'image_{}.png'.format(ch_bbox_id))
						#fpath = os.path.join(char_image_dir_path, '{}_{}.png'.format(ch_bbox_id, ch))
						cv2.imwrite(fpath, patch)
						fd.write('{},{}\n'.format(os.path.relpath(fpath, rrc_mlt_2019_dir_path), ch))
						ch_bbox_id += 1
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(char_image_label_filpath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(char_image_label_filpath))
	print('End generating chars: {} secs.'.format(time.time() - start_time))

def generate_words_from_rrc_mlt_2019_data():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	rrc_mlt_2019_dir_path = data_base_dir_path + '/text/scene_text/icdar_mlt_2019'

	#acceptable_types = ['Korean', 'Latin', 'Chinese', 'Japanese', 'Symbols', 'Mixed', 'None']
	acceptable_types = ['Korean', 'Latin', 'Symbols', 'Mixed', 'None']
	if True:
		lang_tag = 'kr'
	else:
		lang_tag = 'en'
	pkl_filepath = rrc_mlt_2019_dir_path + '/icdar_mlt_2019_{}.pkl'.format(lang_tag)
	word_image_label_filpath = rrc_mlt_2019_dir_path + '/word_images_{}.txt'.format(lang_tag)
	word_image_dir_path = rrc_mlt_2019_dir_path + '/word_images_{}'.format(lang_tag)

	os.makedirs(word_image_dir_path, exist_ok=False)

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	imagefile_box_text_triples = None
	try:
		with open(pkl_filepath, 'rb') as fd:
			imagefile_box_text_triples = pickle.load(fd)
			print('#loaded triples of image file, boxes, and texts =', len(imagefile_box_text_triples))
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	print('Start generating words...')
	start_time = time.time()
	word_bbox_id = 0
	for idx, (imgfile, bboxes_gt, texts_gt) in enumerate(imagefile_box_text_triples):
		fpath = os.path.join(rrc_mlt_2019_dir_path, imgfile)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue

		if False:
			draw_bboxes(bboxes_gt, img.copy())
			cv2.waitKey(0)

		try:
			with open(word_image_label_filpath, 'w' if idx == 0 else 'a', encoding='UTF8') as fd:
				for (bbox, txt) in zip(bboxes_gt, texts_gt):
					if (txt[0] not in acceptable_types) or (txt[1] == '###'): continue

					(x1, y1), (x2, y2) = np.floor(np.min(bbox, axis=0)).astype(np.int32), np.ceil(np.max(bbox, axis=0)).astype(np.int32)
					patch = img[y1:y2+1,x1:x2+1]
					if patch is None or patch.size == 0: continue

					fpath = os.path.join(word_image_dir_path, 'image_{}.png'.format(word_bbox_id))
					#fpath = os.path.join(word_image_dir_path, '{}_{}.png'.format(word_bbox_id, txt[1]))
					cv2.imwrite(fpath, patch)
					fd.write('{},{}\n'.format(os.path.relpath(fpath, rrc_mlt_2019_dir_path), txt[1]))
					word_bbox_id += 1
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(word_image_label_filpath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(word_image_label_filpath))
	print('End generating words: {} secs.'.format(time.time() - start_time))

# REF [site] >> https://rrc.cvc.uab.es/?ch=13
def generate_icdar2019_sroie_task1_train_text_line_data():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/my_dataset'
	else:
		data_base_dir_path = 'E:/dataset'
	data_dir_path = data_base_dir_path + '/pattern_recognition/language_processing/rrc_icdar/0_download/sroie/0325updated.task1train(626p)'
	save_dir_path = './icdar2019_sroie/task1_train_text_line'

	image_filepaths = glob.glob(os.path.join(data_dir_path, 'X???????????.jpg'))
	label_filepaths = glob.glob(os.path.join(data_dir_path, 'X???????????.txt'))

	if len(image_filepaths) != len(label_filepaths):
		print('[SWL] Error: Unmatched numbers of image files and text files, {} != {}.'.format(len(image_filepaths), len(label_filepaths)))
		return

	def separate_line(line):
		pos = [s.start() for s in re.finditer(r',', line)][7]
		return list(int(nn) for nn in line[:pos].split(',')), line[pos+1:]

	os.makedirs(save_dir_path, exist_ok=False)
	save_file_id = 0
	for image_fpath, label_fpath in zip(image_filepaths, label_filepaths):
		try:
			with open(label_fpath, 'r', encoding='UTF8') as fd:
				lines = fd.read().splitlines()
		except FileNotFoundError as ex:
			print('[SWL] Error: File not found: {}.'.format(label_fpath))
			continue
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error: {}.'.format(label_fpath))
			continue

		img = cv2.imread(image_fpath, cv2.IMREAD_COLOR)
		if img is None:
			print('[SWL] Error: Failed to load an image, {}.'.format(image_filepaths))
			continue

		lines = list(separate_line(line) for line in lines)

		for coords, lbl in lines:
			img_fpath, txt_fpath = os.path.join(save_dir_path, 'file_{:06}.jpg'.format(save_file_id)), os.path.join(save_dir_path, 'file_{:06}.txt'.format(save_file_id))
			try:
				with open(txt_fpath, 'w', encoding='UTF8') as fd:
					fd.write(lbl)
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found: {}.'.format(txt_fpath))
				continue
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error: {}.'.format(txt_fpath))
				continue

			x = min(coords[0], coords[2], coords[4], coords[6]), max(coords[0], coords[2], coords[4], coords[6])
			y = min(coords[1], coords[3], coords[5], coords[7]), max(coords[1], coords[3], coords[5], coords[7])
			patch = img[y[0]:y[1]+1,x[0]:x[1]+1]
			cv2.imwrite(img_fpath, patch)

			save_file_id += 1

			#print('Text =', lbl)
			#cv2.imshow('Image', patch)
			#cv2.waitKey(0)

"""
		for coords, lbl in lines:
			cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 1, cv2.LINE_AA)
			cv2.line(img, (coords[2], coords[3]), (coords[4], coords[5]), (0, 255, 0), 1, cv2.LINE_AA)
			cv2.line(img, (coords[4], coords[5]), (coords[6], coords[7]), (255, 0, 0), 1, cv2.LINE_AA)
			cv2.line(img, (coords[6], coords[7]), (coords[0], coords[1]), (255, 0, 255), 1, cv2.LINE_AA)

			print('Text =', lbl)
		cv2.imshow('Image', img)
		cv2.waitKey(0)
	cv2.destroyAllWindows()
"""

def check_label_distribution_of_icdar2019_sroie_task1_train_text_line_data():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/icdar2019_sroie/task1_train_text_line'

	label_filepaths = glob.glob(os.path.join(data_dir_path, '*.txt'))

	lines = list()
	for label_filepath in label_filepaths:
		try:
			with open(label_filepath, 'r', encoding='UTF8') as fd:
				#lines.append(fd.readlines()[0])
				lines.append(fd.read().splitlines()[0])
		except FileNotFoundError:
			print('[SWL] Error: File not found: {}.'.format(label_filepath))
			continue
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error: {}.'.format(label_filepath))
			continue
	if len(label_filepaths) != len(lines):
		print('[SWL] Error: Invalid labels.')
		return

	#--------------------
	from swl.language_processing.util import draw_character_histogram
	draw_character_histogram(lines, charset=None)

# REF [function] >> generate_icdar2019_sroie_task1_train_text_line_data().
def Icdar2019SroieTextLineDataset_test():
	import icdar_data

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/icdar2019_sroie/task1_train_text_line'

	image_height, image_width, image_channel = 64, 640, 1
	train_test_ratio = 0.8
	max_char_count = 100

	import string
	labels = \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))

	label_converter = swl_langproc_util.TokenConverter(labels, pad=None)
	# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
	print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
	print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

	print('Start creating an Icdar2019SroieTextLineDataset...')
	start_time = time.time()
	dataset = icdar_data.Icdar2019SroieTextLineDataset(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating an Icdar2019SroieTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def main():
	# NOTE [info] >> RRC MLT 2019 contains RRC MLT 2017.
	#rrc_mlt_2017_test()
	#rrc_mlt_2019_test()

	#rrc_sroie_test()

	#--------------------
	# Single character data.

	#generate_chars_from_rrc_mlt_2017_data()  # Not yet implemented.
	#generate_words_from_rrc_mlt_2017_data()  # Not yet implemented.
	#generate_chars_from_rrc_mlt_2019_data()
	generate_words_from_rrc_mlt_2019_data()

	#--------------------
	# Text line data.

	#generate_icdar2019_sroie_task1_train_text_line_data()
	#check_label_distribution_of_icdar2019_sroie_task1_train_text_line_data()

	#Icdar2019SroieTextLineDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
