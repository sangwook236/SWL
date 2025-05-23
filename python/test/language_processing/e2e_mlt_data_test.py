#!/usr/bin/env python

# REF [site] >> https://github.com/MichalBusta/E2E-MLT

import os, math, time, glob, csv, pickle
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

def e2e_mlt_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	e2e_mlt_dir_path = data_base_dir_path + '/text/scene_text/e2e_mlt'

	if True:
		e2e_mlt_lang = 'Korean'
		lang_tag = 'kr'
	else:
		e2e_mlt_lang = 'Latin'
		lang_tag = 'en'
	pkl_filepath = e2e_mlt_dir_path + '/e2e_mlt_{}.pkl'.format(lang_tag)

	print('Start loading file list...')
	start_time = time.time()
	img_filepaths = glob.glob(os.path.join(e2e_mlt_dir_path, '{}/*.jpg'.format(e2e_mlt_lang)), recursive=False)
	gt_filepaths = glob.glob(os.path.join(e2e_mlt_dir_path, '{}/*.txt'.format(e2e_mlt_lang)), recursive=False)

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
		fpath = os.path.join(e2e_mlt_dir_path, img_filepath)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
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

	#visualize_data_using_image_file(e2e_mlt_dir_path, img_filepaths, gt_boxes, gt_texts, num_images_to_show=10)

	#--------------------
	# Triples of (image filepath, bboxes, texts).
	if True:
		imagefile_box_text_triples = prepare_and_save_and_load_data_using_image_file(e2e_mlt_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath) 

		#visualize_data_using_image_file(e2e_mlt_dir_path, *list(zip(*imagefile_box_text_triples)), num_images_to_show=10)

	# Triples of (image, bboxes, texts).
	# NOTE [info] >> Cannot save triples of (image, bboxes, texts) to a pickle file.
	if False:
		image_box_text_triples = prepare_and_save_and_load_data_using_image(e2e_mlt_dir_path, img_filepaths, gt_boxes, gt_texts, pkl_filepath)

		#visualize_data_using_image(*list(zip(*image_box_text_triples)), num_images_to_show=10)

def generate_chars_from_e2e_mlt_data():
	import craft.test_utils as test_utils
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	e2e_mlt_dir_path = data_base_dir_path + '/text/scene_text/e2e_mlt'

	if True:
		lang_tag = 'kr'
	else:
		lang_tag = 'en'
	pkl_filepath = e2e_mlt_dir_path + '/e2e_mlt_{}.pkl'.format(lang_tag)
	char_image_label_filpath = e2e_mlt_dir_path + '/char_images_{}.txt'.format(lang_tag)
	char_image_dir_path = e2e_mlt_dir_path + '/char_images_{}'.format(lang_tag)

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
		fpath = os.path.join(e2e_mlt_dir_path, imgfile)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue

		print('Start running CRAFT...')
		start_time = time.time()
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB order.
		bboxes_craft, ch_bboxes_lst_craft, score_text_craft = test_utils.run_craft(rgb, net, refine_net, cuda)
		print('End running CRAFT: {} secs.'.format(time.time() - start_time))

		#print('Texts =', texts)
		matched_bbox_count = 0
		selected_bboxes, selected_ch_bboxes, selected_ch_bbox_text_pairs = [], [], []
		for bbox_gt, txt_gt in zip(bboxes_gt, texts_gt):
			poly_gt = Polygon(bbox_gt)
			for bbox_craft, ch_bboxes_craft in zip(bboxes_craft, ch_bboxes_lst_craft):
				poly_craft = Polygon(bbox_craft)

				#if poly_gt.intersects(poly_craft):
				if len(txt_gt) == len(ch_bboxes_craft) and poly_gt.intersects(poly_craft):
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
							selected_ch_bbox_text_pairs.append([ch_bboxes_craft, txt_gt])

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
						fd.write('{},{}\n'.format(os.path.relpath(fpath, e2e_mlt_dir_path), ch))
						ch_bbox_id += 1
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(char_image_label_filpath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(char_image_label_filpath))
	print('End generating chars: {} secs.'.format(time.time() - start_time))

def generate_words_from_e2e_mlt_data():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	e2e_mlt_dir_path = data_base_dir_path + '/text/scene_text/e2e_mlt'

	if True:
		lang_tag = 'kr'
	else:
		lang_tag = 'en'
	pkl_filepath = e2e_mlt_dir_path + '/e2e_mlt_{}.pkl'.format(lang_tag)
	word_image_label_filpath = e2e_mlt_dir_path + '/word_images_{}.txt'.format(lang_tag)
	word_image_dir_path = e2e_mlt_dir_path + '/word_images_{}'.format(lang_tag)

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
		fpath = os.path.join(e2e_mlt_dir_path, imgfile)
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
					(x1, y1), (x2, y2) = np.floor(np.min(bbox, axis=0)).astype(np.int32), np.ceil(np.max(bbox, axis=0)).astype(np.int32)
					patch = img[y1:y2+1,x1:x2+1]
					if patch is None or patch.size == 0: continue

					fpath = os.path.join(word_image_dir_path, 'image_{}.png'.format(word_bbox_id))
					#fpath = os.path.join(word_image_dir_path, '{}_{}.png'.format(word_bbox_id, ch))
					cv2.imwrite(fpath, patch)
					fd.write('{},{}\n'.format(os.path.relpath(fpath, e2e_mlt_dir_path), txt))
					word_bbox_id += 1
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(word_image_label_filpath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(word_image_label_filpath))
	print('End generating words: {} secs.'.format(time.time() - start_time))

def main():
	#e2e_mlt_test()

	#generate_chars_from_e2e_mlt_data()
	generate_words_from_e2e_mlt_data()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
