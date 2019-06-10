#!/usr/bin/env python

# REF [site] >> http://www.robots.ox.ac.uk/~vgg/data/scenetext/

import os, math, time, glob, csv
import numpy as np
import scipy.io as sio
import cv2

# REF [file] >> ${ssd_detectors_HOME}/data_synthtext.py
def data_loading_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	#synthtext_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/synth_text_vgg/SynthText'
	synthtext_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/SynthText/SynthText'
	gt_filepath = gt_path = os.path.join(synthtext_dir_path, 'gt.mat')
	max_slope = None
	polygon = True

	print('Loading file list...')
	start_time = time.time()
	data = sio.loadmat(gt_filepath)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	print('Loading data...')
	start_time = time.time()
	num_samples = data['imnames'].shape[1]
	image_names, word_bboxes, char_bboxes, gt_texts = list(), list(), list(), list()
	for i in range(num_samples):
		image_name = data['imnames'][0,i][0]
		text = [s.strip().split() for s in data['txt'][0,i]]
		text = [w for s in text for w in s]

		# Read image size only when size changes.
		if image_name.split('_')[-1].split('.')[0] == '0':
			img = cv2.imread(os.path.join(synthtext_dir_path, image_name))
			img_height, img_width = img.shape[:2]

		# data['wordBB'][0,i] has shape (x + y, points, words) = (2, 4, n).
		boxes = data['wordBB'][0,i]
		if 2 == len(boxes.shape):
			boxes = boxes[:,:,None]
		boxes = boxes.transpose(2, 1, 0)
		boxes[:,:,0] /= img_width
		boxes[:,:,1] /= img_height
		boxes = boxes.reshape(boxes.shape[0], -1)

		# data['charBB'][0,i] has shape (x + y, points, words) = (2, 4, n).
		char_boxes = data['charBB'][0,i]
		if 2 == len(char_boxes.shape):
			char_boxes = char_boxes[:,:,None]
		char_boxes = char_boxes.transpose(2, 1, 0)
		char_boxes[:,:,0] /= img_width
		char_boxes[:,:,1] /= img_height
		char_boxes = char_boxes.reshape(char_boxes.shape[0], -1)

		# Fix some bugs in the SynthText dataset.
		eps = 1e-3
		p1, p2, p3, p4 = boxes[:,0:2], boxes[:,2:4], boxes[:,4:6], boxes[:,6:8]
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
		if max_slope is not None:
			angles = np.arctan(np.divide(boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1]))
			angles[angles < 0] += np.pi
			angles = angles / np.pi * 180 - 90
			boxes = boxes[np.abs(angles) < max_slope]

		# Only images with boxes.
		if 0 == len(boxes):
			continue

		if not polygon:
			xmax = np.max(boxes[:,0::2], axis=1)
			xmin = np.min(boxes[:,0::2], axis=1)
			ymax = np.max(boxes[:,1::2], axis=1)
			ymin = np.min(boxes[:,1::2], axis=1)
			boxes = np.array([xmin, ymin, xmax, ymax]).T

			xmax = np.max(char_boxes[:,0::2], axis=1)
			xmin = np.min(char_boxes[:,0::2], axis=1)
			ymax = np.max(char_boxes[:,1::2], axis=1)
			ymin = np.min(char_boxes[:,1::2], axis=1)
			char_boxes = np.array([xmin, ymin, xmax, ymax]).T

		# Append classes.
		image_names.append(image_name)
		#boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
		word_bboxes.append(boxes)
		#char_boxes = np.concatenate([char_boxes, np.ones([char_boxes.shape[0],1])], axis=1)
		char_bboxes.append(char_boxes)
		gt_texts.append(text)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	for image_name, word_boxes, char_boxes, texts in zip(image_names, word_bboxes, char_bboxes, gt_texts):
		img = cv2.imread(os.path.join(synthtext_dir_path, image_name))
		img_height, img_width = img.shape[:2]

		word_boxes = word_boxes.reshape(word_boxes.shape[0], -1, 2)
		char_boxes = char_boxes.reshape(char_boxes.shape[0], -1, 2)
		word_boxes[:,:,0] *= img_width
		word_boxes[:,:,1] *= img_height
		char_boxes[:,:,0] *= img_width
		char_boxes[:,:,1] *= img_height
		word_boxes = np.round(word_boxes).astype(np.int)
		char_boxes = np.round(char_boxes).astype(np.int)

		print('GT texts =', texts)
		rgb = img.copy()
		for box in word_boxes:
			if False:
				cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)
			else:
				cv2.line(rgb, tuple(box[0,:]), tuple(box[1,:]), (0, 0, 255), 2, cv2.LINE_8)
				cv2.line(rgb, tuple(box[1,:]), tuple(box[2,:]), (0, 255, 0), 2, cv2.LINE_8)
				cv2.line(rgb, tuple(box[2,:]), tuple(box[3,:]), (255, 0, 0), 2, cv2.LINE_8)
				cv2.line(rgb, tuple(box[3,:]), tuple(box[0,:]), (255, 0, 255), 2, cv2.LINE_8)

		cv2.imshow('SynthText', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	data_loading_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
