#!/usr/bin/env python

# REF [site] >> https://www.kaggle.com/c/facial-keypoints-detection/

import os, math, time, glob, csv
import numpy as np
import cv2

def data_loading_test():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	facial_keypoints_detection_dir_path = data_home_dir_path + '/pattern_recognition/kaggle/facial_keypoints_detection/facial-keypoints-detection'
	train_filepath = facial_keypoints_detection_dir_path + '/training.csv'
	test_filepath = facial_keypoints_detection_dir_path + '/test.csv'
	image_height, image_width = 96, 96

	def convert_coordinates(elem):
		try:
			return -1 if not elem else float(elem)
		except:
			return -1

	#--------------------
	print('Loading training data...')
	start_time = time.time()
	face_train_keypoints, face_train_images = list(), list()
	# left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y,Image
	with open(train_filepath, newline='', encoding='UTF-8') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar=None)
		idx = 0
		for row in reader:
			if idx > 0:
				if 31 != len(row):
					print('Invalid length in the {}-th row: {}.'.format(idx, row))
				#keypoints = list(float(elem) for elem in row[:30] if not elem else -1)
				keypoints = list(map(convert_coordinates, row[:30]))
				if 30 != len(keypoints):
					print('Invalid keypoints in the {}-th row: {}.'.format(idx, keypoints))
				face_train_keypoints.append(np.array(keypoints, dtype=np.float32).reshape(15, 2))
				pixels = list(int(px) for px in row[30].split(' '))
				if image_height * image_width != len(pixels):
					print('Invalid pixels in the {}-th row: {}.'.format(idx, pixels))
				face_train_images.append(np.array(pixels, dtype=np.uint8).reshape(image_height, image_width))
			idx += 1
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	print('Loading test data...')
	start_time = time.time()
	face_test_images = list()
	# ImageId,Image
	with open(test_filepath, newline='', encoding='UTF-8') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar=None)
		idx = 0
		for row in reader:
			if idx > 0:
				if 2 != len(row):
					print('Invalid length in the {}-th row: {}.'.format(idx, row))
				id = int(row[0])
				pixels = list(int(px) for px in row[1].split(' '))
				if image_height * image_width != len(pixels):
					print('Invalid pixels in the {}-th row: {}.'.format(idx, pixels))
				face_test_images.append(np.array(pixels, dtype=np.uint8).reshape(image_height, image_width))
			idx += 1
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	for keypts, img in zip(face_train_keypoints, face_train_images):
		rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		keypts = np.round(keypts).astype(np.int)
		for pt in keypts:
			cv2.circle(rgb, tuple(pt), 1, (0, 0, 255), 1, cv2.LINE_8)

		cv2.imshow('Train Image', rgb)
		cv2.waitKey(0)

	#--------------------
	for img in face_test_images:
		cv2.imshow('Test Image', img)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	data_loading_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
