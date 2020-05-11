#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, pickle, time
import cv2

def visualize_data_using_image_file(data_dir_path, image_filepaths, labels, num_images_to_show=10):
	for idx, (img_fpath, lbl) in enumerate(zip(image_filepaths, labels)):
		fpath = os.path.join(data_dir_path, '{}.png'.format(img_fpath))
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue

		print('Label =', lbl)
		cv2.imshow('Image', img)
		cv2.waitKey(0)

		if idx >= (num_images_to_show - 1):
			break

	cv2.destroyAllWindows()

def visualize_data_using_image(images, labels, num_images_to_show=10):
	for idx, (img, lbl) in enumerate(zip(images, labels)):
		print('Label =', lbl)
		cv2.imshow('Image', img)
		cv2.waitKey(0)

		if idx >= (num_images_to_show - 1):
			break

	cv2.destroyAllWindows()

def prepare_and_save_and_load_data_using_image_file(data_dir_path, data_info, pkl_filepath):
	print('Start preparing data...')
	start_time = time.time()
	imagefile_label_pairs = []
	for img_fpath, lbl in zip(data_info[0], data_info[2]):
		img_fpath = '{}.png'.format(img_fpath)
		fpath = os.path.join(data_dir_path, img_fpath)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue
		imagefile_label_pairs.append([img_fpath, int(lbl[0])])
	print('End preparing data: {} secs.'.format(time.time() - start_time))

	print('Start saving data to {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'wb') as fd:
			pickle.dump(imagefile_label_pairs, fd)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End saving data: {} secs.'.format(time.time() - start_time))
	#del imagefile_label_pairs

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'rb') as fd:
			loaded_imagefile_label_pairs = pickle.load(fd)
			print('#loaded pairs of image file and label =', len(loaded_imagefile_label_pairs))
			del loaded_imagefile_label_pairs
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	return imagefile_label_pairs

def prepare_and_save_and_load_data_using_image(data_dir_path, data_info, pkl_filepath):
	print('Start preparing data...')
	start_time = time.time()
	image_label_pairs = []
	for img_fpath, lbl in zip(data_info[0], data_info[2]):
		img_fpath = '{}.png'.format(img_fpath)
		fpath = os.path.join(data_dir_path, img_fpath)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image, {}.'.format(fpath))
			continue
		image_label_pairs.append([img, int(lbl[0])])
	print('End preparing data: {} secs.'.format(time.time() - start_time))

	print('Start saving data to {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'wb') as fd:
			pickle.dump(image_label_pairs, fd)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End saving data: {} secs.'.format(time.time() - start_time))
	#del image_label_pairs

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'rb') as fd:
			loaded_image_label_pairs = pickle.load(fd)
			print('#loaded pairs of image and label =', len(loaded_image_label_pairs))
			del loaded_image_label_pairs
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	return image_label_pairs

# REF [site] >> http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
def chars74k_test():
	import scipy.io

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/chars74k/English/Img'
	image_list_filepaths = [
		data_dir_path + '/all_good.txt',
		data_dir_path + '/all__bad.txt'
	]
	mat_filepath = data_dir_path + '/lists.mat'
	pkl_filepath = data_dir_path + '/chars74k.pkl'

	print('Start loading chars74k data...')
	start_time = time.time()
	mat = scipy.io.loadmat(mat_filepath)
	data_info = mat['list'][0,0]
	print('PNG files:', data_info[0])  # A list of PNG files.
	#print('*****', data_info[1])  # ???
	print('Labels:', data_info[2])  # Labels.
	print('Class labels:', data_info[3])  # Class labels. [1, ..., 62].
	print('Directories of good images:', data_info[4])  # Directories of good images.
	print('Number of classes =', data_info[5])  # Number of classes. 62.
	#print('*****', data_info[6])  # ???
	#print('*****', data_info[7])  # ???
	#print('*****', data_info[8])  # ???
	#print('*****', data_info[9])  # ???
	print('End loading chars74k data: {} secs.'.format(time.time() - start_time))

	#visualize_data_using_image_file(data_dir_path, data_info[0], data_info[2], num_images_to_show=10)

	#--------------------
	# Pairs of (image filepath, label).
	if True:
		imagefile_label_pairs = prepare_and_save_and_load_data_using_image_file(data_dir_path, data_info, pkl_filepath)

		#visualize_data_using_image_file(data_dir_path, *list(zip(*imagefile_label_pairs)), num_images_to_show=10)

	# Pairs of (image, label).
	if False:
		image_label_pairs = prepare_and_save_and_load_data_using_image(data_dir_path, data_info, pkl_filepath)

		#visualize_data_using_image(*list(zip(*image_label_pairs)), num_images_to_show=10)

def generate_chars_from_chars74k_data():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/chars74k/English/Img'

	pkl_filepath = data_dir_path + '/chars74k.pkl'
	char_image_label_filpath = data_dir_path + '/char_images.txt'

	print('Start loading data from {}...'.format(pkl_filepath))
	start_time = time.time()
	imagefile_label_pairs = None
	try:
		with open(pkl_filepath, 'rb') as fd:
			imagefile_label_pairs = pickle.load(fd)
			print('#loaded pairs of image file and label =', len(imagefile_label_pairs))
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(pkl_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(pkl_filepath))
	print('End loading data: {} secs.'.format(time.time() - start_time))

	print('Start generating chars...')
	start_time = time.time()
	try:
		with open(char_image_label_filpath, 'w', encoding='UTF8') as fd:
			for idx, (imgfile, lbl) in enumerate(imagefile_label_pairs):
				fpath = os.path.join(data_dir_path, imgfile)
				img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
				if img is None:
					print('Failed to load an image, {}.'.format(fpath))
					continue

				fd.write('{},{}\n'.format(imgfile, lbl))
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(char_image_label_filpath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(char_image_label_filpath))
	print('End generating chars: {} secs.'.format(time.time() - start_time))

def main():
	#chars74k_test()

	generate_chars_from_chars74k_data()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
