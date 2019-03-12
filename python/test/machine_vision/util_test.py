#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os
import numpy as np
import cv2
from PIL import Image
#import keras
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_vision.util as swl_cv_util

def load_images_test():
	if 'posix' == os.name:
		#data_home_dir_path = '/home/sangwook/my_dataset'
		data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'

	train_image_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/train/image'
	train_label_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/trainannot/image'
	val_image_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/val/image'
	val_label_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/valannot/image'
	test_image_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/test/image'
	test_label_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/testannot/image'

	#--------------------
	# Convert image to array.

	image_suffix = ''
	image_extension = 'png'
	label_suffix = ''
	label_extension = 'png'

	#image_width, image_height = None, None
	#image_width, image_height = 480, 360
	image_width, image_height = 224, 224

	if True:
		train_images = swl_cv_util.load_images_by_pil(train_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
		train_labels = swl_cv_util.load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
		val_images = swl_cv_util.load_images_by_pil(val_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
		val_labels = swl_cv_util.load_labels_by_pil(val_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
		test_images = swl_cv_util.load_images_by_pil(test_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
		test_labels = swl_cv_util.load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
	else:
		train_images = swl_cv_util.load_images_by_scipy(train_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
		train_labels = swl_cv_util.load_labels_by_scipy(train_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
		val_images = swl_cv_util.load_images_by_scipy(val_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
		val_labels = swl_cv_util.load_labels_by_scipy(val_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
		test_images = swl_cv_util.load_images_by_scipy(test_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
		test_labels = swl_cv_util.load_labels_by_scipy(test_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)

	#--------------------

	if False:
		num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
		train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
		val_labels = np.uint8(keras.utils.to_categorical(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
		test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))
	else:
		num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
		train_labels = np.uint8(swl_ml_util.to_one_hot_encoding(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
		val_labels = np.uint8(swl_ml_util.to_one_hot_encoding(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
		test_labels = np.uint8(swl_ml_util.to_one_hot_encoding(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

	#--------------------
	# Save a numpy.array to an npy file.
	if True:
		np.save('camvid_data/train_images.npy', train_images)
		np.save('camvid_data/train_labels.npy', train_labels)
		np.save('camvid_data/val_images.npy', val_images)
		np.save('camvid_data/val_labels.npy', val_labels)
		np.save('camvid_data/test_images.npy', test_images)
		np.save('camvid_data/test_labels.npy', test_labels)
	else:
		np.savez('camvid_data/train_images.npz', train_images)
		np.savez('camvid_data/train_labels.npz', train_labels)
		np.savez('camvid_data/val_images.npz', val_images)
		np.savez('camvid_data/val_labels.npz', val_labels)
		np.savez('camvid_data/test_images.npz', test_images)
		np.savez('camvid_data/test_labels.npz', test_labels)

	# Load a numpy.array from an npy file.
	if True:
		train_images0 = np.load('camvid_data/train_images.npy')
		train_labels0 = np.load('camvid_data/train_labels.npy')
		val_images0 = np.load('camvid_data/val_images.npy')
		val_labels0 = np.load('camvid_data/val_labels.npy')
		test_images0 = np.load('camvid_data/test_images.npy')
		test_labels0 = np.load('camvid_data/test_labels.npy')
	else:
		train_images0 = np.load('camvid_data/train_images.npz')
		train_labels0 = np.load('camvid_data/train_labels.npz')
		val_images0 = np.load('camvid_data/val_images.npz')
		val_labels0 = np.load('camvid_data/val_labels.npz')
		test_images0 = np.load('camvid_data/test_images.npz')
		test_labels0 = np.load('camvid_data/test_labels.npz')

def show_image_in_npy():
	if True:
		npy_dir_path = './npy_dir'
		file_prefix = ''
		file_suffix = ''
		npy_filepaths = swl_util.list_files_in_directory(npy_dir_path, file_prefix, file_suffix, 'npy', is_recursive=False)
	else:
		npy_filepaths = [
			'./npy_dir/img.npy',
		]
	np_arr_list = list()
	for filepath in npy_filepaths:
		np_arr_list.append(np.load(filepath))

	print('#loaded npy files =', len(np_arr_list))

	for idx, arr in enumerate(np_arr_list):
		print('ID = {}.'.format(idx))
		print('Shape: {}, dtype: {}.'.format(arr.shape, arr.dtype))
		print('Min = {}, max = {}.'.format(np.min(arr), np.max(arr)))

		for img in arr:
			#img.save('./img.png')
			#img = Image.fromarray(img)
			#img.show()

			#cv2.imwrite('./img.png', img)
			cv2.imshow('Image', img)
			ch = cv2.waitKey(0)
			if 27 == ch:  # ESC.
				break

		if 27 == ch:  # ESC.
			break

	cv2.destroyAllWindows()

def show_image_and_label_in_npy_pair():
	if True:
		npy_dir_path = './npy_dir'
		input_file_prefix, output_file_prefix = 'image', 'label'
		file_suffix = ''
		input_npy_filepaths = swl_util.list_files_in_directory(npy_dir_path, input_file_prefix, file_suffix, 'npy', is_recursive=False)
		output_npy_filepaths = swl_util.list_files_in_directory(npy_dir_path, output_file_prefix, file_suffix, 'npy', is_recursive=False)
		npy_filepath_pairs = [npy_filepath_pair for npy_filepath_pair in zip(input_npy_filepaths, output_npy_filepaths)]
	else:
		npy_filepath_pairs = [
			('./npy_dir/image.npy', './npy_dir/label.npy'),
		]

	for input_npy_filepath, output_npy_filepath in npy_filepath_pairs:
		inputs, outputs = np.load(input_npy_filepath), np.load(output_npy_filepath)

		if len(inputs) != len(outputs):
			print('Unmatched data count: {} != {}.'.format(len(inputs), len(outputs)))
			continue

		print('Image filapath: {}, label filapath: {}.'.format(input_npy_filepath, output_npy_filepath))
		print('Image shape: {}, dtype: {}.'.format(inputs.shape, inputs.dtype))
		print('Image min = {}, max = {}.'.format(np.min(inputs), np.max(inputs)))
		print('Label shape: {}, dtype: {}.'.format(outputs.shape, outputs.dtype))

		for img, lbl in zip(inputs, outputs):
			print('\tLabel = {}'.format(lbl))

			#img.save('./img.png')
			#img = Image.fromarray(img)
			#img.show()

			#cv2.imwrite('./img.png', img)
			cv2.imshow('Image', img)
			ch = cv2.waitKey(0)
			if 27 == ch:  # ESC.
				break

		if 27 == ch:  # ESC.
			break

	cv2.destroyAllWindows()

def show_image_in_npz():
	if True:
		npz_dir_path = './npz_dir'
		file_prefix = ''
		file_suffix = ''
		npz_filepaths = swl_util.list_files_in_directory(npz_dir_path, file_prefix, file_suffix, 'npz', is_recursive=False)
	else:
		npz_filepaths = [
			'./npz_dir/img.npz',
		]

	for npz_filepath in npz_filepaths:
		npzfile = np.load(npz_filepath)

		print('#loaded files =', len(npzfile.files))

		for key, arr in npzfile.items():
			print('ID = {}.'.format(key))
			print('Shape: {}, dtype: {}.'.format(arr.shape, arr.dtype))
			print('Min = {}, max = {}.'.format(np.min(arr), np.max(arr)))

			for img in arr:
				#img.save('./img.png')
				#img = Image.fromarray(img)
				#img.show()

				#cv2.imwrite('./img.png', img)
				cv2.imshow('Image', img)
				ch = cv2.waitKey(0)
				if 27 == ch:  # ESC.
					break

			if 27 == ch:  # ESC.
				break
		if 27 == ch:  # ESC.
			break

	cv2.destroyAllWindows()

def show_image_and_label_in_npz_pair():
	if True:
		npz_dir_path = './npz_dir'
		input_file_prefix, output_file_prefix = 'image', 'label'
		file_suffix = ''
		input_npz_filepaths = swl_util.list_files_in_directory(npz_dir_path, input_file_prefix, file_suffix, 'npz', is_recursive=False)
		output_npz_filepaths = swl_util.list_files_in_directory(npz_dir_path, output_file_prefix, file_suffix, 'npz', is_recursive=False)
		npz_filepath_pairs = [npz_filepath_pair for npz_filepath_pair in zip(input_npz_filepaths, output_npz_filepaths)]
	else:
		npz_filepath_pairs = [
			('./npz_dir/image.npz', './npz_dir/label.npz'),
		]

	for input_npz_filepath, output_npz_filepath in npz_filepath_pairs:
		input_npzfile, output_npzfile = np.load(input_npz_filepath), np.load(output_npz_filepath)

		print('#loaded input files = {}, #loaded output files ={}.'.format(len(input_npzfile.files), len(output_npzfile.files)))

		for (in_key, in_arr), (out_key, out_arr) in zip(input_npzfile.items(), output_npzfile.items()):
			if in_key != out_key:
				print('Unmatched keys: {} != {}.'.format(in_key, out_key))
				continue
			if len(in_arr) != len(out_arr):
				print('Unmatched data count: {} != {}.'.format(len(in_arr), len(out_arr)))
				continue

			print('Image ID = {}, label ID = {}.'.format(in_key, out_key))
			print('Image shape: {}, dtype: {}.'.format(in_arr.shape, in_arr.dtype))
			print('Image min = {}, max = {}.'.format(np.min(in_arr), np.max(in_arr)))
			print('Label shape: {}, dtype: {}.'.format(out_arr.shape, out_arr.dtype))

			for img, lbl in zip(in_arr, out_arr):
				print('\tLabel = {}'.format(lbl))

				#img.save('./img.png')
				#img = Image.fromarray(img)
				#img.show()

				#cv2.imwrite('./img.png', img)
				cv2.imshow('Image', img)
				ch = cv2.waitKey(0)
				if 27 == ch:  # ESC.
					break

			if 27 == ch:  # ESC.
				break
		if 27 == ch:  # ESC.
			break

	cv2.destroyAllWindows()

def main():
	#load_images_test()

	#--------------------
	# Tool.

	#show_image_in_npy()
	#show_image_and_label_in_npy_pair()
	#show_image_in_npz()
	show_image_and_label_in_npz_pair()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
