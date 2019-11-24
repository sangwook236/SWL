#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, glob, re, time
import cv2
import icdar_data

# REF [site] >> https://rrc.cvc.uab.es/?ch=13
def generate_icdar2019_sroie_task1_train_text_line_data():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/my_dataset'
	else:
		data_base_dir_path = 'E:/dataset'
	data_dir_path = data_base_dir_path + '/pattern_recognition/language_processing/rrc_icdar/0_download/sroie/0325updated.task1train(626p)'
	saved_dir_path = './icdar2019_sroie/task1_train_text_line'

	image_filepaths = glob.glob(os.path.join(data_dir_path, 'X???????????.jpg'))
	label_filepaths = glob.glob(os.path.join(data_dir_path, 'X???????????.txt'))

	if len(image_filepaths) != len(label_filepaths):
		print('[SWL] Error: Unmatched numbers of image files and text files, {} != {}.'.format(len(image_filepaths), len(label_filepaths)))
		return

	def separate_line(line):
		pos = [s.start() for s in re.finditer(r',', line)][7]
		return list(int(nn) for nn in line[:pos].split(',')), line[pos+1:]

	os.makedirs(saved_dir_path, exist_ok=False)
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
			img_fpath, txt_fpath = os.path.join(saved_dir_path, 'file_{:06}.jpg'.format(save_file_id)), os.path.join(saved_dir_path, 'file_{:06}.txt'.format(save_file_id))
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

# REF [function] >> generate_icdar2019_sroie_task1_train_text_line_data().
def Icdar2019SroieTextLineDataset_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/icdar2019_sroie/task1_train_text_line'

	image_height, image_width, image_channel = 64, 640, 1
	train_test_ratio = 0.8
	max_char_count = 100

	print('Start creating an Icdar2019SroieTextLineDataset...')
	start_time = time.time()
	dataset = icdar_data.Icdar2019SroieTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating an Icdar2019SroieTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

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
	import string
	charset = \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '

	#charset = sorted(charset)
	charset = ''.join(sorted(charset))

	#--------------------
	char_dict = dict()
	for ch in charset:
		char_dict[ch] = 0

	for line in lines:
		if not line:
			continue

		for ch in line:
			try:
				char_dict[ch] += 1
			except KeyError:
				print('[SWL] Warning: Invalid character {}.'.format(ch))

	#--------------------
	import numpy as np
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(10, 6))
	x_label = np.arange(len(char_dict.keys()))
	plt.bar(x_label, char_dict.values(), align='center', alpha=0.5)
	plt.xticks(x_label, char_dict.keys())
	plt.show()

	fig.savefig('./label_distribution.png')
	plt.close(fig)

def main():
	#generate_icdar2019_sroie_task1_train_text_line_data()

	Icdar2019SroieTextLineDataset_test()

	#check_label_distribution_of_icdar2019_sroie_task1_train_text_line_data()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
