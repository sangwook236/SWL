import csv
import numpy as np
import pandas as pd
import cv2

def generate_phd08_dataset_info(phd08_conversion_result_filepath, phd08_npy_dataset_info_filepath):
	with open(phd08_conversion_result_filepath, 'r', encoding='UTF8') as fd:
		lines = fd.readlines()

	dataset_info_list = list()

	#line_idx = 0
	line_idx = 1  # Removes the first line.
	while True:
		first_line, second_line = lines[line_idx], lines[line_idx + 1]

		end_idx = first_line.find('.txt')
		start_idx = end_idx - 1
		assert start_idx >= 0 and end_idx < len(first_line)
		label = first_line[start_idx:end_idx]

		start_idx = second_line.rfind(':') + 2
		assert start_idx >= 0 and start_idx < len(second_line)
		data_filepath = second_line[start_idx:].strip('\n') + '.npy'
		label_filepath = data_filepath.replace('data', 'labels')

		img = np.load(data_filepath)
		dataset_info_list.append((data_filepath, label_filepath, label, ord(label)) + img.shape)

		line_idx += 2
		#if line_idx >= len(lines):
		if line_idx >= len(lines) - 1:
			break

	with open(phd08_npy_dataset_info_filepath, 'w', newline='', encoding='UTF8') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(('data_filepath', 'label_filepath', 'label', 'label_unicode', 'data_count', 'height', 'width'))  # Writes a header.
		for info in dataset_info_list:
			writer.writerow(info)

def load_phd08_image(phd08_image_dataset_info_filepath, is_dark_background=False):
	"""
	with open(phd08_image_dataset_info_filepath, 'r', encoding='UTF8') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		#reader = csv.DictReader(csvfile, delimiter=',')
		# Row: filepath, label, label_unicode, height, width, channels.
		for row in reader:
			print(row)
	"""
	df = pd.read_csv(phd08_image_dataset_info_filepath)
	#labels = pd.unique(df.label)
	#labels = pd.unique(df.label_unicode)

	#df['image'] = None
	letter_dict = dict()
	for row in df.iterrows():
		img_filepath, label, label_unicode, height, width, channels = row[1]
		if not label in letter_dict:
			letter_dict[label] = list()
		img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
		if img is None:
			print('Failed to load an image:', img_filepath)
			continue
		if img.shape != (height, width, channels):
			print('Different image shape: {} != {} in {}'.format(img.shape, (height, width, channels), img_filepath))
		if is_dark_background:
			img = cv2.bitwise_not(img)
		letter_dict[label].append(img)

	return letter_dict

def load_phd08_npy(phd08_npy_dataset_info_filepath, is_dark_background=False):
	"""
	with open(phd08_npy_dataset_info_filepath, 'r', encoding='UTF8') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		#reader = csv.DictReader(csvfile, delimiter=',')
		# Row: data_filepath, label_filepath, label, label_unicode, data_count, height, width.
		for row in reader:
			print(row)
	"""
	df = pd.read_csv(phd08_npy_dataset_info_filepath)
	#labels = pd.unique(df.label)
	#labels = pd.unique(df.label_unicode)

	#df['image'] = None
	letter_dict = dict()
	for row in df.iterrows():
		data_filepath, label_filepath, label, label_unicode, data_count, height, width = row[1]
		#data_npy, label_npy = np.load(data_filepath), np.load(label_filepath)
		data_npy = np.load(data_filepath)
		if data_npy is None:
			print('Failed to load an image:', data_filepath)
			continue
		if data_npy.shape != (data_count, height, width):
			print('Different image shape: {} != {} in {}'.format(data_npy.shape, (data_count, height, width), data_filepath))
		if not is_dark_background:
			data_npy = cv2.bitwise_not(data_npy)
		#if not label in letter_dict:
		#	letter_dict[label] = list()
		#for ii in range(len(data_npy)):
		#	letter_dict[label].append(data_npy[ii])
		letter_dict[label] = data_npy.astype(np.uint8)

	return letter_dict
