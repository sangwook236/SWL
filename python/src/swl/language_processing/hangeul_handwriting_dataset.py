import os, csv
import numpy as np
import pandas as pd
import cv2

def generate_phd08_dataset_info(data_dir_path, phd08_conversion_result_filepath, phd08_npy_dataset_info_filepath):
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

		img = np.load(os.path.join(data_dir_path, data_filepath))
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

#--------------------------------------------------------------------

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

#--------------------------------------------------------------------

# REF [site] >> http://cv.jbnu.ac.kr/index.php?mid=notice&document_srl=189
def load_phd08_dataset(data_dir_path, dataset_info_filepath):
	dataset_dict = dict()
	with open(dataset_info_filepath, 'r', encoding='UTF8') as fd:
		reader = csv.reader(fd, delimiter=',')
		headers = next(reader, None)  # Read a header.
		for row in reader:
			data_filepath, label_filepath = os.path.join(data_dir_path, row[0]), os.path.join(data_dir_path, row[1])
			label_char, label_unicode, data_count, height, width = row[2], int(row[3]), int(row[4]), int(row[5]), int(row[6])

			data = np.load(data_filepath)
			#data, labels = np.load(data_filepath), np.load(label_filepath)
			#labels = np.argmax(labels, axis=-1)  # Labels are data indices.

			#if data_count != len(data) or data_count != len(labels):
			#	print('Invalid data length: {} != {} != {} in {}.'.format(data_count, len(data), len(labels), data_filepath))
			if data_count != len(data):
				print('Invalid data length: {} != {} in {}.'.format(data_count, len(data), data_filepath))
			if label_unicode != ord(label_char):
				print('Invalid label: {} != {} in {}.'.format(label_char, label_unicode, data_filepath))
			#if not all(labels == ord(label_char)):
			#	print('Invalid labels: {} != {} in {}.'.format(label_char, np.unique(labels), data_filepath))

			dataset_dict[label_char] = data

	return dataset_dict

# REF [site] >> https://github.com/callee2006/HangulDB
# HanDB dataset:
#	REF [file] >> HanDB_character_count.csv
#	#train data = 514201670, #test data = 6424572.
#		The number of data in each class(character) is very different.
#	#classes = 2350.
def load_handb_dataset(data_dir_path, dataset_info_filepath):
	dataset_dict = dict()
	# FIXME [delete] >>
	count_dict = dict()
	#with open(dataset_info_filepath, 'r', encoding='UTF8') as fd:
	with open(dataset_info_filepath, 'r') as fd:
		reader = csv.reader(fd, delimiter=',')
		for row in reader:
			data_filepath, label_filepath = os.path.join(data_dir_path, row[0]), os.path.join(data_dir_path, row[1])
			# Height and width are actual ones.
			data_idx, height, width, label_char = int(row[2]), int(row[3]), int(row[4]), row[5]

			# data = (num_examples, 100, 100, 1), labels = (num_examples, 1).
			#	Unicode is used as label.
			data, labels = np.load(data_filepath), np.load(label_filepath)
			labels = labels.reshape(labels.shape[:-1])  # Unicode.

			if len(data) != len(labels):
				print('Invalid data length: {} != {} in {}.'.format(len(data), len(labels), data_filepath))
			if not all(labels == ord(label_char)):
				print('Invalid labels: {} != {} in {}.'.format(label_char, np.unique(labels), data_filepath))

			# FIXME [restore] >>
			#if not label_char in dataset_dict:
			#	dataset_dict[label_char] = list()
			#dataset_dict[label_char].append(data)

			# FIXME [delete] >>
			if not label_char in count_dict:
				count_dict[label_char] = 0
			count_dict[label_char] += len(data)

	# FIXME [delete] >>
	for key, val in count_dict.items():
		print('{}, {}'.format(key, val))

	for key, val in dataset_dict.items():
		dataset_dict[key] = np.vstack(dataset_dict[key])

	return dataset_dict

# REF [site] >> https://github.com/callee2006/HangulDB
# PE92 dataset:
#	REF [file] >> PE92_character_count.csv
#	#train data = 17070923, #test data = 233119.
#		The number of data in each class(character) is different.
#	#classes = 2350.
def load_pe92_dataset(data_dir_path, dataset_info_filepath):
	return load_handb_dataset(data_dir_path, dataset_info_filepath)

# REF [site] >> https://github.com/callee2006/HangulDB
# SERI dataset:
#	REF [file] >> SERI_character_count.csv
#	#train data = 417032887, #test data = 5157335.
#		The number of data in each class(character) is different.
#	#classes = 520.
def load_seri_dataset(data_dir_path, dataset_info_filepath):
	return load_handb_dataset(data_dir_path, dataset_info_filepath)
