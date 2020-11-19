#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, math, itertools, functools, glob, json, time
import numpy as np
import torch, torchvision
import cv2
import labelme_data

def LabelMeDataset_test():
	image_channel = 3

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'

	json_filepaths = glob.glob(data_dir_path + '/*.json', recursive=False)
	print('#loaded JSON files = {}.'.format(len(json_filepaths)))

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	dataset = labelme_data.LabelMeDataset(json_filepaths, image_channel)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#files = {}.'.format(len(dataset)))

	if False:
		for idx, dat in enumerate(dataset):
			print('Data #{}:'.format(idx))
			print('\tversion = {}.'.format(dat['version']))
			print('\tflags = {}.'.format(dat['flags']))
			print('\tlineColor = {}.'.format(dat['lineColor']))
			print('\tfillColor = {}.'.format(dat['fillColor']))
			print('\timagePath = {}.'.format(dat['imagePath']))
			print('\timageData = {}.'.format(dat['imageData']))
			print('\timageWidth = {}.'.format(dat['imageWidth']))
			print('\timageHeight = {}.'.format(dat['imageHeight']))

			for sidx, shape in enumerate(dat['shapes']):
				print('\tShape #{}:'.format(sidx))
				print('\t\tlabel = {}.'.format(shape['label']))
				print('\t\tline_color = {}.'.format(shape['line_color']))
				print('\t\tfill_color = {}.'.format(shape['fill_color']))
				print('\t\tpoints = {}.'.format(shape['points']))
				print('\t\tgroup_id = {}.'.format(shape['group_id']))
				print('\t\tshape_type = {}.'.format(shape['shape_type']))

			if idx >= 2: break

	#--------------------
	num_shapes = functools.reduce(lambda nn, dat: nn + len(dat['shapes']), dataset, 0)
	print('#shapes = {}.'.format(num_shapes))

	shape_counts = dict()
	for dat in dataset:
		for shape in dat['shapes']:
			if shape['label'] in shape_counts:
				shape_counts[shape['label']] += 1
			else:
				shape_counts[shape['label']] = 1
	print('Shape labels = {}.'.format(list(shape_counts.keys())))
	print('#total examples = {}.'.format(sum(shape_counts.values())))
	print('#examples of each shape = {}.'.format(shape_counts))

def create_augmenter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	augmenter = iaa.Sequential([
		iaa.Sometimes(0.25,
			iaa.Grayscale(alpha=(0.0, 1.0)),  # Requires RGB images.
		),
		#iaa.Sometimes(0.5, iaa.OneOf([
		#	iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
		#	iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
		#	#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
		#	#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
		#])),
		iaa.Sometimes(0.8, iaa.Sequential([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # Translate by -5 to +5 percent along x-axis and -5 to +5 percent along y-axis.
				#rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				#shear=(-2, 2),  # Shear by -2 to +2 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			iaa.Sometimes(0.75, iaa.OneOf([
				iaa.Sequential([
					iaa.ShearX((-45, 45)),
					iaa.ShearY((-5, 5))
				]),
				iaa.PiecewiseAffine(scale=(0.01, 0.03)),  # Move parts of the image around. Slow.
				#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
				iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
			])),
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
			iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.GaussianBlur(sigma=(0.5, 1.5)),
			iaa.AverageBlur(k=(2, 4)),
			iaa.MedianBlur(k=(3, 3)),
			iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0), order=1),
		])),
		#iaa.Sometimes(0.8, iaa.OneOf([
		#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
		#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
		#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

		#	iaa.Invert(p=1, per_channel=False),

		#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
		#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
		#])),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

	return augmenter

class ConvertPILMode(object):
	def __init__(self, mode='RGB'):
		self.mode = mode

	def __call__(self, x):
		return x.convert(self.mode)

class ResizeImageToFixedSizeWithPadding(object):
	def __init__(self, height, width, is_pil=True):
		self.height, self.width = height, width
		self.resize_functor = self._resize_by_pil if is_pil else self._resize_by_opencv

	def __call__(self, x):
		return self.resize_functor(x, self.height, self.width)

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_opencv() in text_line_data.py.
	def _resize_by_opencv(self, input, height, width, *args, **kwargs):
		interpolation = cv2.INTER_AREA
		"""
		hi, wi = input.shape[:2]
		if wi >= width:
			return cv2.resize(input, (width, height), interpolation=interpolation)
		else:
			aspect_ratio = height / hi
			min_width = min(width, int(wi * aspect_ratio))
			input = cv2.resize(input, (min_width, height), interpolation=interpolation)
			if min_width < width:
				image_zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
				image_zeropadded[:,:min_width] = input[:,:min_width]
				return image_zeropadded
			else:
				return input
		"""
		hi, wi = input.shape[:2]
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
		zeropadded[:,:min_width] = cv2.resize(input, (min_width, height), interpolation=interpolation)
		return zeropadded
		"""
		return cv2.resize(input, (width, height), interpolation=interpolation)
		"""

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_pil() in text_line_data.py.
	def _resize_by_pil(self, input, height, width, *args, **kwargs):
		import PIL.Image

		interpolation = PIL.Image.BICUBIC
		wi, hi = input.size
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = PIL.Image.new(input.mode, (width, height), color=0)
		zeropadded.paste(input.resize((min_width, height), resample=interpolation), (0, 0, min_width, height))
		return zeropadded
		"""
		return input.resize((width, height), resample=interpolation)
		"""

class MySubsetDataset(torch.utils.data.Dataset):
	def __init__(self, subset, transform=None, target_transform=None):
		self.subset = subset
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, idx):
		inp, outp = self.subset[idx]
		if self.transform:
			inp = self.transform(inp)
		if self.target_transform:
			outp = self.target_transform(outp)
		return inp, outp

	def __len__(self):
		return len(self.subset)

def visualize_data(dataloader, num_data=None):
	data_iter = iter(dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	images = images.transpose(0, 2, 3, 1)
	for idx, (img, lbl) in enumerate(zip(images, labels)):
		print('Label: {}.'.format(lbl))
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		if num_data and idx >= (num_data - 1): break
	cv2.destroyAllWindows()

def visualize_detection_data(dataloader, num_data=None):
	colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

	data_iter = iter(dataloader)
	images, targets = data_iter.next()  # tuple of torch.Tensor's & tuple of dicts.
	for idx, (img, tgt) in enumerate(zip(images, targets)):
		img, boxes, labels = img.numpy().transpose(1, 2, 0), tgt['boxes'].numpy(), tgt['labels'].numpy()
		# NOTE [info] >> In order to deal with "TypeError: an integer is required (got type tuple)" error.
		img = np.ascontiguousarray(img)

		print('Labels: {}.'.format(labels))
		for ii, (left, top, right, bottom) in enumerate(boxes):
			# FIXME [fix] >> Need resizing.
			#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
			left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
			cv2.rectangle(img, (left, top), (right, bottom), colors[ii % len(colors)], 2, cv2.LINE_8)
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		if num_data and idx >= (num_data - 1): break
	cv2.destroyAllWindows()

def sminds_figure_data_worker_proc(json_filepath, data_dir_path, classes, flag, is_preloaded_image_used):
	try:
		with open(json_filepath, 'r') as fd:
			json_data = json.load(fd)
	except UnicodeDecodeError as ex:
		print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
		return None
	except FileNotFoundError as ex:
		print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
		return None

	if not json_data['shapes']:
		print('[SWL] Warning: No shape in a JSON file, {}.'.format(json_filepath))
		return None

	try:
		json_dir_path = os.path.dirname(json_filepath)
		image_filepath = os.path.join(json_dir_path, json_data['imagePath'])
		image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

		img = cv2.imread(image_filepath, flag)
		if img is None:
			print('[SWL] Warning: Failed to load an image, {}.'.format(image_filepath))
			return None
		if img.shape[0] != image_height or img.shape[1] != image_width:
			print('[SWL] Warning: Invalid image shape, ({}, {}) != ({}, {}).'.format(img.shape[0], img.shape[1], image_height, image_width))
			return None

		figures, labels = list(), list()
		for shape in json_data['shapes']:
			label, shape_type = shape['label'], shape['shape_type']
			if label not in classes:
				print('[SWL] Warning: Invalid label, {} in {}.'.format(label, json_filepath))
				return None
			if shape_type != 'rectangle':
				print('[SWL] Warning: Invalid shape type, {} in {}.'.format(shape_type, json_filepath))
				return None

			(left, top), (right, bottom) = shape['points']
			#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
			left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
			figures.append(img[top:bottom,left:right] if is_preloaded_image_used else (os.path.relpath(image_filepath, data_dir_path), (left, top, right, bottom)))
			labels.append([classes.index(label)])
		return figures, labels
	except KeyError as ex:
		print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
		return None, None

def sminds_figure_data_json_loading_test():
	def _load_data_from_json(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used):
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		figures, labels = list(), list()
		for json_filepath in json_filepaths:
			try:
				with open(json_filepath, 'r') as fd:
					json_data = json.load(fd)
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
				continue
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
				continue

			if not json_data['shapes']:
				print('[SWL] Warning: No shape in a JSON file, {}.'.format(json_filepath))
				continue

			try:
				json_dir_path = os.path.dirname(json_filepath)
				image_filepath = os.path.join(json_dir_path, json_data['imagePath'])
				image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

				img = cv2.imread(image_filepath, flag)
				if img is None:
					print('[SWL] Warning: Failed to load an image, {}.'.format(image_filepath))
					continue
				if img.shape[0] != image_height or img.shape[1] != image_width:
					print('[SWL] Warning: Invalid image shape, ({}, {}) != ({}, {}).'.format(img.shape[0], img.shape[1], image_height, image_width))
					continue

				for shape in json_data['shapes']:
					label, shape_type = shape['label'], shape['shape_type']
					if label not in classes:
						print('[SWL] Warning: Invalid label, {} in {}.'.format(label, json_filepath))
						continue
					if shape_type != 'rectangle':
						print('[SWL] Warning: Invalid shape type, {} in {}.'.format(shape_type, json_filepath))
						continue

					(left, top), (right, bottom) = shape['points']
					#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
					left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
					figures.append(img[top:bottom,left:right] if is_preloaded_image_used else (os.path.relpath(image_filepath, data_dir_path), (left, top, right, bottom)))
					labels.append([classes.index(label)])
			except KeyError as ex:
				print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
				figures.append(None)
				labels.append(None)

		return figures, labels

	def _load_data_from_json_async(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used):
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		#--------------------
		import multiprocessing as mp

		async_results = list()
		def async_callback(result):
			# This is called whenever sqr_with_sleep(i) returns a result.
			# async_results is modified only by the main process, not the pool workers.
			async_results.append(result)

		num_processes = 8
		#timeout = 10
		timeout = None
		#with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
		with mp.Pool(processes=num_processes) as pool:
			#results = pool.map_async(functools.partial(sminds_figure_data_worker_proc, data_dir_path=data_dir_path, classes=classes, flag=flag, is_preloaded_image_used=is_preloaded_image_used), json_filepaths)
			results = pool.map_async(functools.partial(sminds_figure_data_worker_proc, data_dir_path=data_dir_path, classes=classes, flag=flag, is_preloaded_image_used=is_preloaded_image_used), json_filepaths, callback=async_callback)

			results.get(timeout)

		async_results = list(res for res in async_results[0] if res is not None)
		figures, labels = zip(*async_results)
		figures, labels = list(itertools.chain(*figures)), list(itertools.chain(*labels))

		return figures, labels

	#--------------------
	image_channel = 3
	is_preloaded_image_used = False

	#classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear']
	classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear', 'table-all_unclear', 'table-partial_unclear', 'table-hv_unclear', 'table-bare_unclear', 'picture_unclear', 'diagram_unclear']

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	pkl_filepath = data_dir_path + '/sminds_figures.pkl'
	json_filepaths = glob.glob(data_dir_path + '/labelme_??/*.json', recursive=False)

	#--------------------
	print('Start loading SMinds figure data...')
	start_time = time.time()
	#figures, labels = _load_data_from_json(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used)
	figures, labels = _load_data_from_json_async(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used)
	print('End loading SMinds figure data: {} secs.'.format(time.time() - start_time))
	assert (figures is None and labels is None) or len(figures) == len(labels)

	#--------------------
	if True:
		# Save to a pickle file.
		import pickle
		print('Start saving SMinds figure data to {}...'.format(pkl_filepath))
		start_time = time.time()
		try:
			with open(pkl_filepath, 'wb') as fd:
				pickle.dump((figures, labels), fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(pkl_filepath, ex))
		print('End saving SMinds figure data: {} secs.'.format(time.time() - start_time))

def sminds_figure_data_pickle_loading_test():
	image_channel = 3
	is_preloaded_image_used = False

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	pkl_filepath = data_dir_path + '/sminds_figures.pkl'

	# Load from a pickle file.
	import pickle
	print('Start loading SMinds figure data from {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'rb') as fd:
			figures, labels = pickle.load(fd)
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pkl_filepath, ex))
	print('End loading SMinds figure data : {} secs.'.format(time.time() - start_time))
	assert (figures is None and labels is None) or len(figures) == len(labels)
	print('#loaded data = {}.'.format(len(figures)))

	#--------------------
	if True:
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		num_data = 10
		for idx, (fig, lbl) in enumerate(zip(figures, labels)):
			print('Label: {}.'.format(lbl))
			if is_preloaded_image_used:
				img = fig
			else:
				img_fpath, (left, top, right, bottom) = fig
				img_fpath = os.path.join(data_dir_path, img_fpath)
				img = cv2.imread(img_fpath, flag)
				#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
				left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
				img = img[top:bottom,left:right]
			cv2.imshow('Image', img)
			cv2.waitKey(0)
			if num_data and idx >= (num_data - 1): break
		cv2.destroyAllWindows()

def FigureLabelMeDataset_test():
	#classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear']
	classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear', 'table-all_unclear', 'table-partial_unclear', 'table-hv_unclear', 'table-bare_unclear', 'picture_unclear', 'diagram_unclear']

	image_height, image_width, image_channel = 512, 512, 3
	is_preloaded_image_used = False
	train_test_ratio = 0.8
	batch_size = 64
	shuffle = True
	num_workers = 4

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	pkl_filepath = data_dir_path + '/sminds_figures.pkl'

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(create_augmenter()),
		#ConvertPILMode(mode='RGB'),
		ResizeImageToFixedSizeWithPadding(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		#ConvertPILMode(mode='RGB'),
		ResizeImageToFixedSizeWithPadding(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = torch.IntTensor

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	dataset = labelme_data.FigureLabelMeDataset(pkl_filepath, image_channel, classes, is_preloaded_image_used)

	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#examples = {}, #train examples = {}, #test examples = {}.'.format(len(dataset), len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))
	print('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	#--------------------
	# Show data info.
	data_iter = iter(train_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	visualize_data(train_dataloader, num_data=10)
	visualize_data(test_dataloader, num_data=10)

def sminds_figure_detection_data_worker_proc(json_filepath, data_dir_path, classes, flag, is_preloaded_image_used):
	try:
		with open(json_filepath, 'r') as fd:
			json_data = json.load(fd)
	except UnicodeDecodeError as ex:
		print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
		return None
	except FileNotFoundError as ex:
		print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
		return None

	if not json_data['shapes']:
		print('[SWL] Warning: No shape in a JSON file, {}.'.format(json_filepath))
		return None

	try:
		json_dir_path = os.path.dirname(json_filepath)
		image_filepath = os.path.join(json_dir_path, json_data['imagePath'])
		image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

		img = cv2.imread(image_filepath, flag)
		if img is None:
			print('[SWL] Warning: Failed to load an image, {}.'.format(image_filepath))
			return None
		if img.shape[0] != image_height or img.shape[1] != image_width:
			print('[SWL] Warning: Invalid image shape, ({}, {}) != ({}, {}).'.format(img.shape[0], img.shape[1], image_height, image_width))
			return None

		bboxes, labels = list(), list()
		for shape in json_data['shapes']:
			label, shape_type = shape['label'], shape['shape_type']
			if label not in classes:
				print('[SWL] Warning: Invalid label, {} in {}.'.format(label, json_filepath))
				return None
			if shape_type != 'rectangle':
				print('[SWL] Warning: Invalid shape type, {} in {}.'.format(shape_type, json_filepath))
				return None

			(left, top), (right, bottom) = shape['points']
			#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
			left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
			bboxes.append((left, top, right, bottom))
			labels.append(classes.index(label))
		return [(img if is_preloaded_image_used else os.path.relpath(image_filepath, data_dir_path)), np.array(bboxes), np.array(labels)] if bboxes and labels else None
	except KeyError as ex:
		print('[SWL] Warning: Key error in a JSON file, {}: {}.'.format(json_filepath, ex))
		return None

def sminds_figure_detection_data_json_loading_test():
	def _load_data_from_json(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used):
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		figures = list()
		for json_filepath in json_filepaths:
			try:
				with open(json_filepath, 'r') as fd:
					json_data = json.load(fd)
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
				continue
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
				continue

			if not json_data['shapes']:
				print('[SWL] Warning: No shape in a JSON file, {}.'.format(json_filepath))
				continue

			try:
				json_dir_path = os.path.dirname(json_filepath)
				image_filepath = os.path.join(json_dir_path, json_data['imagePath'])
				image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

				img = cv2.imread(image_filepath, flag)
				if img is None:
					print('[SWL] Warning: Failed to load an image, {}.'.format(image_filepath))
					continue
				if img.shape[0] != image_height or img.shape[1] != image_width:
					print('[SWL] Warning: Invalid image shape, ({}, {}) != ({}, {}).'.format(img.shape[0], img.shape[1], image_height, image_width))
					continue

				bboxes, labels = list(), list()
				for shape in json_data['shapes']:
					label, shape_type = shape['label'], shape['shape_type']
					if label not in classes:
						print('[SWL] Warning: Invalid label, {} in {}.'.format(label, json_filepath))
						continue
					if shape_type != 'rectangle':
						print('[SWL] Warning: Invalid shape type, {} in {}.'.format(shape_type, json_filepath))
						continue

					(left, top), (right, bottom) = shape['points']
					#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
					left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
					bboxes.append([left, top, right, bottom])
					labels.append(classes.index(label))
				figures.append([img if is_preloaded_image_used else os.path.relpath(image_filepath, data_dir_path), np.array(bboxes), np.array(labels)] if bboxes and labels else None)
			except KeyError as ex:
				print('[SWL] Warning: Key error in a JSON file, {}: {}.'.format(json_filepath, ex))
				figures.append(None)

		return figures

	def _load_data_from_json_async(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used):
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		#--------------------
		import multiprocessing as mp

		async_results = list()
		def async_callback(result):
			# This is called whenever sqr_with_sleep(i) returns a result.
			# async_results is modified only by the main process, not the pool workers.
			async_results.append(result)

		num_processes = 8
		#timeout = 10
		timeout = None
		#with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
		with mp.Pool(processes=num_processes) as pool:
			#results = pool.map_async(functools.partial(sminds_figure_detection_data_worker_proc, data_dir_path=data_dir_path, classes=classes, flag=flag, is_preloaded_image_used=is_preloaded_image_used), json_filepaths)
			results = pool.map_async(functools.partial(sminds_figure_detection_data_worker_proc, data_dir_path=data_dir_path, classes=classes, flag=flag, is_preloaded_image_used=is_preloaded_image_used), json_filepaths, callback=async_callback)

			results.get(timeout)

		figures = list(res for res in async_results[0] if res is not None)
		return figures

	#--------------------
	image_channel = 3
	is_preloaded_image_used = False

	#classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear']
	classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear', 'table-all_unclear', 'table-partial_unclear', 'table-hv_unclear', 'table-bare_unclear', 'picture_unclear', 'diagram_unclear']

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	pkl_filepath = data_dir_path + '/sminds_figure_detection.pkl'
	json_filepaths = glob.glob(data_dir_path + '/labelme_??/*.json', recursive=False)

	#--------------------
	print('Start loading SMinds figure data...')
	start_time = time.time()
	#figures = _load_data_from_json(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used)
	figures = _load_data_from_json_async(json_filepaths, data_dir_path, image_channel, classes, is_preloaded_image_used)
	print('End loading SMinds figure data: {} secs.'.format(time.time() - start_time))

	#--------------------
	if True:
		# Save figure infos to a pickle file.
		import pickle
		print('Start saving SMinds figure detection data to {}...'.format(pkl_filepath))
		start_time = time.time()
		try:
			with open(pkl_filepath, 'wb') as fd:
				pickle.dump(figures, fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(pkl_filepath, ex))
		print('End saving SMinds figure detection data: {} secs.'.format(time.time() - start_time))

def sminds_figure_detection_data_pickle_loading_test():
	image_channel = 3
	is_preloaded_image_used = False
	
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	pkl_filepath = data_dir_path + '/sminds_figure_detection.pkl'

	# Load from a pickle file.
	import pickle
	print('Start loading SMinds figure detection data from {}...'.format(pkl_filepath))
	start_time = time.time()
	try:
		with open(pkl_filepath, 'rb') as fd:
			figure_detection_data = pickle.load(fd)
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pkl_filepath, ex))
	print('End loading SMinds figure detection data : {} secs.'.format(time.time() - start_time))
	print('#figure detection data = {}.'.format(len(figure_detection_data)))

	#--------------------
	if True:
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

		num_data = 10
		for idx, (img, boxes, labels) in enumerate(figure_detection_data):
			if not is_preloaded_image_used:
				img_fpath = os.path.join(data_dir_path, img)
				img = cv2.imread(img_fpath, flag)
				if img is None:
					print('File not found, {}.'.format(img_fpath))
					continue

			print('Labels: {}.'.format(labels))
			for ii, (left, top, right, bottom) in enumerate(boxes):
				#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
				left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
				cv2.rectangle(img, (left, top), (right, bottom), colors[ii % len(colors)], 2, cv2.LINE_8)
			cv2.imshow('Image', img)
			cv2.waitKey(0)
			if num_data and idx >= (num_data - 1): break
		cv2.destroyAllWindows()

def FigureDetectionLabelMeDataset_test():
	# NOTE [info] >> In order to deal with "OSError: [Errno 24] Too many open files" error. 
	torch.multiprocessing.set_sharing_strategy('file_system')

	#classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear']
	classes = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear', 'table-all_unclear', 'table-partial_unclear', 'table-hv_unclear', 'table-bare_unclear', 'picture_unclear', 'diagram_unclear']

	image_height, image_width, image_channel = 1024, 1024, 3
	is_preloaded_image_used = False
	train_test_ratio = 0.8
	batch_size = 64
	shuffle = True
	num_workers = 8

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	pkl_filepath = data_dir_path + '/sminds_figure_detection.pkl'

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(create_augmenter()),
		#ConvertPILMode(mode='RGB'),
		ResizeImageToFixedSizeWithPadding(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		#ConvertPILMode(mode='RGB'),
		ResizeImageToFixedSizeWithPadding(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	dataset = labelme_data.FigureDetectionLabelMeDataset(pkl_filepath, image_channel, classes)

	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#examples = {}, #train examples = {}, #test examples = {}.'.format(len(dataset), len(train_dataset), len(test_dataset)))

	#--------------------
	# REF [function] >> collate_fn() in https://github.com/pytorch/vision/tree/master/references/detection/utils.py
	def collate_fn(batch):
		return tuple(zip(*batch))

	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))
	print('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	#--------------------
	# Show data info.
	data_iter = iter(train_dataloader)
	images, targets = data_iter.next()  # tuple of torch.Tensor's & tuple of dicts.
	image0, target0 = images[0].numpy(), targets[0]
	bboxes0, labels0 = target0['boxes'].numpy(), target0['labels'].numpy()
	print('Train data: #images = {}, #targets = {}.'.format(len(images), len(targets)))
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(image0.shape, image0.dtype, np.min(image0), np.max(image0)))
	print('Train target: Keys = {}.'.format(list(target0.keys())))
	print("Train target: Boxes' shape = {}, boxes' dtype = {}, labels' shape = {}, labels' dtype = {}.".format(bboxes0.shape, bboxes0.dtype, labels0.shape, labels0.dtype))

	data_iter = iter(test_dataloader)
	images, targets = data_iter.next()  # tuple of torch.Tensor's & tuple of dicts.
	image0, target0 = images[0].numpy(), targets[0]
	bboxes0, labels0 = target0['boxes'].numpy(), target0['labels'].numpy()
	print('Test data: #images = {}, #targets = {}.'.format(len(images), len(targets)))
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(image0.shape, image0.dtype, np.min(image0), np.max(image0)))
	print('Test target: Keys = {}.'.format(list(target0.keys())))
	print("Test target: Boxes' shape = {}, boxes' dtype = {}, labels' shape = {}, labels' dtype = {}.".format(bboxes0.shape, bboxes0.dtype, labels0.shape, labels0.dtype))

	#--------------------
	# Visualize.
	visualize_detection_data(train_dataloader, num_data=10)
	visualize_detection_data(test_dataloader, num_data=10)

def main():
	#LabelMeDataset_test()

	#--------------------
	#sminds_figure_data_json_loading_test()
	#sminds_figure_data_pickle_loading_test()

	#FigureLabelMeDataset_test()

	#--------------------
	#sminds_figure_detection_data_json_loading_test()
	#sminds_figure_detection_data_pickle_loading_test()

	FigureDetectionLabelMeDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
