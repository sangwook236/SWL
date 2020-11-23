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

def create_imgaug_augmenter():
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
				rotate=(-5, 5),  # Rotate by -5 to +5 degrees.
				#shear=(-5, 5),  # Shear by -5 to +5 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			iaa.Sometimes(0.75, iaa.OneOf([
				iaa.Sequential([
					iaa.ShearX((-5, 5)),
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

class AugmentByImgaug(object):
	def __init__(self, augmenter, is_pil=True):
		if is_pil:
			import PIL.Image
			self.augment_functor = lambda x: PIL.Image.fromarray(augmenter.augment_image(np.array(x)))
			#self.augment_functor = lambda x: PIL.Image.fromarray(augmenter.augment_images(np.array(x)))
		else:
			self.augment_functor = lambda x: augmenter.augment_image(x)
			#self.augment_functor = lambda x: augmenter.augment_images(x)

	def __call__(self, x):
		return self.augment_functor(x)

class ConvertPILMode(object):
	def __init__(self, mode='RGB'):
		self.mode = mode

	def __call__(self, x):
		return x.convert(self.mode)

class ResizeToFixedSize(object):
	def __init__(self, height, width, warn_about_small_image=False, is_pil=True, logger=None):
		self.height, self.width = height, width
		self.resize_functor = self._resize_by_pil if is_pil else self._resize_by_opencv
		self.logger = logger

		self.min_height_threshold, self.min_width_threshold = 30, 30
		self.warn = self._warn_about_small_image if warn_about_small_image else lambda *args, **kwargs: None

	def __call__(self, x):
		return self.resize_functor(x, self.height, self.width)

	@staticmethod
	def _compute_scale_factor(canvas_height, canvas_width, image_height, image_width, max_scale_factor=3, re_scale_factor=0.5):
		h_scale_factor, w_scale_factor = canvas_height / image_height, canvas_width / image_width
		#scale_factor = min(h_scale_factor, w_scale_factor)
		scale_factor = min(h_scale_factor, w_scale_factor, max_scale_factor)
		#return scale_factor, scale_factor
		return max(scale_factor, min(h_scale_factor, re_scale_factor)), max(scale_factor, min(w_scale_factor, re_scale_factor))

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_opencv() in ${SWL_PYTHON_HOME}/test/language_processing/text_line_data.py.
	def _resize_by_opencv(self, image, canvas_height, canvas_width, *args, **kwargs):
		min_height, min_width = canvas_height // 2, canvas_width // 2

		image_height, image_width = image.shape[:2]
		self.warn(image_height, image_width)
		image_height, image_width = max(image_height, 1), max(image_width, 1)

		h_scale_factor, w_scale_factor = self._compute_scale_factor(canvas_height, canvas_width, image_height, image_width)

		#tgt_height, tgt_width = image_height, canvas_width
		tgt_height, tgt_width = int(image_height * h_scale_factor), int(image_width * w_scale_factor)
		#tgt_height, tgt_width = max(int(image_height * h_scale_factor), min_height), max(int(image_width * w_scale_factor), min_width)
		assert tgt_height > 0 and tgt_width > 0

		zeropadded = np.zeros((canvas_height, canvas_width) + image.shape[2:], dtype=image.dtype)
		zeropadded[:tgt_height,:tgt_width] = cv2.resize(image, (tgt_width, tgt_height), interpolation=cv2.INTER_AREA)
		return zeropadded

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_pil() in ${SWL_PYTHON_HOME}/test/language_processing/text_line_data.py.
	def _resize_by_pil(self, image, canvas_height, canvas_width, *args, **kwargs):
		min_height, min_width = canvas_height // 2, canvas_width // 2

		image_width, image_height = image.size
		self.warn(image_height, image_width)
		image_height, image_width = max(image_height, 1), max(image_width, 1)

		h_scale_factor, w_scale_factor = self._compute_scale_factor(canvas_height, canvas_width, image_height, image_width)

		#tgt_height, tgt_width = image_height, canvas_width
		tgt_height, tgt_width = int(image_height * h_scale_factor), int(image_width * w_scale_factor)
		#tgt_height, tgt_width = max(int(image_height * h_scale_factor), min_height), max(int(image_width * w_scale_factor), min_width)
		assert tgt_height > 0 and tgt_width > 0

		import PIL.Image
		zeropadded = PIL.Image.new(image.mode, (canvas_width, canvas_height), color=0)
		zeropadded.paste(image.resize((tgt_width, tgt_height), resample=PIL.Image.BICUBIC), (0, 0, tgt_width, tgt_height))
		return zeropadded

	def _warn_about_small_image(self, height, width):
		if height < self.min_height_threshold:
			if self.logger: self.logger.warning('Too small image: The image height {} should be larger than or equal to {}.'.format(height, self.min_height_threshold))
		#if width < self.min_width_threshold:
		#	if self.logger: self.logger.warning('Too small image: The image width {} should be larger than or equal to {}.'.format(width, self.min_width_threshold))

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

class MyPairSubsetDataset(torch.utils.data.Dataset):
	def __init__(self, subset, transform=None):
		self.subset = subset
		self.transform = transform

	def __getitem__(self, idx):
		inp, outp = self.subset[idx]
		if self.transform:
			inp, outp = self.transform(inp, outp)
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
		if img is None:
			print('Invalid image: image = {}.'.format(img))
			continue
		#img, boxes, keypoints, labels = img.numpy().transpose(1, 2, 0), tgt['boxes'].numpy(), tgt['keypoints'].numpy(), tgt['labels'].numpy()
		img, boxes, keypoints, labels = img.numpy().transpose(1, 2, 0), tgt['boxes'], tgt['keypoints'], tgt['labels']
		# NOTE [info] >> In order to deal with "TypeError: an integer is required (got type tuple)" error.
		img = np.ascontiguousarray(img)

		print('Labels: {}.'.format(labels))
		for ii, (left, top, right, bottom) in enumerate(boxes):
			#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), (img.shape))
			cv2.rectangle(img, (math.floor(left), math.floor(top)), (math.ceil(right), math.ceil(bottom)), colors[ii % len(colors)], 2, cv2.LINE_8)
		for ii, pts in enumerate(keypoints):
			for x, y, visibility in pts:
				#assert x >= 0 and y >= 0 and x <= img.shape[1] and y <= img.shape[0], ((x, y), (img.shape))
				cv2.circle(img, (math.floor(x), math.floor(y)), 2, colors[ii % len(colors)], 2, cv2.FILLED)
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		if num_data and idx >= (num_data - 1): break
	cv2.destroyAllWindows()

def LabelMeDataset_test():
	image_channel = 3

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'
	json_filepaths = sorted(glob.glob(data_dir_path + '/labelme_??/*.json', recursive=False))
	print('#JSON files = {}.'.format(len(json_filepaths)))

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
	print('Shape labels = {}.'.format(sorted(shape_counts.keys())))
	print('#total examples = {}.'.format(sum(shape_counts.values())))
	print('#examples of each shape label = {}.'.format({k: v for k, v in sorted(shape_counts.items())}))

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
			left, right, top, bottom = min(left, right), max(left, right), min(top, bottom), max(top, bottom)
			#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0] and left < right and top < bottom, ((left, top, right, bottom), (img.shape))
			left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
			figures.append(img[top:bottom,left:right] if is_preloaded_image_used else (os.path.relpath(image_filepath, data_dir_path), (left, top, right, bottom)))
			labels.append([classes.index(label)])
		return figures, labels
	except KeyError as ex:
		print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
		return None, None

def sminds_figure_data_json_loading_test():
	def _load_data_from_json_files(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used):
		data = list(sminds_figure_data_worker_proc(json_filepath, data_dir_path, classes, flag, is_preloaded_image_used) for json_filepath in json_filepaths)
		data = list(dat for dat in data if dat is not None)
		figures, labels = zip(*data)
		figures, labels = list(itertools.chain(*figures)), list(itertools.chain(*labels))
		return figures, labels

	def _load_data_from_json_files_async(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used):
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
	json_filepaths = sorted(glob.glob(data_dir_path + '/labelme_??/*.json', recursive=False))
	print('#JSON files = {}.'.format(len(json_filepaths)))

	#--------------------
	if 1 == image_channel:
		flag = cv2.IMREAD_GRAYSCALE
	elif 3 == image_channel:
		flag = cv2.IMREAD_COLOR
	elif 4 == image_channel:
		flag = cv2.IMREAD_ANYCOLOR  # ?
	else:
		flag = cv2.IMREAD_UNCHANGED

	print('Start loading SMinds figure data...')
	start_time = time.time()
	#figures, labels = _load_data_from_json_files(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used)
	figures, labels = _load_data_from_json_files_async(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used)
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
		AugmentByImgaug(create_imgaug_augmenter()),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width),
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
			left, right, top, bottom = min(left, right), max(left, right), min(top, bottom), max(top, bottom)
			#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0] and left < right and top < bottom, ((left, top, right, bottom), (img.shape))
			left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
			bboxes.append((left, top, right, bottom))
			labels.append(classes.index(label))
		return [(img if is_preloaded_image_used else os.path.relpath(image_filepath, data_dir_path)), np.array(bboxes), np.array(labels)] if bboxes and labels else None
	except KeyError as ex:
		print('[SWL] Warning: Key error in a JSON file, {}: {}.'.format(json_filepath, ex))
		return None

def sminds_figure_detection_data_json_loading_test():
	def _load_data_from_json_files(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used):
		figures = list(sminds_figure_detection_data_worker_proc(json_filepath, data_dir_path, classes, flag, is_preloaded_image_used) for json_filepath in json_filepaths)
		return list(fig for fig in figures if fig is not None)

	def _load_data_from_json_files_async(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used):
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
	json_filepaths = sorted(glob.glob(data_dir_path + '/labelme_??/*.json', recursive=False))
	print('#JSON files = {}.'.format(len(json_filepaths)))

	#--------------------
	if 1 == image_channel:
		flag = cv2.IMREAD_GRAYSCALE
	elif 3 == image_channel:
		flag = cv2.IMREAD_COLOR
	elif 4 == image_channel:
		flag = cv2.IMREAD_ANYCOLOR  # ?
	else:
		flag = cv2.IMREAD_UNCHANGED

	print('Start loading SMinds figure data...')
	start_time = time.time()
	#figures = _load_data_from_json_files(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used)
	figures = _load_data_from_json_files_async(json_filepaths, data_dir_path, classes, flag, is_preloaded_image_used)
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

	image_height, image_width, image_channel = 512, 512, 3
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
	import swl.machine_learning.pair_transforms as pair_transforms
	train_transform = pair_transforms.Compose([
		pair_transforms.AugmentByImgaug(create_imgaug_augmenter()),
		#pair_transforms.ConvertPILMode(mode='RGB'),
		pair_transforms.ResizeToFixedSize(image_height, image_width),
		pair_transforms.ToTensor(),
		#pair_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = pair_transforms.Compose([
		#pair_transforms.ConvertPILMode(mode='RGB'),
		#pair_transforms.ResizeToFixedSize(image_height, image_width),
		pair_transforms.ToTensor(),
		#pair_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	dataset = labelme_data.FigureDetectionLabelMeDataset(pkl_filepath, image_channel, classes, is_preloaded_image_used)

	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)
	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MyPairSubsetDataset(train_subset, transform=train_transform)
	test_dataset = MyPairSubsetDataset(test_subset, transform=test_transform)
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
	#bboxes0, labels0 = target0['boxes'].numpy(), target0['labels'].numpy()
	bboxes0, labels0 = target0['boxes'], target0['labels']
	print('Train data: #images = {}, #targets = {}.'.format(len(images), len(targets)))
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(image0.shape, image0.dtype, np.min(image0), np.max(image0)))
	print('Train target: Keys = {}.'.format(list(target0.keys())))
	print("Train target: Boxes' shape = {}, boxes' dtype = {}, labels' shape = {}, labels' dtype = {}.".format(bboxes0.shape, bboxes0.dtype, labels0.shape, labels0.dtype))

	data_iter = iter(test_dataloader)
	images, targets = data_iter.next()  # tuple of torch.Tensor's & tuple of dicts.
	image0, target0 = images[0].numpy(), targets[0]
	#bboxes0, labels0 = target0['boxes'].numpy(), target0['labels'].numpy()
	bboxes0, labels0 = target0['boxes'], target0['labels']
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
