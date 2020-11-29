#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, math, random, functools, pickle, time
import numpy as np
import torch, torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util
import coco_text_data

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

class AugmentByImgaug(object):
	def __init__(self, augmenter, is_pil=True):
		if is_pil:
			self.augment_functor = lambda x: Image.fromarray(augmenter.augment_image(np.array(x)))
			#self.augment_functor = lambda x: Image.fromarray(augmenter.augment_images(np.array(x)))
		else:
			self.augment_functor = lambda x: augmenter.augment_image(x)
			#self.augment_functor = lambda x: augmenter.augment_images(x)

	def __call__(self, x):
		return self.augment_functor(x)

class RandomInvert(object):
	def __call__(self, x):
		return ImageOps.invert(x) if random.randrange(2) else x

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

		self.min_height_threshold, self.min_width_threshold = 20, 20
		self.warn = self._warn_about_small_image if warn_about_small_image else lambda *args, **kwargs: None

	def __call__(self, x):
		return self.resize_functor(x, self.height, self.width)

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_opencv() in text_line_data.py.
	def _resize_by_opencv(self, image, height, width, *args, **kwargs):
		interpolation = cv2.INTER_AREA
		"""
		hi, wi = image.shape[:2]
		if wi >= width:
			return cv2.resize(image, (width, height), interpolation=interpolation)
		else:
			scale_factor = height / hi
			#min_width = min(width, int(wi * scale_factor))
			min_width = max(min(width, int(wi * scale_factor)), height // 2)
			assert min_width > 0 and height > 0
			image = cv2.resize(image, (min_width, height), interpolation=interpolation)
			if min_width < width:
				image_zeropadded = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
				image_zeropadded[:,:min_width] = image[:,:min_width]
				return image_zeropadded
			else:
				return image
		"""
		hi, wi = image.shape[:2]
		self.warn(hi, wi)
		scale_factor = height / hi
		#min_width = min(width, int(wi * scale_factor))
		min_width = max(min(width, int(wi * scale_factor)), height // 2)
		assert min_width > 0 and height > 0
		zeropadded = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
		zeropadded[:,:min_width] = cv2.resize(image, (min_width, height), interpolation=interpolation)
		return zeropadded
		"""
		return cv2.resize(image, (width, height), interpolation=interpolation)
		"""

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_pil() in text_line_data.py.
	def _resize_by_pil(self, image, height, width, *args, **kwargs):
		interpolation = Image.BICUBIC
		wi, hi = image.size
		self.warn(hi, wi)
		scale_factor = height / hi
		#min_width = min(width, int(wi * scale_factor))
		min_width = max(min(width, int(wi * scale_factor)), height // 2)
		assert min_width > 0 and height > 0
		zeropadded = Image.new(image.mode, (width, height), color=0)
		zeropadded.paste(image.resize((min_width, height), resample=interpolation), (0, 0, min_width, height))
		return zeropadded
		"""
		return image.resize((width, height), resample=interpolation)
		"""

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
		inp, outp, outp_len = self.subset[idx]
		if self.transform:
			inp = self.transform(inp)
		if self.target_transform:
			outp = self.target_transform(outp)
		return inp, outp, outp_len

	def __len__(self):
		return len(self.subset)

def visualize_data_with_length(dataloader, label_converter, num_data=None):
	data_iter = iter(dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	images = images.transpose(0, 2, 3, 1)

	num_data = min(num_data, len(images), len(labels), len(label_lens)) if num_data else min(len(images), len(labels), len(label_lens))
	for img, lbl, l in random.sample(list(zip(images, labels, label_lens)), num_data):
		print('Label (len={}): {} (int), {} (str).'.format(l, [ll for ll in lbl if ll != label_converter.pad_id], label_converter.decode(lbl)))
		cv2.imshow('Image', img)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

def coco_text_json_loading_test():
	import json

	image_channel = 3

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	coco_text_dir_path = data_base_dir_path + '/text/scene_text'
	json_filepath = coco_text_dir_path + '/cocotext.v2.json'
	pickle_filepath = coco_text_dir_path + '/cocotext.v2.pkl'

	coco_dir_path = data_base_dir_path + '/coco/train2014'

	#--------------------
	# Show information on data.
	if True:
		try:
			print('Start loading a COCO-Text data from {}...'.format(json_filepath))
			start_time = time.time()
			with open(json_filepath, encoding='UTF8') as fd:
				coco_text = json.load(fd)
			print('End loading a COCO-Text data: {} secs.'.format(time.time() - start_time))
		except UnicodeDecodeError as ex:
			print('Unicode decode error in {}: {}.'.format(json_filepath, ex))
			return
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(json_filepath, ex))
			return

		#print('Dataset: {}.'.format(coco_text))
		print('Dataset keys = {}.'.format(list(coco_text.keys())))  # {'imgs', 'anns', 'cats', 'imgToAnns', 'info'}.
		print('Info: {}.'.format(coco_text['info']))
		print('Category: {}.'.format(coco_text['cats']))
		#print('Image: {}.'.format(coco_text['imgs']))
		#print('Annotation: {}.'.format(coco_text['anns']))
		#print('Image to Annotation: {}.'.format(coco_text['imgToAnns']))

		print('Image keys = {}.'.format(list(coco_text['imgs'][list(coco_text['imgs'].keys())[0]].keys())))  # {'id', 'file_name', 'width', 'height', 'set'}.
		print('Annotation keys = {}.'.format(list(coco_text['anns'][list(coco_text['anns'].keys())[0]].keys())))  # {'id', 'mask', 'class', 'bbox', 'image_id', 'language', 'area', 'utf8_string', 'legibility'}.

		print("Set of values of 'set' key in images = {}.".format(set(list(val['set'] for val in coco_text['imgs'].values()))))  # {'train', 'val'}.
		print("Set of values of 'language' key in annotations = {}.".format(set(list(val['language'] for val in coco_text['anns'].values()))))  # {'english', 'not english'}.
		print("Set of values of 'class' key in annotations = {}.".format(set(list(val['class'] for val in coco_text['anns'].values()))))  # {'machine printed', 'handwritten'}.
		print("Set of values of 'legibility' key in annotations = {}.".format(set(list(val['legibility'] for val in coco_text['anns'].values()))))  # {'legible', 'illegible'}.

		#images = 53686.
		#annotations = 201126.
		print('#images = {}.'.format(len(coco_text['imgs'])))
		print('\t#train images = {}.'.format(len(list(val for val in coco_text['imgs'].values() if val['set'] == 'train'))))
		print('\t#validation images = {}.'.format(len(list(val for val in coco_text['imgs'].values() if val['set'] == 'val'))))
		print('#annotations = {}.'.format(len(coco_text['anns'])))
		print('#image-to-annotations = {}.'.format(len(coco_text['imgToAnns'])))

		for key, val in coco_text['imgs'].items():
			if int(key) != val['id']:
				print('Unmatched image key, {}: {}.'.format(key, val))
		for key, val in coco_text['anns'].items():
			if int(key) != val['id']:
				print('Unmatched annotation key, {}: {}.'.format(key, val))
		for key, val in coco_text['imgToAnns'].items():
			if key not in coco_text['imgs']:
				print('Invalid image key, {}: {}.'.format(key, val))

		print('Max text length of annotations = {}.'.format(functools.reduce(lambda ll, val: max(ll, len(val['utf8_string'])), coco_text['anns'].values(), 0)))
		print('Min and max IDs of images = ({}, {}).'.format(min(list(int(key) for key in coco_text['imgs'].keys())), max(list(int(key) for key in coco_text['imgs'].keys()))))
		print('Min and max IDs of annotations = ({}, {}).'.format(min(list(int(key) for key in coco_text['anns'].keys())), max(list(int(key) for key in coco_text['anns'].keys()))))

		print('Image = {}.'.format(coco_text['imgs'][list(coco_text['imgs'].keys())[0]]))
		print('Annotation = {}.'.format(coco_text['anns'][list(coco_text['anns'].keys())[0]]))
		print('Annotation in an image = {}.'.format(coco_text['imgToAnns'][list(coco_text['imgs'].keys())[2]]))

	#--------------------
	# Check data.
	for key, image_info in coco_text['imgs'].items():
		if 'train2014' not in image_info['file_name']:
			print('Unexpected tag, {}.'.format(image_info['file_name']))

	#--------------------
	# Save to a pickle file.
	if True:
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		pkl_dir_path = os.path.dirname(pickle_filepath)

		print('Start extracting from COCO-Text dataset...')
		start_time = time.time()
		data_dicts = list()
		for key, image_info in coco_text['imgs'].items():
			img_fpath = image_info['file_name']
			image_id = image_info['id']
			image_height, image_width = image_info['height'], image_info['width']
			tag = image_info['set']

			img_fpath = os.path.join(coco_dir_path, img_fpath)

			img = cv2.imread(img_fpath, flag)
			if img is None:
				print('[SWL] Warning: Failed to load an image, {}.'.format(img_fpath))
				continue
			if img.shape[0] != image_height or img.shape[1] != image_width:
				print('[SWL] Warning: Invalid image shape, ({}, {}) != ({}, {}).'.format(img.shape[0], img.shape[1], image_height, image_width))
				continue

			annotation_ids = coco_text['imgToAnns'][str(image_id)]
			annotations = list()
			for ann_id in annotation_ids:
				coco_text_ann = coco_text['anns'][str(ann_id)]

				annotation = {
					'bbox': coco_text_ann['bbox'],  # [left, right, width, height].
					'mask': coco_text_ann['mask'],  # [x1, y1, ..., xn, yn].
					'area': coco_text_ann['area'],
					'utf8_string': coco_text_ann['utf8_string'],
					'language': coco_text_ann['language'],
					'class': coco_text_ann['class'],
					'legibility': coco_text_ann['legibility'],
				}
				annotations.append(annotation)

			data_dict = {
				#'file_name': img_fpath,
				#'file_name': image_info['file_name'],
				'file_name': os.path.relpath(img_fpath, pkl_dir_path),
				'height': image_height,
				'width': image_width,
				'image_id': image_id,
				'annotations': annotations
			}
			data_dicts.append(data_dict)
		print('End extracting from COCO-Text dataset: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('Start saving COCO-Text dataset to {}...'.format(pickle_filepath))
		start_time = time.time()
		try:
			with open(pickle_filepath, 'wb') as fd:
				pickle.dump(data_dicts, fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(pickle_filepath, ex))
		print('End saving COCO-Text dataset: {} secs.'.format(time.time() - start_time))

def coco_text_pickle_loading_test():
	image_channel = 3

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	coco_text_dir_path = data_base_dir_path + '/text/scene_text'
	pickle_filepath = coco_text_dir_path + '/cocotext.v2.pkl'

	coco_dir_path = data_base_dir_path + '/coco/train2014'

	print('Start loading COCO-Text dataset from {}...'.format(pickle_filepath))
	start_time = time.time()
	try:
		with open(pickle_filepath, 'rb') as fd:
			data_dicts = pickle.load(fd)
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pickle_filepath, ex))
		data_dicts = None
	print('End loading COCO-Text dataset: {} secs.'.format(time.time() - start_time))
	assert data_dicts
	print('#loaded data = {}.'.format(len(data_dicts)))

	#--------------------
	# Visualize data.
	if True:
		#data_dir_path = coco_dir_path
		data_dir_path = os.path.dirname(pickle_filepath)
	
		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		num_data_to_visualize = 20
		for dat in random.sample(data_dicts, min(num_data_to_visualize, len(data_dicts))):
			img_fpath = dat['file_name']
			#image_height, image_width = dat['height'], dat['width']
			#image_id = dat['image_id']
			annotations = dat['annotations']

			img_fpath = os.path.join(data_dir_path, img_fpath)
			img = cv2.imread(img_fpath, flag)
			if img is None:
				print('File not found, {}.'.format(img_fpath))
				continue

			print('Labels = {}.'.format(list(annotation['utf8_string'] for annotation in annotations)))
			for annotation in annotations:
				bbox = annotation['bbox']
				mask = annotation['mask']
				#area = annotation['area']
				utf8_string = annotation['utf8_string']
				#language = annotation['language']
				#class = annotation['class']
				#legibility = annotation['legibility']

				left, top, right, bottom = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
				#assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0], ((left, top, right, bottom), img.shape, img_fpath)
				left, top, right, bottom = max(math.floor(left), 0), max(math.floor(top), 0), min(math.ceil(right), img.shape[1] - 1), min(math.ceil(bottom), img.shape[0] - 1)
				cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2, cv2.LINE_8)
				mask = np.expand_dims(np.round(np.array(list(mask[si:si + 2] for si in range(0, len(mask), 2)))).astype(np.int), axis=1)
				overlay = np.zeros_like(img)
				cv2.drawContours(overlay, [mask], 0, (0, 0, 255), cv2.FILLED, cv2.LINE_8)
				img = cv2.addWeighted(img, 1.0, overlay, 0.25, 0)
				"""
				keypoints = list(keypoints[ki:ki + 3] for ki in range(0, len(keypoints), 3))
				for x, y, visibility in keypoints:
					#assert x >= 0 and y >= 0 and x <= img.shape[1] and y <= img.shape[0], ((x, y), (img.shape))
					cv2.circle(img, (round(x), round(y)), 2, (0, 0, 255), 2, cv2.FILLED)
				"""
			cv2.imshow('Image', img)
			cv2.waitKey(0)
		cv2.destroyAllWindows()

# REF [function] >> FigureDetectronDataset_test() in ${SWL_PYTHON_HOME}/test/language_processing/figure_data_test.py
def CocoTextDataset_test():
	image_height, image_width, image_channel = 64, 640, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	max_text_len = 65
	batch_size = 64
	shuffle = True
	num_workers = 8

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	coco_text_dir_path = data_base_dir_path + '/text/scene_text'
	coco_dir_path = data_base_dir_path + '/coco/train2014'

	if True:
		json_filepath = coco_text_dir_path + '/cocotext.v2.json'
		data_dir_path = coco_dir_path
	else:
		pickle_filepath = coco_text_dir_path + '/cocotext.v2.pkl'
		data_dir_path = os.path.dirname(pickle_filepath)

	#--------------------
	charset = tg_util.construct_charset(space=False, hangeul=True)

	label_converter = swl_langproc_util.TokenConverter(list(charset))
	#print('Classes:\n{}.'.format(label_converter.tokens))
	print('#classes = {}.'.format(label_converter.num_tokens))
	print('<PAD> = {}. <UNK> = {}.'.format(label_converter.pad_id, label_converter.encode([label_converter.UNKNOWN], is_bare_output=True)[0]))

	#--------------------
	train_transform = torchvision.transforms.Compose([
		AugmentByImgaug(create_imgaug_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = torch.IntTensor

	#--------------------
	print('Start creating COCO-Text datasets...')
	start_time = time.time()
	if True:
		train_dataset = coco_text_data.CocoTextDataset('train', json_filepath, data_dir_path, label_converter, image_channel, max_text_len, transform=train_transform, target_transform=train_target_transform)
		test_dataset = coco_text_data.CocoTextDataset('val', json_filepath, data_dir_path, label_converter, image_channel, max_text_len, transform=test_transform, target_transform=test_target_transform)
	else:
		train_dataset = coco_text_data.CocoTextDataset('train', pickle_filepath, data_dir_path, label_converter, image_channel, max_text_len, transform=train_transform, target_transform=train_target_transform)
		test_dataset = coco_text_data.CocoTextDataset('val', pickle_filepath, data_dir_path, label_converter, image_channel, max_text_len, transform=test_transform, target_transform=test_target_transform)
	print('End creating COCO-Text datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	# REF [function] >> collate_fn() in https://github.com/pytorch/vision/tree/master/references/detection/utils.py
	print('Start creating COCO-Text data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating COCO-Text data loaders: {} secs.'.format(time.time() - start_time))
	print('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	#--------------------
	# Show data info.
	data_iter = iter(train_dataloader)
	images, labels, label_lens = data_iter.next()
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Train label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	data_iter = iter(test_dataloader)
	images, labels, label_lens = data_iter.next()
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Test label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	#--------------------
	# Visualize.
	visualize_data_with_length(train_dataloader, label_converter, num_data=10)
	visualize_data_with_length(test_dataloader, label_converter, num_data=10)

def main():
	#coco_text_json_loading_test()
	#coco_text_pickle_loading_test()

	CocoTextDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
