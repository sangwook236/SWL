#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, random, functools, itertools, shutil, glob, datetime, time
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util
import text_data
#import mixup.vgg, mixup.resnet

def save_model(model_filepath, model):
	#torch.save(model.state_dict(), model_filepath)
	torch.save({'state_dict': model.state_dict()}, model_filepath)
	print('Saved a model to {}.'.format(model_filepath))

def load_model(model_filepath, model, device='cpu'):
	loaded_data = torch.load(model_filepath, map_location=device)
	#model.load_state_dict(loaded_data)
	model.load_state_dict(loaded_data['state_dict'])
	print('Loaded a model from {}.'.format(model_filepath))
	return model

# REF [function] >> construct_font() in font_test.py.
def construct_font(korean=True, english=True):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	font_dir_paths = list()
	if korean:
		font_dir_paths.append(font_base_dir_path + '/kor')
		#font_dir_paths.append(font_base_dir_path + '/kor_small')
		#font_dir_paths.append(font_base_dir_path + '/kor_large')
		#font_dir_paths.append(font_base_dir_path + '/kor_receipt')
	if english:
		font_dir_paths.append(font_base_dir_path + '/eng')
		#font_dir_paths.append(font_base_dir_path + '/eng_small')
		#font_dir_paths.append(font_base_dir_path + '/eng_large')
		#font_dir_paths.append(font_base_dir_path + '/eng_receipt')

	return tg_util.construct_font(font_dir_paths)

def create_char_augmenter():
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
		iaa.Sometimes(0.8, iaa.OneOf([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent along x-axis and -20 to +20 percent along y-axis.
				rotate=(-30, 30),  # Rotate by -10 to +10 degrees.
				shear=(-10, 10),  # Shear by -10 to +10 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
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

def create_word_augmenter():
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
		iaa.Sometimes(0.8, iaa.OneOf([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent along x-axis and -10 to +10 percent along y-axis.
				rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
				shear=(-5, 5),  # Shear by -5 to +5 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
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

def create_text_line_augmenter():
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
		iaa.Sometimes(0.8, iaa.OneOf([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # Translate by -5 to +5 percent along x-axis and -5 to +5 percent along y-axis.
				rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				shear=(-2, 2),  # Shear by -2 to +2 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
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

def generate_font_colors(image_depth):
	#random_val = random.randrange(1, 255)

	#font_color = (255,) * image_depth  # White font color.
	font_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB font color.
	#font_color = (random.randrange(256),) * image_depth  # A grayscale font color.
	#font_color = (random_val,) * image_depth  # A grayscale font color.
	#font_color = (random.randrange(random_val),) * image_depth  # A darker grayscale font color.
	#font_color = (random.randrange(random_val + 1, 256),) * image_depth  # A lighter grayscale font color.
	#font_color = (random.randrange(128),) * image_depth  # A dark grayscale font color.
	#font_color = (random.randrange(128, 256),) * image_depth  # A light grayscale font color.

	#bg_color = (0,) * image_depth  # Black background color.
	bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB background color.
	#bg_color = (random.randrange(256),) * image_depth  # A grayscale background color.
	#bg_color = (random_val,) * image_depth  # A grayscale background color.
	#bg_color = (random.randrange(random_val),) * image_depth  # A darker grayscale background color.
	#bg_color = (random.randrange(random_val + 1, 256),) * image_depth  # A lighter grayscale background color.
	#bg_color = (random.randrange(128),) * image_depth  # A dark grayscale background color.
	#bg_color = (random.randrange(128, 256),) * image_depth  # A light grayscale background color.
	return font_color, bg_color

class RandomAugment(object):
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

class ConvertNumpyToRGB(object):
	def __call__(self, x):
		if x.ndim == 1:
			return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
		elif x.ndim == 3:
			return x
		else: raise ValueError('Invalid dimension, {}.'.format(x.ndim))

class ResizeImage(object):
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
			#min_width = min(width, int(wi * aspect_ratio))
			min_width = max(min(width, int(wi * aspect_ratio)), height // 2)
			assert min_width > 0 and height > 0
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
		#min_width = min(width, int(wi * aspect_ratio))
		min_width = max(min(width, int(wi * aspect_ratio)), height // 2)
		assert min_width > 0 and height > 0
		zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
		zeropadded[:,:min_width] = cv2.resize(input, (min_width, height), interpolation=interpolation)
		return zeropadded
		"""
		return cv2.resize(input, (width, height), interpolation=interpolation)
		"""

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_pil() in text_line_data.py.
	def _resize_by_pil(self, input, height, width, *args, **kwargs):
		interpolation = Image.BICUBIC
		wi, hi = input.size
		aspect_ratio = height / hi
		#min_width = min(width, int(wi * aspect_ratio))
		min_width = max(min(width, int(wi * aspect_ratio)), height // 2)
		assert min_width > 0 and height > 0
		zeropadded = Image.new(input.mode, (width, height), color=0)
		zeropadded.paste(input.resize((min_width, height), resample=interpolation), (0, 0, min_width, height))
		return zeropadded
		"""
		return input.resize((width, height), resample=interpolation)
		"""

class ToIntTensor(object):
	def __call__(self, lst):
		return torch.IntTensor(lst)

class MySubsetDataset(torch.utils.data.Dataset):
	def __init__(self, subset, transform=None, target_transform=None):
		self.subset = subset
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, idx):
		dat = list(self.subset[idx])
		if self.transform:
			dat[0] = self.transform(dat[0])
		if self.target_transform:
			dat[1] = self.target_transform(dat[1])
		return dat

	def __len__(self):
		return len(self.subset)

def create_char_data_loaders(char_type, label_converter, charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_char_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])

	print('Start creating datasets...')
	start_time = time.time()
	if char_type == 'simple_char':
		chars = list(charset * num_train_examples_per_class)
		random.shuffle(chars)
		train_dataset = text_data.SimpleCharacterDataset(label_converter, chars, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=train_transform)
		chars = list(charset * num_test_examples_per_class)
		random.shuffle(chars)
		test_dataset = text_data.SimpleCharacterDataset(label_converter, chars, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=test_transform)
	elif char_type == 'noisy_char':
		chars = list(charset * num_train_examples_per_class)
		random.shuffle(chars)
		train_dataset = text_data.NoisyCharacterDataset(label_converter, chars, image_channel, char_clipping_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform)
		chars = list(charset * num_test_examples_per_class)
		random.shuffle(chars)
		test_dataset = text_data.NoisyCharacterDataset(label_converter, chars, image_channel, char_clipping_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform)
	elif char_type == 'file_based_char':
		if 'posix' == os.name:
			data_base_dir_path = '/home/sangwook/work/dataset'
		else:
			data_base_dir_path = 'D:/work/dataset'

		datasets = list()
		if True:
			# REF [function] >> generate_chars_from_chars74k_data() in chars74k_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/chars74k/English/Img/char_images.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_kr.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_en.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_kr.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_en.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
		assert datasets, 'NO Dataset'

		dataset = torch.utils.data.ConcatDataset(datasets)
		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)

		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform)
	else:
		raise ValueError('Invalid dataset type: {}'.format(char_type))
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def create_mixed_char_data_loaders(label_converter, charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_char_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	print('Start creating datasets...')
	start_time = time.time()
	datasets = list()
	if True:
		chars = list(charset * num_simple_char_examples_per_class)
		random.shuffle(chars)
		datasets.append(text_data.SimpleCharacterDataset(label_converter, chars, image_channel, font_list, font_size_interval, color_functor=color_functor))
	if True:
		chars = list(charset * num_noisy_examples_per_class)
		random.shuffle(chars)
		datasets.append(text_data.NoisyCharacterDataset(label_converter, chars, image_channel, char_clipping_ratio_interval, font_list, font_size_interval, color_functor=color_functor))
	if True:
		# REF [function] >> generate_chars_from_chars74k_data() in chars74k_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/chars74k/English/Img/char_images.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_kr.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_en.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_kr.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_en.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used))
	assert datasets, 'NO Dataset'

	dataset = torch.utils.data.ConcatDataset(datasets)
	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	test_target_transform = ToIntTensor()

	print('Start creating datasets...')
	start_time = time.time()
	if word_type == 'simple_word':
		train_dataset = text_data.SimpleWordDataset(label_converter, wordset, num_train_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
		test_dataset = text_data.SimpleWordDataset(label_converter, wordset, num_test_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	elif word_type == 'random_word':
		train_dataset = text_data.RandomWordDataset(label_converter, chars, num_train_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
		test_dataset = text_data.RandomWordDataset(label_converter, chars, num_test_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	elif word_type == 'file_based_word':
		if 'posix' == os.name:
			data_base_dir_path = '/home/sangwook/work/dataset'
		else:
			data_base_dir_path = 'D:/work/dataset'

		datasets = list()
		if True:
			# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_kr.txt'
			is_preloaded_image_used = False
			datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_en.txt'
			is_preloaded_image_used = False
			datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_kr.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_en.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
		assert datasets, 'NO Dataset'

		dataset = torch.utils.data.ConcatDataset(datasets)
		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)

		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	else:
		raise ValueError('Invalid dataset type: {}'.format(word_type))
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	test_target_transform = ToIntTensor()

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	print('Start creating datasets...')
	start_time = time.time()
	datasets = list()
	if True:
		datasets.append(text_data.SimpleWordDataset(label_converter, wordset, num_simple_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor))
	if True:
		datasets.append(text_data.RandomWordDataset(label_converter, chars, num_random_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor))
	if True:
		# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_kr.txt'
		is_preloaded_image_used = False
		datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_en.txt'
		is_preloaded_image_used = False
		datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_kr.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_en.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used))
	assert datasets, 'NO Dataset'

	dataset = torch.utils.data.ConcatDataset(datasets)
	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def create_text_line_data_loaders(textline_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_text_line_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	test_target_transform = ToIntTensor()

	print('Start creating datasets...')
	start_time = time.time()
	if textline_type == 'simple_textline':
		train_dataset = text_data.SimpleTextLineDataset(label_converter, wordset, num_train_examples, image_channel, max_textline_len, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
		test_dataset = text_data.SimpleTextLineDataset(label_converter, wordset, num_test_examples, image_channel, max_textline_len, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	elif textline_type == 'random_textline':
		train_dataset = text_data.RandomTextLineDataset(label_converter, chars, num_train_examples, image_channel, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
		test_dataset = text_data.RandomTextLineDataset(label_converter, chars, num_test_examples, image_channel, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	elif textline_type == 'trdg_textline':
		font_size = image_height
		num_words = word_count_interval[1]  # TODO [check] >>
		is_variable_length = True

		generator_kwargs = {
			'skewing_angle': 0, 'random_skew': False,  # In degrees counter clockwise.
			#'blur': 0, 'random_blur': False,  # Blur radius.
			'blur': 2, 'random_blur': True,  # Blur radius.
			#'distorsion_type': 0, 'distorsion_orientation': 0,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
			'distorsion_type': 3, 'distorsion_orientation': 2,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
			'background_type': 0,  # background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
			'width': -1,  # Specify a background width when width > 0.
			'alignment': 1,  # Align an image in a background image. alignment = 0 (left), 1 (center), the rest (right).
			'image_dir': None,  # Background image directory which is used when background_type = 3.
			'is_handwritten': False,
			#'text_color': '#282828',
			'text_color': '#000000,#FFFFFF',  # (0x00, 0x00, 0x00) ~ (0xFF, 0xFF, 0xFF).
			'orientation': 0,  # orientation = 0 (horizontal), 1 (vertical).
			'space_width': 1.0,  # The ratio of space width.
			'character_spacing': 0,  # Control space between characters (in pixels).
			'margins': (5, 5, 5, 5),  # For finer layout control. (top, left, bottom, right).
			'fit': False,  # For finer layout control. Specify if images and masks are cropped or not.
			'output_mask': False,  # Specify if a character-level mask for each image is outputted or not.
			'word_split': False  # Split on word instead of per-character. This is useful for ligature-based languages.
		}

		if True:
			lang = 'en'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
			#font_filepaths = trdg.utils.load_fonts(lang)
			font_filepaths = list()

			is_randomly_generated = False
			train_dataset_en = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_train_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=train_transform, target_transform=train_target_transform, **generator_kwargs)
			test_dataset_en = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_test_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=test_transform, target_transform=test_target_transform, **generator_kwargs)
			is_randomly_generated = True
			train_dataset_en_rnd = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_train_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=train_transform, target_transform=train_target_transform, **generator_kwargs)
			test_dataset_en_rnd = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_test_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=test_transform, target_transform=test_target_transform, **generator_kwargs)
		if True:
			lang = 'kr'
			font_filepaths = construct_font(korean=True, english=False)
			font_filepaths, _ = zip(*font_filepaths)

			is_randomly_generated = False
			train_dataset_kr = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_train_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=train_transform, target_transform=train_target_transform, **generator_kwargs)
			test_dataset_kr = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_test_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=test_transform, target_transform=test_target_transform, **generator_kwargs)
			is_randomly_generated = True
			train_dataset_kr_rnd = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_train_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=train_transform, target_transform=train_target_transform, **generator_kwargs)
			test_dataset_kr_rnd = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_test_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=test_transform, target_transform=test_target_transform, **generator_kwargs)
		train_dataset = torch.utils.data.ConcatDataset([train_dataset_en, train_dataset_en_rnd, train_dataset_kr, train_dataset_kr_rnd])
		test_dataset = torch.utils.data.ConcatDataset([test_dataset_en, test_dataset_en_rnd, test_dataset_kr, test_dataset_kr_rnd])
	elif textline_type == 'file_based_textline':
		if 'posix' == os.name:
			data_base_dir_path = '/home/sangwook/work/dataset'
		else:
			data_base_dir_path = 'D:/work/dataset'

		datasets = list()
		if True:
			# ICDAR 2019 SROIE dataset.
			is_preloaded_image_used = False
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.jpg', recursive=False))
			labels_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.txt', recursive=False))
			datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, labels_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
			if True:
				image_label_info_filepath = data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_test_text_line/labels.txt'
				datasets.append(text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.png', recursive=False))
			label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.txt', recursive=False))
			is_preloaded_image_used = True
			datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			image_label_info_filepath = data_base_dir_path + '/text/receipt/sminds/receipt_text_line/labels.txt'
			is_preloaded_image_used = True
			datasets.append(text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
		assert datasets, 'NO Dataset'

		dataset = torch.utils.data.ConcatDataset(datasets)
		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)

		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	else:
		raise ValueError('Invalid dataset type: {}'.format(textline_type))
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def create_mixed_text_line_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, num_trdg_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_text_line_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
	])
	test_target_transform = ToIntTensor()

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	print('Start creating datasets...')
	start_time = time.time()
	datasets = list()
	if True:
		datasets.append(text_data.SimpleTextLineDataset(label_converter, wordset, num_simple_examples, image_channel, max_textline_len, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor))
	if True:
		datasets.append(text_data.RandomTextLineDataset(label_converter, chars, num_random_examples, image_channel, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor))
	if True:
		font_size = image_height
		num_words = word_count_interval[1]  # TODO [check] >>
		is_variable_length = True

		generator_kwargs = {
			'skewing_angle': 0, 'random_skew': False,  # In degrees counter clockwise.
			#'blur': 0, 'random_blur': False,  # Blur radius.
			'blur': 2, 'random_blur': True,  # Blur radius.
			#'distorsion_type': 0, 'distorsion_orientation': 0,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
			'distorsion_type': 3, 'distorsion_orientation': 2,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
			'background_type': 0,  # background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
			'width': -1,  # Specify a background width when width > 0.
			'alignment': 1,  # Align an image in a background image. alignment = 0 (left), 1 (center), the rest (right).
			'image_dir': None,  # Background image directory which is used when background_type = 3.
			'is_handwritten': False,
			#'text_color': '#282828',
			'text_color': '#000000,#FFFFFF',  # (0x00, 0x00, 0x00) ~ (0xFF, 0xFF, 0xFF).
			'orientation': 0,  # orientation = 0 (horizontal), 1 (vertical).
			'space_width': 1.0,  # The ratio of space width.
			'character_spacing': 0,  # Control space between characters (in pixels).
			'margins': (5, 5, 5, 5),  # For finer layout control. (top, left, bottom, right).
			'fit': False,  # For finer layout control. Specify if images and masks are cropped or not.
			'output_mask': False,  # Specify if a character-level mask for each image is outputted or not.
			'word_split': False  # Split on word instead of per-character. This is useful for ligature-based languages.
		}

		if True:
			lang = 'en'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
			#font_filepaths = trdg.utils.load_fonts(lang)
			font_filepaths = list()

			is_randomly_generated = False
			dataset_en = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_trdg_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=None, target_transform=None, **generator_kwargs)
			is_randomly_generated = True
			dataset_en_rnd = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_trdg_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=None, target_transform=None, **generator_kwargs)
		if True:
			lang = 'kr'
			font_filepaths = construct_font(korean=True, english=False)
			font_filepaths, _ = zip(*font_filepaths)

			is_randomly_generated = False
			dataset_kr = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_trdg_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=None, target_transform=None, **generator_kwargs)
			is_randomly_generated = True
			dataset_kr_rnd = text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_trdg_examples // 4, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=None, target_transform=None, **generator_kwargs)
		datasets.append(torch.utils.data.ConcatDataset([dataset_en, dataset_en_rnd, dataset_kr, dataset_kr_rnd]))
	if True:
		# ICDAR 2019 SROIE dataset.
		is_preloaded_image_used = False
		image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.jpg', recursive=False))
		labels_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.txt', recursive=False))
		datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, labels_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
		if True:
			image_label_info_filepath = data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_test_text_line/labels.txt'
			datasets.append(text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.png', recursive=False))
		label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.txt', recursive=False))
		is_preloaded_image_used = True
		datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
	if True:
		image_label_info_filepath = data_base_dir_path + '/text/receipt/sminds/receipt_text_line/labels.txt'
		is_preloaded_image_used = True
		datasets.append(text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used))
	assert datasets, 'NO Dataset'

	dataset = torch.utils.data.ConcatDataset(datasets)
	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def concatenate_labels(labels, eos_value, lengths=None):
	concat_labels = list()
	if lengths == None:
		for lbl in labels:
			try:
				concat_labels.append(lbl[:lbl.index(eos_value)+1])
			except ValueError as ex:
				concat_labels.append(lbl)
	else:
		for lbl, ll in zip(labels, lengths):
			concat_labels.append(lbl[:ll])
	return list(itertools.chain(*concat_labels))

def show_image(img):
	img = img / 2 + 0.5  # Unnormalize.
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def show_char_data_info(dataloader, label_converter, visualize=True, mode='Train'):
	dataiter = iter(dataloader)
	images, labels = dataiter.next()
	images_np, labels_np = images.numpy(), labels.numpy()

	print('{} image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, images_np.shape, images_np.dtype, np.min(images_np), np.max(images_np)))
	print('{} label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, labels_np.shape, labels_np.dtype, np.min(labels_np), np.max(labels_np)))

	if visualize:
		print('Labels: {}.'.format(' '.join(label_converter.decode(labels_np))))
		show_image(torchvision.utils.make_grid(images))

def show_text_data_info(dataloader, label_converter, visualize=True, mode='Train'):
	dataiter = iter(dataloader)
	images, labels, label_lens = dataiter.next()
	images_np, labels_np, label_lens_np = images.numpy(), labels.numpy(), label_lens.numpy()

	print('{} image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, images_np.shape, images_np.dtype, np.min(images_np), np.max(images_np)))
	print('{} label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, labels_np.shape, labels_np.dtype, np.min(labels_np), np.max(labels_np)))
	print('{} label length: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, label_lens_np.shape, label_lens_np.dtype, np.min(label_lens_np), np.max(label_lens_np)))

	if visualize:
		#print('Labels: {}.'.format(' '.join([label_converter.decode(lbl) for lbl in images_np])))
		for idx, (lbl, ll) in enumerate(zip(labels_np, label_lens_np)):
			print('Label #{} (len = {}): {} (int), {} (str).'.format(idx, ll, lbl, label_converter.decode(lbl)))
		show_image(torchvision.utils.make_grid(images))

def show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char=False):
	#for idx in range(num_classes):
	#	print('Accuracy of {:5s} = {:2d} %.'.format(classes[idx], 100 * correct_char_class_count[idx] / total_char_class_count[idx] if total_char_class_count[idx] > 0 else -1))
	accuracies = [100 * correct_char_class_count[idx] / total_char_class_count[idx] if total_char_class_count[idx] > 0 else -1 for idx in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	#hist, bin_edges = np.histogram(accuracies, bins=range(0, 101), density=False)
	#print('Per-character accuracy histogram: {}.'.format({bb: hh for bb, hh in zip(bin_edges, hist)}))
	print('Per-character accuracy histogram: {}.'.format({bb: hh for bb, hh in zip(bin_edges, hist) if hh > 0}))

	if show_acc_per_char:
		valid_accuracies = [100 * correct_char_class_count[idx] / total_char_class_count[idx] for idx in range(num_classes) if total_char_class_count[idx] > 0]
		print('Per-character accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
		acc_thresh = 98
		print('Per-character accuracy (< {}) = {}.'.format(acc_thresh, {classes[idx]: round(acc, 2) for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]) if acc < acc_thresh}))

def train_char_recognition_model(model, forward_functor, criterion, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler=None, max_gradient_norm=None, model_params=None, device='cpu'):
	best_measure = 0.0
	best_model_filepath = None
	for epoch in range(num_epochs):  # Loop over the dataset multiple times.
		start_time = time.time()
		model.train()
		running_loss = 0.0
		for idx, batch in enumerate(train_dataloader):
			# Zero the parameter gradients.
			optimizer.zero_grad()

			# Forward + backward + optimize.
			loss = forward_functor(model, batch, device)
			loss.backward()
			if max_gradient_norm: torch.nn.utils.clip_grad_norm_(model_params, max_norm=max_gradient_norm)  # Gradient clipping.
			optimizer.step()

			# Print statistics.
			running_loss += loss.item()
			if idx % log_print_freq == (log_print_freq - 1):
				print('[{}, {:5d}] loss = {:.6g}: {:.3f} secs.'.format(epoch + 1, idx + 1, running_loss / log_print_freq, time.time() - start_time))
				running_loss = 0.0

			sys.stdout.flush()
			time.sleep(0)
		print('Epoch {} completed: {} secs.'.format(epoch + 1, time.time() - start_time))

		print('Start evaluating...')
		start_time = time.time()
		model.eval()
		acc = evaluate_char_recognition_model(model, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=False, is_error_cases_saved=False, device=device)
		print('End evaluating: {} secs.'.format(time.time() - start_time))

		if scheduler: scheduler.step()

		if acc >= best_measure:
			ckpt_fpath = model_filepath_format.format('_acc{:.4f}_epoch{}'.format(acc, epoch + 1))
			# Save a checkpoint.
			save_model(ckpt_fpath, model)
			best_measure = acc
			best_model_filepath = ckpt_fpath

	return model, best_model_filepath

def train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler=None, max_gradient_norm=None, model_params=None, device='cpu'):
	best_measure = 0.0
	best_model_filepath = None
	for epoch in range(num_epochs):  # Loop over the dataset multiple times.
		start_time = time.time()
		model.train()
		running_loss = 0.0
		for idx, batch in enumerate(train_dataloader):
			# Zero the parameter gradients.
			optimizer.zero_grad()

			# Forward + backward + optimize.
			loss = forward_functor(model, criterion, batch, device)
			loss.backward()
			if max_gradient_norm: torch.nn.utils.clip_grad_norm_(model_params, max_norm=max_gradient_norm)  # Gradient clipping.
			optimizer.step()

			# Print statistics.
			running_loss += loss.item()
			if idx % log_print_freq == (log_print_freq - 1):
				print('[{}, {:5d}] loss = {:.6g}: {:.3f} secs.'.format(epoch + 1, idx + 1, running_loss / log_print_freq, time.time() - start_time))
				running_loss = 0.0

			sys.stdout.flush()
			time.sleep(0)
		print('Epoch {} completed: {} secs.'.format(epoch + 1, time.time() - start_time))

		print('Start evaluating...')
		start_time = time.time()
		model.eval()
		acc = evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=False, is_error_cases_saved=False, device=device)
		print('End evaluating: {} secs.'.format(time.time() - start_time))

		if scheduler: scheduler.step()

		if acc >= best_measure:
			ckpt_fpath = model_filepath_format.format('_acc{:.4f}_epoch{}'.format(acc, epoch + 1))
			# Save a checkpoint.
			save_model(ckpt_fpath, model)
			best_measure = acc
			best_model_filepath = ckpt_fpath

	return model, best_model_filepath

def evaluate_char_recognition_model(model, label_converter, dataloader, is_case_sensitive=False, show_acc_per_char=False, is_error_cases_saved=False, device='cpu'):
	classes, num_classes = label_converter.tokens, label_converter.num_tokens

	error_cases_dir_path = './char_error_cases'
	if is_error_cases_saved:
		os.makedirs(error_cases_dir_path, exist_ok=True)

	correct_char_count, total_char_count = 0, 0
	correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
	error_cases = list()
	error_idx = 0
	is_first = True
	with torch.no_grad():
		for images, labels in dataloader:
			predictions = model(images.to(device))

			_, predictions = torch.max(predictions, 1)
			predictions = predictions.cpu().numpy()
			gts = labels.numpy()

			for gl, pl in zip(gts, predictions):
				if gl == pl: correct_char_class_count[gl] += 1
				total_char_class_count[gl] += 1

			gts, predictions = label_converter.decode(gts), label_converter.decode(predictions)
			gts_case, predictions_case = (gts, predictions) if is_case_sensitive else (gts.lower(), predictions.lower())

			total_char_count += max(len(gts), len(predictions))
			#correct_char_count += (gts_case == predictions_case).sum()
			correct_char_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gts_case, predictions_case))))

			if is_error_cases_saved:
				images_np = images.numpy()
				if images_np.ndim == 4: images_np = images_np.transpose(0, 2, 3, 1)
				#minval, maxval = np.min(images_np), np.max(images_np)
				minval, maxval = -1, 1
				images_np = np.round((images_np - minval) * 255 / (maxval - minval)).astype(np.uint8)

				for img, gt, pred, gt_case, pred_case in zip(images_np.numpy(), gts, predictions, gts_case, predictions_case):
					if gt_case != pred_case:
						cv2.imwrite(os.path.join(error_cases_dir_path, 'image_{}.png'.format(error_idx)), img)
						error_cases.append((gt, pred))
						error_idx += 1

			if is_first:
				# Show images.
				#show_image(torchvision.utils.make_grid(images))

				#print('G/T:        {}.'.format(' '.join(gts)))
				#print('Prediction: {}.'.format(' '.join(predictions)))
				#for gt, pred in zip(gts, predictions):
				#	print('G/T - prediction: {}, {}.'.format(gt, pred))
				print('G/T - prediction:\n{}.'.format([(gt, pred) for gt, pred in zip(gts, predictions)]))

				is_first = False

	if is_error_cases_saved:
		err_fpath = os.path.join(error_cases_dir_path, 'error_cases.txt')
		try:
			with open(err_fpath, 'w', encoding='UTF8') as fd:
				for idx, (gt, pred) in enumerate(error_cases):
					fd.write('{}: {}\t{}\n'.format(idx, gt, pred))
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(err_fpath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(err_fpath))

	show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char)
	print('Char accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

	return correct_char_count / total_char_count

def evaluate_text_recognition_model(model, infer_functor, label_converter, dataloader, is_case_sensitive=False, show_acc_per_char=False, is_error_cases_saved=False, device='cpu'):
	classes, num_classes = label_converter.tokens, label_converter.num_tokens

	error_cases_dir_path = './text_error_cases'
	if is_error_cases_saved:
		os.makedirs(error_cases_dir_path, exist_ok=True)

	correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0, 0, 0
	correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
	error_cases = list()
	error_idx = 0
	is_first = True
	with torch.no_grad():
		for images, labels, label_lens in dataloader:
			predictions, gts = infer_functor(model, images, labels, label_lens, device)

			images_np = images.numpy()
			if images_np.ndim == 4: images_np = images_np.transpose(0, 2, 3, 1)
			#minval, maxval = np.min(images_np), np.max(images_np)
			minval, maxval = -1, 1
			images_np = np.round((images_np - minval) * 255 / (maxval - minval)).astype(np.uint8)

			total_text_count += len(gts)
			for img, gt, pred in zip(images_np, gts, predictions):
				for gl, pl in zip(gt, pred):
					if gl == pl: correct_char_class_count[gl] += 1
					total_char_class_count[gl] += 1

				gt, pred = label_converter.decode(gt), label_converter.decode(pred)
				gt_case, pred_case = (gt, pred) if is_case_sensitive else (gt.lower(), pred.lower())

				if gt_case == pred_case:
					correct_text_count += 1
				elif is_error_cases_saved:
					cv2.imwrite(os.path.join(error_cases_dir_path, 'image_{}.png'.format(error_idx)), img)
					error_cases.append((gt, pred))
					error_idx += 1

				gt_words, pred_words = gt_case.split(' '), pred_case.split(' ')
				total_word_count += max(len(gt_words), len(pred_words))
				correct_word_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gt_words, pred_words))))

				total_char_count += max(len(gt), len(pred))
				correct_char_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gt_case, pred_case))))

			if is_first:
				# Show images.
				#show_image(torchvision.utils.make_grid(images))

				#print('G/T:        {}.'.format(' '.join([label_converter.decode(lbl) for lbl in gts])))
				#print('Prediction: {}.'.format(' '.join([label_converter.decode(lbl) for lbl in predictions])))
				#for gt, pred in zip(gts, predictions):
				#	print('G/T - prediction: {}, {}.'.format(label_converter.decode(gt), label_converter.decode(pred)))
				print('G/T - prediction:\n{}.'.format([(label_converter.decode(gt), label_converter.decode(pred)) for gt, pred in zip(gts, predictions)]))

				is_first = False

	if is_error_cases_saved:
		err_fpath = os.path.join(error_cases_dir_path, 'error_cases.txt')
		try:
			with open(err_fpath, 'w', encoding='UTF8') as fd:
				for idx, (gt, pred) in enumerate(error_cases):
					fd.write('{}: {}\t{}\n'.format(idx, gt, pred))
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(err_fpath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(err_fpath))

	show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char)
	print('Text accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count))
	print('Word accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
	print('Char accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

	return correct_char_count / total_char_count

def infer_by_text_recognition_model(model, infer_functor, label_converter, inputs, outputs=None, batch_size=None, is_case_sensitive=False, show_acc_per_char=False, is_error_cases_saved=False, device='cpu'):
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	if batch_size is None: batch_size = len(inputs)

	with torch.no_grad():
		predictions = list()
		for idx in range(0, len(inputs), batch_size):
			predictions.append(infer_functor(model, inputs[idx:idx+batch_size], device=device)[0])
	inputs, predictions = inputs.numpy(), np.vstack(predictions)
	print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(predictions.shape, predictions.dtype, np.min(predictions), np.max(predictions)))

	if outputs is None:
		num_iters = 0
		for idx in range(0, len(predictions), batch_size):
			# Show images.
			#show_image(torchvision.utils.make_grid(inputs[idx:idx+batch_size]))

			print('Prediction:\n{}.'.format('\n'.join([label_converter.decode(pred) for pred in predictions[idx:idx+batch_size]])))

			num_iters += 1
			if num_iters >= 5: break
	else:
		outputs = outputs.numpy()

		num_iters = 0
		for idx in range(0, len(predictions), batch_size):
			inps, outps, preds = inputs[idx:idx+batch_size], outputs[idx:idx+batch_size], predictions[idx:idx+batch_size]

			# Show images.
			#show_image(torchvision.utils.make_grid(inps))

			#print('G/T:        {}.'.format(' '.join([label_converter.decode(lbl) for lbl in outps])))
			#print('Prediction: {}.'.format(' '.join([label_converter.decode(lbl) for lbl in preds])))
			#for gt, pred in zip(outps, preds):
			#	print('G/T - prediction: {}, {}.'.format(label_converter.decode(gt), label_converter.decode(pred)))
			print('G/T - prediction:\n{}.'.format([(label_converter.decode(gt), label_converter.decode(pred)) for gt, pred in zip(outps, preds)]))

			num_iters += 1
			if num_iters >= 5: break

		#--------------------
		error_cases_dir_path = './text_error_cases'
		if is_error_cases_saved:
			os.makedirs(error_cases_dir_path, exist_ok=True)

		correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0, 0, 0
		correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
		error_cases = list()
		error_idx = 0

		if inputs.ndim == 4: inputs = inputs.transpose(0, 2, 3, 1)
		#minval, maxval = np.min(inputs), np.max(inputs)
		minval, maxval = -1, 1
		inputs = np.round((inputs - minval) * 255 / (maxval - minval)).astype(np.uint8)

		total_text_count += len(outputs)
		for img, gt, pred in zip(inputs, outputs, predictions):
			for gl, pl in zip(gt, pred):
				if gl == pl: correct_char_class_count[gl] += 1
				total_char_class_count[gl] += 1

			gt, pred = label_converter.decode(gt), label_converter.decode(pred)
			gt_case, pred_case = (gt, pred) if is_case_sensitive else (gt.lower(), pred.lower())

			if gt_case == pred_case:
				correct_text_count += 1
			elif is_error_cases_saved:
				cv2.imwrite(os.path.join(error_cases_dir_path, 'image_{}.png'.format(error_idx)), img)
				error_cases.append((gt, pred))
				error_idx += 1

			gt_words, pred_words = gt_case.split(' '), pred_case.split(' ')
			total_word_count += max(len(gt_words), len(pred_words))
			correct_word_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gt_words, pred_words))))

			total_char_count += max(len(gt), len(pred))
			correct_char_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gt_case, pred_case))))

		if is_error_cases_saved:
			err_fpath = os.path.join(error_cases_dir_path, 'error_cases.txt')
			try:
				with open(err_fpath, 'w', encoding='UTF8') as fd:
					for idx, (gt, pred) in enumerate(error_cases):
						fd.write('{}: {}\t{}\n'.format(idx, gt, pred))
			except FileNotFoundError as ex:
				print('File not found: {}.'.format(err_fpath))
			except UnicodeDecodeError as ex:
				print('Unicode decode error: {}.'.format(err_fpath))

		show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char)
		print('Text accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count))
		print('Word accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
		print('Char accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

def build_char_model(label_converter, image_channel, loss_type, lang, device='cpu'):
	model_name = 'ResNet'  # {'VGG', 'ResNet', 'RCNN'}.
	input_channel, output_channel = image_channel, 1024

	# Define a loss function.
	if loss_type == 'xent':
		criterion = torch.nn.CrossEntropyLoss().to(device)
	elif loss_type == 'nll':
		criterion = torch.nn.NLLLoss(reduction='sum').to(device)
	else:
		raise ValueError('Invalid loss type, {}'.format(loss_type))

	def forward(model, criterion, batch, device):
		inputs, outputs = batch
		inputs, outputs = inputs.to(device), outputs.to(device)

		model_outputs = model(inputs)
		return criterion(model_outputs, outputs)

	import rare.model_char
	model = rare.model_char.create_model(model_name, input_channel, output_channel, label_converter.num_tokens)

	return model, forward, criterion

def build_char_mixup_model(label_converter, image_channel, loss_type, lang, device='cpu'):
	model_name = 'ResNet'  # {'VGG', 'ResNet', 'RCNN'}.
	input_channel, output_channel = image_channel, 1024

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	# Define a loss function.
	if loss_type == 'xent':
		criterion = torch.nn.CrossEntropyLoss().to(device)
	elif loss_type == 'nll':
		criterion = torch.nn.NLLLoss(reduction='sum').to(device)
	else:
		raise ValueError('Invalid loss type, {}'.format(loss_type))

	def forward(model, criterion, batch, device):
		inputs, outputs = batch
		inputs, outputs = inputs.to(device), outputs.to(device)

		model_outputs, outputs = model(inputs, outputs, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
		return criterion(model_outputs, torch.argmax(outputs, dim=1))

	# REF [function] >> mnist_predefined_mixup_test() in ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/run_mnist_cnn.py.
	import rare.model_char
	model = rare.model_char.create_mixup_model(model_name, input_channel, output_channel, label_converter.num_tokens)

	return model, forward, criterion

def build_rare1_model(label_converter, image_height, image_width, image_channel, loss_type, lang, max_text_len, num_suffixes, sos_value, blank_label=None, device='cpu'):
	#num_fiducials = 20  # The number of fiducial points of TPS-STN.
	num_fiducials = None
	input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	if lang == 'kor':
		hidden_size = 1024  # The size of the LSTM hidden states.
	else:
		hidden_size = 512  # The size of the LSTM hidden states.
	transformer = None  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.
	if loss_type == 'ctc':
		decoder = 'CTC'  # The type of decoder. {'CTC', 'Attn'}.
	elif loss_type in ['xent', 'nll']:
		decoder = 'Attn'  # The type of decoder. {'CTC', 'Attn'}.

	if loss_type == 'ctc':
		# Define a loss function.
		criterion = torch.nn.CTCLoss(blank=blank_label, zero_infinity=True).to(device)  # The BLANK label.

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch

			model_outputs = model(inputs.to(device), None, is_train=True, device=device).log_softmax(2)

			N, T = model_outputs.shape[:2]
			model_outputs = model_outputs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C).
			model_output_lens = torch.full([N], T, dtype=torch.int32, device=device)

			# TODO [check] >> To avoid CTC loss issue, disable cuDNN for the computation of the CTC loss.
			# https://github.com/jpuigcerver/PyLaia/issues/16
			torch.backends.cudnn.enabled = False
			cost = criterion(model_outputs, outputs.to(device), model_output_lens, output_lens.to(device))
			torch.backends.cudnn.enabled = True
			return cost

		def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
			raise NotImplementedError

	elif loss_type in ['xent', 'nll']:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			# Construct inputs for one-step look-ahead.
			decoder_inputs = outputs[:,:-1]
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> token.
			decoder_output_lens = output_lens - 1

			model_outputs = model(inputs.to(device), decoder_inputs.to(device), is_train=True, device=device)

			# TODO [check] >> How to compute loss?
			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			#return criterion(model_outputs.view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
			"""
			mask = torch.full(decoder_outputs.shape[:2], False, dtype=torch.bool)
			for idx, ll in enumerate(decoder_output_lens):
				mask[idx,:ll].fill_(True)
			model_outputs[mask == False] = label_converter.pad_value
			return criterion(model_outputs.view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
			"""
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device))

		def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
			model_outputs = model(inputs.to(device), None, is_train=False, device=device)

			_, model_outputs = torch.max(model_outputs, dim=-1)
			if outputs is None or output_lens is None:
				return model_outputs.cpu().numpy(), None
			else:
				return model_outputs.cpu().numpy(), outputs.numpy()

	import rare.model
	model = rare.model.Model(image_height, image_width, label_converter.num_tokens, num_fiducials, input_channel, output_channel, hidden_size, max_text_len + num_suffixes, sos_value, label_converter.pad_value, transformer, feature_extractor, sequence_model, decoder)

	return model, infer, forward, criterion

def build_rare1_mixup_model(label_converter, image_height, image_width, image_channel, loss_type, lang, max_text_len, num_suffixes, sos_value, blank_label=None, device='cpu'):
	#num_fiducials = 20  # The number of fiducial points of TPS-STN.
	num_fiducials = None
	input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	if lang == 'kor':
		hidden_size = 1024  # The size of the LSTM hidden states.
	else:
		hidden_size = 512  # The size of the LSTM hidden states.
	transformer = None  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.
	if loss_type == 'ctc':
		decoder = 'CTC'  # The type of decoder. {'CTC', 'Attn'}.
	elif loss_type in ['xent', 'nll']:
		decoder = 'Attn'  # The type of decoder. {'CTC', 'Attn'}.

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	if loss_type == 'ctc':
		# Define a loss function.
		criterion = torch.nn.CTCLoss(blank=blank_label, zero_infinity=True).to(device)  # The BLANK label.

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch

			model_outputs = model(inputs.to(device), None, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, is_train=True, device=device).log_softmax(2)

			N, T = model_outputs.shape[:2]
			model_outputs = model_outputs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C).
			model_output_lens = torch.full([N], T, dtype=torch.int32, device=device)

			# TODO [check] >> To avoid CTC loss issue, disable cuDNN for the computation of the CTC loss.
			# https://github.com/jpuigcerver/PyLaia/issues/16
			torch.backends.cudnn.enabled = False
			cost = criterion(model_outputs, outputs.to(device), model_output_lens, output_lens.to(device))
			torch.backends.cudnn.enabled = True
			return cost

		def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
			raise NotImplementedError

	elif loss_type in ['xent', 'nll']:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			# Construct inputs for one-step look-ahead.
			decoder_inputs = outputs[:,:-1]
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> token.
			decoder_output_lens = output_lens - 1

			model_outputs = model(inputs.to(device), decoder_inputs.to(device), is_train=True, device=device)

			# TODO [check] >> How to compute loss?
			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			#return criterion(model_outputs.view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
			"""
			mask = torch.full(decoder_outputs.shape[:2], False, dtype=torch.bool)
			for idx, ll in enumerate(decoder_output_lens):
				mask[idx,:ll].fill_(True)
			model_outputs[mask == False] = label_converter.pad_value
			return criterion(model_outputs.view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
			"""
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device))

		def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
			model_outputs = model(inputs.to(device), None, is_train=False, device=device)

			_, model_outputs = torch.max(model_outputs, dim=-1)
			if outputs is None or output_lens is None:
				return model_outputs.cpu().numpy(), None
			else:
				return model_outputs.cpu().numpy(), outputs.numpy()

	# FIXME [error] >> rare.model.Model_MixUp is not working.
	import rare.model
	model = rare.model.Model_MixUp(image_height, image_width, label_converter.num_tokens, num_fiducials, input_channel, output_channel, hidden_size, max_text_len + num_suffixes, sos_value, label_converter.pad_value, transformer, feature_extractor, sequence_model, decoder)

	return model, infer, forward, criterion

def build_rare2_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, max_time_steps=0, sos_value=0, device='cpu'):
	if lang == 'kor':
		hidden_size = 512  # The size of the LSTM hidden states.
	else:
		hidden_size = 256  # The size of the LSTM hidden states.
	num_rnns = 2
	embedding_size = 256
	use_leaky_relu = False

	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			# Construct inputs for one-step look-ahead.
			decoder_inputs = outputs[:,:-1]
			decoder_input_lens = output_lens - 1
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> token.
			decoder_output_lens = output_lens - 1

			model_outputs = model(inputs.to(device), decoder_inputs.to(device), decoder_input_lens.to(device), device=device)

			# TODO [check] >> How to compute loss?
			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			#return criterion(model_outputs.view(-1, model_outputs.shape[-1]), outputs.to(device).contiguous().view(-1))
			"""
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device))
			"""
			"""
			concat_decoder_outputs = list()
			for do, dl in zip(decoder_outputs, decoder_output_lens):
				concat_decoder_outputs.append(do[:dl])
			return criterion(model_outputs, torch.cat(concat_decoder_outputs, 0).to(device))
			"""
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device))
	else:
		criterion = None
		forward = None

	def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
		#model_outputs = model(inputs.to(device), decoder_inputs.to(device), decoder_input_lens.to(device), device=device)
		model_outputs = model(inputs.to(device), None, None, device=device)

		_, model_outputs = torch.max(model_outputs, dim=-1)
		model_outputs = model_outputs.cpu().numpy()

		if outputs is None or output_lens is None:
			return model_outputs, None
		else:
			outputs = outputs.long()

			# Construct inputs for one-step look-ahead.
			#decoder_inputs = outputs[:,:-1]
			#decoder_input_lens = output_lens - 1
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> token.
			decoder_output_lens = output_lens - 1

			"""
			separated_model_outputs = np.zeros(decoder_outputs.shape, model_outputs.dtype)
			start_idx = 0
			for idx, dl in enumerate(decoder_output_lens):
				end_idx = start_idx + dl
				separated_model_outputs[idx,:dl] = model_outputs[start_idx:end_idx]
				start_idx = end_idx
			return separated_model_outputs, decoder_outputs.numpy()
			"""
			return model_outputs, decoder_outputs.numpy()

	import rare.crnn_lang
	model = rare.crnn_lang.CRNN(imgH=image_height, nc=image_channel, nclass=label_converter.num_tokens, nh=hidden_size, n_rnn=num_rnns, num_embeddings=embedding_size, leakyRelu=use_leaky_relu, max_time_steps=max_time_steps, sos_value=sos_value)

	return model, infer, forward, criterion

def build_aster_model(label_converter, image_height, image_width, image_channel, lang, max_text_len, eos_value, device='cpu'):
	if lang == 'kor':
		hidden_size = 512  # The size of the LSTM hidden states.
	else:
		hidden_size = 256  # The size of the LSTM hidden states.

	import aster.config
	sys_args = aster.config.get_args(sys.argv[1:])
	sys_args.with_lstm = True
	#sys_args.STN_ON = True

	print('Config options: {}.'.format(vars(sys_args)))

	# Define a loss function.
	#if loss_type == 'xent':
	#	criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
	#elif loss_type == 'nll':
	#	criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.
	#else:
	#	raise ValueError('Invalid loss type, {}'.format(loss_type))

	def forward(model, criterion, batch, device):
		inputs, outputs, output_lens = batch

		"""
		# Construct inputs for one-step look-ahead.
		decoder_inputs = outputs.clone()
		for idx, ll in enumerate(output_lens):
			decoder_inputs[idx, ll-1] = label_converter.pad_value  # Remove <EOS> token.
		decoder_inputs = decoder_inputs[:,:-1]
		"""
		# Construct outputs for one-step look-ahead.
		decoder_outputs = outputs[:,1:]  # Remove <SOS> token.
		decoder_output_lens = output_lens - 1

		input_dict = dict()
		input_dict['images'] = inputs.to(device)
		input_dict['rec_targets'] = decoder_outputs.to(device)
		input_dict['rec_lengths'] = decoder_output_lens.to(device)

		model_output_dict = model(input_dict, device=device)

		loss = model_output_dict['losses']['loss_rec']  # aster.sequence_cross_entropy_loss.SequenceCrossEntropyLoss.
		return loss

	def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
		if outputs is None or output_lens is None:
			input_dict = dict()
			input_dict['images'] = inputs.to(device)
			input_dict['rec_targets'] = None
			input_dict['rec_lengths'] = None

			model_output_dict = model(input_dict, device=device)

			model_outputs = model_output_dict['output']['pred_rec']  # [batch size, max label len].
			#model_output_scores = model_output_dict['output']['pred_rec_score']  # [batch size, max label len].

			return model_outputs.cpu().numpy(), None
		else:
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> token.
			decoder_output_lens = output_lens - 1

			input_dict = dict()
			input_dict['images'] = inputs.to(device)
			input_dict['rec_targets'] = decoder_outputs.to(device)
			input_dict['rec_lengths'] = decoder_output_lens.to(device)

			model_output_dict = model(input_dict, device=device)

			#loss = model_output_dict['losses']['loss_rec']
			model_outputs = model_output_dict['output']['pred_rec']  # [batch size, max label len].
			#model_output_scores = model_output_dict['output']['pred_rec_score']  # [batch size, max label len].

			# TODO [check] >>
			#return model_outputs.cpu().numpy(), outputs.numpy()
			return model_outputs.cpu().numpy(), decoder_outputs.numpy()

	import aster.model_builder
	model = aster.model_builder.ModelBuilder(
		sys_args, arch=sys_args.arch, input_height=image_height, input_channel=image_channel,
		hidden_size=hidden_size, rec_num_classes=label_converter.num_tokens,
		sDim=sys_args.decoder_sdim, attDim=sys_args.attDim,
		max_len_labels=max_text_len + label_converter.num_affixes, eos=eos_value,
		STN_ON=sys_args.STN_ON
	)

	return model, infer, forward, sys_args

def build_opennmt_submodels(input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size):
	import onmt

	bidirectional_encoder = True
	embedding_dropout = 0.3
	encoder_num_layers = 2
	encoder_rnn_size = encoder_rnn_size
	encoder_dropout = 0.3
	decoder_rnn_type = 'LSTM'
	decoder_num_layers = 2
	decoder_hidden_size = decoder_hidden_size
	decoder_dropout = 0.3

	src_embeddings = None
	tgt_embeddings = onmt.modules.Embeddings(
		word_vec_size=word_vec_size,
		word_vocab_size=num_classes,
		word_padding_idx=1,
		position_encoding=False,
		feat_merge='concat',
		feat_vec_exponent=0.7,
		feat_vec_size=-1,
		feat_padding_idx=[],
		feat_vocab_sizes=[],
		dropout=embedding_dropout,
		sparse=False,
		fix_word_vecs=False
	)

	encoder = onmt.encoders.ImageEncoder(
		num_layers=encoder_num_layers, bidirectional=bidirectional_encoder,
		rnn_size=encoder_rnn_size, dropout=encoder_dropout, image_chanel_size=input_channel
	)
	decoder = onmt.decoders.InputFeedRNNDecoder(
		rnn_type=decoder_rnn_type, bidirectional_encoder=bidirectional_encoder,
		num_layers=decoder_num_layers, hidden_size=decoder_hidden_size,
		attn_type='general', attn_func='softmax',
		coverage_attn=False, context_gate=None,
		copy_attn=False, dropout=decoder_dropout, embeddings=tgt_embeddings,
		reuse_copy_attn=False, copy_attn_type='general'
	)
	generator = torch.nn.Sequential(
		torch.nn.Linear(in_features=decoder_hidden_size, out_features=num_classes, bias=True),
		onmt.modules.util_class.Cast(dtype=torch.float32),
		torch.nn.LogSoftmax(dim=-1)
	)
	return encoder, decoder, generator

def build_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, device='cpu'):
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 1024, 1024
	else:
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 512, 512

	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:].to(device)
			outputs.unsqueeze_(dim=-1)
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model.generator(model_output_tuple[0]).transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().view(-1))
	else:
		criterion = None
		forward = None

	def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
		if outputs is None or output_lens is None:
			# FIXME [check] >>
			model_output_tuple = model(inputs.to(device))

			model_outputs = model.generator(model_output_tuple[0]).transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), None
		else:
			deconder_outputs = outputs[:,1:]
			outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model.generator(model_output_tuple[0]).transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), deconder_outputs.numpy()

	import onmt
	encoder, decoder, generator = build_opennmt_submodels(image_channel, label_converter.num_tokens, word_vec_size, encoder_rnn_size, decoder_hidden_size)
	model = onmt.models.NMTModel(encoder, decoder)
	model.add_module('generator', generator)

	return model, infer, forward, criterion

def build_rare1_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, device='cpu'):
	#num_fiducials = 20  # The number of fiducial points of TPS-STN.
	num_fiducials = None
	#input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	transformer = None  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.

	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 1024, 1024
	else:
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 512, 512

	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:].to(device)
			outputs.unsqueeze_(dim=-1)
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().view(-1))
	else:
		criterion = None
		forward = None

	def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
		if outputs is None or output_lens is None:
			# FIXME [check] >>
			model_output_tuple = model(inputs.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), None
		else:
			decoder_outputs = outputs[:,1:]
			outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), decoder_outputs.numpy()

	class MyCompositeModel(torch.nn.Module):
		def __init__(self, image_height, image_width, input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size):
			super().__init__()

			self.encoder = self._create_encoder(image_height, image_width, input_channel, encoder_rnn_size // 2)
			_, self.decoder, self.generator = build_opennmt_submodels(input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size)

		def _create_encoder(self, image_height, image_width, input_channel, hidden_size):
			from rare.modules.transformation import TPS_SpatialTransformerNetwork
			from rare.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
			from rare.modules.feature_extraction import VGG_FeatureExtractor_MixUp, RCNN_FeatureExtractor_MixUp, ResNet_FeatureExtractor_MixUp
			from rare.modules.sequence_modeling import BidirectionalLSTM

			# Transformer.
			if transformer == 'TPS':
				self.transformer = TPS_SpatialTransformerNetwork(F=num_fiducials, I_size=(image_height, image_width), I_r_size=(image_height, image_width), I_channel_num=input_channel)
			else:
				print('No transformer specified.')
				self.transformer = None

			# Feature extraction.
			if feature_extractor == 'VGG':
				self.feature_extractor = VGG_FeatureExtractor(input_channel, output_channel)
			elif feature_extractor == 'RCNN':
				self.feature_extractor = RCNN_FeatureExtractor(input_channel, output_channel)
			elif feature_extractor == 'ResNet':
				self.feature_extractor = ResNet_FeatureExtractor(input_channel, output_channel)
			else:
				raise ValueError("The argument, feature_extractor has to be one of 'VGG', 'RCNN', or 'ResNet': {}".format(feature_extractor))
			feature_extractor_output_size = output_channel  # int(image_height / 16 - 1) * output_channel.
			self.avg_pool = torch.nn.AdaptiveAvgPool2d((None, 1))  # Transform final (image_height / 16 - 1) -> 1.

			# Sequence model.
			if sequence_model == 'BiLSTM':
				#self.sequence_model = torch.nn.Sequential(
				#	BidirectionalLSTM(feature_extractor_output_size, hidden_size, hidden_size, batch_first=True),
				#	BidirectionalLSTM(hidden_size, hidden_size, hidden_size, batch_first=True)
				#)
				self.sequence_rnn = torch.nn.LSTM(feature_extractor_output_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
				self.sequence_projector = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
			else:
				print('No sequence model specified.')
				self.sequence_model = None

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def forward(self, src, tgt, lengths, bptt=False, with_align=False):
			# Transform.
			if self.transformer: src = self.transformer(src, device)  # [b, c, h, w].

			# Extract feature.
			visual_feature = self.feature_extractor(src)  # [b, c_out, h/32, w/4-1].
			visual_feature = self.avg_pool(visual_feature.permute(0, 3, 1, 2))  # [b, w/4-1, c_out, 1].
			#visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, w/4-1, c_out, h/32].
			assert visual_feature.shape[3] == 1
			visual_feature = visual_feature.squeeze(3)  # [b, w/4-1, c_out].
			#visual_feature = visual_feature.reshape(visual_feature.shape[0], visual_feature.shape[1], -1)  # [b, w/4-1, c_out * h/32].

			# When batch is not in the first order.
			#visual_feature = visual_feature.transpose(0, 1)  # [w/4-1, b, c_out * h/32].

			# Sequence model.
			# TODO [check] >> The hidden size is too small?
			#enc_outputs, enc_hiddens = self.sequence_model((visual_feature, None))  # [b, w/4-1, #directions * hidden size] or [w/4-1, b, #directions * hidden size], ([#directions, b, hidden size], [#directions, b, hidden size]).
			enc_outputs, enc_hiddens = self.sequence_rnn(visual_feature)  # [b, w/4-1, #directions * hidden size], ([#layers * #directions, b, hidden size], [#layers * #directions, b, hidden size]).
			enc_outputs = self.sequence_projector(enc_outputs)  # [b, w/4-1, hidden size].
			enc_outputs = enc_outputs.transpose(0, 1)  # [w/4-1, b, hidden size]

			dec_in = tgt[:-1]  # Exclude last target from inputs.

			# TODO [check] >> Is it proper to use enc_outputs & enc_hiddens?
			if bptt is False:
				self.decoder.init_state(src, enc_outputs, enc_hiddens)
			dec_outs, attns = self.decoder(dec_in, enc_outputs, memory_lengths=lengths, with_align=with_align)
			outs = self.generator(dec_outs)
			return outs, attns

	model = MyCompositeModel(image_height, image_width, image_channel, label_converter.num_tokens, word_vec_size, encoder_rnn_size, decoder_hidden_size)

	return model, infer, forward, criterion

def build_rare2_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, device='cpu'):
	#num_fiducials = 20  # The number of fiducial points of TPS-STN.
	num_fiducials = None
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 1024, 1024
	else:
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 512, 512

	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:].to(device)
			outputs.unsqueeze_(dim=-1)
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().view(-1))
	else:
		criterion = None
		forward = None

	def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
		if outputs is None or output_lens is None:
			# FIXME [check] >>
			model_output_tuple = model(inputs.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), None
		else:
			decoder_outputs = outputs[:,1:]
			outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), decoder_outputs.numpy()

	class MyCompositeModel(torch.nn.Module):
		def __init__(self, image_height, image_width, input_channel, num_classes, num_fiducials, word_vec_size, encoder_rnn_size, decoder_hidden_size):
			super().__init__()

			if num_fiducials:
				import rare.modules.transformation
				self.transformer = rare.modules.transformation.TPS_SpatialTransformerNetwork(F=num_fiducials, I_size=(image_height, image_width), I_r_size=(image_height, image_width), I_channel_num=input_channel)
			else:
				self.transformer = None

			self.encoder = self._create_encoder(image_height, image_width, input_channel, encoder_rnn_size // 2)
			_, self.decoder, self.generator = build_opennmt_submodels(input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size)

		def _create_encoder(self, image_height, image_width, input_channel, hidden_size):
			assert image_height % 16 == 0, 'image_height has to be a multiple of 16'

			# This implementation assumes that input size is h x w.
			self.cnn = torch.nn.Sequential(
				torch.nn.Conv2d(image_channel, 64, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d(2, 2),  # 64 x h/2 x w/2.
				torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d(2, 2),  # 128 x h/4 x w/4.
				torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),  # 256 x h/4 x w/4.
				torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256 x h/8 x w/4+1.
				torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),  # 512 x h/8 x w/4+1.
				torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512 x h/16 x w/4+2.
				torch.nn.Conv2d(512, 512, 2, 1, 0), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True)  # 512 x h/16-1 x w/4+1.
			)
			num_features = (image_height // 16 - 1) * 512
			#import rare.crnn_lang
			#self.rnn = torch.nn.Sequential(
			#	rare.crnn_lang.BidirectionalLSTM(num_features, hidden_size, hidden_size),
			#	rare.crnn_lang.BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
			#)
			self.sequence_rnn = torch.nn.LSTM(num_features, hidden_size, num_layers=2, bidirectional=True, batch_first=False)
			self.sequence_projector = torch.nn.Linear(hidden_size * 2, hidden_size * 2)

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def forward(self, src, tgt, lengths, bptt=False, with_align=False):
			if self.transformer: src = self.transformer(src, device)  # [B, C, H, W].

			# Conv features.
			conv = self.cnn(src)  # [b, c_out, h/16-1, w/4+1].
			b, c, h, w = conv.size()
			#assert h == 1, 'The height of conv must be 1'
			#conv = conv.squeeze(2)  # [b, c_out, w/4+1].
			conv = conv.reshape(b, -1, w)  # [b, c_out * h/16-1, w/4+1].
			conv = conv.permute(2, 0, 1)  # [w/4+1, b, c_out * h/16-1].

			# RNN features.
			#enc_outputs, enc_hiddens = self.rnn((conv, None))  # [w/4+1, b, hidden size], ([#directions, b, hidden size], [#directions, b, hidden size]).
			enc_outputs, enc_hiddens = self.sequence_rnn(conv)  # [w/4+1, b, #directions * hidden size], ([#layers * #directions, b, hidden size], [#layers * #directions, b, hidden size]).
			enc_outputs = self.sequence_projector(enc_outputs)  # [w/4+1, b, hidden size].

			dec_in = tgt[:-1]  # Exclude last target from inputs.

			# TODO [check] >> Is it proper to use enc_outputs & enc_hiddens?
			if bptt is False:
				self.decoder.init_state(src, enc_outputs, enc_hiddens)
			dec_outs, attns = self.decoder(dec_in, enc_outputs, memory_lengths=lengths, with_align=with_align)
			outs = self.generator(dec_outs)
			return outs, attns

	model = MyCompositeModel(image_height, image_width, image_channel, label_converter.num_tokens, num_fiducials, word_vec_size, encoder_rnn_size, decoder_hidden_size)

	return model, infer, forward, criterion

def build_aster_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, device='cpu'):
	#num_fiducials = 20  # The number of fiducial points of TPS-STN.
	num_fiducials = None
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 1024, 1024
	else:
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 512, 512

	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_value).to(device)  # Ignore the pad value.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_value, reduction='sum').to(device)  # Ignore the pad value.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))

		def forward(model, criterion, batch, device):
			inputs, outputs, output_lens = batch
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:].to(device)
			outputs.unsqueeze_(dim=-1)
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().view(-1))
	else:
		criterion = None
		forward = None

	def infer(model, inputs, outputs=None, output_lens=None, device='cpu'):
		if outputs is None or output_lens is None:
			# FIXME [check] >>
			model_output_tuple = model(inputs.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), None
		else:
			decoder_outputs = outputs[:,1:]
			outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T, F] -> [T, B, F].

			model_output_tuple = model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T, B, F] -> [B, T, F].
			#attentions = model_output_tuple[1]['std']

			_, model_outputs = torch.max(model_outputs, dim=-1)
			return model_outputs.cpu().numpy(), decoder_outputs.numpy()

	class MyCompositeModel(torch.nn.Module):
		def __init__(self, image_height, image_width, input_channel, num_classes, num_fiducials, word_vec_size, encoder_rnn_size, decoder_hidden_size):
			super().__init__()

			if num_fiducials:
				import rare.modules.transformation
				self.transformer = rare.modules.transformation.TPS_SpatialTransformerNetwork(F=num_fiducials, I_size=(image_height, image_width), I_r_size=(image_height, image_width), I_channel_num=input_channel)
			else:
				self.transformer = None

			import aster.resnet_aster
			self.encoder = aster.resnet_aster.ResNet_ASTER(with_lstm=True, in_height=image_height, in_channels=input_channel, hidden_size=encoder_rnn_size // 2)
			_, self.decoder, self.generator = build_opennmt_submodels(input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size)

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def forward(self, src, tgt, lengths, bptt=False, with_align=False):
			dec_in = tgt[:-1]  # Exclude last target from inputs.

			if self.transformer: src = self.transformer(src, device)  # [B, C, H, W].

			enc_outputs, enc_hiddens = self.encoder(src)
			enc_outputs = enc_outputs.transpose(0, 1)  # [B, T, F] -> [T, B, F].

			# TODO [check] >> Is it proper to use enc_outputs & enc_hiddens?
			if bptt is False:
				self.decoder.init_state(src, enc_outputs, enc_hiddens)
			dec_outs, attns = self.decoder(dec_in, enc_outputs, memory_lengths=lengths, with_align=with_align)
			outs = self.generator(dec_outs)
			return outs, attns

	model = MyCompositeModel(image_height, image_width, image_channel, label_converter.num_tokens, num_fiducials, word_vec_size, encoder_rnn_size, decoder_hidden_size)

	return model, infer, forward, criterion

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_character():
	image_height, image_width, image_channel = 64, 64, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based chars: 78,838.
	is_mixed_chars_used = True
	if is_mixed_chars_used:
		num_simple_char_examples_per_class, num_noisy_examples_per_class = 300, 300  # For mixed chars.
	else:
		char_type = 'simple_char'  # {'simple_char', 'noisy_char', 'file_based_char'}.
		num_train_examples_per_class, num_test_examples_per_class = 500, 50  # For simple and noisy chars.
	char_clipping_ratio_interval = (0.8, 1.25)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 5  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 100
	batch_size = 128
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	model_filepath_base = './char_recognition_{}_{}_{}_{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, lang, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset = tg_util.construct_charset()
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset = tg_util.construct_charset(hangeul=False)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	label_converter = swl_langproc_util.TokenConverter(list(charset))
	if is_mixed_chars_used:
		train_dataloader, test_dataloader = create_mixed_char_data_loaders(label_converter, charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_char_data_loaders(char_type, label_converter, charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))

	# Show data info.
	show_char_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_char_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, forward_functor, criterion = build_char_model(label_converter, image_channel, loss_type, lang, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'initialized_variable_name' in name:
			#	print(f'Skip {name} as it has already been initialized.')
			#	continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		optimizer = torch.optim.SGD(model_params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = None

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_char_recognition_model(model, forward_functor, criterion, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_char_recognition_model(model, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_character_using_mixup():
	image_height, image_width, image_channel = 64, 64, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based chars: 78,838.
	is_mixed_chars_used = True
	if is_mixed_chars_used:
		num_simple_char_examples_per_class, num_noisy_examples_per_class = 300, 300  # For mixed chars.
	else:
		char_type = 'simple_char'  # {'simple_char', 'noisy_char', 'file_based_char'}.
		num_train_examples_per_class, num_test_examples_per_class = 500, 50  # For simple and noisy chars.
	char_clipping_ratio_interval = (0.8, 1.25)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 5  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 100
	batch_size = 128
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	model_filepath_base = './char_recognition_mixup_{}_{}_{}_{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, lang, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset = tg_util.construct_charset()
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset = tg_util.construct_charset(hangeul=False)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	label_converter = swl_langproc_util.TokenConverter(list(charset))
	if is_mixed_chars_used:
		train_dataloader, test_dataloader = create_mixed_char_data_loaders(label_converter, charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_char_data_loaders(char_type, label_converter, charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))

	# Show data info.
	show_char_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_char_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, forward_functor, criterion = build_char_mixup_model(label_converter, image_channel, loss_type, lang, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'initialized_variable_name' in name:
			#	print(f'Skip {name} as it has already been initialized.')
			#	continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		optimizer = torch.optim.SGD(model_params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = None

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_char_recognition_model(model, forward_functor, criterion, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_char_recognition_model(model, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_rare1():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'ctc', 'xent', 'nll'}.
	max_gradient_norm = 5  # Gradient clipping value.
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	if loss_type == 'ctc':
		model_filepath_base = './word_recognition_rare1_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	elif loss_type in ['xent', 'nll']:
		model_filepath_base = './word_recognition_rare1_attn_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	else:
		raise ValueError('Invalid loss type, {}'.format(loss_type))
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if loss_type == 'ctc':
		BLANK_LABEL = '<BLANK>'  # The BLANK label for CTC.
		label_converter = swl_langproc_util.TokenConverter([BLANK_LABEL] + list(charset), pad_value=None)  # NOTE [info] >> It's a trick. The ID of the BLANK label is set to 0.
		assert label_converter.encode([BLANK_LABEL], is_bare_output=True)[0] == 0, '{} != 0'.format(label_converter.encode([BLANK_LABEL], is_bare_output=True)[0])
		BLANK_LABEL_INT = 0 #label_converter.encode([BLANK_LABEL], is_bare_output=True)[0]
		SOS_VALUE, EOS_VALUE = None, None
		num_suffixes = 0
	elif loss_type in ['xent', 'nll']:
		if is_individual_pad_value_used:
			# When the pad value is the ID of a valid token.
			PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
			PAD_TOKEN = '<PAD>'
			label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
			assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
			assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
		else:
			# When the pad value = the ID of <SOS> token.
			label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
		SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, SOS_VALUE, EOS_VALUE))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_rare1_model(label_converter, image_height, image_width, image_channel, loss_type, lang, max_word_len, num_suffixes, SOS_VALUE, BLANK_LABEL if loss_type == 'ctc' else None, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
				print(f'Skip {name} as it has already been initialized.')
				continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = None

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_rare2():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	max_gradient_norm = 5  # Gradient clipping value.
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './word_recognition_rare2_attn_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
	SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
	num_suffixes = 1

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, SOS_VALUE, EOS_VALUE))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_rare2_model(label_converter, image_height, image_width, image_channel, lang, loss_type, max_word_len + num_suffixes, SOS_VALUE, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
			#	print(f'Skip {name} as it has already been initialized.')
			#	continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = None

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_aster():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	#loss_type = 'sxent'  # Sequence cross entropy.
	#max_gradient_norm = 5  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './word_recognition_aster_sxent_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
	SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, SOS_VALUE, EOS_VALUE))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, sys_args = build_aster_model(label_converter, image_height, image_width, image_channel, lang, max_word_len, EOS_VALUE, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
			#	print(f'Skip {name} as it has already been initialized.')
			#	continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		#optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		optimizer = torch.optim.Adadelta(model_params, lr=sys_args.lr, weight_decay=sys_args.weight_decay)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, None, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_opennmt():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 20  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './word_recognition_onmt_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
			#	print(f'Skip {name} as it has already been initialized.')
			#	continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)
	model.generator = model.generator.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_rare1_and_opennmt():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 20  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './word_recognition_rare1+onmt_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_rare1_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
				print(f'Skip {name} as it has already been initialized.')
				continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_rare2_and_opennmt():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 20  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './word_recognition_rare2+onmt_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_rare2_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
				print(f'Skip {name} as it has already been initialized.')
				continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_by_aster_and_opennmt():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 5  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 20  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './word_recognition_aster+onmt_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_aster_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
				print(f'Skip {name} as it has already been initialized.')
				continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_word_using_mixup():
	image_height, image_width, image_channel = 32, 100, 3
	#image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	# File-based words: 504,279.
	is_mixed_words_used = True
	if is_mixed_words_used:
		num_simple_examples, num_random_examples = int(5e5), int(5e5)  # For mixed words.
	else:
		word_type = 'simple_word'  # {'simple_word', 'random_word', 'file_based_word'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
	max_word_len = 25  # Max. word length.
	word_len_interval = (1, max_word_len)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'ctc', 'xent', 'nll'}.
	max_gradient_norm = 5  # Gradient clipping value.
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	if loss_type == 'ctc':
		model_filepath_base = './word_recognition_mixup_rare1_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	elif loss_type in ['xent', 'nll']:
		model_filepath_base = './word_recognition_mixup_rare1_attn_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_word_len, image_height, image_width, image_channel)
	else:
		raise ValueError('Invalid loss type, {}'.format(loss_type))
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if loss_type == 'ctc':
		BLANK_LABEL = '<BLANK>'  # The BLANK label for CTC.
		label_converter = swl_langproc_util.TokenConverter([BLANK_LABEL] + list(charset), pad_value=None)  # NOTE [info] >> It's a trick. The ID of the BLANK label is set to 0.
		assert label_converter.encode([BLANK_LABEL], is_bare_output=True)[0] == 0, '{} != 0'.format(label_converter.encode([BLANK_LABEL], is_bare_output=True)[0])
		BLANK_LABEL_INT = 0 #label_converter.encode([BLANK_LABEL], is_bare_output=True)[0]
		SOS_VALUE, EOS_VALUE = None, None
		num_suffixes = 0
	elif loss_type in ['xent', 'nll']:
		if is_individual_pad_value_used:
			# When the pad value is the ID of a valid token.
			PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
			PAD_TOKEN = '<PAD>'
			label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
			assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
			assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
		else:
			# When the pad value = the ID of <SOS> token.
			label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
		SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_words_used:
		train_dataloader, test_dataloader = create_mixed_word_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_word_data_loaders(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, SOS_VALUE, EOS_VALUE))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_rare1_mixup_model(label_converter, image_height, image_width, image_channel, loss_type, lang, max_word_len, num_suffixes, SOS_VALUE, BLANK_LABEL if loss_type == 'ctc' else None, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
				print(f'Skip {name} as it has already been initialized.')
				continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = None

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def evaluate_word_recognizer():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.
	max_label_len = 10
	is_preloaded_image_used = False

	lang = 'kor'  # {'kor', 'eng'}.
	shuffle = True
	num_workers = 8
	batch_size = 64
	is_individual_pad_value_used = False

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
	SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
	num_suffixes = 1

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
	aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

	test_transform = torchvision.transforms.Compose([
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	import aihub_data

	print('Start creating a dataset and a dataloader...')
	start_time = time.time()
	test_dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used, transform=test_transform, target_transform=test_target_transform)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('End creating a dataset and a dataloader: {} secs.'.format(time.time() - start_time))
	print('#examples = {}.'.format(len(test_dataset)))
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, SOS_VALUE, EOS_VALUE))

	# Show data info.
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	print('Start building a model...')
	start_time = time.time()
	if True:
		# For RARE2.
		model_filepath_to_load = './training_outputs_word_recognition/word_recognition_rare2_attn_xent_gradclip_allparams_nopad_kor_large_ch20_64x1280x3_acc0.9514_epoch3.pth'
		assert model_filepath_to_load is not None

		model, infer_functor, _, _ = build_rare2_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, max_time_steps=max_label_len + num_suffixes, sos_value=SOS_VALUE, device=device)
	elif False:
		# For ASTER + OpenNMT.
		model_filepath_to_load = './training_outputs_word_recognition/word_recognition_aster_sxent_nogradclip_allparams_nopad_kor_ch5_64x640x3_acc0.8449_epoch3.pth'
		assert model_filepath_to_load is not None

		model, infer_functor, _, _ = build_aster_model(label_converter, image_height, image_width, image_channel, lang, max_label_len, EOS_VALUE, device=device)
	elif False:
		# For ASTER + OpenNMT.
		model_filepath_to_load = './training_outputs_word_recognition/word_recognition_aster+onmt_xent_nogradclip_allparams_nopad_kor_large_ch20_64x1280x3_acc0.9325_epoch2.pth'
		assert model_filepath_to_load is not None

		model, infer_functor, _, _ = build_aster_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, device=device)
	else:
		raise ValueError('Undefined model.')
	print('End building a model: {} secs.'.format(time.time() - start_time))

	# Load a model.
	print('Start loading a pretrained model from {}.'.format(model_filepath_to_load))
	start_time = time.time()
	model = load_model(model_filepath_to_load, model, device=device)
	print('End loading a pretrained model: {} secs.'.format(time.time() - start_time))

	model = model.to(device)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=True, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def infer_by_word_recognizer():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.
	max_label_len = 10
	is_preloaded_image_used = False

	lang = 'kor'  # {'kor', 'eng'}.
	shuffle = True
	num_workers = 8
	batch_size = 64
	is_individual_pad_value_used = False

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
	SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
	num_suffixes = 1

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
	aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

	test_transform = torchvision.transforms.Compose([
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	import aihub_data

	print('Start creating a dataset and a dataloader...')
	start_time = time.time()
	test_dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used, transform=test_transform, target_transform=test_target_transform)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('End creating a dataset and a dataloader: {} secs.'.format(time.time() - start_time))
	print('#examples = {}.'.format(len(test_dataset)))
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, SOS_VALUE, EOS_VALUE))

	# Show data info.
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	inputs, outputs = list(), list()
	try:
		for images, labels, _ in test_dataloader:
			inputs.append(images)
			outputs.append(labels)
	except Exception as ex:
		print('Exception raised: {}.'.format(ex))
	inputs = torch.cat(inputs)
	outputs = torch.cat(outputs)

	#--------------------
	# Build a model.

	print('Start building a model...')
	start_time = time.time()
	if True:
		# For RARE2.
		model_filepath_to_load = './training_outputs_word_recognition/word_recognition_rare2_attn_xent_gradclip_allparams_nopad_kor_large_ch20_64x1280x3_acc0.9514_epoch3.pth'
		assert model_filepath_to_load is not None

		model, infer_functor, _, _ = build_rare2_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, max_time_steps=max_label_len + num_suffixes, sos_value=SOS_VALUE, device=device)
	elif False:
		# For ASTER.
		model_filepath_to_load = './training_outputs_word_recognition/word_recognition_aster_sxent_nogradclip_allparams_nopad_kor_ch5_64x640x3_acc0.8449_epoch3.pth'
		assert model_filepath_to_load is not None

		model, infer_functor, _, _ = build_aster_model(label_converter, image_height, image_width, image_channel, lang, max_label_len, EOS_VALUE, device=device)
	elif False:
		# For ASTER + OpenNMT.
		model_filepath_to_load = './training_outputs_word_recognition/word_recognition_aster+onmt_xent_nogradclip_allparams_nopad_kor_large_ch20_64x1280x3_acc0.9325_epoch2.pth'
		assert model_filepath_to_load is not None

		model, infer_functor, _, _ = build_aster_and_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type=None, device=device)
	else:
		raise ValueError('Undefined model.')
	print('End building a model: {} secs.'.format(time.time() - start_time))

	# Load a model.
	print('Start loading a pretrained model from {}.'.format(model_filepath_to_load))
	start_time = time.time()
	model = load_model(model_filepath_to_load, model, device=device)
	print('End loading a pretrained model: {} secs.'.format(time.time() - start_time))

	model = model.to(device)

	#--------------------
	# Infer by the model.

	print('Start inferring...')
	start_time = time.time()
	model.eval()
	infer_by_text_recognition_model(model, infer_functor, label_converter, inputs, outputs=None, batch_size=batch_size, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=True, device=device)
	print('End inferring: {} secs.'.format(time.time() - start_time))

def recognize_textline_by_opennmt():
	#image_height, image_width, image_channel = 32, 100, 3
	image_height, image_width, image_channel = 64, 640, 3
	#image_height, image_width, image_channel = 64, 1280, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	is_mixed_textlines_used = True
	if is_mixed_textlines_used:
		num_simple_examples, num_random_examples, num_trdg_examples = int(5e5), int(5e5), int(5e5)  # For mixed text lines.
	else:
		textline_type = 'simple_textline'  # {'simple_textline', 'random_textline', 'trdg_textline', 'file_based_textline'}.
		num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple, random, and TRDG text lines.
	max_textline_len = 50  # Max. text line length.
	word_len_interval = (1, 20)
	word_count_interval = (1, 5)
	space_count_interval = (1, 3)
	char_space_ratio_interval = (0.8, 1.25)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	lang = 'kor'  # {'kor', 'eng'}.
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	loss_type = 'xent'  # {'xent', 'nll'}.
	#max_gradient_norm = 20  # Gradient clipping value.
	max_gradient_norm = None
	num_epochs = 20
	batch_size = 64
	log_print_freq = 1000

	is_trained = True
	is_model_loaded = False
	is_model_initialized = True
	is_all_model_params_optimized = True
	is_individual_pad_value_used = False

	gradclip_nogradclip = 'gradclip' if max_gradient_norm else 'nogradclip'
	allparams_gradparams = 'allparams' if is_all_model_params_optimized else 'gradparams'
	pad_nopad = 'pad' if is_individual_pad_value_used else 'nopad'
	model_filepath_base = './text_line_recognition_onmt_{}_{}_{}_{}_{}_ch{}_{}x{}x{}'.format(loss_type, gradclip_nogradclip, allparams_gradparams, pad_nopad, lang, max_textline_len, image_height, image_width, image_channel)
	model_filepath_format = model_filepath_base + '{}.pth'
	print('Model filepath: {}.'.format(model_filepath_format.format('')))

	if is_model_loaded:
		model_filepath_to_load = None
	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	if lang == 'kor':
		charset, wordset = tg_util.construct_charset(), tg_util.construct_word_set(korean=True, english=True)
		font_list = construct_font(korean=True, english=False)
	elif lang == 'eng':
		charset, wordset = tg_util.construct_charset(hangeul=False), tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font(korean=False, english=True)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	#--------------------
	# Prepare data.

	if is_individual_pad_value_used:
		# When the pad value is the ID of a valid token.
		PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
		PAD_TOKEN = '<PAD>'
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
		assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
	else:
		# When the pad value = the ID of <SOS> token.
		label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)

	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if is_mixed_textlines_used:
		train_dataloader, test_dataloader = create_mixed_text_line_data_loaders(label_converter, wordset, chars, num_simple_examples, num_random_examples, num_trdg_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	else:
		train_dataloader, test_dataloader = create_text_line_data_loaders(textline_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, batch_size, shuffle, num_workers)
	classes, num_classes = label_converter.tokens, label_converter.num_tokens
	print('#classes = {}.'.format(num_classes))
	print('Pad value = {}, <SOS> = {}, <EOS> = {}.'.format(label_converter.pad_value, label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, mode='Train')
	show_text_data_info(test_dataloader, label_converter, visualize=False, mode='Test')

	#--------------------
	# Build a model.

	model, infer_functor, forward_functor, criterion = build_opennmt_model(label_converter, image_height, image_width, image_channel, lang, loss_type, device)

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
			#	print(f'Skip {name} as it has already been initialized.')
			#	continue
			try:
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.kaiming_normal_(param)
			except Exception as ex:  # For batch normalization.
				if 'weight' in name:
					param.data.fill_(1)
				continue
	if is_model_loaded:
		# Load a model.
		model = load_model(model_filepath_to_load, model, device=device)

	model = model.to(device)
	model.generator = model.generator.to(device)

	#--------------------
	# Train the model.

	if is_trained:
		if is_all_model_params_optimized:
			model_params = list(model.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, model.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			print('#trainable model parameters = {}.'.format(num_model_params))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

		# Define an optimizer.
		#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
		#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
		#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
		#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		model, best_model_filepath = train_text_recognition_model(model, criterion, forward_functor, infer_functor, label_converter, train_dataloader, test_dataloader, optimizer, num_epochs, log_print_freq, model_filepath_format, scheduler, max_gradient_norm, model_params, device)
		print('End training: {} secs.'.format(time.time() - start_time))

		# Save a model.
		if best_model_filepath:
			model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				print('Copied the best trained model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				print('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
		else:
			if model:
				model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
				save_model(model_filepath, model)

	#--------------------
	# Evaluate the model.

	print('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, infer_functor, label_converter, test_dataloader, is_case_sensitive=False, show_acc_per_char=True, is_error_cases_saved=False, device=device)
	print('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_text_using_craft_and_character_recognizer():
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils
	import craft.test_utils as test_utils

	image_height, image_width, image_channel = 64, 64, 3

	model_name = 'ResNet'  # {'VGG', 'ResNet', 'RCNN'}.
	input_channel, output_channel = image_channel, 1024

	charset = tg_util.construct_charset()

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	# For CRAFT.
	craft_trained_model_filepath = './craft/craft_mlt_25k.pth'
	craft_refiner_model_filepath = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.
	craft_refine = False  # Enable link refiner.
	craft_cuda = gpu >= 0  # Use cuda for inference.

	# For char recognizer.
	#recognizer_model_filepath = './craft/char_recognition.pth'
	recognizer_model_filepath = './craft/char_recognition_mixup.pth'

	#image_filepath = './craft/images/I3.jpg'
	image_filepath = './craft/images/book_1.png'
	#image_filepath = './craft/images/book_2.png'

	output_dir_path = './char_recog_results'

	#--------------------
	label_converter = swl_langproc_util.TokenConverter(list(charset))
	num_classes = label_converter.num_tokens

	print('Start loading CRAFT...')
	start_time = time.time()
	craft_net, craft_refine_net = test_utils.load_craft(craft_trained_model_filepath, craft_refiner_model_filepath, craft_refine, craft_cuda)
	print('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	print('Start loading char recognizer...')
	start_time = time.time()
	import rare.model_char
	recognizer = rare.model_char.create_model(model_name, input_channel, output_channel, num_classes)

	recognizer = load_model(recognizer_model_filepath, recognizer, device=device)
	recognizer = recognizer.to(device)
	print('End loading char recognizer: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, ch_bboxes_lst, score_text = test_utils.run_char_craft(rgb, craft_net, craft_refine_net, craft_cuda)
	print('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
		os.makedirs(output_dir_path, exist_ok=True)

		print('Start inferring...')
		start_time = time.time()
		image = cv2.imread(image_filepath)

		"""
		cv2.imshow('Input', image)
		rgb1, rgb2 = image.copy(), image.copy()
		for bbox, ch_bboxes in zip(bboxes, ch_bboxes_lst):
			color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
			cv2.drawContours(rgb1, [np.round(np.expand_dims(bbox, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
			for bbox in ch_bboxes:
				cv2.drawContours(rgb2, [np.round(np.expand_dims(bbox, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
		cv2.imshow('Word BBox', rgb1)
		cv2.imshow('Char BBox', rgb2)
		cv2.waitKey(0)
		"""

		ch_images = list()
		rgb = image.copy()
		for i, ch_bboxes in enumerate(ch_bboxes_lst):
			imgs = list()
			color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
			for j, bbox in enumerate(ch_bboxes):
				(x1, y1), (x2, y2) = np.min(bbox, axis=0), np.max(bbox, axis=0)
				x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
				img = image[y1:y2+1,x1:x2+1]
				imgs.append(img)

				cv2.imwrite(os.path.join(output_dir_path, 'ch_{}_{}.png'.format(i, j)), img)

				cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)
			ch_images.append(imgs)
		cv2.imwrite(os.path.join(output_dir_path, 'char_bbox.png'), rgb)

		#--------------------
		transform = torchvision.transforms.Compose([
			#RandomInvert(),
			ConvertPILMode(mode='RGB'),
			ResizeImage(image_height, image_width),
			#torchvision.transforms.Resize((image_height, image_width)),
			#torchvision.transforms.CenterCrop((image_height, image_width)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
		])

		recognizer.eval()
		with torch.no_grad():
			for idx, imgs in enumerate(ch_images):
				imgs = torch.stack([transform(Image.fromarray(img)) for img in imgs]).to(device)

				predictions = recognizer(imgs)

				_, predictions = torch.max(predictions, 1)
				predictions = predictions.cpu().numpy()
				print('\t{}: {} (int), {} (str).'.format(idx, predictions, ''.join(label_converter.decode(predictions))))
		print('End inferring: {} secs.'.format(time.time() - start_time))
	else:
		print('No text detected.')

def recognize_word_using_craft_and_word_recognizer():
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils
	import craft.test_utils as test_utils

	image_height, image_width, image_channel = 64, 64, 3

	num_fiducials = 20  # The number of fiducial points of TPS-STN.
	input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	hidden_size = 256  # The size of the LSTM hidden states.
	transformer = 'TPS'  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.
	decoder = 'Attn'  # The type of decoder. {'CTC', 'Attn'}.
	max_word_len = 25  # Max. word length.

	charset = tg_util.construct_charset()

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu >= 0 else 'cpu')
	print('Device: {}.'.format(device))

	# For CRAFT.
	craft_trained_model_filepath = './craft/craft_mlt_25k.pth'
	craft_refiner_model_filepath = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.
	craft_refine = False  # Enable link refiner.
	craft_cuda = gpu >= 0  # Use cuda for inference.

	# For word recognizer.
	recognizer_model_filepath = './craft/word_recognition.pth'
	#recognizer_model_filepath = './craft/word_recognition_mixup.pth'

	#image_filepath = './craft/images/I3.jpg'
	image_filepath = './craft/images/book_1.png'
	#image_filepath = './craft/images/book_2.png'

	output_dir_path = './word_recog_results'

	#--------------------
	if decoder == 'CTC':
		BLANK_LABEL = '<BLANK>'  # The BLANK label for CTC.
		label_converter = swl_langproc_util.TokenConverter([BLANK_LABEL] + list(charset), pad_value=None)  # NOTE [info] >> It's a trick. The ID of the BLANK label is set to 0.
		assert label_converter.encode([BLANK_LABEL], is_bare_output=True)[0] == 0, '{} != 0'.format(label_converter.encode([BLANK_LABEL], is_bare_output=True)[0])
		BLANK_LABEL_INT = 0 #label_converter.encode([BLANK_LABEL], is_bare_output=True)[0]
		SOS_VALUE, EOS_VALUE = None, None
		num_suffixes = 0
	else:
		is_individual_pad_value_used = False
		if is_individual_pad_value_used:
			# When the pad value is the ID of a valid token.
			PAD_VALUE = len(charset)  # NOTE [info] >> It's a trick which makes the pad value the ID of a valid token.
			PAD_TOKEN = '<PAD>'
			label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], use_sos=True, use_eos=True, pad_value=PAD_VALUE)
			assert label_converter.pad_value == PAD_VALUE, '{} != {}'.format(label_converter.pad_value, PAD_VALUE)
			assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_VALUE, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_VALUE)
		else:
			# When the pad value = the ID of <SOS> token.
			label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=swl_langproc_util.TokenConverter.SOS)
		SOS_VALUE, EOS_VALUE = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1
	num_classes = label_converter.num_tokens

	print('Start loading CRAFT...')
	start_time = time.time()
	craft_net, craft_refine_net = test_utils.load_craft(craft_trained_model_filepath, craft_refiner_model_filepath, craft_refine, craft_cuda)
	print('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	print('Start loading word recognizer...')
	start_time = time.time()
	import rare.model
	recognizer = rare.model.Model(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_word_len + num_suffixes, SOS_VALUE, label_converter.pad_value, transformer, feature_extractor, sequence_model, decoder)

	recognizer = load_model(recognizer_model_filepath, recognizer, device=device)
	recognizer = recognizer.to(device)
	print('End loading word recognizer: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, polys, score_text = test_utils.run_word_craft(rgb, craft_net, craft_refine_net, craft_cuda)
	print('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
		os.makedirs(output_dir_path, exist_ok=True)

		print('Start inferring...')
		start_time = time.time()
		image = cv2.imread(image_filepath)

		"""
		cv2.imshow('Input', image)
		rgb1, rgb2 = image.copy(), image.copy()
		for bbox, poly in zip(bboxes, polys):
			color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
			cv2.drawContours(rgb1, [np.round(np.expand_dims(bbox, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
			cv2.drawContours(rgb2, [np.round(np.expand_dims(poly, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
		cv2.imshow('BBox', rgb1)
		cv2.imshow('Poly', rgb2)
		cv2.waitKey(0)
		"""

		word_images = list()
		rgb = image.copy()
		for idx, bbox in enumerate(bboxes):
			(x1, y1), (x2, y2) = np.min(bbox, axis=0), np.max(bbox, axis=0)
			x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
			img = image[y1:y2+1,x1:x2+1]
			word_images.append(img)

			cv2.imwrite(os.path.join(output_dir_path, 'word_{}.png'.format(idx)), img)

			cv2.rectangle(rgb, (x1, y1), (x2, y2), (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_4)
		cv2.imwrite(os.path.join(output_dir_path, 'word_bbox.png'), rgb)

		#--------------------
		transform = torchvision.transforms.Compose([
			#RandomInvert(),
			ConvertPILMode(mode='RGB'),
			ResizeImage(image_height, image_width),
			#torchvision.transforms.Resize((image_height, image_width)),
			#torchvision.transforms.CenterCrop((image_height, image_width)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5,) * image_channel, (0.5,) * image_channel)
		])

		recognizer.eval()
		with torch.no_grad():
			imgs = torch.stack([transform(Image.fromarray(img)) for img in word_images]).to(device)

			predictions = recognizer(imgs)

			_, predictions = torch.max(predictions, 1)
			predictions = predictions.cpu().numpy()
			for idx, pred in enumerate(predictions):
				print('\t{}: {} (int), {} (str).'.format(idx, pred, ''.join(label_converter.decode(pred))))
		print('End inferring: {} secs.'.format(time.time() - start_time))
	else:
		print('No text detected.')

def main():
	#recognize_character()
	#recognize_character_using_mixup()

	# Recognize text using CRAFT (scene text detector) + character recognizer.
	#recognize_text_using_craft_and_character_recognizer()

	#--------------------
	#recognize_word_by_rare1()  # Use RARE #1.
	#recognize_word_by_rare2()  # Use RARE #2.
	#recognize_word_by_aster()  # Use ASTER.
	#recognize_word_by_opennmt()  # Use OpenNMT.
	#recognize_word_by_rare1_and_opennmt()  # Use RARE #1 (encoder) + OpenNMT (decoder).
	#recognize_word_by_rare2_and_opennmt()  # Use RARE #2 (encoder) + OpenNMT (decoder).
	#recognize_word_by_aster_and_opennmt()  # Use ASTER (encoder) + OpenNMT (decoder).
	#recognize_word_using_mixup()  # Use RARE #1. Not working.

	#evaluate_word_recognizer()
	infer_by_word_recognizer()

	# Recognize word using CRAFT (scene text detector) + word recognizer.
	#recognize_word_using_craft_and_word_recognizer()  # Use RARE #1.

	#--------------------
	#recognize_textline_by_opennmt()  # Use OpenNMT.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
