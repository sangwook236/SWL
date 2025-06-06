#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, collections, math, random, functools, itertools, operator, pickle, shutil, glob, datetime, time
import argparse, logging, logging.handlers
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import matplotlib.pyplot as plt
import swl.machine_learning.util as swl_ml_util
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util
import text_data, aihub_data
import opennmt_util
#import mixup.vgg, mixup.resnet

def save_model(model_filepath, model, logger=None):
	#torch.save(model.state_dict(), model_filepath)
	torch.save({'state_dict': model.state_dict()}, model_filepath)
	if logger: logger.info('Saved a model to {}.'.format(model_filepath))

def load_model(model_filepath, model, logger=None, device='cuda'):
	loaded_data = torch.load(model_filepath, map_location=device)
	#model.load_state_dict(loaded_data)
	model.load_state_dict(loaded_data['state_dict'])
	if logger: logger.info('Loaded a model from {}.'.format(model_filepath))
	return model

# REF [function] >> construct_font() in font_test.py.
def construct_font(font_types):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	font_dir_paths = list()
	if 'kor-small' in font_types:
		font_dir_paths.append(font_base_dir_path + '/kor_small')
	if 'kor-large' in font_types:
		font_dir_paths.append(font_base_dir_path + '/kor_large')
	if 'kor-receipt' in font_types:
		font_dir_paths.append(font_base_dir_path + '/kor_receipt')
	if 'eng-small' in font_types:
		font_dir_paths.append(font_base_dir_path + '/eng_small')
	if 'eng-large' in font_types:
		font_dir_paths.append(font_base_dir_path + '/eng_large')
	if 'eng-receipt' in font_types:
		font_dir_paths.append(font_base_dir_path + '/eng_receipt')
	assert font_dir_paths

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
		iaa.Sometimes(0.8, iaa.Sequential([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent along x-axis and -20 to +20 percent along y-axis.
				rotate=(-30, 30),  # Rotate by -10 to +10 degrees.
				#shear=(-10, 10),  # Shear by -10 to +10 degrees.
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
		iaa.Sometimes(0.8, iaa.Sequential([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent along x-axis and -10 to +10 percent along y-axis.
				#rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
				rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
				#shear=(-5, 5),  # Shear by -5 to +5 degrees.
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

def create_textline_augmenter():
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

class EnlargeImageForGeometricTransformation(object):
	def __init__(self, height=None, width=None, is_pil=True):
		self.height, self.width = height, width
		self.resize_functor = self._enlarge_background_by_pil if is_pil else self._enlarge_background_by_opencv

	def __call__(self, x):
		return self.resize_functor(x, self.height, self.width)

	@staticmethod
	def _enlarge_background_by_pil(image, height=None, width=None):
		if height is None or width is None:
			width = height = math.ceil(math.sqrt(image.width**2 + image.height**2))
		sx, sy = (width - image.width) // 2, (height - image.height) // 2
		enlarged = Image.new(image.mode, (width, height), color=0)
		enlarged.paste(image, (sx, sy))
		return enlarged

	@staticmethod
	def _enlarge_background_by_opencv(image, height=None, width=None):
		if height is None or width is None:
			height = width = math.ceil(math.sqrt(image.shape[0]**2 + image.shape[1]**2))
		sy, sx = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
		enlarged = np.zeros((height, width) + image.shape[2:], dtype=image.dtype)
		enlarged[sy:sy+image.shape[0],sx:sx+image.shape[1]] = image
		return enlarged

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

class ConvertNumpyToRGB(object):
	def __call__(self, x):
		if x.ndim == 1:
			return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
		elif x.ndim == 3:
			return x
		else: raise ValueError('Invalid dimension, {}'.format(x.ndim))

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

class ResizeToBelowMaxWidth(object):
	def __init__(self, height, max_width, warn_about_small_image, is_pil=True, logger=None):
		self.height, self.max_width = height, max_width
		self.resize_functor = self._resize_by_pil if is_pil else self._resize_by_opencv
		self.logger = logger

		self.min_height_threshold, self.min_width_threshold = 20, 20
		self.warn = self._warn_about_small_image if warn_about_small_image else lambda *args, **kwargs: None

	def __call__(self, x):
		return self.resize_functor(x, self.height, self.max_width)

	def _resize_by_opencv(self, image, height, max_width, *args, **kwargs):
		interpolation = cv2.INTER_AREA
		hi, wi = image.shape[:2]
		self.warn(hi, wi)
		scale_factor = height / hi
		#min_width = min(max_width, int(wi * scale_factor))
		min_width = max(min(max_width, int(wi * scale_factor)), height // 2)
		assert min_width > 0 and height > 0
		return cv2.resize(image, (min_width, height), interpolation=interpolation)

	def _resize_by_pil(self, image, height, max_width, *args, **kwargs):
		interpolation = Image.BICUBIC
		wi, hi = image.size
		self.warn(hi, wi)
		scale_factor = height / hi
		#min_width = min(max_width, int(wi * scale_factor))
		min_width = max(min(max_width, int(wi * scale_factor)), height // 2)
		assert min_width > 0 and height > 0
		return image.resize((min_width, height), resample=interpolation)

	def _warn_about_small_image(self, height, width):
		if height < self.min_height_threshold:
			if self.logger: self.logger.warning('Too small image: The image height {} should be larger than or equal to {}.'.format(height, self.min_height_threshold))
		#if width < self.min_width_threshold:
		#	if self.logger: self.logger.warning('Too small image: The image width {} should be larger than or equal to {}.'.format(width, self.min_width_threshold))

class ToPaddedIntTensor(object):
	def __init__(self, pad, max_len):
		self.pad, self.max_len = pad, max_len

	def __call__(self, lst):
		padded = [self.pad] * self.max_len
		min_len = min(len(lst), self.max_len)
		padded[:min_len] = lst[:min_len]
		return torch.IntTensor(padded)

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

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
def generate_trdg_datasets(num_train_examples, num_test_examples, image_channel, lang_infos, background_infos, distortion_types, distortion_directions, label_converter, max_text_len, font_size, num_words, is_variable_length, train_transform, train_target_transform, test_transform, test_target_transform):
	generator_kwargs = {
		'skewing_angle': 0, 'random_skew': False,  # In degrees counter clockwise.
		#'blur': 0, 'random_blur': False,  # Blur radius.
		'blur': 2, 'random_blur': True,  # Blur radius.
		'distorsion_type': 0, 'distorsion_orientation': 0,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
		#'distorsion_type': 3, 'distorsion_orientation': 2,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
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

	train_datasets, test_datasets = list(), list()
	divisor = len(lang_infos) * len(background_infos) * len(distortion_types) * len(distortion_directions) * 2  # Words in dictionary or randomly generated words.
	for lang, font_filepaths in lang_infos:
		for background_type, background_image_dir in background_infos:
			generator_kwargs['background_type'] = background_type
			generator_kwargs['image_dir'] = background_image_dir
			for distortion_type in distortion_types:
				generator_kwargs['distorsion_type'] = distortion_type
				for distortion_direction in distortion_directions:
					generator_kwargs['distorsion_orientation'] = distortion_direction
					for is_randomly_generated in [False, True]:
						train_datasets.append(text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_train_examples // divisor, image_channel, max_text_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=train_transform, target_transform=train_target_transform, **generator_kwargs))
						test_datasets.append(text_data.TextRecognitionDataGeneratorTextLineDataset(label_converter, lang, num_test_examples // divisor, image_channel, max_text_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated, transform=test_transform, target_transform=test_target_transform, **generator_kwargs))

	return train_datasets, test_datasets

def create_char_datasets(char_type, label_converter, charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, is_pil=True, logger=None):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_char_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop, logger=logger),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	test_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		#AugmentByImgaug(create_char_augmenter()),
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width, logger=logger),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

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
		train_datasets, test_datasets = list(), list()
		if True:
			# REF [function] >> generate_chars_from_chars74k_data() in chars74k_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/chars74k/English/Img/char_images.txt'
			is_preloaded_image_used = True
			dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_dataset = MySubsetDataset(train_subset, transform=train_transform)
			test_dataset = MySubsetDataset(test_subset, transform=test_transform)
		if True:
			# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/char_images_kr.txt'
			is_preloaded_image_used = True
			dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_dataset = MySubsetDataset(train_subset, transform=train_transform)
			test_dataset = MySubsetDataset(test_subset, transform=test_transform)
		if True:
			# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/char_images_en.txt'
			is_preloaded_image_used = True
			dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_dataset = MySubsetDataset(train_subset, transform=train_transform)
			test_dataset = MySubsetDataset(test_subset, transform=test_transform)
		if True:
			# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/char_images_kr.txt'
			is_preloaded_image_used = True
			dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_dataset = MySubsetDataset(train_subset, transform=train_transform)
			test_dataset = MySubsetDataset(test_subset, transform=test_transform)
		if True:
			# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/char_images_en.txt'
			is_preloaded_image_used = True
			dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_dataset = MySubsetDataset(train_subset, transform=train_transform)
			test_dataset = MySubsetDataset(test_subset, transform=test_transform)
		assert train_datasets, 'NO train dataset'
		assert test_datasets, 'NO test dataset'

		train_dataset = torch.utils.data.ConcatDataset(train_datasets)
		test_dataset = torch.utils.data.ConcatDataset(test_datasets)
	else:
		raise ValueError('Invalid dataset type: {}'.format(char_type))

	return train_dataset, test_dataset

def create_mixed_char_datasets(label_converter, charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, is_pil=True, logger=None):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_char_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop, logger=logger),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	test_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		#AugmentByImgaug(create_char_augmenter()),
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width, logger=logger),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	train_datasets, test_datasets = list(), list()
	if True:
		chars = list(charset * num_simple_char_examples_per_class)
		random.shuffle(chars)
		num_train_examples = int(len(chars) * train_test_ratio)
		train_datasets.append(text_data.SimpleCharacterDataset(label_converter, chars[:num_train_examples], image_channel, font_list, font_size_interval, color_functor=color_functor, transform=train_transform))
		test_datasets.append(text_data.SimpleCharacterDataset(label_converter, chars[num_train_examples:], image_channel, font_list, font_size_interval, color_functor=color_functor, transform=test_transform))
	if True:
		chars = list(charset * num_noisy_examples_per_class)
		random.shuffle(chars)
		num_train_examples = int(len(chars) * train_test_ratio)
		train_datasets.append(text_data.NoisyCharacterDataset(label_converter, chars[:num_train_examples], image_channel, char_clipping_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform))
		test_datasets.append(text_data.NoisyCharacterDataset(label_converter, chars[num_train_examples:], image_channel, char_clipping_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform))
	if True:
		# REF [function] >> generate_chars_from_chars74k_data() in chars74k_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/chars74k/English/Img/char_images.txt'
		is_preloaded_image_used = True
		dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform))
	if True:
		# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/char_images_kr.txt'
		is_preloaded_image_used = True
		dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform))
		test_datasets.append( MySubsetDataset(test_subset, transform=test_transform))
	if True:
		# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/char_images_en.txt'
		is_preloaded_image_used = True
		dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform))
	if True:
		# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/char_images_kr.txt'
		is_preloaded_image_used = True
		dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform))
	if True:
		# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/char_images_en.txt'
		is_preloaded_image_used = True
		dataset = text_data.FileBasedCharacterDataset(label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform))
	assert train_datasets, 'NO train dataset'
	assert test_datasets, 'NO test dataset'

	train_dataset = torch.utils.data.ConcatDataset(train_datasets)
	test_dataset = torch.utils.data.ConcatDataset(test_datasets)

	return train_dataset, test_dataset

def create_word_datasets(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, is_pil=True, logger=None):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		EnlargeImageForGeometricTransformation(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop, logger=logger),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		EnlargeImageForGeometricTransformation(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_word_augmenter()),
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width, logger=logger),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	test_target_transform = torch.IntTensor

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	if word_type == 'simple_word':
		train_dataset = text_data.SimpleWordDataset(label_converter, wordset, num_train_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
		test_dataset = text_data.SimpleWordDataset(label_converter, wordset, num_test_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	elif word_type == 'random_word':
		train_dataset = text_data.RandomWordDataset(label_converter, chars, num_train_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
		test_dataset = text_data.RandomWordDataset(label_converter, chars, num_test_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	elif word_type == 'trdg_word':
		font_size = image_height
		num_words = 1
		is_variable_length = False

		lang_infos = list()
		if True:
			lang = 'kr'
			font_types = ['kor-large']  # {'kor-small', 'kor-large', 'kor-receipt'}.
			font_filepaths = construct_font(font_types)
			font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if True:
			lang = 'en'
			if False:
				#font_filepaths = trdg.utils.load_fonts(lang)
				font_filepaths = list()
			else:
				font_types = ['eng-large']  # {'eng-small', 'eng-large', 'eng-receipt'}.
				font_filepaths = construct_font(font_types)
				font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if False:
			lang = 'cn'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
			#font_filepaths = trdg.utils.load_fonts(lang)
			font_filepaths = list()
			lang_infos.append((lang, font_filepaths))
		# background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
		background_infos = [(0, None)]
		#background_infos = [(0, None), (3, os.path.join(data_base_dir_path, 'background_image'))]
		# distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random).
		# distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
		distortion_types, distortion_directions = (1, 2, 3), (0, 1, 2)

		train_datasets, test_datasets = generate_trdg_datasets(num_train_examples, num_test_examples, image_channel, lang_infos, background_infos, distortion_types, distortion_directions, label_converter, max_word_len, font_size, num_words, is_variable_length, train_transform, train_target_transform, test_transform, test_target_transform)
		train_dataset = torch.utils.data.ConcatDataset(train_datasets)
		test_dataset = torch.utils.data.ConcatDataset(test_datasets)
	elif word_type == 'aihub_word':
		# AI-Hub printed text dataset.
		#	#syllables = 558,600, #words = 277,150, #sentences = 42,350.
		aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
		aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

		image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.
		is_preloaded_image_used = False
		dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_word_len, is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	elif word_type == 'file_based_word':
		# File-based words: 504,279.
		train_datasets, test_datasets = list(), list()
		if True:
			# E2E-MLT Korean dataset.
			# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/word_images_kr.txt'
			is_preloaded_image_used = False
			dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
			test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
		if True:
			# E2E-MLT English dataset.
			# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/word_images_en.txt'
			is_preloaded_image_used = False
			dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
			test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
		if True:
			# ICDAR RRC-MLT 2019 Korean dataset.
			# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/word_images_kr.txt'
			is_preloaded_image_used = True
			dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
			test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
		if True:
			# ICDAR RRC-MLT 2019 English dataset.
			# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
			image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/word_images_en.txt'
			is_preloaded_image_used = True
			dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
			test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
		assert train_datasets, 'NO train dataset'
		assert test_datasets, 'NO test dataset'

		train_dataset = torch.utils.data.ConcatDataset(train_datasets)
		test_dataset = torch.utils.data.ConcatDataset(test_datasets)
	else:
		raise ValueError('Invalid dataset type: {}'.format(word_type))

	return train_dataset, test_dataset

def create_mixed_word_datasets(label_converter, wordset, chars, num_simple_examples, num_random_examples, num_trdg_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_word_len, word_len_interval, font_list, font_size_interval, color_functor, is_pil=True, logger=None):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		EnlargeImageForGeometricTransformation(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop, logger=logger),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		EnlargeImageForGeometricTransformation(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_word_augmenter()),
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width, logger=logger),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	test_target_transform = torch.IntTensor

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	train_datasets, test_datasets = list(), list()
	if True:
		num_train_examples = int(num_simple_examples * train_test_ratio)
		train_datasets.append(text_data.SimpleWordDataset(label_converter, wordset, num_train_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(text_data.SimpleWordDataset(label_converter, wordset, num_simple_examples - num_train_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform))
	if True:
		num_train_examples = int(num_random_examples * train_test_ratio)
		train_datasets.append(text_data.RandomWordDataset(label_converter, chars, num_train_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(text_data.RandomWordDataset(label_converter, chars, num_random_examples - num_train_examples, image_channel, max_word_len, word_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform))
	if True:
		font_size = image_height
		num_words = 1
		is_variable_length = True

		num_train_examples = int(num_trdg_examples * train_test_ratio)
		num_test_examples = num_trdg_examples - num_train_examples

		lang_infos = list()
		if True:
			lang = 'kr'
			font_types = ['kor-large']  # {'kor-small', 'kor-large', 'kor-receipt'}.
			font_filepaths = construct_font(font_types)
			font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if True:
			lang = 'en'
			if False:
				#font_filepaths = trdg.utils.load_fonts(lang)
				font_filepaths = list()
			else:
				font_types = ['eng-large']  # {'eng-small', 'eng-large', 'eng-receipt'}.
				font_filepaths = construct_font(font_types)
				font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if False:
			lang = 'cn'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
			#font_filepaths = trdg.utils.load_fonts(lang)
			font_filepaths = list()
			lang_infos.append((lang, font_filepaths))
		# background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
		background_infos = [(0, None)]
		#background_infos = [(0, None), (3, os.path.join(data_base_dir_path, 'background_image'))]
		# distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random).
		# distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
		distortion_types, distortion_directions = (1, 2, 3), (0, 1, 2)

		trdg_train_datasets, trdg_test_datasets = generate_trdg_datasets(num_train_examples, num_test_examples, image_channel, lang_infos, background_infos, distortion_types, distortion_directions, label_converter, max_word_len, font_size, num_words, is_variable_length, train_transform, train_target_transform, test_transform, test_target_transform)
		train_datasets.extend(trdg_train_datasets)
		test_datasets.extend(trdg_test_datasets)
	if False:
		# AI-Hub printed text dataset.
		#	#syllables = 558,600, #words = 277,150, #sentences = 42,350.
		aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
		aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

		image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.
		is_preloaded_image_used = False
		dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_word_len, is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	# File-based words: 504,279.
	if True:
		# E2E-MLT Korean dataset.
		# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/word_images_kr.txt'
		is_preloaded_image_used = False
		dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	if True:
		# E2E-MLT English dataset.
		# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/word_images_en.txt'
		is_preloaded_image_used = False
		dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	if True:
		# ICDAR RRC-MLT 2019 Korean dataset.
		# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/word_images_kr.txt'
		is_preloaded_image_used = True
		dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	if True:
		# ICDAR RRC-MLT 2019 English dataset.
		# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/word_images_en.txt'
		is_preloaded_image_used = True
		dataset = text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	assert train_datasets, 'NO train dataset'
	assert test_datasets, 'NO test dataset'

	train_dataset = torch.utils.data.ConcatDataset(train_datasets)
	test_dataset = torch.utils.data.ConcatDataset(test_datasets)

	return train_dataset, test_dataset

def create_textline_datasets(textline_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, is_pil=True, logger=None):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_textline_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop, logger=logger),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		#AugmentByImgaug(create_textline_augmenter()),
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width, logger=logger),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	test_target_transform = torch.IntTensor

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

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

		lang_infos = list()
		if True:
			lang = 'kr'
			font_types = ['kor-large']  # {'kor-small', 'kor-large', 'kor-receipt'}.
			font_filepaths = construct_font(font_types)
			font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if True:
			lang = 'en'
			if False:
				#font_filepaths = trdg.utils.load_fonts(lang)
				font_filepaths = list()
			else:
				font_types = ['eng-large']  # {'eng-small', 'eng-large', 'eng-receipt'}.
				font_filepaths = construct_font(font_types)
				font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if False:
			lang = 'cn'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
			#font_filepaths = trdg.utils.load_fonts(lang)
			font_filepaths = list()
			lang_infos.append((lang, font_filepaths))
		# background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
		background_infos = [(0, None)]
		#background_infos = [(0, None), (3, os.path.join(data_base_dir_path, 'background_image'))]
		# distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random).
		# distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
		distortion_types, distortion_directions = (1, 2, 3), (0, 1, 2)

		train_datasets, test_datasets = generate_trdg_datasets(num_train_examples, num_test_examples, image_channel, lang_infos, background_infos, distortion_types, distortion_directions, label_converter, max_textline_len, font_size, num_words, is_variable_length, train_transform, train_target_transform, test_transform, test_target_transform)
		train_dataset = torch.utils.data.ConcatDataset(train_datasets)
		test_dataset = torch.utils.data.ConcatDataset(test_datasets)
	elif textline_type == 'aihub_textline':
		# AI-Hub printed text dataset.
		#	#syllables = 558,600, #words = 277,150, #sentences = 42,350.
		aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
		aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

		image_types_to_load = ['sentence']  # {'syllable', 'word', 'sentence'}.
		#image_types_to_load = ['word', 'sentence']  # {'syllable', 'word', 'sentence'}.
		is_preloaded_image_used = False
		dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_textline_len, is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	elif textline_type == 'file_based_textline':
		# File-based text lines: 55,835.
		train_datasets, test_datasets = list(), list()
		if True:
			# ICDAR 2019 SROIE dataset.
			is_preloaded_image_used = False
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.jpg', recursive=False))
			label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.txt', recursive=False))
			train_datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used, transform=train_transform, target_transform=train_target_transform))
			image_label_info_filepath = data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_test_text_line/labels.txt'
			test_datasets.append(text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used, transform=test_transform, target_transform=test_target_transform))
		if True:
			# SiliconMinds receipt data.
			image_label_info_filepath = data_base_dir_path + '/text/receipt/sminds/receipt_text_line/labels.txt'
			is_preloaded_image_used = True
			dataset = text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
			test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
		if True:
			# ePapyrus receipt data.
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.png', recursive=False))
			label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.txt', recursive=False))
			is_preloaded_image_used = True
			dataset = text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used)

			num_examples = len(dataset)
			num_train_examples = int(num_examples * train_test_ratio)
			train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
			train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
			test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
		assert train_datasets, 'NO train dataset'
		assert test_datasets, 'NO test dataset'

		train_dataset = torch.utils.data.ConcatDataset(train_datasets)
		test_dataset = torch.utils.data.ConcatDataset(test_datasets)
	else:
		raise ValueError('Invalid dataset type: {}'.format(textline_type))

	return train_dataset, test_dataset

def create_mixed_textline_datasets(label_converter, wordset, chars, num_simple_examples, num_random_examples, num_trdg_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, is_pil=True, logger=None):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		AugmentByImgaug(create_textline_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height_before_crop, image_width_before_crop, logger=logger),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		#EnlargeBackground(height=None, width=None, is_pil=is_pil),
		#AugmentByImgaug(create_textline_augmenter()),
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeToFixedSize(image_height, image_width, logger=logger),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	test_target_transform = torch.IntTensor

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	train_datasets, test_datasets = list(), list()
	if True:
		num_train_examples = int(num_simple_examples * train_test_ratio)
		train_datasets.append(text_data.SimpleTextLineDataset(label_converter, wordset, num_train_examples, image_channel, max_textline_len, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(text_data.SimpleTextLineDataset(label_converter, wordset, num_simple_examples - num_train_examples, image_channel, max_textline_len, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform))
	if True:
		num_train_examples = int(num_random_examples * train_test_ratio)
		train_datasets.append(text_data.RandomTextLineDataset(label_converter, chars, num_train_examples, image_channel, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(text_data.RandomTextLineDataset(label_converter, chars, num_random_examples - num_train_examples, image_channel, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform))
	if True:
		font_size = image_height
		num_words = word_count_interval[1]  # TODO [check] >>
		is_variable_length = True

		num_train_examples = int(num_trdg_examples * train_test_ratio)
		num_test_examples = num_trdg_examples - num_train_examples

		lang_infos = list()
		if True:
			lang = 'kr'
			font_types = ['kor-large']  # {'kor-small', 'kor-large', 'kor-receipt'}.
			font_filepaths = construct_font(font_types)
			font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if True:
			lang = 'en'
			if False:
				#font_filepaths = trdg.utils.load_fonts(lang)
				font_filepaths = list()
			else:
				font_types = ['eng-large']  # {'eng-small', 'eng-large', 'eng-receipt'}.
				font_filepaths = construct_font(font_types)
				font_filepaths, _ = zip(*font_filepaths)
			lang_infos.append((lang, font_filepaths))
		if False:
			lang = 'cn'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
			#font_filepaths = trdg.utils.load_fonts(lang)
			font_filepaths = list()
			lang_infos.append((lang, font_filepaths))
		# background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
		background_infos = [(0, None)]
		#background_infos = [(0, None), (3, os.path.join(data_base_dir_path, 'background_image'))]
		# distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random).
		# distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
		distortion_types, distortion_directions = (1, 2, 3), (0, 1, 2)

		trdg_train_datasets, trdg_test_datasets = generate_trdg_datasets(num_train_examples, num_test_examples, image_channel, lang_infos, background_infos, distortion_types, distortion_directions, label_converter, max_textline_len, font_size, num_words, is_variable_length, train_transform, train_target_transform, test_transform, test_target_transform)
		train_datasets.extend(trdg_train_datasets)
		test_datasets.extend(trdg_test_datasets)
	if False:
		# AI-Hub printed text dataset.
		#	#syllables = 558,600, #words = 277,150, #sentences = 42,350.
		aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
		aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

		image_types_to_load = ['sentence']  # {'syllable', 'word', 'sentence'}.
		#image_types_to_load = ['word', 'sentence']  # {'syllable', 'word', 'sentence'}.
		is_preloaded_image_used = False
		dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_textline_len, is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	# File-based text lines: 55,835.
	if True:
		# ICDAR 2019 SROIE dataset.
		is_preloaded_image_used = False
		image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.jpg', recursive=False))
		label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.txt', recursive=False))
		train_datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used, transform=train_transform, target_transform=train_target_transform))
		image_label_info_filepath = data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_test_text_line/labels.txt'
		test_datasets.append(text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used, transform=test_transform, target_transform=test_target_transform))
	if True:
		# SiliconMinds receipt data.
		image_label_info_filepath = data_base_dir_path + '/text/receipt/sminds/receipt_text_line/labels.txt'
		is_preloaded_image_used = True
		dataset = text_data.InfoFileBasedTextLineDataset(label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	if True:
		# ePapyrus receipt data.
		image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.png', recursive=False))
		label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/epapyrus/epapyrus_20190618/receipt_text_line/*.txt', recursive=False))
		is_preloaded_image_used = True
		dataset = text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=is_preloaded_image_used)

		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)
		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_datasets.append(MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform))
		test_datasets.append(MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform))
	assert train_datasets, 'NO train dataset'
	assert test_datasets, 'NO test dataset'

	train_dataset = torch.utils.data.ConcatDataset(train_datasets)
	test_dataset = torch.utils.data.ConcatDataset(test_datasets)

	return train_dataset, test_dataset

def concatenate_labels(labels, eos_id, lengths=None):
	concat_labels = list()
	if lengths == None:
		for lbl in labels:
			try:
				concat_labels.append(lbl[:lbl.index(eos_id)+1])
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

def show_char_data_info(dataloader, label_converter, visualize=True, nrow=8, mode='Train', logger=None):
	dataiter = iter(dataloader)
	images, labels = dataiter.next()
	images_np, labels_np = images.numpy(), labels.numpy()

	if logger: logger.info('{} image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, images_np.shape, images_np.dtype, np.min(images_np), np.max(images_np)))
	if logger: logger.info('{} label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, labels_np.shape, labels_np.dtype, np.min(labels_np), np.max(labels_np)))

	if visualize:
		if logger: logger.info('Labels: {}.'.format(' '.join(label_converter.decode(labels_np))))
		show_image(torchvision.utils.make_grid(images, nrow))

def show_text_data_info(dataloader, label_converter, visualize=True, nrow=2, mode='Train', logger=None):
	dataiter = iter(dataloader)
	images, labels, label_lens = dataiter.next()
	images_np, labels_np, label_lens_np = images.numpy(), labels.numpy(), label_lens.numpy()

	if logger: logger.info('{} image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, images_np.shape, images_np.dtype, np.min(images_np), np.max(images_np)))
	if logger: logger.info('{} label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, labels_np.shape, labels_np.dtype, np.min(labels_np), np.max(labels_np)))
	if logger: logger.info('{} label length: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, label_lens_np.shape, label_lens_np.dtype, np.min(label_lens_np), np.max(label_lens_np)))

	if visualize:
		#if logger: logger.info('Labels: {}.'.format(' '.join([label_converter.decode(lbl) for lbl in images_np])))
		for idx, (lbl, ll) in enumerate(zip(labels_np, label_lens_np)):
			if logger: logger.info('Label #{} (len = {}): {} (int), {} (str).'.format(idx, ll, lbl, label_converter.decode(lbl)))
		show_image(torchvision.utils.make_grid(images, nrow))

def show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char=False, logger=None):
	#for idx in range(num_classes):
	#	if logger: logger.info('Accuracy of {:5s} = {:2d} %.'.format(classes[idx], 100 * correct_char_class_count[idx] / total_char_class_count[idx] if total_char_class_count[idx] > 0 else -1))
	accuracies = [100 * correct_char_class_count[idx] / total_char_class_count[idx] if total_char_class_count[idx] > 0 else -1 for idx in range(num_classes)]
	#if logger: logger.info('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	#hist, bin_edges = np.histogram(accuracies, bins=range(0, 101), density=False)
	#if logger: logger.info('Per-character accuracy histogram: {}.'.format({bb: hh for bb, hh in zip(bin_edges, hist)}))
	if logger: logger.info('Per-character accuracy histogram: {}.'.format({bb: hh for bb, hh in zip(bin_edges, hist) if hh > 0}))

	if show_acc_per_char:
		valid_accuracies = [100 * correct_char_class_count[idx] / total_char_class_count[idx] for idx in range(num_classes) if total_char_class_count[idx] > 0]
		acc_thresh = 98
		if logger: logger.info('Per-character accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
		if logger: logger.info('Per-character accuracy (< {}) = {}.'.format(acc_thresh, {classes[idx]: round(acc, 2) for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]) if acc < acc_thresh}))

def compute_per_char_accuracy(inputs, outputs, predictions, num_classes):
	import difflib

	isjunk = None
	#isjunk = lambda x: x == '\n\r'
	#isjunk = lambda x: x == ' \t\n\r'

	correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
	for img, gt, pred in zip(inputs, outputs, predictions):
		matcher = difflib.SequenceMatcher(isjunk, gt, pred)
		for mth in matcher.get_matching_blocks():
			if mth.size != 0:
				for idx in range(mth.a, mth.a + mth.size):
					correct_char_class_count[gt[idx]] += 1
		for gl in gt:
			total_char_class_count[gl] += 1
	return correct_char_class_count, total_char_class_count
	"""
	correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
	for img, gt, pred in zip(inputs, outputs, predictions):
		for gl, pl in zip(gt, pred):
			if gl == pl: correct_char_class_count[gl] += 1
			total_char_class_count[gl] += 1
	return correct_char_class_count, total_char_class_count
	"""

def compute_simple_matching_accuracy(inputs, outputs, predictions, label_converter, is_case_sensitive, error_cases_dir_path=None, error_idx=0):
	total_text_count = len(outputs)
	correct_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0, 0
	error_cases = list()
	for img, gt, pred in zip(inputs, outputs, predictions):
		gt, pred = label_converter.decode(gt), label_converter.decode(pred)
		gt_case, pred_case = (gt, pred) if is_case_sensitive else (gt.lower(), pred.lower())

		if gt_case == pred_case:
			correct_text_count += 1
		elif error_cases_dir_path is not None:
			cv2.imwrite(os.path.join(error_cases_dir_path, 'image_{}.png'.format(error_idx)), img)
			error_cases.append((gt, pred))
			error_idx += 1

		gt_words, pred_words = gt_case.split(' '), pred_case.split(' ')
		total_word_count += max(len(gt_words), len(pred_words))
		correct_word_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gt_words, pred_words))))

		total_char_count += max(len(gt), len(pred))
		correct_char_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gt_case, pred_case))))

	return correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count, error_cases

def compute_sequence_matching_ratio(inputs, outputs, predictions, label_converter, is_case_sensitive, error_cases_dir_path=None, error_idx=0):
	import difflib

	isjunk = None
	#isjunk = lambda x: x == '\n\r'
	#isjunk = lambda x: x == ' \t\n\r'

	total_matching_ratio = 0
	error_cases = list()
	for img, gt, pred in zip(inputs, outputs, predictions):
		gt, pred = label_converter.decode(gt), label_converter.decode(pred)
		gt_case, pred_case = (gt, pred) if is_case_sensitive else (gt.lower(), pred.lower())

		matching_ratio = difflib.SequenceMatcher(isjunk, gt_case, pred_case).ratio()
		total_matching_ratio += matching_ratio
		if matching_ratio < 1 and error_cases_dir_path is not None:
			cv2.imwrite(os.path.join(error_cases_dir_path, 'image_{}.png'.format(error_idx)), img)
			error_cases.append((gt, pred))
			error_idx += 1

	return total_matching_ratio, error_cases

def build_char_model(label_converter, image_channel, loss_type):
	model_name = 'ResNet'  # {'VGG', 'ResNet', 'RCNN'}.
	input_channel, output_channel = image_channel, 1024

	class MySimpleModelBase(torch.nn.Module):
		def __init__(self, model_name, input_channel, output_channel, num_classes):
			super().__init__()

			import rare.model_char
			self.model = rare.model_char.create_model(model_name, input_channel, output_channel, num_classes)

		def forward(self, x, *args, **kwargs):
			return self.model(x)

		def train_forward(self, criterion, inputs, outputs, device='cuda'):
			inputs, outputs = inputs.to(device), outputs.to(device)

			model_outputs = self.model(inputs)
			return criterion(model_outputs, outputs), model_outputs

	model = MySimpleModelBase(model_name, input_channel, output_channel, label_converter.num_tokens)

	#--------------------
	# Define a loss function.
	if loss_type == 'xent':
		criterion = torch.nn.CrossEntropyLoss()
	elif loss_type == 'nll':
		criterion = torch.nn.NLLLoss(reduction='sum')
	else:
		raise ValueError('Invalid loss type, {}'.format(loss_type))

	return model, criterion

def build_char_mixup_model(label_converter, image_channel, loss_type):
	model_name = 'ResNet'  # {'VGG', 'ResNet', 'RCNN'}.
	input_channel, output_channel = image_channel, 1024

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	class MySimpleModelBase(torch.nn.Module):
		def __init__(self, model_name, input_channel, output_channel, num_classes):
			super().__init__()

			# REF [function] >> mnist_predefined_mixup_test() in ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/run_mnist_cnn.py.
			import rare.model_char
			self.model = rare.model_char.create_mixup_model(model_name, input_channel, output_channel, num_classes)

		def forward(self, x, target=None, mixup_input=False, mixup_hidden=False, mixup_alpha=None, cutout=False, cutout_size=None, device=None, *args, **kwargs):
			return self.model(x, target, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)

		def train_forward(self, criterion, inputs, outputs, device='cuda'):
			inputs, outputs = inputs.to(device), outputs.to(device)

			model_outputs, outputs = self.model(inputs, outputs, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
			return criterion(model_outputs, torch.argmax(outputs, dim=1)), model_outputs

	model = MySimpleModelBase(model_name, input_channel, output_channel, label_converter.num_tokens)

	#--------------------
	# Define a loss function.
	if loss_type == 'xent':
		criterion = torch.nn.CrossEntropyLoss()
	elif loss_type == 'nll':
		criterion = torch.nn.NLLLoss(reduction='sum')
	else:
		raise ValueError('Invalid loss type, {}'.format(loss_type))

	return model, criterion

def build_rare1_model(image_height, image_width, image_channel, max_time_steps, num_classes, pad_id, sos_id, blank_label, lang, loss_type=None):
	transformer = None  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.
	if loss_type and loss_type == 'ctc':
		decoder = 'CTC'  # The type of decoder. {'CTC', 'Attn'}.
	else:
		decoder = 'Attn'  # The type of decoder. {'CTC', 'Attn'}.

	num_fiducials = 20  # The number of fiducial points of TPS-STN.
	input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	if lang == 'kor':
		hidden_size = 1024  # The size of the LSTM hidden states.
	else:
		hidden_size = 512  # The size of the LSTM hidden states.

	class MySimpleModelBase(torch.nn.Module):
		def __init__(self, image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder):
			super().__init__()

			import rare.model
			self.model = rare.model.Model(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

		def forward(self, inputs, outputs=None, is_train=False, device='cuda', *args, **kwargs):
			raise NotImplementedError

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			raise NotImplementedError

	if not loss_type and loss_type == 'ctc':
		class MySimpleModelForCTC(MySimpleModelBase):
			def __init__(self, image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder):
				super().__init__(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

			def forward(self, inputs, outputs=None, is_train=False, device='cuda', *args, **kwargs):
				# TODO [check] >> Not yet tested.
				return self.model(inputs, outputs, is_train, device)

			def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
				model_outputs = self.model(inputs.to(device), None, is_train=True, device=device).log_softmax(2)

				N, T = model_outputs.shape[:2]
				model_outputs = model_outputs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C).
				model_output_lens = torch.full([N], T, dtype=torch.int32, device=device)

				# TODO [check] >> To avoid CTC loss issue, disable cuDNN for the computation of the CTC loss.
				# https://github.com/jpuigcerver/PyLaia/issues/16
				torch.backends.cudnn.enabled = False
				cost = criterion(model_outputs, outputs.to(device), model_output_lens, output_lens.to(device))
				torch.backends.cudnn.enabled = True
				return cost, model_outputs

		model = MySimpleModelForCTC(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)
	else:
		class MySimpleModel(MySimpleModelBase):
			def __init__(self, image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder):
				super().__init__(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

			def forward(self, inputs, outputs=None, is_train=False, device='cuda', *args, **kwargs):
				model_outputs = self.model(inputs, outputs, is_train=is_train, device=device)

				model_outputs = torch.argmax(model_outputs, dim=-1)
				if outputs is None:
					return model_outputs
				else:
					return model_outputs, outputs

			def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
				outputs = outputs.long()

				# Construct inputs for one-step look-ahead.
				decoder_inputs = outputs[:,:-1]
				# Construct outputs for one-step look-ahead.
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				decoder_output_lens = output_lens - 1

				model_outputs = self.model(inputs.to(device), decoder_inputs.to(device), is_train=True, device=device)

				# TODO [check] >> How to compute loss?
				# NOTE [info] >> All examples in a batch are concatenated together.
				#	Can each example be handled individually?
				#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
				"""
				mask = torch.full(decoder_outputs.shape[:2], False, dtype=torch.bool)
				for idx, ll in enumerate(decoder_output_lens):
					mask[idx,:ll].fill_(True)
				model_outputs[mask == False] = pad_id
				return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
				"""
				concat_model_outputs, concat_decoder_outputs = list(), list()
				for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
					concat_model_outputs.append(mo[:dl])
					concat_decoder_outputs.append(do[:dl])
				return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

		model = MySimpleModel(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

	#--------------------
	if loss_type is not None:
		if loss_type == 'ctc':
			# Define a loss function.
			criterion = torch.nn.CTCLoss(blank=blank_label, zero_infinity=True)  # The BLANK label.

		elif loss_type in ['xent', 'nll']:
			# Define a loss function.
			if loss_type == 'xent':
				criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore the PAD ID.
			elif loss_type == 'nll':
				criterion = torch.nn.NLLLoss(ignore_index=pad_id, reduction='sum')  # Ignore the PAD ID.
	else:
		criterion = None

	return model, criterion

def build_rare1_mixup_model(image_height, image_width, image_channel, max_time_steps, num_classes, pad_id, sos_id, blank_label, lang, loss_type=None):
	transformer = None  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.
	if loss_type and loss_type == 'ctc':
		decoder = 'CTC'  # The type of decoder. {'CTC', 'Attn'}.
	else:
		decoder = 'Attn'  # The type of decoder. {'CTC', 'Attn'}.

	num_fiducials = 20  # The number of fiducial points of TPS-STN.
	input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	if lang == 'kor':
		hidden_size = 1024  # The size of the LSTM hidden states.
	else:
		hidden_size = 512  # The size of the LSTM hidden states.

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	class MySimpleModelBase(torch.nn.Module):
		def __init__(self, image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder):
			super().__init__()

			# FIXME [error] >> rare.model.Model_MixUp is not working.
			import rare.model
			self.model = rare.model.Model_MixUp(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

		def forward(self, inputs, outputs=None, is_train=False, device='cuda', *args, **kwargs):
			raise NotImplementedError

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			raise NotImplementedError

	if not loss_type and loss_type == 'ctc':
		class MySimpleModelForCTC(MySimpleModelBase):
			def __init__(self, image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder):
				super().__init__(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

			def forward(self, inputs, outputs=None, is_train=False, device='cuda', *args, **kwargs):
				# TODO [check] >> Not yet tested.
				return self.model(inputs, outputs, is_train, device)

			def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
				model_outputs = self.model(inputs.to(device), None, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, is_train=True, device=device).log_softmax(2)

				N, T = model_outputs.shape[:2]
				model_outputs = model_outputs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C).
				model_output_lens = torch.full([N], T, dtype=torch.int32, device=device)

				# TODO [check] >> To avoid CTC loss issue, disable cuDNN for the computation of the CTC loss.
				# https://github.com/jpuigcerver/PyLaia/issues/16
				torch.backends.cudnn.enabled = False
				cost = criterion(model_outputs, outputs.to(device), model_output_lens, output_lens.to(device))
				torch.backends.cudnn.enabled = True
				return cost, model_outputs

		model = MySimpleModelForCTC(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)
	else:
		class MySimpleModel(MySimpleModelBase):
			def __init__(self, image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder):
				super().__init__(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

			def forward(self, inputs, outputs=None, is_train=False, device='cuda', *args, **kwargs):
				model_outputs = self.model(inputs, outputs, is_train=is_train, device=device)

				model_outputs = torch.argmax(model_outputs, dim=-1)
				if outputs is None:
					return model_outputs, None
				else:
					return model_outputs, outputs

			def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
				outputs = outputs.long()

				# Construct inputs for one-step look-ahead.
				decoder_inputs = outputs[:,:-1]
				# Construct outputs for one-step look-ahead.
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				decoder_output_lens = output_lens - 1

				model_outputs = self.model(inputs.to(device), decoder_inputs.to(device), is_train=True, device=device)

				# TODO [check] >> How to compute loss?
				# NOTE [info] >> All examples in a batch are concatenated together.
				#	Can each example be handled individually?
				#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
				"""
				mask = torch.full(decoder_outputs.shape[:2], False, dtype=torch.bool)
				for idx, ll in enumerate(decoder_output_lens):
					mask[idx,:ll].fill_(True)
				model_outputs[mask == False] = pad_id
				return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.to(device).contiguous().view(-1))
				"""
				concat_model_outputs, concat_decoder_outputs = list(), list()
				for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
					concat_model_outputs.append(mo[:dl])
					concat_decoder_outputs.append(do[:dl])
				return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

		model = MySimpleModel(image_height, image_width, num_classes, num_fiducials, input_channel, output_channel, hidden_size, max_time_steps, sos_id, pad_id, transformer, feature_extractor, sequence_model, decoder)

	#--------------------
	if loss_type is not None:
		if loss_type == 'ctc':
			# Define a loss function.
			criterion = torch.nn.CTCLoss(blank=blank_label, zero_infinity=True)  # The BLANK label.

		elif loss_type in ['xent', 'nll']:
			# Define a loss function.
			if loss_type == 'xent':
				criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore the PAD ID.
			elif loss_type == 'nll':
				criterion = torch.nn.NLLLoss(ignore_index=pad_id, reduction='sum')  # Ignore the PAD ID.
	else:
		criterion = None

	return model, criterion

def build_rare2_model(image_height, image_width, image_channel, max_time_steps, num_classes, pad_id, sos_id, lang, loss_type=None):
	if lang == 'kor':
		hidden_size = 512  # The size of the LSTM hidden states.
	else:
		hidden_size = 256  # The size of the LSTM hidden states.
	num_rnns = 2
	embedding_size = 256
	use_leaky_relu = False

	class MySimpleModel(torch.nn.Module):
		def __init__(self, image_height, image_channel, num_classes, max_time_steps, hidden_size, num_rnns, embedding_size, use_leaky_relu, sos_id):
			super().__init__()

			import rare.crnn_lang
			self.model = rare.crnn_lang.CRNN(imgH=image_height, nc=image_channel, nclass=num_classes, nh=hidden_size, n_rnn=num_rnns, num_embeddings=embedding_size, leakyRelu=use_leaky_relu, max_time_steps=max_time_steps, sos_id=sos_id)

		def forward(self, inputs, outputs=None, output_lens=None, device='cuda', *args, **kwargs):
			#model_outputs = self.model(inputs, decoder_inputs, decoder_input_lens, device=device)
			model_outputs = self.model(inputs, outputs, output_lens, device=device)

			model_outputs = torch.argmax(model_outputs, dim=-1)

			if outputs is None or output_lens is None:
				return model_outputs
			else:
				# Construct inputs for one-step look-ahead.
				#decoder_inputs = outputs[:,:-1]
				#decoder_input_lens = output_lens - 1
				# Construct outputs for one-step look-ahead.
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				#decoder_output_lens = output_lens - 1

				"""
				separated_model_outputs = np.zeros(decoder_outputs.shape, model_outputs.dtype)
				start_idx = 0
				for idx, dl in enumerate(decoder_output_lens):
					end_idx = start_idx + dl
					separated_model_outputs[idx,:dl] = model_outputs[start_idx:end_idx]
					start_idx = end_idx
				return separated_model_outputs, decoder_outputs
				"""
				return model_outputs, decoder_outputs

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			outputs = outputs.long()

			# Construct inputs for one-step look-ahead.
			decoder_inputs = outputs[:,:-1]
			decoder_input_lens = output_lens - 1
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
			decoder_output_lens = output_lens - 1

			model_outputs = self.model(inputs.to(device), decoder_inputs.to(device), decoder_input_lens.to(device), device=device)

			# TODO [check] >> How to compute loss?
			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), outputs.to(device).contiguous().view(-1))
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
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

	model = MySimpleModel(image_height, image_channel, num_classes, max_time_steps, hidden_size, num_rnns, embedding_size, use_leaky_relu, sos_id)

	#--------------------
	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore the PAD ID.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=pad_id, reduction='sum')  # Ignore the PAD ID.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))
	else:
		criterion = None

	return model, criterion

def build_aster_model(image_height, image_width, image_channel, max_time_steps, num_classes, pad_id, eos_id, lang, logger=None):
	if lang == 'kor':
		hidden_size = 512  # The size of the LSTM hidden states.
	else:
		hidden_size = 256  # The size of the LSTM hidden states.

	import aster.config
	#sys_args = aster.config.get_args(sys.argv[1:])
	sys_args = aster.config.get_args([])
	sys_args.with_lstm = True
	#sys_args.STN_ON = True

	if logger: logger.info('ASTER options: {}.'.format(vars(sys_args)))

	class MySimpleModel(torch.nn.Module):
		def __init__(self, image_height, image_channel, num_classes, max_time_steps, hidden_size, eos_id, sys_args):
			super().__init__()

			import aster.model_builder
			self.model = aster.model_builder.ModelBuilder(
				sys_args, arch=sys_args.arch, input_height=image_height, input_channel=image_channel,
				hidden_size=hidden_size, rec_num_classes=num_classes,
				sDim=sys_args.decoder_sdim, attDim=sys_args.attDim,
				max_len_labels=max_time_steps, eos=eos_id,
				STN_ON=sys_args.STN_ON
			)

		def forward(self, inputs, outputs=None, output_lens=None, device='cuda', *args, **kwargs):
			if outputs is None or output_lens is None:
				input_dict =  {
					'images': inputs,
					'rec_targets': None,
					'rec_lengths': None
				}

				model_output_dict = self.model(input_dict, device=device)

				model_outputs = model_output_dict['output']['pred_rec']  # [batch size, max label len].
				#model_output_scores = model_output_dict['output']['pred_rec_score']  # [batch size, max label len].

				return model_outputs
			else:
				# Construct outputs for one-step look-ahead.
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				decoder_output_lens = output_lens - 1

				input_dict = {
					'images': inputs,
					'rec_targets': decoder_outputs,
					'rec_lengths': decoder_output_lens
				}

				model_output_dict = self.model(input_dict, device=device)

				#loss = model_output_dict['losses']['loss_rec']
				model_outputs = model_output_dict['output']['pred_rec']  # [batch size, max label len].
				#model_output_scores = model_output_dict['output']['pred_rec_score']  # [batch size, max label len].

				# TODO [check] >>
				#return model_outputs, outputs
				return model_outputs, decoder_outputs

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			"""
			# Construct inputs for one-step look-ahead.
			if eos_id != pad_id:
				decoder_inputs = outputs[:,:-1].clone()
				decoder_inputs[decoder_inputs == eos_id] = pad_id  # Remove <EOS> tokens.
			else: decoder_inputs = outputs[:,:-1]
			"""
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
			decoder_output_lens = output_lens - 1

			input_dict = {
				'images': inputs.to(device),
				'rec_targets': decoder_outputs.to(device),
				'rec_lengths': decoder_output_lens.to(device)
			}

			model_output_dict = self.model(input_dict, device=device)

			loss = model_output_dict['losses']['loss_rec']  # aster.sequence_cross_entropy_loss.SequenceCrossEntropyLoss.
			model_outputs = model_output_dict['output']['pred_rec']  # [batch size, max label len].
			return loss, model_outputs

	model = MySimpleModel(image_height, image_channel, num_classes, max_time_steps, hidden_size, eos_id, sys_args)

	#--------------------
	# Define a loss function.
	#if loss_type == 'xent':
	#	criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore the PAD ID.
	#elif loss_type == 'nll':
	#	criterion = torch.nn.NLLLoss(ignore_index=pad_id, reduction='sum')  # Ignore the PAD ID.
	#else:
	#	raise ValueError('Invalid loss type, {}'.format(loss_type))

	return model, sys_args

def build_decoder_and_generator_for_opennmt(num_classes, word_vec_size, hidden_size, num_layers=2, bidirectional_encoder=True):
	import onmt

	embedding_dropout = 0.3
	rnn_type = 'LSTM'
	num_layers = num_layers
	#hidden_size = hidden_size
	dropout = 0.3

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

	decoder = onmt.decoders.InputFeedRNNDecoder(
		rnn_type=rnn_type, bidirectional_encoder=bidirectional_encoder,
		num_layers=num_layers, hidden_size=hidden_size,
		attn_type='general', attn_func='softmax',
		coverage_attn=False, context_gate=None,
		copy_attn=False, dropout=dropout, embeddings=tgt_embeddings,
		reuse_copy_attn=False, copy_attn_type='general'
	)
	generator = torch.nn.Sequential(
		torch.nn.Linear(in_features=hidden_size, out_features=num_classes, bias=True),
		onmt.modules.util_class.Cast(dtype=torch.float32),
		torch.nn.LogSoftmax(dim=-1)
	)
	return decoder, generator

def build_opennmt_model(image_height, image_width, image_channel, max_time_steps, encoder_type, label_converter, lang, loss_type=None):
	bidirectional_encoder = True
	num_encoder_layers, num_decoder_layers = 2, 2
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 1024, 1024
	else:
		word_vec_size = 80
		encoder_rnn_size, decoder_hidden_size = 512, 512

	#--------------------
	import onmt, onmt.translate
	import torchtext
	import torchtext_util

	tgt_field = torchtext.data.Field(
		sequential=True, use_vocab=True, init_token=label_converter.SOS, eos_token=label_converter.EOS, fix_length=None,
		dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
		tokenize=None, tokenizer_language='kr',  # TODO [check] >> tokenizer_language is not valid.
		#tokenize=functools.partial(onmt.inputters.inputter._feature_tokenize, layer=0, feat_delim=None, truncate=None), tokenizer_language='en',
		include_lengths=False, batch_first=False, pad_token=label_converter.PAD, pad_first=False, unk_token=label_converter.UNKNOWN,
		truncate_first=False, stop_words=None, is_target=False
	)
	#tgt_field.build_vocab([label_converter.tokens], specials=[label_converter.UNKNOWN, label_converter.PAD], specials_first=False)  # Sort vocabulary + add special tokens, <unknown>, <pad>, <bos>, and <eos>.
	if label_converter.PAD in [label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN]:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	else:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.PAD, label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	assert label_converter.num_tokens == len(tgt_field.vocab.itos)
	assert len(list(filter(lambda pair: pair[0] != pair[1], zip(label_converter.tokens, tgt_field.vocab.itos)))) == 0

	tgt_vocab = tgt_field.vocab
	tgt_unk = tgt_vocab.stoi[tgt_field.unk_token]
	tgt_bos = tgt_vocab.stoi[tgt_field.init_token]
	tgt_eos = tgt_vocab.stoi[tgt_field.eos_token]
	tgt_pad = tgt_vocab.stoi[tgt_field.pad_token]

	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')

	is_beam_search_used = True
	if is_beam_search_used:
		beam_size = 30
		n_best = 1
		ratio = 0.0
	else:
		beam_size = 1
		random_sampling_topk, random_sampling_temp = 1, 1
		n_best = 1  # Fixed. For handling translation results.
	min_length, max_length = 0, max_time_steps
	block_ngram_repeat = 0
	#ignore_when_blocking = frozenset()
	#exclusion_idxs = {tgt_vocab.stoi[t] for t in ignore_when_blocking}
	exclusion_idxs = set()

	if encoder_type == 'onmt':
		#embedding_dropout = 0.3
		dropout = 0.3

		#src_embeddings = None

		encoder = onmt.encoders.ImageEncoder(
			num_layers=num_encoder_layers, bidirectional=bidirectional_encoder,
			rnn_size=encoder_rnn_size, dropout=dropout, image_chanel_size=image_channel
		)
	elif encoder_type == 'rare1':
		transformer = None  # The type of transformer. {None, 'TPS'}.
		feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
		sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.

		num_fiducials = 20  # The number of fiducial points of TPS-STN.
		output_channel = 512  # The number of output channel of feature extractor.

		encoder = opennmt_util.Rare1ImageEncoder(
			image_height, image_width, image_channel, output_channel,
			hidden_size=encoder_rnn_size, num_layers=num_encoder_layers, bidirectional=bidirectional_encoder,
			transformer=transformer, feature_extractor=feature_extractor, sequence_model=sequence_model,
			num_fiducials=num_fiducials
		)
	elif encoder_type == 'rare2':
		is_stn_used = False
		if is_stn_used:
			num_fiducials = 20  # The number of fiducial points of TPS-STN.
		else:
			num_fiducials = 0  # No TPS-STN.

		encoder = opennmt_util.Rare2ImageEncoder(
			image_height, image_width, image_channel,
			hidden_size=encoder_rnn_size, num_layers=num_encoder_layers, bidirectional=bidirectional_encoder,
			num_fiducials=num_fiducials
		)
	elif encoder_type == 'aster':
		is_stn_used = False
		if is_stn_used:
			num_fiducials = 20  # The number of fiducial points of TPS-STN.
		else:
			num_fiducials = 0  # No TPS-STN.

		encoder = opennmt_util.AsterImageEncoder(
			image_height, image_width, image_channel, label_converter.num_tokens,
			hidden_size=encoder_rnn_size,
			num_fiducials=num_fiducials
		)
	else:
		raise ValueError('Invalid encoder type: {}'.format(encoder_type))
	decoder, generator = build_decoder_and_generator_for_opennmt(label_converter.num_tokens, word_vec_size, hidden_size=decoder_hidden_size, num_layers=num_decoder_layers, bidirectional_encoder=bidirectional_encoder)

	class MySimpleModel(torch.nn.Module):
		def __init__(self, encoder, decoder, generator):
			super().__init__()

			import onmt
			self.model = onmt.models.NMTModel(encoder, decoder)
			self.model.add_module('generator', generator)

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def forward(self, inputs, outputs=None, output_lens=None, bptt=False, with_align=False, *args, **kwargs):
			if outputs is None or output_lens is None:
				batch_size = len(inputs)

				if is_beam_search_used:
					decode_strategy = opennmt_util.create_beam_search_strategy(batch_size, scorer, beam_size, n_best, ratio, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)
				else:
					decode_strategy = opennmt_util.create_greedy_search_strategy(batch_size, random_sampling_topk, random_sampling_temp, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)

				model_output_dict = opennmt_util.translate_batch_with_strategy(self.model, decode_strategy, inputs, batch_size, beam_size, tgt_unk, tgt_vocab, src_vocabs=[])

				model_outputs = model_output_dict['predictions']
				#scores = model_output_dict['scores']
				#attentions = model_output_dict['attention']
				#alignment = model_output_dict['alignment']

				rank_id = 0  # rank_id < n_best.
				#max_time_steps = functools.reduce(lambda x, y: x if x >= len(y[rank_id]) else len(y[rank_id]), model_outputs, 0)
				new_model_outputs = torch.full((len(model_outputs), max_time_steps), tgt_pad, dtype=torch.int)
				for idx, moutp in enumerate(model_outputs):
					new_model_outputs[idx,:len(moutp[rank_id])] = moutp[rank_id]

				return new_model_outputs
			else:
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T] -> [T, B, 1]. No one-hot encoding.

				model_output_tuple = self.model(inputs, outputs, output_lens)

				model_outputs = self.model.generator(model_output_tuple[0]).transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes].
				#attentions = model_output_tuple[1]['std']

				model_outputs = torch.argmax(model_outputs, dim=-1)
				return model_outputs, decoder_outputs

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
			decoder_output_lens = output_lens - 1
			outputs.unsqueeze_(dim=-1)  # [B, T] -> [B, T, 1]. No one-hot encoding.
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, 1] -> [T, B, 1].

			model_output_tuple = self.model(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = self.model.generator(model_output_tuple[0]).transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes] where T-1 is for one-step look-ahead.
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			# TODO [decide] >> Which is better, tensor.contiguous().to(device) or tensor.to(device).contiguous()?
			#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().to(device).view(-1))
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

	model = MySimpleModel(encoder, decoder, generator)

	#--------------------
	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_id)  # Ignore the PAD ID.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_id, reduction='sum')  # Ignore the PAD ID.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))
	else:
		criterion = None

	return model, criterion

def build_rare1_and_opennmt_model(image_height, image_width, image_channel, max_time_steps, label_converter, lang, loss_type=None, device='cuda'):
	transformer = None  # The type of transformer. {None, 'TPS'}.
	feature_extractor = 'VGG'  # The type of feature extractor. {'VGG', 'RCNN', 'ResNet'}.
	sequence_model = 'BiLSTM'  # The type of sequence model. {None, 'BiLSTM'}.

	num_fiducials = 20  # The number of fiducial points of TPS-STN.
	#input_channel = image_channel  # The number of input channel of feature extractor.
	output_channel = 512  # The number of output channel of feature extractor.
	bidirectional_encoder = True
	num_encoder_layers, num_decoder_layers = 2, 2
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size = 512
		decoder_hidden_size = encoder_rnn_size * 2 if bidirectional_encoder else encoder_rnn_size
	else:
		word_vec_size = 80
		encoder_rnn_size = 256
		decoder_hidden_size = encoder_rnn_size * 2 if bidirectional_encoder else encoder_rnn_size

	#--------------------
	import onmt, onmt.translate
	import torchtext
	import torchtext_util

	tgt_field = torchtext.data.Field(
		sequential=True, use_vocab=True, init_token=label_converter.SOS, eos_token=label_converter.EOS, fix_length=None,
		dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
		tokenize=None, tokenizer_language='kr',  # TODO [check] >> tokenizer_language is not valid.
		#tokenize=functools.partial(onmt.inputters.inputter._feature_tokenize, layer=0, feat_delim=None, truncate=None), tokenizer_language='en',
		include_lengths=False, batch_first=False, pad_token=label_converter.PAD, pad_first=False, unk_token=label_converter.UNKNOWN,
		truncate_first=False, stop_words=None, is_target=False
	)
	#tgt_field.build_vocab([label_converter.tokens], specials=[label_converter.UNKNOWN, label_converter.PAD], specials_first=False)  # Sort vocabulary + add special tokens, <unknown>, <pad>, <bos>, and <eos>.
	if label_converter.PAD in [label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN]:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	else:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.PAD, label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	assert label_converter.num_tokens == len(tgt_field.vocab.itos)
	assert len(list(filter(lambda pair: pair[0] != pair[1], zip(label_converter.tokens, tgt_field.vocab.itos)))) == 0

	tgt_vocab = tgt_field.vocab
	tgt_unk = tgt_vocab.stoi[tgt_field.unk_token]
	tgt_bos = tgt_vocab.stoi[tgt_field.init_token]
	tgt_eos = tgt_vocab.stoi[tgt_field.eos_token]
	tgt_pad = tgt_vocab.stoi[tgt_field.pad_token]

	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')

	is_beam_search_used = True
	if is_beam_search_used:
		beam_size = 30
		n_best = 1
		ratio = 0.0
	else:
		beam_size = 1
		random_sampling_topk, random_sampling_temp = 1, 1
		n_best = 1  # Fixed. For handling translation results.
	min_length, max_length = 0, max_time_steps
	block_ngram_repeat = 0
	#ignore_when_blocking = frozenset()
	#exclusion_idxs = {tgt_vocab.stoi[t] for t in ignore_when_blocking}
	exclusion_idxs = set()

	class MyCompositeModel(torch.nn.Module):
		def __init__(self, image_height, image_width, input_channel, output_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size, num_encoder_layers, num_decoder_layers, bidirectional_encoder, transformer=None, feature_extractor='VGG', sequence_model='BiLSTM', num_fiducials=0):
			super().__init__()

			self.encoder = opennmt_util.Rare1ImageEncoder(image_height, image_width, input_channel, output_channel, hidden_size=encoder_rnn_size, num_layers=num_encoder_layers, bidirectional=bidirectional_encoder, transformer=transformer, feature_extractor=feature_extractor, sequence_model=sequence_model, num_fiducials=num_fiducials)
			self.decoder, self.generator = build_decoder_and_generator_for_opennmt(num_classes, word_vec_size, hidden_size=decoder_hidden_size, num_layers=num_decoder_layers, bidirectional_encoder=bidirectional_encoder)

		def forward(self, inputs, outputs=None, output_lens=None, *args, **kwargs):
			if outputs is None or output_lens is None:
				batch_size = len(inputs)

				if is_beam_search_used:
					decode_strategy = opennmt_util.create_beam_search_strategy(batch_size, scorer, beam_size, n_best, ratio, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)
				else:
					decode_strategy = opennmt_util.create_greedy_search_strategy(batch_size, random_sampling_topk, random_sampling_temp, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)

				model_output_dict = opennmt_util.translate_batch_with_strategy(self, decode_strategy, inputs, batch_size, beam_size, tgt_unk, tgt_vocab, src_vocabs=[])

				model_outputs = model_output_dict['predictions']
				#scores = model_output_dict['scores']
				#attentions = model_output_dict['attention']
				#alignment = model_output_dict['alignment']

				rank_id = 0  # rank_id < n_best.
				#max_time_steps = functools.reduce(lambda x, y: x if x >= len(y[rank_id]) else len(y[rank_id]), model_outputs, 0)
				new_model_outputs = torch.full((len(model_outputs), max_time_steps), tgt_pad, dtype=torch.int)
				for idx, moutp in enumerate(model_outputs):
					new_model_outputs[idx,:len(moutp[rank_id])] = moutp[rank_id]

				return new_model_outputs
			else:
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T] -> [T, B, 1]. No one-hot encoding.

				model_output_tuple = self._onmt_forward(inputs, outputs, output_lens)

				model_outputs = model_output_tuple[0].transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes].
				#attentions = model_output_tuple[1]['std']

				model_outputs = torch.argmax(model_outputs, dim=-1)
				return model_outputs, decoder_outputs

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
			decoder_output_lens = output_lens - 1
			outputs.unsqueeze_(dim=-1)  # [B, T] -> [B, T, 1]. No one-hot encoding.
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, 1] -> [T, B, 1].

			model_output_tuple = self._onmt_forward(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes] where T-1 is for one-step look-ahead.
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			# TODO [decide] >> Which is better, tensor.contiguous().to(device) or tensor.to(device).contiguous()?
			#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().to(device).view(-1))
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def _onmt_forward(self, src, tgt=None, lengths=None, bptt=False, with_align=False):
			enc_hiddens, enc_outputs, lengths = self.encoder(src, lengths=lengths, device=device)

			dec_in = tgt[:-1]  # Exclude last target from inputs.

			# TODO [check] >> Is it proper to use enc_outputs & enc_hiddens?
			if bptt is False:
				self.decoder.init_state(src, enc_outputs, enc_hiddens)
			dec_outs, attns = self.decoder(dec_in, enc_outputs, memory_lengths=lengths, with_align=with_align)
			outs = self.generator(dec_outs)
			return outs, attns

	model = MyCompositeModel(image_height, image_width, image_channel, output_channel, label_converter.num_tokens, word_vec_size, encoder_rnn_size, decoder_hidden_size, num_encoder_layers, num_decoder_layers, bidirectional_encoder, transformer=transformer, feature_extractor=feature_extractor, sequence_model=sequence_model, num_fiducials=num_fiducials)

	#--------------------
	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_id)  # Ignore the PAD ID.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_id, reduction='sum')  # Ignore the PAD ID.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))
	else:
		criterion = None

	return model, criterion

def build_rare2_and_opennmt_model(image_height, image_width, image_channel, max_time_steps, label_converter, lang, loss_type=None, device='cuda'):
	is_stn_used = False
	if is_stn_used:
		num_fiducials = 20  # The number of fiducial points of TPS-STN.
	else:
		num_fiducials = 0  # No TPS-STN.
	bidirectional_encoder = True
	num_encoder_layers, num_decoder_layers = 2, 2
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size = 512
		decoder_hidden_size = encoder_rnn_size * 2 if bidirectional_encoder else encoder_rnn_size
	else:
		word_vec_size = 80
		encoder_rnn_size = 256
		decoder_hidden_size = encoder_rnn_size * 2 if bidirectional_encoder else encoder_rnn_size

	#--------------------
	import onmt, onmt.translate
	import torchtext
	import torchtext_util

	tgt_field = torchtext.data.Field(
		sequential=True, use_vocab=True, init_token=label_converter.SOS, eos_token=label_converter.EOS, fix_length=None,
		dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
		tokenize=None, tokenizer_language='kr',  # TODO [check] >> tokenizer_language is not valid.
		#tokenize=functools.partial(onmt.inputters.inputter._feature_tokenize, layer=0, feat_delim=None, truncate=None), tokenizer_language='en',
		include_lengths=False, batch_first=False, pad_token=label_converter.PAD, pad_first=False, unk_token=label_converter.UNKNOWN,
		truncate_first=False, stop_words=None, is_target=False
	)
	#tgt_field.build_vocab([label_converter.tokens], specials=[label_converter.UNKNOWN, label_converter.PAD], specials_first=False)  # Sort vocabulary + add special tokens, <unknown>, <pad>, <bos>, and <eos>.
	if label_converter.PAD in [label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN]:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	else:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.PAD, label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	assert label_converter.num_tokens == len(tgt_field.vocab.itos)
	assert len(list(filter(lambda pair: pair[0] != pair[1], zip(label_converter.tokens, tgt_field.vocab.itos)))) == 0

	tgt_vocab = tgt_field.vocab
	tgt_unk = tgt_vocab.stoi[tgt_field.unk_token]
	tgt_bos = tgt_vocab.stoi[tgt_field.init_token]
	tgt_eos = tgt_vocab.stoi[tgt_field.eos_token]
	tgt_pad = tgt_vocab.stoi[tgt_field.pad_token]

	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')

	is_beam_search_used = True
	if is_beam_search_used:
		beam_size = 30
		n_best = 1
		ratio = 0.0
	else:
		beam_size = 1
		random_sampling_topk, random_sampling_temp = 1, 1
		n_best = 1  # Fixed. For handling translation results.
	min_length, max_length = 0, max_time_steps
	block_ngram_repeat = 0
	#ignore_when_blocking = frozenset()
	#exclusion_idxs = {tgt_vocab.stoi[t] for t in ignore_when_blocking}
	exclusion_idxs = set()

	class MyCompositeModel(torch.nn.Module):
		def __init__(self, image_height, image_width, input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size, num_encoder_layers, num_decoder_layers, bidirectional_encoder, num_fiducials):
			super().__init__()

			self.encoder = opennmt_util.Rare2ImageEncoder(image_height, image_width, input_channel, hidden_size=encoder_rnn_size, num_layers=num_encoder_layers, bidirectional=bidirectional_encoder, num_fiducials=num_fiducials)
			self.decoder, self.generator = build_decoder_and_generator_for_opennmt(num_classes, word_vec_size, hidden_size=decoder_hidden_size, num_layers=num_decoder_layers, bidirectional_encoder=bidirectional_encoder)

		def forward(self, inputs, outputs=None, output_lens=None, *args, **kwargs):
			if outputs is None or output_lens is None:
				batch_size = len(inputs)

				if is_beam_search_used:
					decode_strategy = opennmt_util.create_beam_search_strategy(batch_size, scorer, beam_size, n_best, ratio, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)
				else:
					decode_strategy = opennmt_util.create_greedy_search_strategy(batch_size, random_sampling_topk, random_sampling_temp, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)

				model_output_dict = opennmt_util.translate_batch_with_strategy(self, decode_strategy, inputs, batch_size, beam_size, tgt_unk, tgt_vocab, src_vocabs=[])

				model_outputs = model_output_dict['predictions']
				#scores = model_output_dict['scores']
				#attentions = model_output_dict['attention']
				#alignment = model_output_dict['alignment']

				rank_id = 0  # rank_id < n_best.
				#max_time_steps = functools.reduce(lambda x, y: x if x >= len(y[rank_id]) else len(y[rank_id]), model_outputs, 0)
				new_model_outputs = torch.full((len(model_outputs), max_time_steps), tgt_pad, dtype=torch.int)
				for idx, moutp in enumerate(model_outputs):
					new_model_outputs[idx,:len(moutp[rank_id])] = moutp[rank_id]

				return new_model_outputs
			else:
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T] -> [T, B, 1]. No one-hot encoding.

				model_output_tuple = self._onmt_forward(inputs, outputs, output_lens)

				model_outputs = model_output_tuple[0].transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes].
				#attentions = model_output_tuple[1]['std']

				model_outputs = torch.argmax(model_outputs, dim=-1)
				return model_outputs, decoder_outputs

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
			decoder_output_lens = output_lens - 1
			outputs.unsqueeze_(dim=-1)  # [B, T] -> [B, T, 1]. No one-hot encoding.
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, 1] -> [T, B, 1].

			model_output_tuple = self._onmt_forward(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes] where T-1 is for one-step look-ahead.
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			# TODO [decide] >> Which is better, tensor.contiguous().to(device) or tensor.to(device).contiguous()?
			#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().to(device).view(-1))
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def _onmt_forward(self, src, tgt=None, lengths=None, bptt=False, with_align=False):
			enc_hiddens, enc_outputs, lengths = self.encoder(src, lengths=lengths, device=device)

			dec_in = tgt[:-1]  # Exclude last target from inputs.

			# TODO [check] >> Is it proper to use enc_outputs & enc_hiddens?
			if bptt is False:
				self.decoder.init_state(src, enc_outputs, enc_hiddens)
			dec_outs, attns = self.decoder(dec_in, enc_outputs, memory_lengths=lengths, with_align=with_align)
			outs = self.generator(dec_outs)
			return outs, attns

	model = MyCompositeModel(image_height, image_width, image_channel, label_converter.num_tokens, word_vec_size, encoder_rnn_size, decoder_hidden_size, num_encoder_layers, num_decoder_layers, bidirectional_encoder, num_fiducials)

	#--------------------
	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_id)  # Ignore the PAD ID.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_id, reduction='sum')  # Ignore the PAD ID.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))
	else:
		criterion = None

	return model, criterion

def build_aster_and_opennmt_model(image_height, image_width, image_channel, max_time_steps, label_converter, lang, loss_type=None, device='cuda'):
	is_stn_used = False
	if is_stn_used:
		num_fiducials = 20  # The number of fiducial points of TPS-STN.
	else:
		num_fiducials = 0  # No TPS-STN.
	bidirectional_encoder = True
	num_decoder_layers = 2
	if lang == 'kor':
		word_vec_size = 80
		encoder_rnn_size = 512
		decoder_hidden_size = encoder_rnn_size * 2 if bidirectional_encoder else encoder_rnn_size
	else:
		word_vec_size = 80
		encoder_rnn_size = 256
		decoder_hidden_size = encoder_rnn_size * 2 if bidirectional_encoder else encoder_rnn_size

	#--------------------
	import onmt, onmt.translate
	import torchtext
	import torchtext_util

	tgt_field = torchtext.data.Field(
		sequential=True, use_vocab=True, init_token=label_converter.SOS, eos_token=label_converter.EOS, fix_length=None,
		dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False,
		tokenize=None, tokenizer_language='kr',
		#tokenize=functools.partial(onmt.inputters.inputter._feature_tokenize, layer=0, feat_delim=None, truncate=None), tokenizer_language='en',
		include_lengths=False, batch_first=False, pad_token=label_converter.PAD, pad_first=False, unk_token=label_converter.UNKNOWN,
		truncate_first=False, stop_words=None, is_target=False
	)
	#tgt_field.build_vocab([label_converter.tokens], specials=[label_converter.UNKNOWN, label_converter.PAD], specials_first=False)  # Sort vocabulary + add special tokens, <unknown>, <pad>, <bos>, and <eos>.
	if label_converter.PAD in [label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN]:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	else:
		tgt_field.vocab = torchtext_util.build_vocab_from_lexicon(label_converter.tokens, specials=[label_converter.PAD, label_converter.SOS, label_converter.EOS, label_converter.UNKNOWN], specials_first=False, sort=False)
	assert label_converter.num_tokens == len(tgt_field.vocab.itos)
	assert len(list(filter(lambda pair: pair[0] != pair[1], zip(label_converter.tokens, tgt_field.vocab.itos)))) == 0

	tgt_vocab = tgt_field.vocab
	tgt_unk = tgt_vocab.stoi[tgt_field.unk_token]
	tgt_bos = tgt_vocab.stoi[tgt_field.init_token]
	tgt_eos = tgt_vocab.stoi[tgt_field.eos_token]
	tgt_pad = tgt_vocab.stoi[tgt_field.pad_token]

	scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')

	is_beam_search_used = True
	if is_beam_search_used:
		beam_size = 30
		n_best = 1
		ratio = 0.0
	else:
		beam_size = 1
		random_sampling_topk, random_sampling_temp = 1, 1
		n_best = 1  # Fixed. For handling translation results.
	min_length, max_length = 0, max_time_steps
	block_ngram_repeat = 0
	#ignore_when_blocking = frozenset()
	#exclusion_idxs = {tgt_vocab.stoi[t] for t in ignore_when_blocking}
	exclusion_idxs = set()

	class MyCompositeModel(torch.nn.Module):
		def __init__(self, image_height, image_width, input_channel, num_classes, word_vec_size, encoder_rnn_size, decoder_hidden_size, num_decoder_layers, bidirectional_encoder, num_fiducials):
			super().__init__()

			self.encoder = opennmt_util.AsterImageEncoder(image_height, image_width, input_channel, num_classes, hidden_size=encoder_rnn_size, num_fiducials=num_fiducials)
			self.decoder, self.generator = build_decoder_and_generator_for_opennmt(num_classes, word_vec_size, hidden_size=decoder_hidden_size, num_layers=num_decoder_layers, bidirectional_encoder=bidirectional_encoder)

		def forward(self, inputs, outputs=None, output_lens=None, *args, **kwargs):
			if outputs is None or output_lens is None:
				batch_size = len(inputs)

				if is_beam_search_used:
					decode_strategy = opennmt_util.create_beam_search_strategy(batch_size, scorer, beam_size, n_best, ratio, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)
				else:
					decode_strategy = opennmt_util.create_greedy_search_strategy(batch_size, random_sampling_topk, random_sampling_temp, min_length, max_length, block_ngram_repeat, tgt_bos, tgt_eos, tgt_pad, exclusion_idxs)

				model_output_dict = opennmt_util.translate_batch_with_strategy(self, decode_strategy, inputs, batch_size, beam_size, tgt_unk, tgt_vocab, src_vocabs=[])

				model_outputs = model_output_dict['predictions']
				#scores = model_output_dict['scores']
				#attentions = model_output_dict['attention']
				#alignment = model_output_dict['alignment']

				rank_id = 0  # rank_id < n_best.
				#max_time_steps = functools.reduce(lambda x, y: x if x >= len(y[rank_id]) else len(y[rank_id]), model_outputs, 0)
				new_model_outputs = torch.full((len(model_outputs), max_time_steps), tgt_pad, dtype=torch.int)
				for idx, moutp in enumerate(model_outputs):
					new_model_outputs[idx,:len(moutp[rank_id])] = moutp[rank_id]

				return new_model_outputs
			else:
				decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
				outputs = torch.unsqueeze(outputs, dim=-1).transpose(0, 1).long()  # [B, T] -> [T, B, 1]. No one-hot encoding.

				model_output_tuple = self._onmt_forward(inputs, outputs, output_lens)

				model_outputs = model_output_tuple[0].transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes].
				#attentions = model_output_tuple[1]['std']

				model_outputs = torch.argmax(model_outputs, dim=-1)
				return model_outputs, decoder_outputs

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			outputs = outputs.long()

			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.
			decoder_output_lens = output_lens - 1
			outputs.unsqueeze_(dim=-1)  # [B, T] -> [B, T, 1]. No one-hot encoding.
			outputs = torch.transpose(outputs, 0, 1)  # [B, T, 1] -> [T, B, 1].

			model_output_tuple = self._onmt_forward(inputs.to(device), outputs.to(device), output_lens.to(device))

			model_outputs = model_output_tuple[0].transpose(0, 1)  # [T-1, B, #classes] -> [B, T-1, #classes] where T-1 is for one-step look-ahead.
			#attentions = model_output_tuple[1]['std']

			# NOTE [info] >> All examples in a batch are concatenated together.
			#	Can each example be handled individually?
			# TODO [decide] >> Which is better, tensor.contiguous().to(device) or tensor.to(device).contiguous()?
			#return criterion(model_outputs.contiguous().view(-1, model_outputs.shape[-1]), decoder_outputs.contiguous().to(device).view(-1))
			concat_model_outputs, concat_decoder_outputs = list(), list()
			for mo, do, dl in zip(model_outputs, decoder_outputs, decoder_output_lens):
				concat_model_outputs.append(mo[:dl])
				concat_decoder_outputs.append(do[:dl])
			return criterion(torch.cat(concat_model_outputs, 0).to(device), torch.cat(concat_decoder_outputs, 0).to(device)), model_outputs

		# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
		def _onmt_forward(self, src, tgt=None, lengths=None, bptt=False, with_align=False):
			enc_hiddens, enc_outputs, lengths = self.encoder(src, lengths=lengths, device=device)

			if tgt is None or lengths is None:
				raise NotImplementedError
			else:
				dec_in = tgt[:-1]  # Exclude last target from inputs.

				# TODO [check] >> Is it proper to use enc_outputs & enc_hiddens?
				if bptt is False:
					self.decoder.init_state(src, enc_outputs, enc_hiddens)
				dec_outs, attns = self.decoder(dec_in, enc_outputs, memory_lengths=lengths, with_align=with_align)
				outs = self.generator(dec_outs)
				return outs, attns

	model = MyCompositeModel(image_height, image_width, image_channel, label_converter.num_tokens, word_vec_size, encoder_rnn_size, decoder_hidden_size, num_decoder_layers, bidirectional_encoder, num_fiducials)

	#--------------------
	if loss_type is not None:
		# Define a loss function.
		if loss_type == 'xent':
			criterion = torch.nn.CrossEntropyLoss(ignore_index=label_converter.pad_id)  # Ignore the PAD ID.
		elif loss_type == 'nll':
			criterion = torch.nn.NLLLoss(ignore_index=label_converter.pad_id, reduction='sum')  # Ignore the PAD ID.
		else:
			raise ValueError('Invalid loss type, {}'.format(loss_type))
	else:
		criterion = None

	return model, criterion

# REF [site] >> https://github.com/fengxinjie/Transformer-OCR
def build_transformer_model(image_height, image_width, image_channel, max_time_steps, label_converter, lang, is_train=False):
	import transformer_ocr.model, transformer_ocr.train, transformer_ocr.predict, transformer_ocr.dataset

	num_layers = 4
	num_heads = 8  # The number of head for MultiHeadedAttention.
	dropout = 0.1  # Dropout probability. [0, 1].
	if lang == 'kor':
		d_model = 256  # The dimension of keys/values/queries in MultiHeadedAttention, also the input size of the first-layer of the PositionwiseFeedForward.
		d_ff = 1024  # The second-layer of the PositionwiseFeedForward.
		d_feature = 1024  # The dimension of features in FeatureExtractor.
	else:
		d_model = 256  # The dimension of keys/values/queries in MultiHeadedAttention, also the input size of the first-layer of the PositionwiseFeedForward.
		d_ff = 1024  # The second-layer of the PositionwiseFeedForward.
		d_feature = 1024  # The dimension of features in FeatureExtractor.
	smoothing = 0.1
	# TODO [check] >> Check if PAD ID or PAD index is used.
	#pad_id = 0
	pad_id = label_converter.pad_id
	sos_id, eos_id = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]

	class MySimpleModel(torch.nn.Module):
		def __init__(self, num_classes, num_layers, d_model, d_ff, d_feature, num_heads, dropout):
			super().__init__()

			self.model = transformer_ocr.model.make_model(num_classes, N=num_layers, d_model=d_model, d_ff=d_ff, d_feature=d_feature, h=num_heads, dropout=dropout)
			self.d_model = d_model
			self.cnn_downsample_factor = 16**2  # Fixed.

		def forward(self, inputs, outputs=None, device='cuda', *args, **kwargs):
			# FIXME [check] >> Why is a single input but not multiple inputs predicted?
			"""
			# Predict a single input.
			# REF [function] >> transformer_ocr.dataset.Batch.__init__()
			#src_mask = torch.autograd.Variable(torch.from_numpy(np.ones([1, 1, 36], dtype=np.bool)).to(device))
			src_mask = torch.autograd.Variable(torch.full([1, 1, inputs.size(2) * inputs.size(3) // self.cnn_downsample_factor], True, dtype=torch.bool, device=device))
			model_outputs = torch.full((len(inputs), max_time_steps), pad_id, dtype=np.int)
			#for idx, src in enumerate(inputs):
			#	src = src.unsqueeze(dim=0)
			for idx in range(len(inputs)):
				src = inputs[idx:idx+1]
				model_outp = transformer_ocr.predict.greedy_decode(self.model, src, src_mask, max_len=max_time_steps, sos=sos_id, eos=eos_id, device=device)
				model_outputs[idx,:len(model_outp[0])] = model_outp[0]
			"""
			# Predict multiple inputs.
			src_mask = torch.autograd.Variable(torch.full([1, 1, inputs.size(2) * inputs.size(3) // self.cnn_downsample_factor], True, dtype=torch.bool, device=device))
			#model_outputs = transformer_ocr.predict.greedy_decode_multi_simple(self.model, inputs, src_mask, max_len=max_time_steps, sos=sos_id, eos=eos_id, pad=pad_id, device=device)
			model_outputs = transformer_ocr.predict.greedy_decode_multi(self.model, inputs, src_mask, max_len=max_time_steps, sos=sos_id, eos=eos_id, pad=pad_id, device=device)
			#model_outputs = transformer_ocr.predict.greedy_decode_multi_async1(self.model, inputs, src_mask, max_len=max_time_steps, sos=sos_id, eos=eos_id, pad=pad_id, device=device)
			#model_outputs = transformer_ocr.predict.greedy_decode_multi_async2(self.model, inputs, src_mask, max_len=max_time_steps, sos=sos_id, eos=eos_id, pad=pad_id, device=device)  # Slow.

			if outputs is None:
				return model_outputs
			else:
				return model_outputs, outputs[:,1:]

		def train_forward(self, criterion, inputs, outputs, output_lens, device='cuda'):
			outputs = outputs.long()

			# Construct inputs for one-step look-ahead.
			if eos_id != pad_id:
				decoder_inputs = outputs[:,:-1].clone()
				decoder_inputs[decoder_inputs == eos_id] = pad_id  # Remove <EOS> tokens.
			else: decoder_inputs = outputs[:,:-1]
			# Construct outputs for one-step look-ahead.
			decoder_outputs = outputs[:,1:]  # Remove <SOS> tokens.

			batch = transformer_ocr.dataset.Batch(inputs, decoder_inputs, decoder_outputs, self.cnn_downsample_factor, pad=pad_id, device=device)
			model_outputs = self.model(batch.src, batch.tgt_input, batch.src_mask, batch.tgt_input_mask)

			# REF [function] >> transformer_ocr.train.SimpleLossCompute.__call__().
			model_outputs = self.model.generator(model_outputs)
			return criterion(model_outputs.contiguous().view(-1, model_outputs.size(-1)), batch.tgt_output.contiguous().view(-1)) / batch.num_tokens, model_outputs

	model = MySimpleModel(label_converter.num_tokens, num_layers, d_model, d_ff, d_feature, num_heads, dropout)

	#--------------------
	if is_train:
		# Define a loss function.
		# Use KL divergence.
		# TODO [check] >>
		criterion = transformer_ocr.train.LabelSmoothing(size=label_converter.num_tokens, padding_idx=pad_id, smoothing=smoothing)
		#criterion = LabelSmoothingLoss(num_labels=label_converter.num_tokens, pad_id=pad_id, smoothing=smoothing)
	else:
		criterion = None

	return model, criterion

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def build_char_model_for_training(model_filepath_to_load, model_type, image_shape, target_type, font_type, output_dir_path, label_converter, logger=None, device='cuda'):
	image_height, image_width, image_channel = image_shape
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	loss_type = 'xent'  # {'xent', 'nll'}.
	if model_type == 'char': model_name = 'basic'
	elif model_type == 'char-mixup': model_name = 'mixup'
	else: raise ValueError('Invalid model type, {}'.format(model_type))
	#max_gradient_norm = 5  # Gradient clipping value.
	max_gradient_norm = None
	train_test_ratio = 0.8

	is_model_loaded = model_filepath_to_load is not None
	is_model_initialized = True
	is_all_model_params_optimized = True

	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	model_filepath_base = os.path.join(output_dir_path, '{}_{}_{}_{}x{}x{}'.format(target_type, model_name, font_type, image_height, image_width, image_channel))
	model_filepath_format = model_filepath_base + '{}.pth'
	if logger: logger.info('Model filepath: {}.'.format(model_filepath_format.format('')))

	#--------------------
	# Build a model.

	if model_type == 'char':
		model, criterion = build_char_model(label_converter, image_channel, loss_type)
	elif model_type == 'char-mixup':
		model, criterion = build_char_mixup_model(label_converter, image_channel, loss_type)
	else:
		model, criterion = None, None, None

	if is_model_initialized:
		# Initialize model weights.
		for name, param in model.named_parameters():
			#if 'initialized_variable_name' in name:
			#	if logger: logger.info(f'Skip {name} as it has already been initialized.')
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
		model = load_model(model_filepath_to_load, model, logger, device=device)

	model = model.to(device)
	criterion = criterion.to(device)

	#--------------------
	# Define an optimizer.

	if is_all_model_params_optimized:
		model_params = list(model.parameters())
	else:
		# Filter model parameters only that require gradients.
		#model_params = filter(lambda p: p.requires_grad, model.parameters())
		model_params, num_model_params = list(), 0
		for p in filter(lambda p: p.requires_grad, model.parameters()):
			model_params.append(p)
			num_model_params += np.prod(p.size())
		if logger: logger.info('#trainable model parameters = {}.'.format(num_model_params))
		#if logger: logger.info('Trainable model parameters:')
		#[if logger: logger.info(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

	optimizer = torch.optim.SGD(model_params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
	scheduler = None
	is_epoch_based_scheduler = True

	return model, criterion, optimizer, scheduler, is_epoch_based_scheduler, model_params, max_gradient_norm, model_filepath_format

def build_char_model_for_inference(model_filepath_to_load, image_shape, num_classes, logger, device='cuda'):
	model_name = 'ResNet'  # {'VGG', 'ResNet', 'RCNN'}.
	input_channel, output_channel = image_shape[2], 1024

	# For char recognizer.
	#model_filepath = './craft/char_recognition.pth'
	model_filepath = './craft/char_recognition_mixup.pth'

	if logger: logger.info('Start loading character recognizer...')
	start_time = time.time()
	import rare.model_char
	model = rare.model_char.create_model(model_name, input_channel, output_channel, num_classes)

	model = load_model(model_filepath_to_load, model, logger, device=device)
	model = model.to(device)
	if logger: logger.info('End loading character recognizer: {} secs.'.format(time.time() - start_time))

	return model

def build_text_model_for_training(model_filepath_to_load, model_type, image_shape, target_type, font_type, max_label_len, output_dir_path, label_converter, sos_id, eos_id, blank_label, num_suffixes, lang, logger=None, device='cuda'):
	image_height, image_width, image_channel = image_shape
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	if model_type == 'rare1':
		loss_type = 'xent'  # {'ctc', 'xent', 'nll'}.
		if loss_type == 'ctc': model_name = 'rare1'
		elif loss_type in ['xent', 'nll']: model_name = 'rare1_attn'
		else: raise ValueError('Invalid loss type, {}'.format(loss_type))
		max_gradient_norm = 5  # Gradient clipping value.
	elif model_type == 'rare1-mixup':
		loss_type = 'xent'  # {'ctc', 'xent', 'nll'}.
		if loss_type == 'ctc': model_name = 'rare1_mixup'
		elif loss_type in ['xent', 'nll']: model_name = 'rare1_attn_mixup'
		else: raise ValueError('Invalid loss type, {}'.format(loss_type))
		max_gradient_norm = 5  # Gradient clipping value.
	elif model_type == 'rare2':
		loss_type = 'xent'  # {'xent', 'nll'}.
		model_name = 'rare2_attn'
		max_gradient_norm = 5  # Gradient clipping value.
	elif model_type == 'aster':
		loss_type = 'sxent'  # Sequence cross entropy.
		model_name = 'aster'
		#max_gradient_norm = 5  # Gradient clipping value.
		max_gradient_norm = None
	elif model_type == 'onmt':
		loss_type = 'xent'  # {'xent', 'nll'}.
		model_name = 'onmt'
		#max_gradient_norm = 20  # Gradient clipping value.
		max_gradient_norm = None
	elif model_type == 'rare1+onmt':
		loss_type = 'xent'  # {'xent', 'nll'}.
		model_name = 'rare1+onmt'
		#max_gradient_norm = 20  # Gradient clipping value.
		max_gradient_norm = None
	elif model_type == 'rare2+onmt':
		loss_type = 'xent'  # {'xent', 'nll'}.
		model_name = 'rare2+onmt'
		#max_gradient_norm = 20  # Gradient clipping value.
		max_gradient_norm = None
	elif model_type == 'aster+onmt':
		loss_type = 'xent'  # {'xent', 'nll'}.
		model_name = 'aster+onmt'
		#max_gradient_norm = 20  # Gradient clipping value.
		max_gradient_norm = None
	elif model_type == 'transformer':
		loss_type = 'kldiv'  # {'kldiv'}.
		model_name = 'transformer'
		#max_gradient_norm = 20  # Gradient clipping value.
		max_gradient_norm = None
	else:
		raise ValueError('Invalid model type, {}'.format(model_type))

	is_model_loaded = model_filepath_to_load is not None
	is_model_initialized = True
	is_all_model_params_optimized = True

	assert not is_model_loaded or (is_model_loaded and model_filepath_to_load is not None)

	model_filepath_base = os.path.join(output_dir_path, '{}_{}_{}_ch{}_{}x{}x{}'.format(target_type, model_name, font_type, max_label_len, image_height, image_width, image_channel))
	model_filepath_format = model_filepath_base + '{}.pth'
	if logger: logger.info('Model filepath: {}.'.format(model_filepath_format.format('')))

	#--------------------
	# Build a model.

	if logger: logger.info('Start building a model...')
	start_time = time.time()
	if model_type == 'rare1':
		model, criterion = build_rare1_model(image_height, image_width, image_channel, max_label_len + num_suffixes, label_converter.num_tokens, label_converter.pad_id, sos_id, blank_label if loss_type == 'ctc' else None, lang, loss_type)
	elif model_type == 'rare1-mixup':
		model, criterion = build_rare1_mixup_model(image_height, image_width, image_channel, max_label_len + num_suffixes, label_converter.num_tokens, label_converter.pad_id, sos_id, blank_label if loss_type == 'ctc' else None, lang, loss_type)
	elif model_type == 'rare2':
		model, criterion = build_rare2_model(image_height, image_width, image_channel, max_label_len + num_suffixes, label_converter.num_tokens, label_converter.pad_id, sos_id, lang, loss_type)
	elif model_type == 'aster':
		model, sys_args = build_aster_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter.num_tokens, label_converter.pad_id, eos_id, lang, logger)
		criterion = None
	elif model_type == 'onmt':
		encoder_type = 'onmt'  # {'onmt', 'rare1', 'rare2', 'aster'}.
		model, criterion = build_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, encoder_type, label_converter, lang, loss_type)
	elif model_type == 'rare1+onmt':
		model, criterion = build_rare1_and_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, loss_type, device)
	elif model_type == 'rare2+onmt':
		model, criterion = build_rare2_and_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, loss_type, device)
	elif model_type == 'aster+onmt':
		model, criterion = build_aster_and_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, loss_type, device)
	elif model_type == 'transformer':
		model, criterion = build_transformer_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, is_train=True)
	else:
		raise ValueError('Invalid model type, {}'.format(model_type))
	if logger: logger.info('End building a model: {} secs.'.format(time.time() - start_time))

	if is_model_initialized:
		is_rare1 = model_type.find('rare1') != -1
		# Initialize model weights.
		for name, param in model.named_parameters():
			if is_rare1 and 'localization_fc2' in name:  # Exists in rare.modules.transformation.TPS_SpatialTransformerNetwork.
				if logger: logger.info(f'Skip {name} as it has already been initialized.')
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
		model = load_model(model_filepath_to_load, model, logger, device=device)

	model = model.to(device)
	# TODO [check] >>
	#if model_type == 'onmt': model.generator = model.generator.to(device)
	if criterion: criterion = criterion.to(device)

	#--------------------
	# Define an optimizer.

	if is_all_model_params_optimized:
		model_params = list(model.parameters())
	else:
		# Filter model parameters only that require gradients.
		#model_params = filter(lambda p: p.requires_grad, model.parameters())
		model_params, num_model_params = list(), 0
		for p in filter(lambda p: p.requires_grad, model.parameters()):
			model_params.append(p)
			num_model_params += np.prod(p.size())
		if logger: logger.info('#trainable model parameters = {}.'.format(num_model_params))
		#if logger: logger.info('Trainable model parameters:')
		#[if logger: logger.info(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

	#optimizer = torch.optim.SGD(model_params, lr=1.0, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
	#optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	#optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
	#optimizer = torch.optim.Adagrad(model_params, lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
	#optimizer = torch.optim.RMSprop(model_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	is_epoch_based_scheduler = True
	if model_type == 'rare1':
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		scheduler = None
	elif model_type == 'rare1-mixup':
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		scheduler = None
	elif model_type == 'rare2':
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		scheduler = None
	elif model_type == 'aster':
		optimizer = torch.optim.Adadelta(model_params, lr=sys_args.lr, weight_decay=sys_args.weight_decay)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)
	elif model_type == 'onmt':
		optimizer = torch.optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)
	elif model_type == 'rare1+onmt':
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)
	elif model_type == 'rare2+onmt':
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)
	elif model_type == 'aster+onmt':
		optimizer = torch.optim.Adadelta(model_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 5], gamma=0.1)
	elif model_type == 'transformer':
		if False:
			optimizer = torch.optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=False)
			import transformer_ocr.train
			optimizer = transformer_ocr.train.NoamOpt(model.d_model, factor=1, warmup=2000, optimizer=optimizer)  # Warning: NoamOpt is not an actual optimizer.
			scheduler = None
		elif True:
			optimizer = torch.optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=False)
			scheduler = NoamLR(optimizer, dim_feature=model.d_model, warmup_steps=2000, factor=1)  # Batch-step-based, not epoch-based, learning rate policy.
			is_epoch_based_scheduler = False
		elif False:
			optimizer = torch.optim.SGD(model_params, lr=0.1, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
	else:
		optimizer, scheduler = None, None

	return model, criterion, optimizer, scheduler, is_epoch_based_scheduler, model_params, max_gradient_norm, model_filepath_format

def build_text_model_for_inference(model_filepath_to_load, model_type, image_shape, max_label_len, label_converter, sos_id, eos_id, num_suffixes, lang, swa=False, logger=None, device='cuda'):
	# Build a model.
	if logger: logger.info('Start building a model...')
	start_time = time.time()
	image_height, image_width, image_channel = image_shape
	if model_type == 'rare1':
		model, _ = build_rare1_model(image_height, image_width, image_channel, max_label_len + num_suffixes, label_converter.num_tokens, label_converter.pad_id, sos_id, blank_label=None, lang=lang, loss_type=None)
	elif model_type == 'rare2':
		model, _ = build_rare2_model(image_height, image_width, image_channel, max_label_len + num_suffixes, label_converter.num_tokens, label_converter.pad_id, sos_id, lang, loss_type=None)
	elif model_type == 'aster':
		model, _ = build_aster_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter.num_tokens, label_converter.pad_id, eos_id, lang, logger)
	elif model_type == 'onmt':
		encoder_type = 'onmt'  # {'onmt', 'rare1', 'rare2', 'aster'}.
		model, _ = build_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, encoder_type, label_converter, lang, loss_type=None)
	elif model_type == 'rare1+onmt':
		model, _ = build_rare1_and_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, loss_type=None, device=device)
	elif model_type == 'rare2+onmt':
		model, _ = build_rare2_and_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, loss_type=None, device=device)
	elif model_type == 'aster+onmt':
		model, _ = build_aster_and_opennmt_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, loss_type=None, device=device)
	elif model_type == 'transformer':
		model, _ = build_transformer_model(image_height, image_width, image_channel, max_label_len + label_converter.num_affixes, label_converter, lang, is_train=False)
	else:
		raise ValueError('Invalid model type, {}'.format(model_type))
	if logger: logger.info('End building a model: {} secs.'.format(time.time() - start_time))

	if swa:
		import torch.optim.swa_utils
		model = torch.optim.swa_utils.AveragedModel(model, device=device, avg_fn=None)

	# Load a model.
	if logger: logger.info('Start loading a pretrained model from {}.'.format(model_filepath_to_load))
	start_time = time.time()
	model = load_model(model_filepath_to_load, model, logger, device=device)
	if logger: logger.info('End loading a pretrained model: {} secs.'.format(time.time() - start_time))

	model = model.to(device)

	return model

# Noam learning rate decay policy.
#	Batch-step-based, but not epoch-based, learning rate decay policy.
#	REF [paper] >> "Attention Is All You Need", NIPS 2017.
class NoamLR(object):
	def __init__(self, optimizer, dim_feature, warmup_steps, factor=1, last_step=0):
		self.optimizer = optimizer
		self.dim_feature = dim_feature
		self.warmup_steps = warmup_steps
		self.factor = factor
		self.last_step = last_step
		self.learning_rate = 0

		"""
		# Initialize step and base learning rates.
		if last_step == -1:
			for group in optimizer.param_groups:
				group.setdefault('initial_lr', group['lr'])
		else:
			for i, group in enumerate(optimizer.param_groups):
				if 'initial_lr' not in group:
					raise KeyError("param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i))
		self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
		"""

	def get_last_lr(self):
		return self.learning_rate

	def get_lr(self):
		return self.get_last_lr()

	def step(self, step=None):
		if step is None:
			self.last_step += 1
		else:
			self.last_step = step
		self.learning_rate = self._adjust_learning_rate(self.optimizer, self.last_step, self.warmup_steps, self.dim_feature, self.factor)

	@staticmethod
	def _adjust_learning_rate(optimizer, step, warmup_steps, dim_feature, factor):
		lr = factor * (dim_feature**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5)))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

# Label smoothing.
class LabelSmoothingLoss(torch.nn.Module):
	def __init__(self, num_labels, pad_id=0, smoothing=0.0):
		super().__init__()
		self.criterion = torch.nn.KLDivLoss(size_average=False)
		self.num_labels = num_labels
		self.pad_id = pad_id
		self.smoothing = smoothing
		self.confidence = 1.0 - smoothing
		#self.true_dist = None
		
	def forward(self, inputs, targets):
		assert inputs.size(1) == self.num_labels
		#true_dist = inputs.data.clone()
		#true_dist.fill_(self.smoothing / (self.num_labels - 2))
		true_dist = torch.full_like(inputs, self.smoothing / (self.num_labels - 2))
		true_dist.scatter_(1, targets.unsqueeze(dim=1), self.confidence)
		true_dist[:, self.pad_id] = 0
		mask = torch.nonzero(targets == self.pad_id)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		#self.true_dist = true_dist
		return self.criterion(inputs, torch.autograd.Variable(true_dist, requires_grad=False))

def train_char_model_in_epoch(model, criterion, optimizer, dataloader, model_params, max_gradient_norm, scheduler, is_step_based_scheduler, epoch, log_print_freq, logger, device='cuda'):
	#start_epoch_time = time.time()
	batch_time, data_time = swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter()
	losses, top1, top5 = swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter()
	#running_loss = 0.0
	for batch_step, batch in enumerate(dataloader):
		start_batch_time = time.time()

		batch_inputs, batch_outputs = batch

		data_time.update(time.time() - start_batch_time)

		# Zero the parameter gradients.
		optimizer.zero_grad()

		# Forward + backward + optimize.
		loss, model_outputs = model.train_forward(criterion, batch_inputs, batch_outputs, device)
		loss.backward()
		if max_gradient_norm: torch.nn.utils.clip_grad_norm_(model_params, max_norm=max_gradient_norm)  # Gradient clipping.
		optimizer.step()

		if scheduler and is_step_based_scheduler: scheduler.step()

		# Measure accuracy and record loss.
		prec1, prec5 = swl_ml_util.accuracy(torch.argmax(model_outputs.cpu(), dim=-1), batch_outputs[:,1:], topk=(1, 5))
		losses.update(loss.item(), batch_inputs.size(0))
		top1.update(prec1.item(), batch_inputs.size(0))
		top5.update(prec5.item(), batch_inputs.size(0))

		# Measure elapsed time.
		batch_time.update(time.time() - start_batch_time)

		"""
		# Print statistics.
		running_loss += loss.item()
		if batch_step % log_print_freq == (log_print_freq - 1):
			if logger: logger.info('[{}, {:5d}] loss = {:.6g}: {:.3f} secs.'.format(epoch + 1, batch_step + 1, running_loss / log_print_freq, time.time() - start_epoch_time))
			running_loss = 0.0
		"""

		if (batch_step + 1) % log_print_freq == 0:
			if logger: logger.info('\tBatch {}/{}: '
				'Batch time = {batch_time.val:.3f} ({batch_time.avg:.3f}), '
				'Data time = {data_time.val:.3f} ({data_time.avg:.3f}), '
				'Loss = {loss.val:.4f} ({loss.avg:.4f}), '
				'Prec@1 = {top1.val:.4f} ({top1.avg:.4f}), '
				'Prec@5 = {top5.val:.4f} ({top5.avg:.4f}).'.format(
				batch_step + 1, len(dataloader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top5=top5))

		sys.stdout.flush()
		time.sleep(0)
	return losses, top1, top5

def validate_char_model_in_epoch(model, criterion, dataloader, label_converter, is_case_sensitive, logger, device='cuda'):
	losses, top1, top5 = swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter()
	total_matching_ratio, num_examples = 0, 0
	with torch.no_grad():
		show = True
		for batch in dataloader:
			batch_inputs, batch_outputs = batch

			"""
			# One-hot encoding.
			batch_outputs_onehot = torch.LongTensor(batch_outputs.shape[0], self._dataset.num_classes)
			batch_outputs_onehot.zero_()
			batch_outputs_onehot.scatter_(1, batch_outputs.view(batch_outputs.shape[0], -1), 1)
			"""

			# Forward.
			loss, model_outputs = model.train_forward(criterion, batch_inputs, batch_outputs, device)

			# Measure accuracy and record loss.
			model_outputs = torch.argmax(model_outputs.cpu(), dim=-1)
			prec1, prec5 = swl_ml_util.accuracy(model_outputs, batch_outputs[:,1:], topk=(1, 5))
			losses.update(loss.item(), batch_inputs.size(0))
			top1.update(prec1.item(), batch_inputs.size(0))
			top5.update(prec5.item(), batch_inputs.size(0))

			batch_total_matching_ratio, _ = compute_sequence_matching_ratio(batch_inputs, batch_outputs, model_outputs, label_converter, is_case_sensitive, error_cases_dir_path=None, error_idx=0)
			total_matching_ratio += batch_total_matching_ratio
			num_examples += len(batch_inputs)

			# Show results.
			if show:
				if logger: logger.info('G/T - prediction:\n{}.'.format([(label_converter.decode(gt), label_converter.decode(pred)) for gt, pred in zip(batch_outputs, model_outputs)]))
				show = False
	avg_matching_ratio = total_matching_ratio / num_examples if num_examples > 0 else total_matching_ratio
	return losses, top1, top5, avg_matching_ratio

def train_text_model_in_epoch(model, criterion, optimizer, dataloader, model_params, max_gradient_norm, label_converter, scheduler, is_step_based_scheduler, is_case_sensitive, epoch, log_print_freq, logger, device='cuda'):
	#start_epoch_time = time.time()
	batch_time, data_time = swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter()
	losses, top1 = swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter()
	#running_loss = 0.0
	for batch_step, batch in enumerate(dataloader):
		start_batch_time = time.time()

		batch_inputs, batch_outputs, batch_output_lens = batch

		data_time.update(time.time() - start_batch_time)

		# Zero the parameter gradients.
		optimizer.zero_grad()

		# Forward + backward + optimize.
		loss, model_outputs = model.train_forward(criterion, batch_inputs, batch_outputs, batch_output_lens, device)
		loss.backward()
		if max_gradient_norm: torch.nn.utils.clip_grad_norm_(model_params, max_norm=max_gradient_norm)  # Gradient clipping.
		optimizer.step()

		if scheduler and is_step_based_scheduler: scheduler.step()

		# Measure accuracy and record loss.
		model_outputs = torch.argmax(model_outputs.cpu(), dim=-1)
		total_matching_ratio, _ = compute_sequence_matching_ratio(batch_inputs, batch_outputs, model_outputs, label_converter, is_case_sensitive, error_cases_dir_path=None, error_idx=0)
		avg_matching_ratio = total_matching_ratio / batch_inputs.size(0) if batch_inputs.size(0) > 0 else total_matching_ratio
		losses.update(loss.item(), batch_inputs.size(0))
		top1.update(avg_matching_ratio * 100, batch_inputs.size(0))

		# Measure elapsed time.
		batch_time.update(time.time() - start_batch_time)

		"""
		# Print statistics.
		running_loss += loss.item()
		if batch_step % log_print_freq == (log_print_freq - 1):
			if logger: logger.info('[{}, {:5d}] loss = {:.6g}: {:.3f} secs.'.format(epoch + 1, batch_step + 1, running_loss / log_print_freq, time.time() - start_epoch_time))
			running_loss = 0.0
		"""

		if (batch_step + 1) % log_print_freq == 0:
			if logger: logger.info('\tBatch {}/{}: '
				'Batch time = {batch_time.val:.3f} ({batch_time.avg:.3f}), '
				'Data time = {data_time.val:.3f} ({data_time.avg:.3f}), '
				'Loss = {loss.val:.4f} ({loss.avg:.4f}), '
				'Prec@1 = {top1.val:.4f} ({top1.avg:.4f}).'.format(
				batch_step + 1, len(dataloader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1))

		sys.stdout.flush()
		time.sleep(0)
	return losses, top1

def validate_text_model_in_epoch(model, criterion, dataloader, label_converter, is_case_sensitive, logger, device='cuda'):
	losses, top1 = swl_ml_util.AverageMeter(), swl_ml_util.AverageMeter()
	with torch.no_grad():
		show = True
		for batch in dataloader:
			batch_inputs, batch_outputs, batch_output_lens = batch

			"""
			# One-hot encoding.
			batch_outputs_onehot = torch.LongTensor(batch_outputs.shape[0], self._dataset.num_classes)
			batch_outputs_onehot.zero_()
			batch_outputs_onehot.scatter_(1, batch_outputs.view(batch_outputs.shape[0], -1), 1)
			"""

			# Forward.
			loss, model_outputs = model.train_forward(criterion, batch_inputs, batch_outputs, batch_output_lens, device)

			# Measure accuracy and record loss.
			model_outputs = torch.argmax(model_outputs.cpu(), dim=-1)
			if model_outputs.ndim == 1:
				model_outputs = torch.unsqueeze(model_outputs, dim=-1)
			total_matching_ratio, _ = compute_sequence_matching_ratio(batch_inputs, batch_outputs, model_outputs, label_converter, is_case_sensitive, error_cases_dir_path=None, error_idx=0)
			avg_matching_ratio = total_matching_ratio / batch_inputs.size(0) if batch_inputs.size(0) > 0 else total_matching_ratio
			losses.update(loss.item(), batch_inputs.size(0))
			top1.update(avg_matching_ratio * 100, batch_inputs.size(0))

			# Show results.
			if show:
				if logger: logger.info('G/T - prediction:\n{}.'.format([(label_converter.decode(gt), label_converter.decode(pred)) for gt, pred in zip(batch_outputs, model_outputs)]))
				show = False
	return losses, top1

def train_char_recognition_model(model, criterion, train_dataloader, test_dataloader, optimizer, label_converter, initial_epoch, final_epoch, log_print_freq, model_filepath_format, output_dir_path, scheduler=None, is_epoch_based_scheduler=True, max_gradient_norm=None, model_params=None, is_case_sensitive=False, logger=None, device='cuda'):
	train_log_filepath = os.path.join(output_dir_path, 'train_log.txt')
	train_history_filepath = os.path.join(output_dir_path, 'train_history.pkl')
	train_result_image_filepath = os.path.join(output_dir_path, 'results.png')

	recorder = swl_ml_util.RecorderMeter(final_epoch)
	history = {
		'acc': list(),
		'loss': list(),
		'val_acc': list(),
		'val_loss': list()
	}

	epoch_time = swl_ml_util.AverageMeter()
	best_performance_measure = 0.0
	best_model_filepath = None
	for epoch in range(initial_epoch, final_epoch):
		current_learning_rate = scheduler.get_last_lr() if scheduler else 0.0
		need_hour, need_mins, need_secs = swl_ml_util.convert_secs2time(epoch_time.avg * (final_epoch - epoch))
		if logger: logger.info('Epoch {}/{}: Need time = {:02d}:{:02d}:{:02d}, Accuracy (best) = {:.4f}, Error (best) = {:.4f}, Learning rate = {}.'.format(epoch + 1, final_epoch, need_hour, need_mins, need_secs, recorder.max_accuracy(False), 100 - recorder.max_accuracy(False), current_learning_rate))
		start_epoch_time = time.time()

		#--------------------
		start_time = time.time()
		model.train()
		losses, top1, top5 = train_char_model_in_epoch(model, criterion, optimizer, train_dataloader, model_params, max_gradient_norm, scheduler, not is_epoch_based_scheduler, epoch, log_print_freq, logger, device)
		if logger: logger.info('Train:      Prec@1 = {top1.avg:.4f}, Prec@5 = {top5.avg:.4f}, Error@1 = {error1:.4f}, Loss = {losses.avg:.4f}: {elapsed_time:.6f} secs.'.format(top1=top1, top5=top5, error1=100 - top1.avg, losses=losses, elapsed_time=time.time() - start_time))

		train_loss, train_acc = losses.avg, top1.avg
		history['loss'].append(train_loss)
		history['acc'].append(train_acc)

		#--------------------
		start_time = time.time()
		model.eval()
		losses, top1, top5, avg_matching_ratio = validate_char_model_in_epoch(model, criterion, test_dataloader, label_converter, is_case_sensitive, logger, device)
		if logger: logger.info('Validation: Prec@1 = {top1.avg:.4f}, Prec@5 = {top5.avg:.4f}, Error@1 = {error1:.4f}, Loss = {losses.avg:.4f}: {elapsed_time:.6f} secs.'.format(top1=top1, top5=top5, error1=100 - top1.avg, losses=losses, elapsed_time=time.time() - start_time))

		val_loss, val_acc = losses.avg, top1.avg
		history['val_loss'].append(val_loss)
		history['val_acc'].append(val_acc)

		if scheduler and is_epoch_based_scheduler: scheduler.step()

		# Measure elapsed time.
		epoch_time.update(time.time() - start_epoch_time)

		#--------------------
		performance_measure = avg_matching_ratio
		#performance_measure = val_acc
		if performance_measure >= best_performance_measure:
			# Save a checkpoint.
			best_model_filepath = model_filepath_format.format('_acc{:.4f}_epoch{}'.format(performance_measure, epoch + 1))
			save_model(best_model_filepath, model, logger)
			best_performance_measure = performance_measure

		dummy = recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
		recorder.plot_curve(train_result_image_filepath)
	
		#import pdb; pdb.set_trace()
		train_log = collections.OrderedDict()
		train_log['train_loss'] = history['loss']
		train_log['train_acc'] = history['acc']
		train_log['val_loss'] = history['val_loss']
		train_log['val_acc'] = history['val_acc']

		pickle.dump(train_log, open(train_history_filepath, 'wb'))
		swl_ml_util.plotting(output_dir_path, train_history_filepath)
		if logger: logger.info('Epoch {}/{} completed: {} secs.'.format(epoch + 1, final_epoch, time.time() - start_epoch_time))

	return model, best_model_filepath

def train_text_recognition_model(model, criterion, optimizer, train_dataloader, test_dataloader, label_converter, initial_epoch, final_epoch, log_print_freq, model_filepath_format, output_dir_path, scheduler=None, is_epoch_based_scheduler=True, max_gradient_norm=None, model_params=None, is_case_sensitive=False, swa=False, logger=None, device='cuda'):
	train_log_filepath = os.path.join(output_dir_path, 'train_log.txt')
	train_history_filepath = os.path.join(output_dir_path, 'train_history.pkl')
	train_result_image_filepath = os.path.join(output_dir_path, 'results.png')

	recorder = swl_ml_util.RecorderMeter(final_epoch)
	history = {
		'acc': list(),
		'loss': list(),
		'val_acc': list(),
		'val_loss': list()
	}

	if swa:
		# REF [site] >>
		#	https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
		#	https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
		import torch.optim.swa_utils
		swa_model = torch.optim.swa_utils.AveragedModel(model, device=device, avg_fn=None)
		swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05, anneal_epochs=5, anneal_strategy='cos', last_epoch=-1)
		swa_start_epoch = 20

	epoch_time = swl_ml_util.AverageMeter()
	best_performance_measure = 0.0
	best_model_filepath = None
	for epoch in range(initial_epoch, final_epoch):
		if swa and epoch >= swa_start_epoch:
			current_learning_rate = swa_scheduler.get_last_lr() if swa_scheduler else 0.0
		else:
			current_learning_rate = scheduler.get_last_lr() if scheduler else 0.0
		need_hour, need_mins, need_secs = swl_ml_util.convert_secs2time(epoch_time.avg * (final_epoch - epoch))
		if logger: logger.info('Epoch {}/{}: Need time = {:02d}:{:02d}:{:02d}, Accuracy (best) = {:.4f}, Error (best) = {:.4f}, Learning rate = {}.'.format(epoch + 1, final_epoch, need_hour, need_mins, need_secs, recorder.max_accuracy(False), 100 - recorder.max_accuracy(False), current_learning_rate))
		start_epoch_time = time.time()

		#--------------------
		start_time = time.time()
		model.train()
		losses, top1 = train_text_model_in_epoch(model, criterion, optimizer, train_dataloader, model_params, max_gradient_norm, label_converter, scheduler, not is_epoch_based_scheduler, is_case_sensitive, epoch, log_print_freq, logger, device)
		if logger: logger.info('Train:      Prec@1 = {top1.avg:.4f}, Error@1 = {error1:.4f}, Loss = {losses.avg:.4f}: {elapsed_time:.6f} secs.'.format(top1=top1, error1=100 - top1.avg, losses=losses, elapsed_time=time.time() - start_time))

		train_loss, train_acc = losses.avg, top1.avg
		history['loss'].append(train_loss)
		history['acc'].append(train_acc)

		#--------------------
		start_time = time.time()
		model.eval()
		losses, top1 = validate_text_model_in_epoch(model, criterion, test_dataloader, label_converter, is_case_sensitive, logger, device)
		if logger: logger.info('Validation: Prec@1 = {top1.avg:.4f}, Error@1 = {error1:.4f}, Loss = {losses.avg:.4f}: {elapsed_time:.6f} secs.'.format(top1=top1, error1=100 - top1.avg, losses=losses, elapsed_time=time.time() - start_time))

		val_loss, val_acc = losses.avg, top1.avg
		history['val_loss'].append(val_loss)
		history['val_acc'].append(val_acc)

		if swa and epoch >= swa_start_epoch:
			swa_model.update_parameters(model)
			swa_scheduler.step()
		else:
			if scheduler and is_epoch_based_scheduler: scheduler.step()

		# Measure elapsed time.
		epoch_time.update(time.time() - start_epoch_time)

		#--------------------
		if val_acc >= best_performance_measure:
			# Save a checkpoint.
			best_model_filepath = model_filepath_format.format('_acc{:.4f}_epoch{}'.format(val_acc, epoch + 1))
			save_model(best_model_filepath, model, logger)
			best_performance_measure = val_acc

		dummy = recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
		recorder.plot_curve(train_result_image_filepath)
	
		#import pdb; pdb.set_trace()
		train_log = collections.OrderedDict()
		train_log['train_loss'] = history['loss']
		train_log['train_acc'] = history['acc']
		train_log['val_loss'] = history['val_loss']
		train_log['val_acc'] = history['val_acc']

		pickle.dump(train_log, open(train_history_filepath, 'wb'))
		swl_ml_util.plotting(output_dir_path, train_history_filepath)
		if logger: logger.info('Epoch {}/{} completed: {} secs.'.format(epoch + 1, final_epoch, time.time() - start_epoch_time))

	#--------------------
	if swa:
		torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)

		swa_model_filepath = model_filepath_format.format('_swa_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
		save_model(swa_model_filepath, swa_model, logger)

	return model, best_model_filepath

def evaluate_char_recognition_model(model, dataloader, label_converter, is_case_sensitive=False, show_acc_per_char=False, error_cases_dir_path=None, logger=None, device='cuda'):
	classes, num_classes = label_converter.tokens, label_converter.num_tokens

	correct_char_count, total_char_count = 0, 0
	correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
	error_cases = list()
	error_idx = 0
	is_visualized = True
	with torch.no_grad():
		for images, labels in dataloader:
			predictions = model(images.to(device))

			predictions = predictions.cpu().numpy()
			predictions = np.argmax(predictions, axis=1)
			gts = labels.numpy()

			for gl, pl in zip(gts, predictions):
				if gl == pl: correct_char_class_count[gl] += 1
				total_char_class_count[gl] += 1

			gts, predictions = label_converter.decode(gts), label_converter.decode(predictions)
			gts_case, predictions_case = (gts, predictions) if is_case_sensitive else (gts.lower(), predictions.lower())

			total_char_count += max(len(gts), len(predictions))
			#correct_char_count += (gts_case == predictions_case).sum()
			correct_char_count += len(list(filter(lambda gp: gp[0] == gp[1], zip(gts_case, predictions_case))))

			if error_cases_dir_path is not None:
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

			if is_visualized:
				# Show images.
				#show_image(torchvision.utils.make_grid(images))

				#if logger: logger.info('G/T:        {}.'.format(' '.join(gts)))
				#if logger: logger.info('Prediction: {}.'.format(' '.join(predictions)))
				#for gt, pred in zip(gts, predictions):
				#	if logger: logger.info('G/T - prediction: {}, {}.'.format(gt, pred))
				if logger: logger.info('G/T - prediction:\n{}.'.format([(gt, pred) for gt, pred in zip(gts, predictions)]))

				is_visualized = False

	if error_cases_dir_path is not None:
		err_fpath = os.path.join(error_cases_dir_path, 'error_cases.txt')
		try:
			with open(err_fpath, 'w', encoding='UTF8') as fd:
				for idx, (gt, pred) in enumerate(error_cases):
					fd.write('{}\t{}\t{}\n'.format(idx, gt, pred))
		except UnicodeDecodeError as ex:
			if logger: logger.warning('Unicode decode error in {}: {}.'.format(err_fpath, ex))
		except FileNotFoundError as ex:
			if logger: logger.warning('File not found, {}: {}.'.format(err_fpath, ex))

	show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char, logger=logger)
	if logger: logger.info('Char accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count if total_char_count > 0 else -1))

	return correct_char_count / total_char_count if total_char_count > 0 else -1

def evaluate_text_recognition_model(model, dataloader, label_converter, is_case_sensitive=False, show_acc_per_char=False, error_cases_dir_path=None, logger=None, device='cuda'):
	classes, num_classes = label_converter.tokens, label_converter.num_tokens

	is_sequence_matching_ratio_used, is_simple_matching_accuracy_used = True, True
	total_matching_ratio, num_examples = 0, 0
	correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0, 0, 0
	correct_char_class_count, total_char_class_count = [0] * num_classes, [0] * num_classes
	error_cases = list()
	error_idx = 0
	is_visualized = True
	with torch.no_grad():
		for images, labels, label_lens in dataloader:
			predictions, gts = model(images.to(device), labels.to(device), output_lens=label_lens.to(device), device=device)

			images_np, predictions, gts = images.numpy(), predictions.cpu().numpy(), gts.cpu().numpy()
			if images_np.ndim == 4: images_np = images_np.transpose(0, 2, 3, 1)
			#minval, maxval = np.min(images_np), np.max(images_np)
			minval, maxval = -1, 1
			images_np = np.round((images_np - minval) * 255 / (maxval - minval)).astype(np.uint8)

			if is_sequence_matching_ratio_used:
				batch_total_matching_ratio, batch_error_cases = compute_sequence_matching_ratio(images_np, gts, predictions, label_converter, is_case_sensitive, error_cases_dir_path, error_idx=error_idx)
				total_matching_ratio += batch_total_matching_ratio
				num_examples += len(images_np)
			if is_simple_matching_accuracy_used:
				batch_correct_text_count, batch_total_text_count, batch_correct_word_count, batch_total_word_count, batch_correct_char_count, batch_total_char_count, batch_error_cases = compute_simple_matching_accuracy(images_np, gts, predictions, label_converter, is_case_sensitive, error_cases_dir_path, error_idx=error_idx)
				correct_text_count += batch_correct_text_count
				total_text_count += batch_total_text_count
				correct_word_count += batch_correct_word_count
				total_word_count += batch_total_word_count
				correct_char_count += batch_correct_char_count
				total_char_count += batch_total_char_count
			error_idx += len(batch_error_cases)
			error_cases += batch_error_cases
			batch_correct_char_class_count, batch_total_char_class_count = compute_per_char_accuracy(images_np, gts, predictions, num_classes)
			correct_char_class_count = list(map(operator.add, correct_char_class_count, batch_correct_char_class_count))
			total_char_class_count = list(map(operator.add, total_char_class_count, batch_total_char_class_count))

			if is_visualized:
				# Show images.
				#show_image(torchvision.utils.make_grid(images))

				#if logger: logger.info('G/T:        {}.'.format(' '.join([label_converter.decode(lbl) for lbl in gts])))
				#if logger: logger.info('Prediction: {}.'.format(' '.join([label_converter.decode(lbl) for lbl in predictions])))
				#for gt, pred in zip(gts, predictions):
				#	if logger: logger.info('G/T - prediction: {}, {}.'.format(label_converter.decode(gt), label_converter.decode(pred)))
				if logger: logger.info('G/T - prediction:\n{}.'.format([(label_converter.decode(gt), label_converter.decode(pred)) for gt, pred in zip(gts, predictions)]))

				is_visualized = False

	if error_cases_dir_path is not None:
		err_fpath = os.path.join(error_cases_dir_path, 'error_cases.txt')
		try:
			with open(err_fpath, 'w', encoding='UTF8') as fd:
				for idx, (gt, pred) in enumerate(error_cases):
					fd.write('{}\t{}\t{}\n'.format(idx, gt, pred))
		except UnicodeDecodeError as ex:
			if logger: logger.warning('Unicode decode error in {}: {}.'.format(err_fpath, ex))
		except FileNotFoundError as ex:
			if logger: logger.warning('File not found, {}: {}.'.format(err_fpath, ex))

	show_per_char_accuracy(correct_char_class_count, total_char_class_count, classes, num_classes, show_acc_per_char, logger=logger)
	if is_sequence_matching_ratio_used:
		#num_examples = len(dataloader)
		avg_matching_ratio = total_matching_ratio / num_examples if num_examples > 0 else total_matching_ratio
		if logger: logger.info('Average sequence matching ratio = {}.'.format(avg_matching_ratio))
	if is_simple_matching_accuracy_used:
		if logger:
			logger.info('Text: Simple matching accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count if total_text_count > 0 else -1))
			logger.info('Word: Simple matching accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count if total_word_count > 0 else -1))
			logger.info('Char: Simple matching accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count if total_char_count > 0 else -1))

	if is_sequence_matching_ratio_used:
		return avg_matching_ratio
	elif is_simple_matching_accuracy_used:
		return correct_char_count / total_char_count if total_char_count > 0 else -1
	else: return -1

def train_char_recognizer(model, criterion, optimizer, scheduler, is_epoch_based_scheduler, train_dataset, test_dataset, output_dir_path, label_converter, model_params, max_gradient_norm, num_epochs, batch_size, num_workers, is_case_sensitive, model_filepath_format, logger=None, device='cuda'):
	initial_epoch, final_epoch = 0, num_epochs
	log_print_freq = 1000

	if logger: logger.info('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	if logger: logger.info('End creating data loaders: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	# Show data info.
	show_char_data_info(train_dataloader, label_converter, visualize=False, nrow=8, mode='Train', logger=logger)
	show_char_data_info(test_dataloader, label_converter, visualize=False, nrow=8, mode='Test', logger=logger)

	#--------------------
	# Train a model.

	if logger: logger.info('Start training...')
	start_time = time.time()
	model, best_model_filepath = train_char_recognition_model(model, criterion, train_dataloader, test_dataloader, optimizer, label_converter, initial_epoch, final_epoch, log_print_freq, model_filepath_format, output_dir_path, scheduler, is_epoch_based_scheduler, max_gradient_norm, model_params, is_case_sensitive, logger, device)
	if logger: logger.info('End training: {} secs.'.format(time.time() - start_time))

	# Save a model.
	if best_model_filepath:
		model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
		try:
			shutil.copyfile(best_model_filepath, model_filepath)
			if logger: logger.info('Copied the best trained model to {}.'.format(model_filepath))
		except (FileNotFoundError, PermissionError) as ex:
			if logger: logger.warning('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
	elif model:
		model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
		save_model(model_filepath, model, logger)
	else: model_filepath = None

	#--------------------
	# Evaluate the model.

	if logger: logger.info('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_char_recognition_model(model, test_dataloader, label_converter, is_case_sensitive, show_acc_per_char=True, error_cases_dir_path=None, logger=logger, device=device)
	if logger: logger.info('End evaluating: {} secs.'.format(time.time() - start_time))

	return model_filepath

def train_text_recognizer(model, criterion, optimizer, scheduler, is_epoch_based_scheduler, train_dataset, test_dataset, output_dir_path, label_converter, model_params, max_gradient_norm, num_epochs, batch_size, num_workers, is_case_sensitive, model_filepath_format, swa=False, logger=None, device='cuda'):
	initial_epoch, final_epoch = 0, num_epochs
	log_print_freq = 1000

	if logger: logger.info('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	if logger: logger.info('End creating data loaders: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	# Show data info.
	show_text_data_info(train_dataloader, label_converter, visualize=False, nrow=2, mode='Train', logger=logger)
	show_text_data_info(test_dataloader, label_converter, visualize=False, nrow=2, mode='Test', logger=logger)

	#--------------------
	# Train a model.
	if logger: logger.info('Start training...')
	start_time = time.time()
	model, best_model_filepath = train_text_recognition_model(model, criterion, optimizer, train_dataloader, test_dataloader, label_converter, initial_epoch, final_epoch, log_print_freq, model_filepath_format, output_dir_path, scheduler, is_epoch_based_scheduler, max_gradient_norm, model_params, is_case_sensitive, swa, logger, device)
	if logger: logger.info('End training: {} secs.'.format(time.time() - start_time))

	# Save a model.
	if best_model_filepath:
		model_filepath = model_filepath_format.format('_best_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
		try:
			shutil.copyfile(best_model_filepath, model_filepath)
			if logger: logger.info('Copied the best trained model to {}.'.format(model_filepath))
		except (FileNotFoundError, PermissionError) as ex:
			if logger: logger.warning('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
	elif model:
		model_filepath = model_filepath_format.format('_final_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))
		save_model(model_filepath, model, logger)
	else: model_filepath = None

	#--------------------
	# Evaluate a model.
	if logger: logger.info('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, test_dataloader, label_converter, is_case_sensitive, show_acc_per_char=True, error_cases_dir_path=None, logger=logger, device=device)
	if logger: logger.info('End evaluating: {} secs.'.format(time.time() - start_time))

	return model_filepath

def evaluate_text_recognizer(model, dataset, output_dir_path, label_converter, batch_size, num_workers, is_case_sensitive=False, logger=None, device='cuda'):
	if logger: logger.info('Start creating a dataloader...')
	start_time = time.time()
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	if logger: logger.info('End creating a dataloader: {} secs.'.format(time.time() - start_time))

	# Show data info.
	show_text_data_info(dataloader, label_converter, visualize=False, nrow=2, mode='Test', logger=logger)

	#--------------------
	error_cases_dir_path = os.path.join(output_dir_path, 'eval_text_error_cases')
	if error_cases_dir_path and error_cases_dir_path.strip() and not os.path.exists(error_cases_dir_path):
		os.makedirs(error_cases_dir_path, exist_ok=True)

	# Evaluate a model.
	if logger: logger.info('Start evaluating...')
	start_time = time.time()
	model.eval()
	evaluate_text_recognition_model(model, dataloader, label_converter, is_case_sensitive, show_acc_per_char=True, error_cases_dir_path=error_cases_dir_path, logger=logger, device=device)
	if logger: logger.info('End evaluating: {} secs.'.format(time.time() - start_time))

def recognize_text(model, inputs, batch_size=None, logger=None, device='cuda'):
	if batch_size is not None and batch_size == 1:
		# Infer one-by-one.
		with torch.no_grad():
			inputs = inputs.to(device)
			predictions = np.array(list(model(inputs[idx:idx+1], device=device)[0].cpu().numpy() for idx in range(len(inputs))))
	else:
		# Infer batch-by-batch.
		if batch_size is None: batch_size = len(inputs)

		with torch.no_grad():
			inputs = inputs.to(device)
			predictions = list()
			for idx in range(0, len(inputs), batch_size):
				predictions.append(model(inputs[idx:idx+batch_size], device=device).cpu().numpy())
		predictions = np.vstack(predictions)

	return predictions

def create_label_converter(converter_type, charset):
	BLANK_LABEL, SOS_ID, EOS_ID = None, None, None
	num_suffixes = 0

	if converter_type == 'basic':
		label_converter = swl_langproc_util.TokenConverter(list(charset))
		assert label_converter.PAD is None
	elif converter_type == 'sos':
		# <SOS> only.
		SOS_TOKEN = '<SOS>'
		label_converter = swl_langproc_util.TokenConverter(list(charset), sos=SOS_TOKEN)
		assert label_converter.PAD is None
		SOS_ID = label_converter.encode([label_converter.SOS], is_bare_output=True)[0]
		del SOS_TOKEN
	elif converter_type == 'eos':
		# <EOS> only.
		EOS_TOKEN = '<EOS>'
		label_converter = swl_langproc_util.TokenConverter(list(charset), eos=EOS_TOKEN)
		assert label_converter.PAD is None
		EOS_ID = label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1
		del EOS_TOKEN
	elif converter_type == 'sos+eos':
		# <SOS> + <EOS> and <PAD> = a separate valid token ID.
		SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = '<SOS>', '<EOS>', '<PAD>'
		PAD_ID = len(charset)  # NOTE [info] >> It's a trick which makes <PAD> token have a separate valid token.
		label_converter = swl_langproc_util.TokenConverter(list(charset) + [PAD_TOKEN], sos=SOS_TOKEN, eos=EOS_TOKEN, pad=PAD_ID)
		assert label_converter.pad_id == PAD_ID, '{} != {}'.format(label_converter.pad_id, PAD_ID)
		assert label_converter.encode([PAD_TOKEN], is_bare_output=True)[0] == PAD_ID, '{} != {}'.format(label_converter.encode([PAD_TOKEN], is_bare_output=True)[0], PAD_ID)
		assert label_converter.PAD is not None
		SOS_ID, EOS_ID = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1
		del SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
	elif converter_type == 'sos/pad+eos':
		# <SOS> + <EOS> and <PAD> = <SOS>.
		SOS_TOKEN, EOS_TOKEN = '<SOS>', '<EOS>'
		label_converter = swl_langproc_util.TokenConverter(list(charset), sos=SOS_TOKEN, eos=EOS_TOKEN, pad=SOS_TOKEN)
		assert label_converter.PAD is not None
		SOS_ID, EOS_ID = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1
		del SOS_TOKEN, EOS_TOKEN
	elif converter_type == 'sos+eos/pad':
		# <SOS> + <EOS> and <PAD> = <EOS>.
		SOS_TOKEN, EOS_TOKEN = '<SOS>', '<EOS>'
		label_converter = swl_langproc_util.TokenConverter(list(charset), sos=SOS_TOKEN, eos=EOS_TOKEN, pad=EOS_TOKEN)
		assert label_converter.PAD is not None
		SOS_ID, EOS_ID = label_converter.encode([label_converter.SOS], is_bare_output=True)[0], label_converter.encode([label_converter.EOS], is_bare_output=True)[0]
		num_suffixes = 1
		del SOS_TOKEN, EOS_TOKEN
	elif converter_type == 'blank':
		# The BLANK label for CTC.
		BLANK_LABEL = '<BLANK>'
		label_converter = swl_langproc_util.TokenConverter([BLANK_LABEL] + list(charset), pad=None)  # NOTE [info] >> It's a trick. The ID of the BLANK label is set to 0.
		assert label_converter.encode([BLANK_LABEL], is_bare_output=True)[0] == 0, '{} != 0'.format(label_converter.encode([BLANK_LABEL], is_bare_output=True)[0])
		assert label_converter.PAD is None
		BLANK_LABEL_ID = 0 #label_converter.encode([BLANK_LABEL], is_bare_output=True)[0]
	else:
		raise ValueError('Invalid label converter type, {}'.format(converter_type))

	return label_converter, SOS_ID, EOS_ID, BLANK_LABEL, num_suffixes

def text_dataset_to_tensor(dataset, batch_size, num_workers, logger):
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	# Show data info.
	#show_text_data_info(dataloader, label_converter, visualize=False, nrow=2, mode='Test', logger=logger)

	inputs, outputs = list(), list()
	try:
		for images, labels, _ in dataloader:
			inputs.append(images)
			outputs.append(labels)
	except Exception as ex:
		if logger: logger.warning('Exception raised: {}.'.format(ex))
	inputs = torch.cat(inputs)
	outputs = torch.cat(outputs)

	return inputs, outputs

def images_to_tensor(images, image_shape, is_pil, logger):
	image_height, image_width, image_channel = image_shape
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	#image_height_before_crop, image_width_before_crop = image_height, image_width

	transform = torchvision.transforms.Compose([
		ResizeToFixedSize(image_height, image_width, warn_about_small_image=True, is_pil=is_pil, logger=logger),
		#ResizeToBelowMaxWidth(image_height, image_width, warn_about_small_image=True, is_pil=is_pil, logger=logger),  # batch_size must be 1.
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])

	inputs = list(transform(img) for img in images)
	return torch.stack(inputs)

def labels_to_tensor(labels, max_label_len, label_converter):
	#target_transform = torch.IntTensor
	target_transform = ToPaddedIntTensor(label_converter.pad_id, max_label_len)

	outputs = list(target_transform(label_converter.encode(lbl)) for lbl in labels)
	return torch.stack(outputs)

def load_text_data_from_file(label_converter, image_channel, target_type, max_label_len, is_pil, logger):
	import text_data_util

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	if False:
		# When using an image-label info file.
		if target_type == 'word':
			raise NotImplementedError('Input data should be assigned')
			image_label_info_filepath = None
		elif target_type == 'textline':
			image_label_info_filepath = data_base_dir_path + '/text/receipt/sminds/receipt_text_line/labels.txt'
		else:
			raise ValueError('Invalid target type, {}'.format(target_type))

		images, labels_str, labels_int = text_data_util.load_data_from_image_label_info(label_converter, image_label_info_filepath, image_channel, max_label_len, image_label_separator=' ', is_pil=is_pil)
	else:
		# When using image-label files.
		if target_type == 'word':
			raise NotImplementedError('Input data should be assigned')
			image_filepaths = None
			label_filepaths = None
		elif target_type == 'textline':
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/general/sminds/20200812/image/*.jpg', recursive=False))
			label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/general/sminds/20200812/label/*.txt', recursive=False))
		else:
			raise ValueError('Invalid target type, {}'.format(target_type))
		assert len(image_filepaths) == len(label_filepaths)

		images, labels_str, labels_int = text_data_util.load_data_from_image_and_label_files(label_converter, image_filepaths, label_filepaths, image_channel, max_label_len, is_pil=is_pil)

	return images, labels_int

def create_datasets_for_training(charset, wordset, font_list, target_type, image_shape, label_converter, max_label_len, train_test_ratio, is_mixed_text_used, is_pil, logger):
	image_height, image_width, image_channel = image_shape
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
	chars = charset.replace(' ', '')  # Remove the blank space. Can make the number of each character different.
	if target_type == 'char':
		char_clipping_ratio_interval = (0.8, 1.25)

		# File-based chars: 78,838.
		if is_mixed_text_used:
			num_simple_char_examples_per_class, num_noisy_examples_per_class = 300, 300  # For mixed chars.
			train_dataset, test_dataset = create_mixed_char_datasets(label_converter, charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, is_pil, logger)
		else:
			char_type = 'simple_char'  # {'simple_char', 'noisy_char', 'file_based_char'}.
			num_train_examples_per_class, num_test_examples_per_class = 500, 50  # For simple and noisy chars.
			train_dataset, test_dataset = create_char_datasets(char_type, label_converter, charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, char_clipping_ratio_interval, font_list, font_size_interval, color_functor, is_pil, logger)
	elif target_type == 'word':
		word_len_interval = (1, max_label_len)
		word_count_interval, space_count_interval, char_space_ratio_interval = None, None, None

		# File-based words: 504,279.
		if is_mixed_text_used:
			num_simple_examples, num_random_examples, num_trdg_examples = int(6e5), int(3e5), int(6e5)  # For mixed words.
			train_dataset, test_dataset = create_mixed_word_datasets(label_converter, wordset, chars, num_simple_examples, num_random_examples, num_trdg_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_label_len, word_len_interval, font_list, font_size_interval, color_functor, is_pil, logger)
		else:
			word_type = 'simple_word'  # {'simple_word', 'random_word', 'trdg_word', 'aihub_word', 'file_based_word'}.
			num_train_examples, num_test_examples = int(1e6), int(1e4)  # For simple and random words.
			train_dataset, test_dataset = create_word_datasets(word_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_label_len, word_len_interval, font_list, font_size_interval, color_functor, is_pil, logger)
	elif target_type == 'textline':
		word_len_interval = (1, 20)
		word_count_interval = (1, 5)
		space_count_interval = (1, 3)
		char_space_ratio_interval = (0.8, 1.25)

		# File-based text lines: 55,835.
		if is_mixed_text_used:
			num_simple_examples, num_random_examples, num_trdg_examples = int(5e4), int(5e4), int(5e4)  # For mixed text lines.
			train_dataset, test_dataset = create_mixed_textline_datasets(label_converter, wordset, chars, num_simple_examples, num_random_examples, num_trdg_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_label_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, is_pil, logger)
		else:
			textline_type = 'simple_textline'  # {'simple_textline', 'random_textline', 'trdg_textline', 'aihub_textline', 'file_based_textline'}.
			num_train_examples, num_test_examples = int(2e5), int(2e3)  # For simple, random, and TRDG text lines.
			train_dataset, test_dataset = create_textline_datasets(textline_type, label_converter, wordset, chars, num_train_examples, num_test_examples, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, max_label_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, font_list, font_size_interval, color_functor, is_pil, logger)
	else:
		raise ValueError('Invalid target type, {}'.format(target_type))

	return train_dataset, test_dataset

def create_text_dataset(label_converter, image_shape, target_type, max_label_len, is_preloaded_image_used, is_pil, logger):
	is_aihub_data_used = False

	image_height, image_width, image_channel = image_shape
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	#image_height_before_crop, image_width_before_crop = image_height, image_width

	transform = torchvision.transforms.Compose([
		ResizeToFixedSize(image_height, image_width, warn_about_small_image=True, is_pil=is_pil, logger=logger),
		#ResizeToBelowMaxWidth(image_height, image_width, warn_about_small_image=True, is_pil=is_pil, logger=logger),  # batch_size must be 1.
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5,) * image_channel, std=(0.5,) * image_channel)  # [0, 1] -> [-1, 1].
	])
	target_transform = torch.IntTensor

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	datasets = list()
	if target_type == 'word':
		# When using a dataset with an image-label info file.
		image_label_info_filepath = data_base_dir_path + '/text/scene_text/e2e_mlt/word_images_kr.txt'
		#image_label_info_filepath = data_base_dir_path + '/text/scene_text/icdar_mlt_2019/word_images_kr.txt'
		datasets.append(text_data.InfoFileBasedWordDataset(label_converter, image_label_info_filepath, image_channel, max_label_len, is_preloaded_image_used, transform=transform, target_transform=target_transform))
	elif target_type == 'textline':
		# When using a dataset with image-label files.
		if False:
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.jpg', recursive=False))
			label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/receipt/icdar2019_sroie/task1_train_text_line/*.txt', recursive=False))
		else:
			image_filepaths = sorted(glob.glob(data_base_dir_path + '/text/general/sminds/20200812/image/*.jpg', recursive=False))
			label_filepaths = sorted(glob.glob(data_base_dir_path + '/text/general/sminds/20200812/label/*.txt', recursive=False))
		if image_filepaths and label_filepaths and len(image_filepaths) == len(label_filepaths):
			if logger: logger.info('#loaded image files = {}, #loaded label files = {}.'.format(len(image_filepaths), len(label_filepaths)))
		else:
			if logger: logger.error('#loaded image files = {}, #loaded label files = {}.'.format(len(image_filepaths), len(label_filepaths)))
			raise RuntimeError('Invalid input images and labels, {} != {}'.format(len(image_filepaths), len(label_filepaths)))

		datasets.append(text_data.ImageLabelFileBasedTextLineDataset(label_converter, image_filepaths, label_filepaths, image_channel, max_label_len, is_preloaded_image_used, transform=transform, target_transform=target_transform))

	if is_aihub_data_used:
		# When using AI-Hub data.
		if target_type == 'word':
			image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.
		elif target_type == 'textline':
			image_types_to_load = ['word', 'sentence']  # {'syllable', 'word', 'sentence'}.

		aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
		aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

		datasets.append(aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used, transform=transform, target_transform=target_transform))
	assert datasets, 'NO dataset'

	return torch.utils.data.ConcatDataset(datasets)

def extract_text_rectangle_from_aabb(image, aabboxes, output_dir_path):
	text_patches = list()
	rgb = image.copy()
	for idx, bbox in enumerate(aabboxes):
		x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
		x1, y1, x2, y2 = math.floor(float(x1)), math.floor(float(y1)), math.ceil(float(x2)), math.ceil(float(y2))
		#x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 1, y2 + 1
		patch = image[y1:y2,x1:x2]
		text_patches.append(patch)

		cv2.imwrite(os.path.join(output_dir_path, 'text_{}.png'.format(idx)), patch)

		cv2.rectangle(rgb, (x1, y1), (x2, y2), (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_8)
	cv2.imwrite(os.path.join(output_dir_path, 'text_bbox.png'), rgb)
	return text_patches

def extract_simple_text_rectangle_from_polygon(image, polygons, output_dir_path):
	text_patches = list()
	rgb = image.copy()
	for idx, poly in enumerate(polygons):
		(x1, y1), (x2, y2) = np.min(poly, axis=0), np.max(poly, axis=0)
		x1, y1, x2, y2 = math.floor(float(x1)), math.floor(float(y1)), math.ceil(float(x2)), math.ceil(float(y2))
		#x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 1, y2 + 1
		patch = image[y1:y2,x1:x2]
		text_patches.append(patch)

		cv2.imwrite(os.path.join(output_dir_path, 'text_{}.png'.format(idx)), patch)

		cv2.rectangle(rgb, (x1, y1), (x2, y2), (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_8)
	cv2.imwrite(os.path.join(output_dir_path, 'text_bbox.png'), rgb)
	return text_patches

def extract_masked_text_rectangle_from_polygon(image, polygons, output_dir_path):
	image_height, image_width = image.shape[:2]
	text_patches = list()
	rgb = image.copy()
	black_image = np.zeros_like(image)
	mask_value = (255,) * image.ndim
	for idx, poly in enumerate(polygons):
		(x1, y1), (x2, y2) = np.min(poly, axis=0), np.max(poly, axis=0)
		x1, y1, x2, y2 = math.floor(float(x1)), math.floor(float(y1)), math.ceil(float(x2)), math.ceil(float(y2))
		x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)
		#x1, y1, x2, y2 = max(0, x1 - 1), max(0, y1 - 1), min(image_width, x2 + 1), min(image_height, y2 + 1)
		patch_height, patch_width = y2 - y1, x2 - x1
		poly = np.expand_dims(np.round(poly).astype(np.int), axis=1)  # For OpenCV.

		mask = np.zeros((patch_height, patch_width) + image.shape[2:], dtype=np.uint8)
		#cv2.fillPoly(mask, poly - np.array([x1, y1]), mask_value, cv2.LINE_8)  # Error: Not working.
		cv2.fillConvexPoly(mask, poly - np.array([x1, y1]), mask_value, cv2.LINE_8)

		patch = np.where(mask > 0, image[y1:y2,x1:x2], black_image[:patch_height,:patch_width])
		text_patches.append(patch)

		cv2.imwrite(os.path.join(output_dir_path, 'text_{}.png'.format(idx)), patch)
		cv2.polylines(rgb, [poly], True, (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_8)
	cv2.imwrite(os.path.join(output_dir_path, 'text_bbox.png'), rgb)
	return text_patches

def extract_rotated_text_rectangle_from_polygon(image, polygons, output_dir_path):
	"""
	def compute_rotation_angle(pts):
		pts -= np.mean(pts, axis=0)
		try:
			# PCA.
			#eigvals, eigvecs = np.linalg.eig(np.matmul(pts.transpose(), pts))
			_, singvals, singvecs = np.linalg.svd(pts, full_matrices=False, compute_uv=True)
			#eigvals = singvals**2
			#eigvecs = singvecs.transpose()
			eigvec = singvecs.transpose()[:,0]
			if eigvec[0] < 0: eigvec = -eigvec
			return math.atan2(eigvec[1], eigvec[0])
		except np.LinAlgError as ex:
			if logger: logger.error('numpy.LinAlgError raised: {}.'.format(ex))
			return None

	def rotate_image(image, angle):
		#return scipy.ndimage.rotate(image, angle * 180 / math.pi)
		ctr = tuple(np.array(image.shape[1::-1]) / 2)
		R = cv2.getRotationMatrix2D(ctr, angle * 180 / math.pi, 1.0)
		return cv2.warpAffine(image, R, image.shape[1::-1], flags=cv2.INTER_LINEAR)

	image_height, image_width = image.shape[:2]
	text_patches = list()
	rgb = image.copy()
	black_image = np.zeros_like(image)
	mask_value = (255,) * image.ndim
	for idx, poly in enumerate(polygons):
		canvas = np.zeros(image.shape[:2], dtype=np.uint8)
		cv2.fillConvexPoly(canvas, np.round(poly).astype(np.int), 1, cv2.LINE_8)
		pts = np.stack(np.nonzero(canvas)).transpose().astype(np.float)[:,[1, 0]]
		angle = compute_rotation_angle(pts)

		(x1, y1), (x2, y2) = np.min(poly, axis=0), np.max(poly, axis=0)
		x1, y1, x2, y2 = math.floor(float(x1)), math.floor(float(y1)), math.ceil(float(x2)), math.ceil(float(y2))
		x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)
		#x1, y1, x2, y2 = max(0, x1 - 1), max(0, y1 - 1), min(image_width, x2 + 1), min(image_height, y2 + 1)
		patch_height, patch_width = y2 - y1, x2 - x1
		poly = np.expand_dims(np.round(poly).astype(np.int), axis=1)  # For OpenCV.

		mask = np.zeros((patch_height, patch_width) + image.shape[2:], dtype=np.uint8)
		#cv2.fillPoly(mask, poly - np.array([x1, y1]), mask_value, cv2.LINE_8)  # Error: Not working.
		cv2.fillConvexPoly(mask, poly - np.array([x1, y1]), mask_value, cv2.LINE_8)

		patch = np.where(mask > 0, image[y1:y2,x1:x2], black_image[:patch_height,:patch_width])
		patch = rotate_image(patch, angle)
		text_patches.append(patch)

		cv2.imwrite(os.path.join(output_dir_path, 'text_{}.png'.format(idx)), patch)
		cv2.polylines(rgb, [poly], True, (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_8)
	cv2.imwrite(os.path.join(output_dir_path, 'text_bbox.png'), rgb)
	return text_patches
	"""
	text_patches = list()
	rgb = image.copy()
	for idx, poly in enumerate(polygons):
		obb_center, obb_size, obb_angle = cv2.minAreaRect(poly)  # Tuple: (center, size, angle).
		# TODO [check] >>
		#if obb_size[0] < obb_size[1]:
		if obb_angle < -10 or obb_angle > 10:
			obb_size, obb_angle = obb_size[1::-1], obb_angle + 90

		radius = math.sqrt(obb_size[0]**2 + obb_size[1]**2) / 2
		dia = math.ceil(radius * 2)

		patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(obb_center[0] - radius) - 1), max(0, math.floor(obb_center[1] - radius) - 1), min(rgb.shape[1], math.ceil(obb_center[0] + radius) + 1), min(rgb.shape[0], math.ceil(obb_center[1] + radius) + 1)
		#patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(obb_center[0] - radius)), max(0, math.floor(obb_center[1] - radius)), min(rgb.shape[1], math.ceil(obb_center[0] + radius) + 1), min(rgb.shape[0], math.ceil(obb_center[1] + radius) + 1)
		patch = rgb[patch_y1:patch_y2, patch_x1:patch_x2]

		ctr = patch.shape[1] / 2, patch.shape[0] / 2
		R = cv2.getRotationMatrix2D(ctr, angle=obb_angle, scale=1)
		rotated = cv2.warpAffine(patch, R, (dia, dia), flags=cv2.INTER_LINEAR)

		rotated_patch = rotated[math.floor(ctr[1] - obb_size[1] / 2):math.ceil(ctr[1] + obb_size[1] / 2), math.floor(ctr[0] - obb_size[0] / 2):math.ceil(ctr[0] + obb_size[0] / 2)]
		text_patches.append(rotated_patch)

		cv2.imwrite(os.path.join(output_dir_path, 'text_{}.png'.format(idx)), rotated_patch)
		cv2.polylines(rgb, [np.int0(poly)], True, (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_8)
	cv2.imwrite(os.path.join(output_dir_path, 'text_bbox.png'), rgb)
	return text_patches

def extract_rectified_text_rectangle_from_polygon(image, polygons, output_dir_path):
	"""
	# REF [site] >> https://docs.opencv.org/4.5.0/db/da4/samples_2dnn_2text_detection_8cpp-example.html
	def fourPointsTransform(image, pts, target_pts=None):
		if target_pts is None:
			height, width = image.shape[:2]
			target_pts = ((0, height - 1), (0, 0), (width - 1, 0), (width - 1, height - 1))
		T = cv2.getPerspectiveTransform(pts, target_pts, solveMethod=cv.DECOMP_LU)
		return cv2.warpPerspective(image, T, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	"""

	text_patches = list()
	rgb = image.copy()
	for idx, poly in enumerate(polygons):
		obb_center, obb_size, obb_angle = cv2.minAreaRect(poly)  # Tuple: (center, size, angle).
		obb_pts = cv2.boxPoints((obb_center, obb_size, obb_angle))  # 4 x 2. np.float32.

		if obb_size[0] >= obb_size[1]:
			target_pts = np.float32([[0, obb_size[1]], [0, 0], [obb_size[0], 0], [obb_size[0], obb_size[1]]])
			canvas_size = round(obb_size[0]), round(obb_size[1])
		else:
			target_pts = np.float32([[obb_size[1], obb_size[0]], [0, obb_size[0]], [0, 0], [obb_size[1], 0]])
			canvas_size = round(obb_size[1]), round(obb_size[0])
		T = cv2.getPerspectiveTransform(obb_pts, target_pts, solveMethod=cv2.DECOMP_LU)  # Four points.
		patch = cv2.warpPerspective(rgb, T, canvas_size, flags=cv2.INTER_LINEAR)
		text_patches.append(patch)

		cv2.imwrite(os.path.join(output_dir_path, 'text_{}.png'.format(idx)), patch)
		cv2.polylines(rgb, [np.int0(poly)], True, (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_8)
	cv2.imwrite(os.path.join(output_dir_path, 'text_bbox.png'), rgb)
	return text_patches

def build_craft_model(craft_refine, craft_cuda, logger=None):
	import craft.test_utils as test_utils

	craft_trained_model_filepath = './craft/craft_mlt_25k.pth'
	craft_refiner_model_filepath = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.

	craft_net, craft_refine_net = test_utils.load_craft(craft_trained_model_filepath, craft_refiner_model_filepath, craft_refine, craft_cuda)

	return craft_net, craft_refine_net

def detect_chars_by_craft(image_filepath, craft_refine, craft_cuda, output_dir_path, logger):
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils
	import craft.test_utils as test_utils

	#--------------------
	if logger: logger.info('Start loading CRAFT...')
	start_time = time.time()
	craft_net, craft_refine_net = build_craft_model(craft_refine, craft_cuda, logger=logger)
	if logger: logger.info('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	#--------------------
	if logger: logger.info('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, ch_bboxes_lst, score_text = test_utils.run_char_craft(rgb, craft_net, craft_refine_net, craft_cuda)
	if logger: logger.info('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
		output_dir_path = os.path.join(output_dir_path, 'char_craft_results')
		os.makedirs(output_dir_path, exist_ok=True)

		if logger: logger.info('Start inferring...')
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

		char_patches = list()
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
			char_patches.append(imgs)
		cv2.imwrite(os.path.join(output_dir_path, 'char_bbox.png'), rgb)
		return char_patches
	else: return None

def detect_texts_by_craft(image_filepath, craft_refine, craft_cuda, output_dir_path, logger):
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils
	import craft.test_utils as test_utils

	#--------------------
	if logger: logger.info('Start loading CRAFT...')
	start_time = time.time()
	craft_net, craft_refine_net = build_craft_model(craft_refine, craft_cuda, logger=logger)
	if logger: logger.info('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	#--------------------
	if logger: logger.info('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, polys, score_text = test_utils.run_word_craft(rgb, craft_net, craft_refine_net, craft_cuda)
	if logger: logger.info('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
		image = cv2.imread(image_filepath)
		if image is None:
			if logger: logger.error('File not found, {}.'.format(image_filepath))
			return None, None

		if False:
			cv2.imshow('Input', image)
			rgb1, rgb2 = image.copy(), image.copy()
			for bbox, poly in zip(bboxes, polys):
				color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
				cv2.drawContours(rgb1, [np.round(np.expand_dims(bbox, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
				cv2.drawContours(rgb2, [np.round(np.expand_dims(poly, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
			cv2.imshow('BBox', rgb1)
			cv2.imshow('Poly', rgb2)
			cv2.waitKey(0)

		output_dir_path = os.path.join(output_dir_path, 'craft_results')
		os.makedirs(output_dir_path, exist_ok=True)

		#return extract_simple_text_rectangle_from_polygon(image, bboxes, output_dir_path), bboxes
		#return extract_masked_text_rectangle_from_polygon(image, bboxes, output_dir_path), bboxes
		#return extract_rotated_text_rectangle_from_polygon(image, bboxes, output_dir_path), bboxes  # FIXME [check] >>
		return extract_rectified_text_rectangle_from_polygon(image, bboxes, output_dir_path), bboxes
	else: return None, None

def crop_text_region_in_image(images):
	min_image_height, min_image_width = 10, 10
	use_laplacian = True
	if use_laplacian:
		sum_threshold = 300
		offset, margin = 5, 5
	else:
		sum_threshold = 1000  # ???
		offset, margin = 2, 5
	cropped_images = list()
	for img in images:
		if use_laplacian:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray,  (5, 5), cv2.BORDER_DEFAULT)
			#gray = 255 - gray
			#_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			#gray = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
			#gray = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
			laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3, borderType=cv2.BORDER_DEFAULT)
			#minval, maxval = np.min(laplacian), np.max(laplacian)
			#laplacian = cv2.adaptiveThreshold(((laplacian - minval) * 255 / (maxval - minval)).astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

			#indices = np.nonzero(np.abs(laplacian[offset:-offset,offset:-offset]) > sum_threshold)
			indices = np.nonzero(np.abs(np.sum(laplacian[offset:-offset], axis=1)) > sum_threshold)[0], np.nonzero(np.abs(np.sum(laplacian[:,offset:-offset], axis=0)) > sum_threshold)[0]

			"""
			minval, maxval = np.min(laplacian), np.max(laplacian)
			cv2.imshow('Laplacian', (laplacian - minval) / (maxval - minval))

			plt.title('Laplacian Sum (X)')
			plt.plot(range(laplacian.shape[1] - 10), np.sum(laplacian[:,5:-5], axis=0), color='red')
			plt.show()
			"""
		else:
			#indices = np.nonzero(np.abs(img[offset:-offset,offset:-offset]) > sum_threshold)
			indices = np.nonzero(np.abs(np.sum(img[offset:-offset], axis=1)) > sum_threshold)[0], np.nonzero(np.abs(np.sum(img[:,offset:-offset], axis=0)) > sum_threshold)[0]

			"""
			plt.title('Image Sum (X)')
			plt.plot(range(img.shape[1] - 4), np.sum(img[:,2:-2], axis=0), color='red')
			plt.show()
			"""

		if indices[0].size != 0 and indices[1].size != 0:
			y1, y2, x1, x2 = np.min(indices[0]) + offset - margin, np.max(indices[0]) + offset + margin, np.min(indices[1]) + offset - margin, np.max(indices[1]) + offset + margin
			cropped = img[y1:y2,x1:x2] if y2 - y1 > 1 and x2 - x1 > 1 else None
		else: cropped = None
		cropped_images.append(cropped)

		"""
		cv2.imshow('Image', img)
		cv2.imshow('Gray', gray)
		if cropped is not None:
			cv2.imshow('Cropped', cropped)
		cv2.waitKey(0)
		"""
	return cropped_images

def extract_cells_in_table(image, cell_contours, output_dir_path, logger):
	if len(cell_contours) <= 0: return None

	output_dir_path = os.path.join(output_dir_path, 'table_results')
	os.makedirs(output_dir_path, exist_ok=True)

	if True:
		aabboxes = list(cv2.boundingRect(contour) for contour in cell_contours)  # (left, top, width, height).
		text_patches = extract_text_rectangle_from_aabb(image, aabboxes, output_dir_path)
		return crop_text_region_in_image(text_patches)
	else:
		#cell_contours = list(cv2.convexHull(contour) for contour in cell_contours)  # Optional.

		if False:
			cv2.imshow('Input', image)
			rgb = image.copy()
			for contour in cell_contours:
				color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
				cv2.drawContours(rgb, [np.round(np.expand_dims(contour, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
			cv2.imshow('Cell Contours', rgb)
			cv2.waitKey(0)

		#return extract_simple_text_rectangle_from_polygon(image, cell_contours, output_dir_path)
		#return extract_masked_text_rectangle_from_polygon(image, cell_contours, output_dir_path)
		return extract_rotated_text_rectangle_from_polygon(image, cell_contours, output_dir_path)
		#return extract_rectified_text_rectangle_from_polygon(image, cell_contours, output_dir_path)

def visualize_table_recognition_results(image_shape, cell_contours, cell_patches, valid_cell_texts, font_filepath, output_dir_path, logger=None):
	if len(cell_contours) <= 0 or len(cell_patches) <= 0 or len(valid_cell_texts) <= 0: return None

	#output_dir_path = os.path.join(output_dir_path, 'table_results')
	#os.makedirs(output_dir_path, exist_ok=True)

	canvas = Image.new(mode='RGB', size=(image_shape[1], image_shape[0]), color=(255, 255, 255))
	draw = ImageDraw.Draw(canvas)
	valid_patch_idx = 0
	for contour, patch in zip(cell_contours, cell_patches):
		aabb = cv2.boundingRect(contour)  # (left, top, width, height).
		left, top, right, bottom = aabb[0], aabb[1], aabb[0] + aabb[2], aabb[1] + aabb[3]
		draw.rectangle((left, top, right, bottom), outline=(127, 127, 127), width=2)

		if patch is not None:
			try:
				font = ImageFont.truetype(font=font_filepath, size=patch.shape[0], index=0)
			except Exception as ex:
				if logger: logger.warning('Invalid font, {}: {}.'.format(font_filepath, ex))
				continue

			txt = valid_cell_texts[valid_patch_idx]
			txt = txt.replace('<UNK>', '')  # Optional.
			draw.text(xy=(left, top), text=txt, font=font, fill=(0, 0, 0))

			valid_patch_idx += 1
	canvas.save(output_dir_path + '/table_result.png')

def visualize_scene_text_recognition_results(image_shape, text_bboxes, texts, output_dir_path, is_2x=False, logger=None):
	if len(text_bboxes) <= 0: return None

	#output_dir_path = os.path.join(output_dir_path, 'scene_text_results')
	#os.makedirs(output_dir_path, exist_ok=True)

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_filepath = font_base_dir_path + '/kor_large/batang.ttf'

	scale_factor = 2 if is_2x else 1
	canvas = Image.new(mode='RGB', size=(image_shape[1] * scale_factor, image_shape[0] * scale_factor), color=(255, 255, 255))
	draw = ImageDraw.Draw(canvas)
	for bbox, txt in zip(text_bboxes, texts):
		(left, top), (right, bottom) = np.min(bbox, axis=0), np.max(bbox, axis=0)
		left, top, right, bottom = math.floor(left * scale_factor), math.floor(top * scale_factor), math.ceil(right * scale_factor), math.ceil(bottom * scale_factor)

		try:
			obb = cv2.minAreaRect(bbox)  # Tuple: (center, size, angle).
			#font_size = int(min(obb[1]) * scale_factor)
			font_size = int(min(obb[1]))
			font = ImageFont.truetype(font=font_filepath, size=font_size, index=0)
		except Exception as ex:
			if logger: logger.warning('Invalid font, {}: {}.'.format(font_filepath, ex))
			return

		txt = txt.replace('<UNK>', '')  # Optional.
		draw.text(xy=(left, top), text=txt, font=font, fill=(0, 0, 0))
	canvas.save(output_dir_path + '/scene_text_result.png')

def visualize_inference_results(predictions, label_converter, inputs, outputs, output_dir_path, is_case_sensitive, num_examples_to_visualize, logger=None):
	if not num_examples_to_visualize or num_examples_to_visualize <= 0:
		num_examples_to_visualize = len(predictions)

	if outputs is None:
		# Show images.
		#show_image(torchvision.utils.make_grid(inputs[:num_examples_to_visualize]))

		if logger: logger.info('Prediction:\n{}.'.format('\n'.join([label_converter.decode(pred) for pred in predictions[:num_examples_to_visualize]])))
	else:
		error_cases_dir_path = os.path.join(output_dir_path, 'inf_text_error_cases')
		if error_cases_dir_path and error_cases_dir_path.strip() and not os.path.exists(error_cases_dir_path):
			os.makedirs(error_cases_dir_path, exist_ok=True)

		if logger:
			#logger.info('G/T:        {}.'.format(' '.join([label_converter.decode(lbl) for lbl in outputs[:num_examples_to_visualize]])))
			#logger.info('Prediction: {}.'.format(' '.join([label_converter.decode(lbl) for lbl in predictions[:num_examples_to_visualize]])))
			#for gt, pred in zip(outputs[:num_examples_to_visualize], predictions[:num_examples_to_visualize]):
			#	logger.info('G/T - prediction: {}, {}.'.format(label_converter.decode(gt), label_converter.decode(pred)))
			logger.info('G/T - prediction:\n{}.'.format([(label_converter.decode(gt), label_converter.decode(pred)) for gt, pred in zip(outputs[:num_examples_to_visualize], predictions[:num_examples_to_visualize])]))

		#--------------------
		if inputs.ndim == 4: inputs = inputs.transpose(0, 2, 3, 1)
		# TODO [decide] >>
		#minval, maxval = np.min(inputs), np.max(inputs)
		#minval, maxval = -1, 1
		minval, maxval = 0, 1
		inputs = np.round((inputs - minval) * 255 / (maxval - minval)).astype(np.uint8)  # [0, 255].

		is_sequence_matching_ratio_used, is_simple_matching_accuracy_used = True, True
		if is_sequence_matching_ratio_used:
			total_matching_ratio, error_cases = compute_sequence_matching_ratio(inputs, outputs, predictions, label_converter, is_case_sensitive, error_cases_dir_path, error_idx=0)
		if is_simple_matching_accuracy_used:
			correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count, error_cases = compute_simple_matching_accuracy(inputs, outputs, predictions, label_converter, is_case_sensitive, error_cases_dir_path, error_idx=0)

		if is_sequence_matching_ratio_used and error_cases_dir_path:
			err_fpath = os.path.join(error_cases_dir_path, 'error_cases.txt')
			try:
				with open(err_fpath, 'w', encoding='UTF8') as fd:
					for idx, (gt, pred) in enumerate(error_cases):
						fd.write('{}\t{}\t{}\n'.format(idx, gt, pred))
			except UnicodeDecodeError as ex:
				if logger: logger.warning('Unicode decode error in {}: {}.'.format(err_fpath, ex))
			except FileNotFoundError as ex:
				if logger: logger.warning('File not found, {}: {}.'.format(err_fpath, ex))

		correct_char_class_count, total_char_class_count = compute_per_char_accuracy(inputs, outputs, predictions, label_converter.num_tokens)
		show_per_char_accuracy(correct_char_class_count, total_char_class_count, label_converter.tokens, label_converter.num_tokens, show_acc_per_char=True, logger=logger)
		if is_sequence_matching_ratio_used:
			#num_examples = len(outputs)
			num_examples = min(len(inputs), len(outputs), len(predictions))
			avg_matching_ratio = total_matching_ratio / num_examples if num_examples > 0 else total_matching_ratio
			if logger: logger.info('Average sequence matching ratio = {}.'.format(avg_matching_ratio))
		if is_simple_matching_accuracy_used:
			if logger:
				logger.info('Text: Simple matching accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count if total_text_count > 0 else -1))
				logger.info('Word: Simple matching accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count if total_word_count > 0 else -1))
				logger.info('Char: Simple matching accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count if total_char_count > 0 else -1))

#--------------------------------------------------------------------

def parse_command_line_options():
	parser = argparse.ArgumentParser(description='Runner for text recognition models.')

	parser.add_argument(
		'--train',
		action='store_true',
		help='Specify whether to train a model'
	)
	parser.add_argument(
		'--eval',
		action='store_true',
		help='Specify whether to evaluate a trained model'
	)
	parser.add_argument(
		'--infer',
		action='store_true',
		help='Specify whether to infer by a trained model'
	)
	parser.add_argument(
		'-tt',
		'--target_type',
		choices={'char', 'word', 'textline'},
		help='Target type',
		default='textline'
	)
	parser.add_argument(
		'-mt',
		'--model_type',
		choices={'char', 'char-mixup', 'rare1', 'rare1-mixup', 'rare2', 'aster', 'onmt', 'rare1+onmt', 'rare2+onmt', 'aster+onmt', 'transformer'},
		help='Model type',
		default='transformer'
	)
	parser.add_argument(
		'-ft',
		'--font_type',
		choices={'kor-small', 'kor-large', 'kor-receipt', 'eng-small', 'eng-large', 'eng-receipt'},
		help='Font type',
		default='kor-large'
	)
	parser.add_argument(
		'-is',
		'--image_shape',
		type=str,
		#help='Image shape, HxWxC where H: height, W: width, C: channel = {1, 3}',
		help='Image shape, HxWxC where H: height, W: width, C: channel = 3',
		default='64x1280x3'
	)
	parser.add_argument(
		'-ml',
		'--max_len',
		type=int,
		help='Max. label length',
		default=50
	)
	parser.add_argument(
		'-mf',
		'--model_file',
		type=str,
		#nargs='?',
		help='The model file path to load a pretrained model',
		#required=True,
		default=None
	)
	parser.add_argument(
		'-o',
		'--out_dir',
		type=str,
		#nargs='?',
		help='The output directory path to save results such as images and log',
		#required=True,
		default=None
	)
	"""
	parser.add_argument(
		'-tr',
		'--train_data_dir',
		type=str,
		#nargs='?',
		help='The directory path of training data',
		default='./train_data'
	)
	parser.add_argument(
		'-te',
		'--test_data_dir',
		type=str,
		#nargs='?',
		help='The directory path of test data',
		default='./test_data'
	)
	"""
	parser.add_argument(
		'-e',
		'--epoch',
		type=int,
		help='Number of epochs to train',
		default=20
	)
	parser.add_argument(
		'-b',
		'--batch',
		type=int,
		help='Batch size',
		default=64
	)
	parser.add_argument(
		'-l',
		'--log',
		type=str,
		help='The name of logger and log files',
		default=None
	)
	parser.add_argument(
		'-ll',
		'--log_level',
		type=int,
		help='Log level, [0, 50]',  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
		default=None
	)
	parser.add_argument(
		'-ld',
		'--log_dir',
		type=str,
		help='The directory path to log',
		default=None
	)
	parser.add_argument(
		'-g',
		'--gpu',
		type=str,
		help='Specify GPU to use',
		default='0'
	)

	return parser.parse_args()

def get_logger(name, log_level=None, log_dir_path=None, is_rotating=True):
	if not log_level: log_level = logging.INFO
	if not log_dir_path: log_dir_path = './log'
	if not os.path.exists(log_dir_path):
		os.makedirs(log_dir_path, exist_ok=True)

	log_filepath = os.path.join(log_dir_path, (name if name else 'swl') + '.log')
	if is_rotating:
		file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10000000, backupCount=10)
	else:
		file_handler = logging.FileHandler(log_filepath)
	stream_handler = logging.StreamHandler()

	formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)s][%(asctime)s] [SWL] %(message)s')
	#formatter = logging.Formatter('[%(levelname)s][%(asctime)s] [SWL] %(message)s')
	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def main():
	args = parse_command_line_options()

	logger = get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else args.out_dir, is_rotating=True)
	logger.info('----------------------------------------------------------------------')
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Python version: {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('Torch version: {}.'.format(torch.__version__))
	logger.info('cuDNN version: {}.'.format(torch.backends.cudnn.version()))

	if not args.train and not args.eval and not args.infer:
		logger.error('At least one of command line options "--train", "--eval", and "--infer" has to be specified.')
		return
	image_shape = list(int(sz) for sz in args.image_shape.split('x'))
	if len(image_shape) != 3:
		logger.error('Invalid image shape, {}: The image shape has the form of HxWxC.'.format(args.image_shape))
		return
	#if image_shape[2] not in [1, 3]:
	#	logger.error('Invalid image channel, {}: The image channel has to be 1 or 3.'.format(image_shape[2]))
	if image_shape[2] != 3:
		logger.error('Invalid image channel, {}: The image channel has to be 3.'.format(image_shape[2]))
		return

	#if args.gpu:
	#	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	device = torch.device(('cuda:{}'.format(args.gpu) if int(args.gpu) >= 0 else 'cuda') if torch.cuda.is_available() else 'cpu')
	logger.info('Device: {}.'.format(device))

	#--------------------
	model_filepath_to_load, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	#if model_filepath_to_load and not output_dir_path:
	#	output_dir_path = os.path.dirname(model_filepath_to_load)
	if not output_dir_path:
		#output_dir_prefix = 'text_recognition'
		output_dir_prefix = '{}_{}_{}_ch{}_{}'.format(args.target_type, args.model_type, args.font_type, args.max_len, args.image_shape)
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
		os.makedirs(output_dir_path, exist_ok=True)

	#model_filepath = os.path.join(output_dir_path, 'model.pth')
	model_filepath = None

	logger.info('Output directory path: {}.'.format(output_dir_path))
	#if model_filepath_to_load: logger.info('Model filepath to load: {}.'.format(model_filepath_to_load))
	#if model_filepath: logger.info('Model filepath to save: {}.'.format(model_filepath))

	#--------------------
	swa = False  # Specified whether Stochastic Weight Averaging (SWA) is applied or not.
	is_case_sensitive = False
	is_pil = True  # Specifies whether PIL or OpenCV is used.
	num_workers = 8

	lang = args.font_type[:3]
	if lang == 'kor':
		charset = tg_util.construct_charset(hangeul=True)
	elif lang == 'eng':
		charset = tg_util.construct_charset(hangeul=False)
	else:
		raise ValueError('Invalid language, {}'.format(lang))

	# Create a label converter.
	if args.target_type == 'char':
		label_converter_type = 'basic'  # Fixed.
	elif args.target_type in ['word', 'textline']:
		label_converter_type = 'sos+eos'  # {'basic', 'sos', 'eos', 'sos+eos', 'sos/pad+eos', 'sos+eos/pad', 'blank'}.
	else:
		raise ValueError('Invalid target type, {}'.format(args.target_type))
	label_converter, SOS_ID, EOS_ID, BLANK_LABEL, num_suffixes = create_label_converter(label_converter_type, charset)
	#logger.info('Classes:\n{}.'.format(label_converter.tokens))
	logger.info('#classes = {}.'.format(label_converter.num_tokens))
	logger.info('<PAD> = {}, <SOS> = {}, <EOS> = {}, <UNK> = {}.'.format(label_converter.pad_id, SOS_ID, EOS_ID, label_converter.encode([label_converter.UNKNOWN], is_bare_output=True)[0]))

	#--------------------
	if args.train:
		#is_resumed = args.model_file is not None

		train_test_ratio = 0.8
		is_mixed_text_used = True
		if args.target_type == 'char':
			wordset = None
		elif args.target_type in ['word', 'textline']:
			if lang == 'kor':
				wordset = tg_util.construct_word_set(korean=True, english=True)
			elif lang == 'eng':
				wordset = tg_util.construct_word_set(korean=False, english=True)
		font_list = construct_font([args.font_type])

		# Create datasets.
		logger.info('Start creating datasets...')
		start_time = time.time()
		train_dataset, test_dataset = create_datasets_for_training(charset, wordset, font_list, args.target_type, image_shape, label_converter, args.max_len, train_test_ratio, is_mixed_text_used, is_pil, logger)
		logger.info('End creating datasets: {} secs.'.format(time.time() - start_time))
		logger.info('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

		#--------------------
		if args.target_type == 'char':
			# Build a model.
			model, criterion, optimizer, scheduler, is_epoch_based_scheduler, model_params, max_gradient_norm, model_filepath_format = build_char_model_for_training(model_filepath_to_load, args.model_type, image_shape, args.target_type, args.font_type, output_dir_path, label_converter, logger, device)
			#logger.info('Model:\n{}.'.format(model))

			# Train the model.
			model_filepath = train_char_recognizer(model, criterion, optimizer, scheduler, is_epoch_based_scheduler, train_dataset, test_dataset, output_dir_path, label_converter, model_params, max_gradient_norm, args.epoch, args.batch, num_workers, is_case_sensitive, model_filepath_format, logger, device)
		elif args.target_type in ['word', 'textline']:
			# Build a model.
			model, criterion, optimizer, scheduler, is_epoch_based_scheduler, model_params, max_gradient_norm, model_filepath_format = build_text_model_for_training(model_filepath_to_load, args.model_type, image_shape, args.target_type, args.font_type, args.max_len, output_dir_path, label_converter, SOS_ID, EOS_ID, BLANK_LABEL, num_suffixes, lang, logger, device)
			#logger.info('Model:\n{}.'.format(model))

			# Train the model.
			model_filepath = train_text_recognizer(model, criterion, optimizer, scheduler, is_epoch_based_scheduler, train_dataset, test_dataset, output_dir_path, label_converter, model_params, max_gradient_norm, args.epoch, args.batch, num_workers, is_case_sensitive, model_filepath_format, swa, logger, device)
	elif not model_filepath: model_filepath = model_filepath_to_load

	#--------------------
	if args.eval or args.infer:
		assert model_filepath

		is_preloaded_image_used = False

		#--------------------
		if args.target_type == 'char':
			# Build a moodel.
			model = build_char_model_for_inference(model_filepath_to_load, image_shape, label_converter.num_classes, logger, device)

			#--------------------
			if args.eval and model:
				raise NotImplementedError

			#--------------------
			if args.infer and model:
				# Create data.
				if True:
					# When detecting chars by CRAFT.
					image_filepath = '/path/to/image.png'
					assert os.path.exists(image_filepath)

					craft_refine = False  # Enable a link refiner.
					craft_cuda = torch.cuda.is_available() and int(args.gpu) >= 0
					patches = detect_chars_by_craft(image_filepath, craft_refine, craft_cuda, output_dir_path, logger)
					if patches is None or len(patches) <= 0:
						logger.warning('No text detected in {}.'.format(image_filepath))
						return

					if is_pil: patches = list(Image.fromarray(patch) for patch in patches)
					inputs = images_to_tensor(patches, image_shape, is_pil, logger)
					outputs = None

				# TODO [check] >> This implementation is not tested.
				# Infer by the model.
				logger.info('Start inferring...')
				start_time = time.time()
				model.eval()
				with torch.no_grad():
					predictions = model(inputs)
				logger.info('End inferring: {} secs.'.format(time.time() - start_time))
				predictions = torch.argmax(predictions, dim=1).cpu().numpy()
				logger.info('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(predictions.shape, predictions.dtype, np.min(predictions), np.max(predictions)))

				# Visualize inference results.
				#outputs = None
				num_examples_to_visualize = 50
				visualize_inference_results(predictions, label_converter, inputs.numpy(), outputs, output_dir_path, is_case_sensitive, num_examples_to_visualize, logger)
		elif args.target_type in ['word', 'textline']:
			# Build a model.
			model = build_text_model_for_inference(model_filepath, args.model_type, image_shape, args.max_len, label_converter, SOS_ID, EOS_ID, num_suffixes, lang, swa, logger=logger, device=device)

			#--------------------
			if args.eval and model:
				# Create a dataset.
				logger.info('Start creating a dataset...')
				start_time = time.time()
				dataset = create_text_dataset(label_converter, image_shape, args.target_type, args.max_len, is_preloaded_image_used, is_pil, logger)
				logger.info('End creating a dataset: {} secs.'.format(time.time() - start_time))
				logger.info('#examples = {}.'.format(len(dataset)))

				# Evaluate the model.
				evaluate_text_recognizer(model, dataset, output_dir_path, label_converter, args.batch, num_workers, is_case_sensitive, logger=logger, device=device)

			#--------------------
			if args.infer and model:
				# Create data.
				if True:
					# When loading data.
					logger.info('Start loading data...')
					start_time = time.time()
					images, labels = load_text_data_from_file(label_converter, image_shape[2], args.target_type, args.max_len, is_pil, logger)
					logger.info('End loading data: {} secs.'.format(time.time() - start_time))
					assert len(images) == len(labels)
					logger.info('#examples = {}.'.format(len(images)))

					inputs = images_to_tensor(images, image_shape, is_pil, logger)
					#outputs = labels_to_tensor(labels, max_label_len, label_converter)
					outputs = labels
				elif False:
					# When using a dataset.
					logger.info('Start creating a dataset...')
					start_time = time.time()
					dataset = create_text_dataset(label_converter, image_shape, args.target_type, args.max_len, is_preloaded_image_used, is_pil, logger)
					logger.info('End creating a dataset: {} secs.'.format(time.time() - start_time))
					logger.info('#examples = {}.'.format(len(dataset)))

					inputs, outputs = text_dataset_to_tensor(dataset, args.batch, num_workers, logger)
					outputs = outputs.numpy()
				elif False:
					# When extracting cells in a table.
					# Table information:
					#	REF [file] >> ${DataAnalysis_HOME}/app/document_image/recognize_table_structure.py
					table_info_filepath = '/path/to/table_info.pkl'
					assert os.path.exists(table_info_filepath)

					if logger: logger.info('Start loading table info...')
					start_time = time.time()
					try:
						with open(table_info_filepath, 'rb') as fd:
							image_filepath, cell_contours, cell_neighbors, table_graph = pickle.load(fd)
						assert len(cell_contours) == len(cell_neighbors)
					except FileNotFoundError as ex:
						if logger: logger.error('File not found, {}: {}.'.format(table_info_filepath, ex))
						return None
					if logger: logger.info('End loading table info: {} secs.'.format(time.time() - start_time))

					image = cv2.imread(image_filepath)
					if image is None:
						if logger: logger.error('File not found, {}.'.format(image_filepath))
						return

					cell_patches = extract_cells_in_table(image, cell_contours, output_dir_path, logger)
					if cell_patches is None or len(cell_patches) <= 0:
						logger.warning('No text cell detected in {}.'.format(table_info_filepath))
						return

					valid_cell_patches = list(patch for patch in cell_patches if patch is not None)
					if is_pil: valid_cell_patches = list(Image.fromarray(patch) for patch in valid_cell_patches)
					inputs = images_to_tensor(valid_cell_patches, image_shape, is_pil, logger)
					outputs = None
				else:
					# When detecting texts by CRAFT.
					image_filepath = '/path/to/image.png'
					assert os.path.exists(image_filepath)

					craft_refine = False  # Enable a link refiner.
					craft_cuda = torch.cuda.is_available() and int(args.gpu) >= 0
					patches, text_bboxes = detect_texts_by_craft(image_filepath, craft_refine, craft_cuda, output_dir_path, logger)
					if patches is None or len(patches) <= 0:
						logger.warning('No text detected in {}.'.format(image_filepath))
						return

					if is_pil: patches = list(Image.fromarray(patch) for patch in patches)
					inputs = images_to_tensor(patches, image_shape, is_pil, logger)
					outputs = None

				batch_size = args.batch  # Infer batch-by-batch.
				#batch_size = 1  # Infer one-by-one.

				# Infer by the model.
				logger.info('Start inferring...')
				start_time = time.time()
				model.eval()
				predictions = recognize_text(model, inputs, batch_size, logger=logger, device=device)
				logger.info('End inferring: {} secs.'.format(time.time() - start_time))
				logger.info('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(predictions.shape, predictions.dtype, np.min(predictions), np.max(predictions)))

				if False:
					# Visualize table recognition results.
					predictions = list(label_converter.decode(pred) for pred in predictions)

					if 'posix' == os.name:
						system_font_dir_path = '/usr/share/fonts'
						font_base_dir_path = '/home/sangwook/work/font'
					else:
						system_font_dir_path = 'C:/Windows/Fonts'
						font_base_dir_path = 'D:/work/font'
					font_filepath = font_base_dir_path + '/kor_large/batang.ttf'

					visualize_table_recognition_results(image.shape, cell_contours, cell_patches, predictions, font_filepath, output_dir_path, logger)
				elif False:
					# Visualize scene text recognition results.
					predictions = list(label_converter.decode(pred) for pred in predictions)

					image = cv2.imread(image_filepath)
					if image is None:
						if logger: logger.error('File not found, {}.'.format(image_filepath))
						return

					visualize_scene_text_recognition_results(image.shape, text_bboxes, predictions, output_dir_path, is_2x=False, logger=logger)
				else:
					# Visualize inference results.
					#outputs = None
					num_examples_to_visualize = 50
					visualize_inference_results(predictions, label_converter, inputs.numpy(), outputs, output_dir_path, is_case_sensitive, num_examples_to_visualize, logger)

#--------------------------------------------------------------------

# Usage:
#	python run_text_recognition.py --train --eval --infer --target_type textline --model_type transformer --font_type kor-large --image_shape 64x1280x3 --max_len 50 --epoch 40 --batch 64 --out_dir text_recognition_outputs --log text_recognition --log_dir ./log --gpu 0
#
#	e.g.)
#		python run_text_recognition.py --train --target_type textline --model_type transformer --font_type kor-large --image_shape 64x1280x3 --max_len 50 --epoch 40 --batch 64 --out_dir ./textline_transformer_train_kor-large_ch50_64x1280x3_20201009 --gpu 0
#		python run_text_recognition.py --train --target_type word --model_type aster+onmt --font_type kor-small --image_shape 64x1280x3 --max_len 20 --epoch 30 --batch 64 --model_file ./train_outputs_word/word_aster+onmt_xent_kor-small_ch10_64x640x3.pth --out_dir ./word_aster_onmt_train_kor-small_ch20_64x1280x3_20201009 --gpu 1
#		python run_text_recognition.py --eval --target_type textline --model_type transformer --image_shape 64x1280x3 --max_len 40 --model_file ./train_outputs_textline/textline_transformer_kldiv_kor-large_ch40_64x1280x3.pth --out_dir ./textline_transformer_eval_kor_ch40_64x1280x3_20201009 --gpu 1
#		python run_text_recognition.py --infer --target_type word --model_type onmt --image_shape 64x640x3 --max_len 10 --model_file ./train_outputs_word/word_onmt_xent_kor-large_ch10_64x640x3.pth --out_dir ./word_onmt_infer_kor_ch10_64x640x3_20201009 --gpu 0

if '__main__' == __name__:
	main()
