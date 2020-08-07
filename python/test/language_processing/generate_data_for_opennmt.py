#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, functools, argparse, time
import numpy as np
import torch, torchvision
from PIL import Image, ImageOps
import cv2
import swl.language_processing.util as swl_langproc_util
import text_data
import text_generation_util as tg_util

def construct_font(korean=True, english=True):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	font_dir_paths = []
	if korean:
		font_dir_paths.append(font_base_dir_path + '/kor')
	if english:
		font_dir_paths.append(font_base_dir_path + '/eng')

	return tg_util.construct_font(font_dir_paths)

def construct_chars():
	import string

	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A string.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of strings.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.
	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	"""
	chars = \
		hangeul_charset * 1 + \
		hangeul_jamo_charset * 10 + \
		string.ascii_lowercase * 100 + \
		string.ascii_uppercase * 30 + \
		string.digits * 50 + \
		string.punctuation * 20
	chars *= 500
	"""
	chars = \
		hangeul_charset * 1 + \
		hangeul_jamo_charset * 0 + \
		string.ascii_lowercase * 100 + \
		string.ascii_uppercase * 50 + \
		string.digits * 100 + \
		string.punctuation * 50
	chars *= 500
	"""
	chars = \
		hangeul_charset * 0 + \
		hangeul_jamo_charset * 0 + \
		string.ascii_lowercase * 100 + \
		string.ascii_uppercase * 30 + \
		string.digits * 50 + \
		string.punctuation * 20
	chars *= 500
	"""
	return chars

def create_char_augmenter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	augmenter = iaa.Sequential([
		iaa.Grayscale(alpha=(0.0, 1.0)),
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
		iaa.Sometimes(0.8, iaa.OneOf([
			iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
			iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
		])),
		iaa.Sometimes(0.8, iaa.OneOf([
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
		iaa.Grayscale(alpha=(0.0, 1.0)),
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
		iaa.Sometimes(0.8, iaa.OneOf([
			iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
			iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
		])),
		iaa.Sometimes(0.8, iaa.OneOf([
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
		iaa.Grayscale(alpha=(0.0, 1.0)),
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
				shear=(-1, 1),  # Shear by -1 to +1 degrees.
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
		#iaa.Sometimes(0.5, iaa.OneOf([
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

class ToIntTensor(object):
	def __call__(self, lst):
		return torch.IntTensor(lst)

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

def visualize_data_with_length(dataloader, label_converter, num_data=10):
	data_iter = iter(dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	images = images.transpose(0, 2, 3, 1)
	for idx, (img, lbl, l) in enumerate(zip(images, labels, label_lens)):
		print('Label (len={}): {} (int), {} (str).'.format(l, [ll for ll in lbl if ll != label_converter.pad_id], label_converter.decode(lbl)))
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		if idx >= (num_data - 1): break
	cv2.destroyAllWindows()

def save_data_and_images(is_train, dataloader, data_dir_path, image_dir_path, label_converter):
	src_filepath = os.path.join(data_dir_path, 'src-{}.txt'.format('train' if is_train else 'val'))
	tgt_filepath = os.path.join(data_dir_path, 'tgt-{}.txt'.format('train' if is_train else 'val'))
	image_file_format = 'train_img_{}.png' if is_train else 'val_img_{}.png'
	try:
		with open(src_filepath, 'w', encoding='UTF8') as src_fd, open(tgt_filepath, 'w', encoding='UTF8') as tgt_fd:
			idx = 0
			for batch in dataloader:
				images, labels, _ = batch  # torch.Tensor & torch.Tensor.
				images, labels = images.numpy(), labels.numpy()
				images = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
				for img, lbl in zip(images, labels):
					image_filename = image_file_format.format(idx)
					image_filepath = os.path.join(image_dir_path, image_filename)
					cv2.imwrite(image_filepath, img)
					#src_fd.write(os.path.relpath(image_filepath, image_dir_path) + os.linesep)
					src_fd.write(image_filename + os.linesep)
					lbl = label_converter.decode(lbl, is_string=False)
					tgt_fd.write(' '.join(lbl) + os.linesep)
					idx += 1
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(ex))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(ex))

# REF [function] >> SimpleWordDataset_test() in text_data_test.py
def generate_simple_word_data(image_height, image_width, image_channel, max_word_len, batch_size, data_dir_path, image_dir_path):
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset, wordset = tg_util.construct_charset(space=False), tg_util.construct_word_set(korean=True, english=True)
	font_list = construct_font(korean=True, english=False)

	#num_train_examples, num_test_examples = int(1e6), int(1e4)
	num_train_examples, num_test_examples = int(1e5), int(1e3)
	#max_word_len = None  # Use max. word length.
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	shuffle = True
	num_workers = 8

	#--------------------
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	label_converter = swl_langproc_util.TokenConverter(list(charset), pad=None)
	#label_converter = swl_langproc_util.TokenConverter(list(charset), sos='<SOS>', eos='<EOS>', pad=None)
	train_dataset = text_data.SimpleWordDataset(label_converter, wordset, num_train_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
	test_dataset = text_data.SimpleWordDataset(label_converter, wordset, num_test_examples, image_channel, max_word_len, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))
	print('#classes = {}.'.format(label_converter.num_tokens))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Train label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Test label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	#--------------------
	# Visualize.
	#visualize_data_with_length(train_dataloader, label_converter, num_data=10)
	#visualize_data_with_length(test_dataloader, label_converter, num_data=10)

	# Save data and images.
	print('Start saving data and images for OpenNMT...')
	start_time = time.time()
	save_data_and_images(True, train_dataloader, data_dir_path, image_dir_path, label_converter)
	save_data_and_images(False, test_dataloader, data_dir_path, image_dir_path, label_converter)
	print('End saving data and images for OpenNMT: {} secs.'.format(time.time() - start_time))

# REF [function] >> RandomWordDataset_test() in text_data_test.py
def generate_random_word_data(image_height, image_width, image_channel, max_word_len, batch_size, data_dir_path, image_dir_path):
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset = tg_util.construct_charset(space=False)
	font_list = construct_font(korean=True, english=False)

	#num_train_examples, num_test_examples = int(1e6), int(1e4)
	num_train_examples, num_test_examples = int(1e5), int(1e3)
	#max_word_len = None  # Use max. word length.
	char_len_interval = (1, 20)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	shuffle = True
	num_workers = 8

	#--------------------
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	label_converter = swl_langproc_util.TokenConverter(list(charset), pad=None)
	#label_converter = swl_langproc_util.TokenConverter(list(charset), sos='<SOS>', eos='<EOS>', pad=None)
	chars = charset  # Can make the number of each character different.
	train_dataset = text_data.RandomWordDataset(label_converter, chars, num_train_examples, image_channel, max_word_len, char_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, target_transform=train_target_transform)
	test_dataset = text_data.RandomWordDataset(label_converter, chars, num_test_examples, image_channel, max_word_len, char_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))
	print('#classes = {}.'.format(label_converter.num_tokens))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Train label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Test label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	#--------------------
	# Visualize.
	#visualize_data_with_length(train_dataloader, label_converter, num_data=10)
	#visualize_data_with_length(test_dataloader, label_converter, num_data=10)

	# Save data and images.
	print('Start saving data and images for OpenNMT...')
	start_time = time.time()
	save_data_and_images(True, train_dataloader, data_dir_path, image_dir_path, label_converter)
	save_data_and_images(False, test_dataloader, data_dir_path, image_dir_path, label_converter)
	print('End saving data and images for OpenNMT: {} secs.'.format(time.time() - start_time))

# REF [function] >> FileBasedWordDataset_test() in text_data_test.py
def generate_file_based_word_data(image_height, image_width, image_channel, max_word_len, batch_size, data_dir_path, image_dir_path):
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset = tg_util.construct_charset(space=False)

	#max_word_len = 30
	train_test_ratio = 0.8
	shuffle = True
	num_workers = 8

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	label_converter = swl_langproc_util.TokenConverter(list(charset), pad=None)
	#label_converter = swl_langproc_util.TokenConverter(list(charset), sos='<SOS>', eos='<EOS>', pad=None)

	datasets = []
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
	print('#classes = {}.'.format(label_converter.num_tokens))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Train label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Test label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	#--------------------
	# Visualize.
	#visualize_data_with_length(train_dataloader, label_converter, num_data=10)
	#visualize_data_with_length(test_dataloader, label_converter, num_data=10)

	# Save data and images.
	print('Start saving data and images for OpenNMT...')
	start_time = time.time()
	save_data_and_images(True, train_dataloader, data_dir_path, image_dir_path, label_converter)
	save_data_and_images(False, test_dataloader, data_dir_path, image_dir_path, label_converter)
	print('End saving data and images for OpenNMT: {} secs.'.format(time.time() - start_time))

# REF [function] >> SimpleTextLineDataset_test() in text_data_test.py
def generate_simple_text_line_data(image_height, image_width, image_channel, max_textline_len, batch_size, data_dir_path, image_dir_path):
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset, wordset = tg_util.construct_charset(space=True), tg_util.construct_word_set(korean=True, english=True)
	font_list = construct_font(korean=True, english=False)

	#num_train_examples, num_test_examples = int(1e6), int(1e4)
	num_train_examples, num_test_examples = int(1e5), int(1e3)
	#max_textline_len = 80
	font_size_interval = (10, 100)
	char_space_ratio_interval = (0.8, 1.25)
	word_count_interval = (1, 5)
	space_count_interval = (1, 3)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	shuffle = True
	num_workers = 8

	#--------------------
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_text_line_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	label_converter = swl_langproc_util.TokenConverter(list(charset), pad=None)
	#label_converter = swl_langproc_util.TokenConverter(list(charset),  sos='<SOS>', eos='<EOS>', pad=None)
	train_dataset = text_data.SimpleTextLineDataset(label_converter, wordset, num_train_examples, image_height, image_width, image_channel, max_textline_len, font_list, font_size_interval, char_space_ratio_interval, word_count_interval, space_count_interval, color_functor, transform=train_transform, target_transform=train_target_transform)
	test_dataset = text_data.SimpleTextLineDataset(label_converter, wordset, num_test_examples, image_height, image_width, image_channel, max_textline_len, font_list, font_size_interval, char_space_ratio_interval, word_count_interval, space_count_interval, color_functor, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))
	print('#classes = {}.'.format(label_converter.num_tokens))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Train label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Test label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	#--------------------
	# Visualize.
	#visualize_data_with_length(train_dataloader, label_converter, num_data=10)
	#visualize_data_with_length(test_dataloader, label_converter, num_data=10)

	# Save data and images.
	print('Start saving data and images for OpenNMT...')
	start_time = time.time()
	save_data_and_images(True, train_dataloader, data_dir_path, image_dir_path, label_converter)
	save_data_and_images(False, test_dataloader, data_dir_path, image_dir_path, label_converter)
	print('End saving data and images for OpenNMT: {} secs.'.format(time.time() - start_time))

#--------------------------------------------------------------------

def parse_command_line_options():
	parser = argparse.ArgumentParser(description='Generate data for OpenNMT.')

	parser.add_argument(
		'-d',
		'--data_type',
		type=str,
		help="The data type to generate. {'simple_word', 'random_word', 'file_based_word', 'simple_text_line'}.",
		required=True,
		default='simple_word'
	)
	parser.add_argument(
		'-dd',
		'--data_dir',
		type=str,
		help='The directory to save data',
		required=True,
		default='openmnt_data'
	)
	parser.add_argument(
		'-id',
		'--image_dir',
		type=str,
		help='The directory to save images',
		required=True,
		default='openmnt_images'
	)
	parser.add_argument(
		'-i',
		'--image_shape',
		type=str,
		help="The image shape to generate. 'height x width x channel'. e.g.) '64x640x3'",
		required=True,
		default='64x640x3'
	)
	parser.add_argument(
		'-t',
		'--max_text_len',
		type=int,
		help='Max. length of words or text lines',
		default=50
	)
	parser.add_argument(
		'-b',
		'--batch_size',
		type=int,
		help='Batch size',
		default=64
	)

	return parser.parse_args()

#--------------------------------------------------------------------

def main():
	args = parse_command_line_options()

	image_shape = [int(v) for v in args.image_shape.split('x')]
	assert len(image_shape) == 3

	os.makedirs(args.data_dir, exist_ok=True)
	os.makedirs(args.image_dir, exist_ok=True)

	if args.data_type == 'simple_word':
		generate_simple_word_data(*image_shape, args.max_text_len, args.batch_size, args.data_dir, args.image_dir)
	elif args.data_type == 'random_word':
		generate_random_word_data(*image_shape, args.max_text_len, args.batch_size, args.data_dir, args.image_dir)
	elif args.data_type == 'file_based_word':
		generate_file_based_word_data(*image_shape, args.max_text_len, args.batch_size, args.data_dir, args.image_dir)
	elif args.data_type == 'simple_text_line':
		generate_simple_text_line_data(*image_shape, args.max_text_len, args.batch_size, args.data_dir, args.image_dir)
	else:
		ValueError('Invalid data type, {}'.format(args.data_type))

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
