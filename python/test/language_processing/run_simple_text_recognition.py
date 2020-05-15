#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, functools, glob, time
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import text_data
import text_generation_util as tg_util
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

def construct_charset():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'
	#font_dir_path = font_base_dir_path + '/eng'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	font_list = tg_util.generate_font_list(font_filepaths)

	#--------------------
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A string.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of strings.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	alphabet_charset = string.ascii_uppercase + string.ascii_lowercase
	digit_charset = string.digits
	symbol_charset = string.punctuation
	#symbol_charset = string.punctuation + ' '

	#charset = alphabet_charset + digit_charset + symbol_charset + hangeul_charset + hangeul_jamo_charset
	charset = alphabet_charset + digit_charset + symbol_charset + hangeul_charset
	return charset, font_list

def construct_word_set():
	korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
	#english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
	english_dictionary_filepath = '../../data/language_processing/wordlist_mono_clean.txt'
	#english_dictionary_filepath = '../../data/language_processing/wordlist_bi_clean.txt'

	print('Start loading a Korean dictionary...')
	start_time = time.time()
	with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#korean_words = fd.readlines()
		#korean_words = fd.read().strip('\n')
		korean_words = fd.read().splitlines()
	print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('Start loading an English dictionary...')
	start_time = time.time()
	with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#english_words = fd.readlines()
		#english_words = fd.read().strip('\n')
		english_words = fd.read().splitlines()
	print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))

	#return set(korean_words)
	#return set(english_words)
	return set(korean_words + english_words)

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
	#font_color = (255,) * image_depth  # White font color.
	font_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB font color.
	#font_color = (random.randrange(256),) * image_depth  # A grayscale font color.
	#gray_val = random.randrange(255)
	#font_color = (gray_val,) * image_depth  # A lighter grayscale font color.
	#font_color = (random.randrange(gray_val, 256),) * image_depth  # A darker grayscale font color.
	#font_color = (random.randrange(128, 256),) * image_depth  # A light grayscale font color.
	#font_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale font color.
	#bg_color = (0,) * image_depth  # Black background color.
	bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB background color.
	#bg_color = (random.randrange(256),) * image_depth  # A grayscale background color.
	#bg_color = (random.randrange(gray_val, 256),) * image_depth  # A lighter grayscale background color.
	#bg_color = (gray_val,) * image_depth  # A darker grayscale background color.
	#bg_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale background color.
	#bg_color = (random.randrange(128, 256),) * image_depth  # A light grayscale background color.
	return font_color, bg_color

class RandomAugment(object):
	def __init__(self, augmenter):
		self.augmenter = augmenter

	def __call__(self, x):
		return Image.fromarray(self.augmenter.augment_image(np.array(x)))
		#return Image.fromarray(self.augmenter.augment_images(np.array(x)))

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
		interpolation = Image.BICUBIC
		wi, hi = input.size
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
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
		x, y = self.subset[idx]
		if self.transform:
			x = self.transform(x)
		if self.target_transform:
			y = self.target_transform(y)
		return x, y

	def __len__(self):
		return len(self.subset)

	@property
	def num_classes(self):
		return self.subset.dataset.num_classes

	@property
	def classes(self):
		return self.subset.dataset.classes

	@property
	def default_value(self):
		return self.subset.dataset.default_value

class MyConcatDataset(torch.utils.data.ConcatDataset):
	def __init__(self, datasets):
		super().__init__(datasets)

		self._classes = datasets[0].classes
		for dataset in datasets[1:]:
			assert set(self._classes) == set(dataset.classes), '{}, {}'.format(set(self._classes).difference(dataset.classes), set(dataset.classes).difference(self._classes))

	@property
	def num_classes(self):
		return len(self._classes)

	@property
	def classes(self):
		return self._classes

def create_char_data_loaders(charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_clipping_ratio_interval, color_functor, batch_size, shuffle, num_workers):
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	if True:
		# REF [function] >> generate_chars_from_chars74k_data() in chars74k_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/chars74k/English/Img/char_images.txt'
		is_image_used = True
	elif False:
		# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_kr.txt'
		is_image_used = True
	elif False:
		# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_en.txt'
		is_image_used = True
	elif False:
		# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_kr.txt'
		is_image_used = True
	elif False:
		# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_en.txt'
		is_image_used = True

	#--------------------
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_char_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	print('Start creating datasets...')
	start_time = time.time()
	if False:
		chars = list(charset * num_train_examples_per_class)
		random.shuffle(chars)
		train_dataset = text_data.SimpleCharacterDataset(chars, charset, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=train_transform)
		chars = list(charset * num_test_examples_per_class)
		random.shuffle(chars)
		test_dataset = text_data.SimpleCharacterDataset(chars, charset, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=test_transform)
	elif False:
		chars = list(charset * num_train_examples_per_class)
		random.shuffle(chars)
		train_dataset = text_data.NoisyCharacterDataset(chars, charset, image_channel, font_list, font_size_interval, char_clipping_ratio_interval, color_functor=color_functor, transform=train_transform)
		chars = list(charset * num_test_examples_per_class)
		random.shuffle(chars)
		test_dataset = text_data.NoisyCharacterDataset(chars, charset, image_channel, font_list, font_size_interval, char_clipping_ratio_interval, color_functor=color_functor, transform=test_transform)
	else:
		dataset = text_data.FileBasedCharacterDataset(image_label_info_filepath, charset, image_channel, is_image_used=is_image_used)
		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)

		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	classes = train_dataset.classes
	num_classes = train_dataset.num_classes

	return train_dataloader, test_dataloader, classes, num_classes

def create_mixed_char_data_loaders(charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_clipping_ratio_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_char_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	print('Start creating datasets...')
	start_time = time.time()
	datasets = []
	chars = list(charset * num_simple_char_examples_per_class)
	random.shuffle(chars)
	datasets.append(text_data.SimpleCharacterDataset(chars, charset, image_channel, font_list, font_size_interval, color_functor=color_functor))
	chars = list(charset * num_noisy_examples_per_class)
	random.shuffle(chars)
	datasets.append(text_data.NoisyCharacterDataset(chars, charset, image_channel, font_list, font_size_interval, char_clipping_ratio_interval, color_functor=color_functor))
	# REF [function] >> generate_chars_from_chars74k_data() in chars74k_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/chars74k/English/Img/char_images.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedCharacterDataset(image_label_info_filepath, charset, image_channel, is_image_used=is_image_used))
	# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_kr.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedCharacterDataset(image_label_info_filepath, charset, image_channel, is_image_used=is_image_used))
	# REF [function] >> generate_chars_from_e2e_mlt_data() in e2e_mlt_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/char_images_en.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedCharacterDataset(image_label_info_filepath, charset, image_channel, is_image_used=is_image_used))
	# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_kr.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedCharacterDataset(image_label_info_filepath, charset, image_channel, is_image_used=is_image_used))
	# REF [function] >> generate_chars_from_rrc_mlt_2019_data() in icdar_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/char_images_en.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedCharacterDataset(image_label_info_filepath, charset, image_channel, is_image_used=is_image_used))

	dataset = MyConcatDataset(datasets)
	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	classes = train_dataset.classes
	num_classes = train_dataset.num_classes

	return train_dataloader, test_dataloader, classes, num_classes

def create_word_data_loaders(wordset, charset, num_train_examples, num_test_examples, train_test_ratio, max_word_len, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_len_interval, color_functor, batch_size, shuffle, num_workers):
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	if True:
		# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_kr.txt'
		is_image_used = False
	elif False:
		# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_en.txt'
		is_image_used = False
	elif False:
		# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_kr.txt'
		is_image_used = True
	elif False:
		# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
		image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_en.txt'
		is_image_used = True

	#--------------------
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	print('Start creating datasets...')
	start_time = time.time()
	if True:
		train_dataset = text_data.SimpleWordDataset(wordset, charset, num_train_examples, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, default_value=-1)
		#train_dataset = text_data.SimpleWordDataset(wordset, charset, num_train_examples, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, label_prefix=[text_data.SimpleWordDataset.SOS], label_suffix=[text_data.SimpleWordDataset.EOS], default_value=-1)
		test_dataset = text_data.SimpleWordDataset(wordset, charset, num_test_examples, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, default_value=-1)
		#test_dataset = text_data.SimpleWordDataset(wordset, charset, num_test_examples, image_channel, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, label_prefix=[text_data.SimpleWordDataset.SOS], label_suffix=[text_data.SimpleWordDataset.EOS], default_value=-1)
	elif False:
		chars = charset  # Can make the number of each character different.
		train_dataset = text_data.RandomWordDataset(chars, num_train_examples, image_channel, char_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, default_value=-1)
		#train_dataset = text_data.RandomWordDataset(chars, num_train_examples, image_channel, char_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=train_transform, label_prefix=[text_data.RandomWordDataset.SOS], label_suffix=[text_data.RandomWordDataset.EOS], default_value=-1)
		test_dataset = text_data.RandomWordDataset(chars, num_train_examples, image_channel, char_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, default_value=-1)
		#test_dataset = text_data.RandomWordDataset(chars, num_train_examples, image_channel, char_len_interval, font_list, font_size_interval, color_functor=color_functor, transform=test_transform, label_prefix=[text_data.RandomWordDataset.SOS], label_suffix=[text_data.RandomWordDataset.EOS], default_value=-1)
	else:
		dataset = text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, default_value=-1)
		#dataset = text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, label_prefix=[text_data.FileBasedWordDataset.SOS], label_suffix=[text_data.FileBasedWordDataset.EOS], default_value=-1)
		num_examples = len(dataset)
		num_train_examples = int(num_examples * train_test_ratio)

		train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
		train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
		test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	classes = train_dataset.classes
	num_classes = train_dataset.num_classes

	return train_dataloader, test_dataloader, classes, num_classes

def create_mixed_word_data_loaders(wordset, charset, num_simple_examples, num_random_examples, train_test_ratio, max_word_len, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_len_interval, color_functor, batch_size, shuffle, num_workers):
	# Load and normalize datasets.
	train_transform = torchvision.transforms.Compose([
		RandomAugment(create_word_augmenter()),
		RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertPILMode(mode='RGB'),
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	print('Start creating datasets...')
	start_time = time.time()
	datasets = []
	datasets.append(text_data.SimpleWordDataset(wordset, charset, num_simple_examples, image_channel, font_list, font_size_interval, color_functor=color_functor, default_value=-1))
	#datasets.append(text_data.SimpleWordDataset(wordset, charset, num_simple_examples, image_channel, font_list, font_size_interval, color_functor=color_functor, label_prefix=[text_data.SimpleWordDataset.SOS], label_suffix=[text_data.SimpleWordDataset.EOS], default_value=-1))
	chars = charset  # Can make the number of each character different.
	datasets.append(text_data.RandomWordDataset(chars, num_random_examples, image_channel, char_len_interval, font_list, font_size_interval, color_functor=color_functor, default_value=-1))
	#datasets.append(text_data.RandomWordDataset(chars, num_random_examples, image_channel, char_len_interval, font_list, font_size_interval, color_functor=color_functor, label_prefix=[text_data.RandomWordDataset.SOS], label_suffix=[text_data.RandomWordDataset.EOS], default_value=-1))
	# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_kr.txt'
	is_image_used = False
	datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, default_value=-1))
	#datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, label_prefix=[text_data.FileBasedWordDataset.SOS], label_suffix=[text_data.FileBasedWordDataset.EOS], default_value=-1))
	# REF [function] >> generate_words_from_e2e_mlt_data() in e2e_mlt_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/e2e_mlt/word_images_en.txt'
	is_image_used = False
	datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, default_value=-1))
	#datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, label_prefix=[text_data.FileBasedWordDataset.SOS], label_suffix=[text_data.FileBasedWordDataset.EOS], default_value=-1))
	# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_kr.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, default_value=-1))
	#datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, label_prefix=[text_data.FileBasedWordDataset.SOS], label_suffix=[text_data.FileBasedWordDataset.EOS], default_value=-1))
	# REF [function] >> generate_words_from_rrc_mlt_2019_data() in icdar_data_test.py
	image_label_info_filepath = data_base_dir_path + '/text/icdar_mlt_2019/word_images_en.txt'
	is_image_used = True
	datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, default_value=-1))
	#datasets.append(text_data.FileBasedWordDataset(image_label_info_filepath, charset, image_channel, max_word_len, is_image_used=is_image_used, label_prefix=[text_data.FileBasedWordDataset.SOS], label_suffix=[text_data.FileBasedWordDataset.EOS], default_value=-1))

	dataset = MyConcatDataset(datasets)
	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	classes = train_dataset.classes
	num_classes = train_dataset.num_classes

	return train_dataloader, test_dataloader, classes, num_classes

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_character():
	image_height, image_width, image_channel = 64, 64, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset, font_list = construct_charset()

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	num_simple_char_examples_per_class, num_noisy_examples_per_class = 100, 100
	font_size_interval = (10, 100)
	char_clipping_ratio_interval = (0.8, 1.25)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	train_test_ratio = 0.8
	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4
	log_print_freq = 1000

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './single_char_recognition.pth'

	#--------------------
	#train_dataloader, test_dataloader, classes, num_classes = create_char_data_loaders(charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_clipping_ratio_interval, color_functor, batch_size, shuffle, num_workers)
	train_dataloader, test_dataloader, classes, num_classes = create_mixed_char_data_loaders(charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_clipping_ratio_interval, color_functor, batch_size, shuffle, num_workers)

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(train_dataloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a convolutional neural network.

	if False:
		model = torchvision.models.vgg19(pretrained=False, num_classes=num_classes)
		#model = torchvision.models.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		#model = torchvision.models.vgg19(pretrained=True)
		model = torchvision.models.vgg19_bn(pretrained=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
	else:
		model = torchvision.models.resnet18(pretrained=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	if True:
		model.train()
		for epoch in range(num_epochs):  # Loop over the dataset multiple times.
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(train_dataloader, 0):
				# Get the inputs.
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if i % log_print_freq == (log_print_freq - 1):
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_print_freq))
					running_loss = 0.0
			#scheduler.step()

			# Save a checkpoint.
			save_model(model_filepath, model)

		print('Finished training.')

		# Save a model.
		save_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test images: {} %%.'.format(100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	#for i in range(num_classes):
	#	print('Accuracy of %5s : %2d %%.' % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
	accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	print('Accuracy frequency: {}.'.format(hist))
	valid_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] > 0]
	print('Accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
	accuracy_threshold = 98
	for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]):
		if acc < accuracy_threshold:
			print('\tChar = {}: accuracy = {}.'.format(classes[idx], acc))

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_character_using_mixup():
	image_height, image_width, image_channel = 64, 64, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset, font_list = construct_charset()

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	num_simple_char_examples_per_class, num_noisy_examples_per_class = 100, 100
	font_size_interval = (10, 100)
	char_clipping_ratio_interval = (0.8, 1.25)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	train_test_ratio = 0.8
	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4
	log_print_freq = 1000

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './single_char_recognition_mixup.pth'

	#--------------------
	#train_dataloader, test_dataloader, classes, num_classes = create_char_data_loaders(charset, num_train_examples_per_class, num_test_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_clipping_ratio_interval, color_functor, batch_size, shuffle, num_workers)
	train_dataloader, test_dataloader, classes, num_classes = create_mixed_char_data_loaders(charset, num_simple_char_examples_per_class, num_noisy_examples_per_class, train_test_ratio, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_clipping_ratio_interval, color_functor, batch_size, shuffle, num_workers)

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(train_dataloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a convolutional neural network.

	if False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/vgg.py
		import mixup.vgg
		# NOTE [info] >> Hard to train.
		model = mixup.vgg.vgg19(pretrained=False, num_classes=num_classes)
		#model = mixup.vgg.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/vgg.py
		import mixup.vgg
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = mixup.vgg.vgg19(pretrained=True, progress=True)
		model = mixup.vgg.vgg19_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/resnet.py
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=False, num_classes=num_classes)
	else:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/resnet.py
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	if True:
		model.train()
		for epoch in range(num_epochs):  # Loop over the dataset multiple times.
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(train_dataloader, 0):
				# Get the inputs.
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				outputs, labels = model(inputs, labels, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
				loss = criterion(outputs, torch.argmax(labels, dim=1))
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if i % log_print_freq == (log_print_freq - 1):
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_print_freq))
					running_loss = 0.0
			#scheduler.step()

			# Save a checkpoint.
			save_model(model_filepath, model)

		print('Finished training.')

		# Save a model.
		save_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test images: {} %%.'.format(100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	#for i in range(num_classes):
	#	print('Accuracy of %5s : %2d %%.' % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
	accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	print('Accuracy frequency: {}.'.format(hist))
	valid_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] > 0]
	print('Accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
	accuracy_threshold = 98
	for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]):
		if acc < accuracy_threshold:
			print('\tChar = {}: accuracy = {}.'.format(classes[idx], acc))

def recognize_word():
	image_height, image_width, image_channel = 64, 640, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset, font_list = construct_charset()
	wordset = construct_word_set()

	max_word_len = 30
	num_train_examples, num_test_examples = int(1e6), int(1e4)
	num_simple_examples, num_random_examples = int(1e4), int(1e4)
	char_len_interval = (1, 20)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	train_test_ratio = 0.8
	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4
	stem = False
	log_print_freq = 1000

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './single_word_recognition.pth'

	#--------------------
	#train_dataloader, test_dataloader, classes, num_classes = create_word_data_loaders(wordset, charset, num_train_examples, num_test_examples, train_test_ratio, max_word_len, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_len_interval, color_functor, batch_size, shuffle, num_workers)
	train_dataloader, test_dataloader, classes, num_classes = create_mixed_word_data_loaders(wordset, charset, num_simple_examples, num_random_examples, train_test_ratio, max_word_len, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_len_interval, color_functor, batch_size, shuffle, num_workers)

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(train_dataloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a model.

	import sasa.model
	model = sasa.model.ResNet26(num_classes=num_classes, stem=stem)
	#model = sasa.model.ResNet38(num_classes=num_classes, stem=stem)
	#model = sasa.model.ResNet50(num_classes=num_classes, stem=stem)

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	if True:
		model.train()
		for epoch in range(num_epochs):  # Loop over the dataset multiple times.
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(train_dataloader, 0):
				# Get the inputs.
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if i % log_print_freq == (log_print_freq - 1):
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_print_freq))
					running_loss = 0.0
			#scheduler.step()

			# Save a checkpoint.
			save_model(model_filepath, model)

		print('Finished training.')

		# Save a model.
		save_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test images: {} %%.'.format(100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	#for i in range(num_classes):
	#	print('Accuracy of %5s : %2d %%.' % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
	accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	print('Accuracy frequency: {}.'.format(hist))
	valid_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] > 0]
	print('Accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
	accuracy_threshold = 98
	for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]):
		if acc < accuracy_threshold:
			print('\tChar = {}: accuracy = {}.'.format(classes[idx], acc))

def recognize_word_using_mixup():
	image_height, image_width, image_channel = 64, 640, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	charset, font_list = construct_charset()
	wordset = construct_word_set()

	max_word_len = 30
	num_train_examples, num_test_examples = int(1e6), int(1e4)
	num_simple_examples, num_random_examples = int(1e4), int(1e4)
	char_len_interval = (1, 20)
	font_size_interval = (10, 100)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	train_test_ratio = 0.8
	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4
	stem = False
	log_print_freq = 1000

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './single_word_recognition_mixup.pth'

	#--------------------
	#train_dataloader, test_dataloader, classes, num_classes = create_word_data_loaders(wordset, charset, num_train_examples, num_test_examples, train_test_ratio, max_word_len, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_len_interval, color_functor, batch_size, shuffle, num_workers)
	train_dataloader, test_dataloader, classes, num_classes = create_mixed_word_data_loaders(wordset, charset, num_simple_examples, num_random_examples, train_test_ratio, max_word_len, image_height, image_width, image_channel, image_height_before_crop, image_width_before_crop, font_list, font_size_interval, char_len_interval, color_functor, batch_size, shuffle, num_workers)

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(train_dataloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a model.

	import sasa.model_mixup
	model = sasa.model_mixup.ResNet26(num_classes=num_classes, stem=stem)
	#model = sasa.model_mixup.ResNet38(num_classes=num_classes, stem=stem)
	#model = sasa.model_mixup.ResNet50(num_classes=num_classes, stem=stem)

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	if True:
		model.train()
		for epoch in range(num_epochs):  # Loop over the dataset multiple times.
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(train_dataloader, 0):
				# Get the inputs.
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				outputs, labels = model(inputs, labels, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
				loss = criterion(outputs, torch.argmax(labels, dim=1))
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if i % log_print_freq == (log_print_freq - 1):
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_print_freq))
					running_loss = 0.0
			#scheduler.step()

			# Save a checkpoint.
			save_model(model_filepath, model)

		print('Finished training.')

		# Save a model.
		save_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test images: {} %%.'.format(100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	#for i in range(num_classes):
	#	print('Accuracy of %5s : %2d %%.' % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
	accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	print('Accuracy frequency: {}.'.format(hist))
	valid_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] > 0]
	print('Accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
	accuracy_threshold = 98
	for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]):
		if acc < accuracy_threshold:
			print('\tChar = {}: accuracy = {}.'.format(classes[idx], acc))

def recognize_text_using_craft_and_character_recognizer():
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils
	import craft.test_utils as test_utils

	image_height, image_width = 64, 64

	charset, _ = construct_charset()

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	#model_filepath = './craft/single_char_recognition.pth'
	model_filepath = './craft/single_char_recognition_mixup.pth'

	#classes = charset
	classes = list(charset) + [text_data.TextDatasetBase.UNKNOWN]
	num_classes = len(classes)

	#--------------------
	# Define a convolutional neural network.

	if False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/vgg.py
		import mixup.vgg
		# NOTE [info] >> Hard to train.
		model = mixup.vgg.vgg19(pretrained=False, num_classes=num_classes)
		#model = mixup.vgg.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/vgg.py
		import mixup.vgg
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = mixup.vgg.vgg19(pretrained=True, progress=True)
		model = mixup.vgg.vgg19_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/resnet.py
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=False, num_classes=num_classes)
	else:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/resnet.py
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	#image_filepath = './craft/images/I3.jpg'
	image_filepath = './craft/images/book_1.png'
	#image_filepath = './craft/images/book_2.png'

	print('Start loading CRAFT...')
	start_time = time.time()
	trained_model = './craft/craft_mlt_25k.pth'
	refiner_model = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.
	refine = False  # Enable link refiner.
	cuda = True  # Use cuda for inference.
	net, refine_net = test_utils.load_craft(trained_model, refiner_model, refine, cuda)
	print('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	print('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, ch_bboxes_lst, score_text = test_utils.run_char_craft(rgb, net, refine_net, cuda)
	print('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
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

		os.makedirs('./char_recog_results', exist_ok=True)

		print('Start inferring...')
		start_time = time.time()
		ch_images = []
		rgb = image.copy()
		for i, ch_bboxes in enumerate(ch_bboxes_lst):
			imgs = []
			color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
			for j, bbox in enumerate(ch_bboxes):
				(x1, y1), (x2, y2) = np.min(bbox, axis=0), np.max(bbox, axis=0)
				x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
				img = image[y1:y2+1,x1:x2+1]
				imgs.append(img)

				cv2.imwrite('./char_recog_results/ch_{}_{}.png'.format(i, j), img)

				cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)
			ch_images.append(imgs)
		cv2.imwrite('./char_recog_results/char_bbox.png', rgb)

		#--------------------
		transform = torchvision.transforms.Compose([
			#RandomInvert(),
			ConvertPILMode(mode='RGB'),
			ResizeImage(image_height, image_width),
			#torchvision.transforms.Resize((image_height, image_width)),
			#torchvision.transforms.CenterCrop((image_height, image_width)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		with torch.no_grad():
			for idx, imgs in enumerate(ch_images):
				imgs = torch.stack([transform(Image.fromarray(img)) for img in imgs]).to(device)
				
				outputs = model(imgs)
				_, predictions = torch.max(outputs, 1)
				predictions = predictions.cpu().numpy()
				print('\t{}: {} (int), {} (str).'.format(idx, predictions, ''.join([classes[id] for id in predictions])))
		print('End inferring: {} secs.'.format(time.time() - start_time))
	else:
		print('No text detected.')

def recognize_text_using_craft_and_word_recognizer():
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils
	import craft.test_utils as test_utils

	image_height, image_width = 64, 64

	charset, _ = construct_charset()

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	#model_filepath = './craft/single_word_recognition.pth'
	model_filepath = './craft/single_word_recognition_mixup.pth'

	#classes = charset
	classes = list(charset) + [text_data.TextDatasetBase.UNKNOWN]
	num_classes = len(classes)

	#--------------------
	# Define a convolutional neural network.

	if False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/vgg.py
		import mixup.vgg
		# NOTE [info] >> Hard to train.
		model = mixup.vgg.vgg19(pretrained=False, num_classes=num_classes)
		#model = mixup.vgg.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/vgg.py
		import mixup.vgg
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = mixup.vgg.vgg19(pretrained=True, progress=True)
		model = mixup.vgg.vgg19_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/resnet.py
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=False, num_classes=num_classes)
	else:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/mixup/resnet.py
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	#image_filepath = './craft/images/I3.jpg'
	image_filepath = './craft/images/book_1.png'
	#image_filepath = './craft/images/book_2.png'

	print('Start loading CRAFT...')
	start_time = time.time()
	trained_model = './craft/craft_mlt_25k.pth'
	refiner_model = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.
	refine = False  # Enable link refiner.
	cuda = True  # Use cuda for inference.
	net, refine_net = test_utils.load_craft(trained_model, refiner_model, refine, cuda)
	print('End loading CRAFT: {} secs.'.format(time.time() - start_time))

	print('Start running CRAFT...')
	start_time = time.time()
	rgb = imgproc.loadImage(image_filepath)  # RGB order.
	bboxes, polys, score_text = test_utils.run_word_craft(rgb, net, refine_net, cuda)
	print('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
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

		os.makedirs('./word_recog_results', exist_ok=True)

		print('Start inferring...')
		start_time = time.time()
		word_images = []
		rgb = image.copy()
		for i, bbox in enumerate(bboxes):
			(x1, y1), (x2, y2) = np.min(bbox, axis=0), np.max(bbox, axis=0)
			x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
			img = image[y1:y2+1,x1:x2+1]
			word_images.append(img)

			cv2.imwrite('./word_recog_results/word_{}.png'.format(i), img)

			cv2.rectangle(rgb, (x1, y1), (x2, y2), (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)), 1, cv2.LINE_4)
		cv2.imwrite('./word_recog_results/word_bbox.png', rgb)

		#--------------------
		transform = torchvision.transforms.Compose([
			#RandomInvert(),
			ConvertPILMode(mode='RGB'),
			ResizeImage(image_height, image_width),
			#torchvision.transforms.Resize((image_height, image_width)),
			#torchvision.transforms.CenterCrop((image_height, image_width)),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		with torch.no_grad():
			imgs = torch.stack([transform(Image.fromarray(img)) for img in word_images]).to(device)

			outputs = model(imgs)
			_, predictions = torch.max(outputs, 1)
			predictions = predictions.cpu().numpy()
			for idx, pred in enumerate(predictions):
				print('\t{}: {} (int), {} (str).'.format(idx, pred, ''.join([classes[id] for id in pred])))
		print('End inferring: {} secs.'.format(time.time() - start_time))
	else:
		print('No text detected.')

def main():
	#recognize_character()
	#recognize_character_using_mixup()

	# Recognize text using CRAFT (scene text detector) + character recognizer.
	#recognize_text_using_craft_and_character_recognizer()

	#--------------------
	recognize_word()  # Use RARE.
	#recognize_word_using_mixup()  # Use RARE.

	# Recognize text using CRAFT (scene text detector) + word recognizer.
	#recognize_text_using_craft_and_word_recognizer()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
