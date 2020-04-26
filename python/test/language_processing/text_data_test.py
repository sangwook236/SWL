#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, functools, glob, time
import numpy as np
import torch, torchvision
from PIL import Image, ImageOps
import cv2
import text_data
import text_generation_util as tg_util

def create_augmenter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	augmenter = iaa.Sequential([
		#iaa.Sometimes(0.5, iaa.OneOf([
		#	iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
		#	iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
		#	#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
		#	#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
		#])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (0.0, 0.1), 'y': (-0.05, 0.05)},  # Translate by 0 to +10 percent along x-axis and -5 to +5 percent along y-axis.
				rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				shear=(-10, 10),  # Shear by -10 to +10 degrees.
				#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.OneOf([
				iaa.GaussianBlur(sigma=(0.5, 1.5)),
				iaa.AverageBlur(k=(2, 4)),
				iaa.MedianBlur(k=(3, 3)),
				iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0), order=1),
			]),
			iaa.Sequential([
				iaa.OneOf([
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
					iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
				]),
				iaa.GaussianBlur(sigma=(0.7, 1.0)),
			]),
			#iaa.OneOf([
			#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
			#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
			#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

			#	iaa.Invert(p=1, per_channel=False),

			#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
			#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
			#]),
		])),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

	return augmenter

def SingleCharacterDataset_test():
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
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
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

	#--------------------
	image_height, image_width = 32, 32
	#image_height_before_crop, image_width_before_crop = 36, 36
	image_height_before_crop, image_width_before_crop = image_height, image_width

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	font_size_interval = (10, 100)

	batch_size = 64
	shuffle = True
	num_workers = 4

	#--------------------
	class RandomAugment(object):
		def __init__(self):
			self.augmenter = create_augmenter()

		def __call__(self, x):
			return Image.fromarray(self.augmenter.augment_images(np.array(x)))
	class RandomInvert(object):
		def __call__(self, x):
			return ImageOps.invert(x) if random.randrange(2) else x
	class ConvertChannel(object):
		def __call__(self, x):
			return x.convert('RGB')
			#return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
			#return torch.repeat_interleave(x, 3, dim=0)
			#return torch.repeat_interleave(torch.unsqueeze(x, dim=3), 3, dim=0)

	train_transform = torchvision.transforms.Compose([
		RandomAugment(),
		RandomInvert(),
		#ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, transform=train_transform)
	test_dataset = text_data.SingleCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

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
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	for dataloader in [train_dataloader, test_dataloader]:
		data_iter = iter(dataloader)
		images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
		images, labels = images.numpy(), labels.numpy()
		for idx, (img, lbl) in enumerate(zip(images, labels)):
			print('Label: (int) = {}, (str) = {}.'.format(lbl, charset[lbl]))
			cv2.imshow('Image', img[0])
			cv2.waitKey(0)
			if idx >= 9: break
	cv2.destroyAllWindows()

def SingleWordDataset_test():
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
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
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

	#--------------------
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

	korean_word_set = set(korean_words)
	english_word_set = set(english_words)
	all_word_set = set(korean_words + english_words)

	word_set = all_word_set

	#--------------------
	image_height, image_width = 32, 320
	#image_height_before_crop, image_width_before_crop = 36, 324
	image_height_before_crop, image_width_before_crop = image_height, image_width

	num_train_examples, num_test_examples = int(1e6), int(1e4)
	font_size_interval = (10, 100)

	batch_size = 64
	shuffle = True
	num_workers = 4

	#--------------------
	class RandomAugment(object):
		def __init__(self):
			self.augmenter = create_augmenter()

		def __call__(self, x):
			return Image.fromarray(self.augmenter.augment_images(np.array(x)))
	class RandomInvert(object):
		def __call__(self, x):
			return ImageOps.invert(x) if random.randrange(2) else x
	class ConvertChannel(object):
		def __call__(self, x):
			return x.convert('RGB')
			#return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
			#return torch.repeat_interleave(x, 3, dim=0)
			#return torch.repeat_interleave(torch.unsqueeze(x, dim=3), 3, dim=0)
	class ToIntTensor(object):
		def __call__(self, lst):
			return torch.IntTensor(lst)

	train_transform = torchvision.transforms.Compose([
		RandomAugment(),
		RandomInvert(),
		#ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleWordDataset(num_train_examples, word_set, charset, font_list, font_size_interval, transform=train_transform, target_transform=train_target_transform, default_value=-1)
	test_dataset = text_data.SingleWordDataset(num_test_examples, word_set, charset, font_list, font_size_interval, transform=test_transform, target_transform=test_target_transform, default_value=-1)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

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
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	for dataloader in [train_dataloader, test_dataloader]:
		data_iter = iter(dataloader)
		images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
		images, labels = images.numpy(), labels.numpy()
		for idx, (img, lbl) in enumerate(zip(images, labels)):
			print('Label: (int) = {}, (str) = {}.'.format([ll for ll in lbl if ll != dataloader.dataset.default_value], train_dataset.decode_label(lbl)))
			cv2.imshow('Image', img[0])
			cv2.waitKey(0)
			if idx >= 9: break
	cv2.destroyAllWindows()

def generate_font_colors(image_depth):
	import random
	#font_color = (255,) * image_depth  # White font color.
	#font_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB font color.
	#font_color = (random.randrange(256),) * image_depth  # A grayscale font color.
	#gray_val = random.randrange(255)
	#font_color = (gray_val,) * image_depth  # A lighter grayscale font color.
	font_color = (random.randrange(128, 256),) * image_depth  # A light grayscale font color.
	#bg_color = (0,) * image_depth  # Black background color.
	#bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB background color.
	#bg_color = (random.randrange(256),) * image_depth  # A grayscale background color.
	#bg_color = (random.randrange(gray_val + 1, 256),) * image_depth  # A darker grayscale background color.
	bg_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale background color.
	return font_color, bg_color

def SingleTextLineDataset_test():
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
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
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

	#--------------------
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

	korean_word_set = set(korean_words)
	english_word_set = set(english_words)
	all_word_set = set(korean_words + english_words)

	word_set = all_word_set

	#--------------------
	image_height, image_width, image_channel = 64, 640, 1
	#image_height_before_crop, image_width_before_crop = 68, 644
	image_height_before_crop, image_width_before_crop = image_height, image_width

	num_train_examples, num_test_examples = int(1e6), int(1e4)
	max_word_len = 80
	word_count_interval = (1, 5)
	space_count_interval = (1, 3)
	font_size_interval = (10, 100)
	char_space_ratio_interval = (0.8, 1.25)
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	batch_size = 64
	shuffle = True
	num_workers = 4

	#--------------------
	class RandomAugment(object):
		def __init__(self):
			self.augmenter = create_augmenter()

		def __call__(self, x):
			return Image.fromarray(self.augmenter.augment_images(np.array(x)))
	class RandomInvert(object):
		def __call__(self, x):
			return ImageOps.invert(x) if random.randrange(2) else x
	class ConvertChannel(object):
		def __call__(self, x):
			return x.convert('RGB')
			#return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
			#return torch.repeat_interleave(x, 3, dim=0)
			#return torch.repeat_interleave(torch.unsqueeze(x, dim=3), 3, dim=0)
	class ToIntTensor(object):
		def __call__(self, lst):
			return torch.IntTensor(lst)

	train_transform = torchvision.transforms.Compose([
		RandomAugment(),
		RandomInvert(),
		#ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		RandomInvert(),
		#ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleTextLineDataset(num_train_examples, image_height, image_width, image_channel, word_set, charset, font_list, max_word_len, word_count_interval, space_count_interval, font_size_interval, char_space_ratio_interval, color_functor, transform=train_transform, target_transform=train_target_transform, default_value=-1)
	test_dataset = text_data.SingleTextLineDataset(num_test_examples, image_height, image_width, image_channel, word_set, charset, font_list, max_word_len, word_count_interval, space_count_interval, font_size_interval, char_space_ratio_interval, color_functor, transform=test_transform, target_transform=test_target_transform, default_value=-1)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)
	print('#classes = {}.'.format(train_dataset.num_classes))

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
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	for dataloader in [train_dataloader, test_dataloader]:
		data_iter = iter(dataloader)
		images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
		images, labels = images.numpy(), labels.numpy()
		for idx, (img, lbl) in enumerate(zip(images, labels)):
			print('Label: (int) = {}, (str) = {}.'.format([ll for ll in lbl if ll != dataloader.dataset.default_value], train_dataset.decode_label(lbl)))
			cv2.imshow('Image', img[0])
			cv2.waitKey(0)
			if idx >= 9: break
	cv2.destroyAllWindows()

def main():
	#SingleCharacterDataset_test()
	#SingleWordDataset_test()
	SingleTextLineDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()