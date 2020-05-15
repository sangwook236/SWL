#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, random, time, datetime, functools, itertools, glob, csv
import numpy as np
import torch, torchvision
import cv2

class TextLineDatasetBase(torch.utils.data.Dataset):
	#SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
	#EOS = '<EOS>'  # All strings will end with the End-Of-String token.
	#SOJC = '<SOJC>'  # All Hangeul jamo strings will start with the Start-Of-Jamo-Character token.
	EOJC = '<EOJC>'  # All Hangeul jamo strings will end with the End-Of-Jamo-Character token.
	UNKNOWN = '<UNK>'  # Unknown label token.

	def __init__(self, labels, num_classes, default_value=-1):
		self.labels, self._num_classes, self._default_value = labels, num_classes, default_value

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def default_value(self):
		return self._default_value

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		def label2index(ch):
			try:
				return self.labels.index(ch)
			except ValueError:
				print('[SWL] Error: Failed to encode a label, {} in {}.'.format(ch, label_str))
				return self.labels.index(TextLineDatasetBase.UNKNOWN)
		return list(label2index(ch) for ch in label_str)

	# Integer label -> string label.
	def decode_label(self, label_int, *args, **kwargs):
		def index2label(idx):
			try:
				return self.labels[idx]
			except IndexError:
				print('[SWL] Error: Failed to decode a label, {} in {}.'.format(idx, label_int))
				return TextLineDatasetBase.UNKNOWN  # TODO [check] >> Is it correct?
		return ''.join(list(index2label(idx) for idx in label_int if idx != self._default_value))

	# String labels -> Integer labels.
	def encode_labels(self, labels_str, dtype=np.int16, *args, **kwargs):
		max_label_len = functools.reduce(lambda x, y: max(x, len(y)), labels_str, 0)
		labels_int = np.full((len(labels_str), max_label_len), self._default_value, dtype=dtype)
		for (idx, lbl) in enumerate(labels_str):
			try:
				labels_int[idx,:len(lbl)] = np.array(list(self.labels.index(ch) for ch in lbl))
			except ValueError:
				pass
		return labels_int

	# Integer labels -> string labels.
	def decode_labels(self, labels_int, *args, **kwargs):
		def int2str(label):
			try:
				label = list(self.labels[lid] for lid in label if lid != self._default_value)
				return ''.join(label)
			except ValueError:
				return None
		return list(map(int2str, labels_int))

class MyRunTimeTextLineDataset(TextLineDatasetBase):
	def __init__(self, num_examples, text_set, text_generator, color_functor, transform, labels, num_classes, default_value=-1):
		super().__init__(labels, num_classes, default_value)
	
		self.num_examples = num_examples
		self.transform = transform

		# TODO [check] >> Is it good to use a generator of batch size 1?
		batch_size = 1  # Fixed. batch_size sets to 1 to generate one by one.
		self.data_generator = text_generator.create_subset_generator(text_set, batch_size, color_functor)
		#self.data_generator = text_generator.create_whole_generator(list(text_set), batch_size, color_functor, shuffle=True)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		texts, scenes, _ = next(self.data_generator)
		text_int = self.encode_label(texts[0])

		sample = {'input': scenes[0], 'output': (texts[0], text_int)}
		return self.transform(sample) if self.transform else sample

class MyFileBasedTextLineDataset(TextLineDatasetBase):
	def __init__(self, data, transform, labels, num_classes, default_value=-1):
		super().__init__(labels, num_classes, default_value)
	
		self.transform = transform

		# TODO [enhance] >>
		#	It, even if temporarily, consumes too much memory to create a new data list.
		#	It is computed repeatedly each epoch.
		self.data = list()
		for image, label_str in data:
			try:
				label_int = self.encode_label(label_str)
			except Exception:
				#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.decode_label(label_int):
				print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.decode_label(label_int)))
				continue

			self.data.append((image, label_str, label_int))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""
		image, label_str = self.data[idx]
		label_int = self.encode_label(label_str)
		"""
		image, label_str, label_int = self.data[idx]

		sample = {'input': image, 'output': (label_str, label_int)}
		return self.transform(sample) if self.transform else sample

#--------------------------------------------------------------------

def construct_chars():
	import string, random

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

def generate_text_set(max_label_len):
	import text_generation_util as tg_util

	word_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'

	print('[SWL] Info: Start loading a Korean dictionary...')
	start_time = time.time()
	with open(word_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#dictionary_words = fd.read().strip('\n')
		#dictionary_words = fd.readlines()
		dictionary_words = fd.read().splitlines()
	print('[SWL] Info: End loading a Korean dictionary, {} words loaded: {} secs.'.format(len(dictionary_words), time.time() - start_time))

	print('[SWL] Info: Start generating random words...')
	start_time = time.time()
	chars = construct_chars()
	random_words = tg_util.generate_random_words(chars, min_char_len=1, max_char_len=10)
	print('[SWL] Info: End generating random words, {} words generated: {} secs.'.format(len(random_words), time.time() - start_time))

	print('[SWL] Info: Start generating text lines...')
	#texts = tg_util.generate_random_text_lines(dictionary_words + random_words, min_word_len=1, max_word_len=5)
	texts = tg_util.generate_random_text_lines(random_words, min_word_len=1, max_word_len=5)
	print('[SWL] Info: End generating text lines, {} text lines generated: {} secs.'.format(len(texts), time.time() - start_time))

	if max_label_len > 0:
		texts = set(filter(lambda txt: len(txt) <= max_label_len, texts))

	if False:
		from swl.language_processing.util import draw_character_histogram
		draw_character_histogram(texts, charset=None)

	labels = functools.reduce(lambda x, txt: x.union(txt), texts, set())
	labels.add(TextLineDatasetBase.UNKNOWN)
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))

	return texts, labels

def generate_font_colors(image_depth):
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

def create_printed_text_generator(image_height, image_channel):
	import text_generation_util as tg_util

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'
	#font_dir_path = font_base_dir_path + '/receipt_kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
	char_images_dict = None

	min_font_size, max_font_size = round(image_height * 0.8), round(image_height * 1.25)
	min_char_space_ratio, max_char_space_ratio = 0.8, 1.25
	if 1 == image_channel:
		mode = 'L'
	elif 3 == image_channel:
		mode = 'RGB'
	else:
		raise ValueError('Invalid image channel: {}'.format(image_channel))

	#return tg_util.BasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), None, mode=mode, mask_mode='1')
	return tg_util.BasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), mode=mode, mask_mode='1')

# REF [function] >> FileBasedTextLineDatasetBase._load_data_from_image_and_label_files() in text_line_data.py
def load_data_from_file_pairs(file_pairs, max_label_len, is_grayscale=False):
	"""Loads data from image and label file pairs.
	"""

	for img_fpath, lbl_fpath in file_pairs:
		img_fname, lbl_fname = os.path.splitext(os.path.basename(img_fpath))[0], os.path.splitext(os.path.basename(lbl_fpath))[0]
		if img_fname != lbl_fname:
			print('[SWL] Warning: Different file names of image and label pair, {} != {}.'.format(img_fname, lbl_fname))
			continue

	flags = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
	data = list()
	for img_fpath, lbl_fpath in file_pairs:
		try:
			with open(lbl_fpath, 'r', encoding='UTF8') as fd:
				#label_str = fd.read()
				#label_str = fd.read().rstrip()
				label_str = fd.read().rstrip('\n')
		except FileNotFoundError as ex:
			print('[SWL] Error: File not found: {}.'.format(lbl_fpath))
			continue
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error: {}.'.format(lbl_fpath))
			continue

		if len(label_str) > max_label_len:
			print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
			continue

		img = cv2.imread(img_fpath, flags)
		if img is None:
			print('[SWL] Error: Failed to load an image: {}.'.format(img_fpath))
			continue

		data.append((img, label_str))

	return data

# REF [function] >>
#	FileBasedTextLineDatasetBase._load_data_from_image_label_info() in text_line_data.py
#	TextRecognitionDataGeneratorTextLineDatasetBase._load_data_with_label_file() in TextRecognitionDataGenerator_data.py
def load_data_from_label_file(label_filepath, max_label_len, is_grayscale=False, image_label_separator=','):
	"""Loads data from a label file, in which each filepath and label are separated by a separator like white space or comma.
	Each line consists of 'image-filepath + image-label-separator + label'.
	"""

	try:
		with open(label_filepath, 'r', encoding='UTF8') as fd:
			#lines = fd.readlines()  # A list of strings.
			lines = fd.read().splitlines()  # A list of strings.
	except FileNotFoundError as ex:
		print('[SWL] Error: File not found: {}.'.format(label_filepath))
		raise
	except UnicodeDecodeError as ex:
		print('[SWL] Error: Unicode decode error: {}.'.format(label_filepath))
		raise

	dir_path = os.path.dirname(label_filepath)
	flags = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
	data = list()
	for line in lines:
		img_fpath, label_str = line.split(image_label_separator, 1)

		if len(label_str) > max_label_len:
			print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
			continue

		img_fpath = os.path.join(dir_path, img_fpath)
		img = cv2.imread(img_fpath, flags)
		if img is None:
			print('[SWL] Error: Failed to load an image: {}.'.format(img_fpath))
			continue

		data.append((img, label_str))

	return data

# REF [function] >> TextRecognitionDataGeneratorTextLineDatasetBase._load_data_with_label_in_filename() in TextRecognitionDataGenerator_data.py
def load_data_from_image_files(image_filepaths, max_label_len, is_grayscale=False):
	"""Loads data from image files, whose labels are extracted from their filename.
	"""

	flags = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
	data = list()
	for fpath in image_filepaths:
		label_str = os.path.basename(fpath).split('_')
		if 2 != len(label_str):
			print('[SWL] Warning: Invalid file name: {}.'.format(fpath))
			continue
		label_str = label_str[0]

		if len(label_str) > max_label_len:
			print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
			continue

		img = cv2.imread(fpath, flags)
		if img is None:
			print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
			continue

		data.append((img, label_str))

	return data

def load_data(max_label_len, is_grayscale=False):
	if True:
		if 'posix' == os.name:
			data_base_dir_path = '/home/sangwook/work/dataset'
		else:
			data_base_dir_path = 'D:/work/dataset'

		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_vision/pascal_voc_test.py
		data_dir_path = data_base_dir_path + '/text/receipt_epapyrus/epapyrus_20190618/receipt_text_line'

		image_filepaths, label_filepaths = sorted(glob.glob(os.path.join(data_dir_path, '*.png'), recursive=False)), sorted(glob.glob(os.path.join(data_dir_path, '*.txt'), recursive=False))
		if not image_filepaths or not label_filepaths:
			raise IOError('Failed to load data from {}.'.format(data_dir_path))
		if len(image_filepaths) != len(label_filepaths):
			raise IOError('Unmatched image and label files: {} != {}.'.format(len(image_filepaths), len(label_filepaths)))
		file_pairs = list(zip(image_filepaths, label_filepaths))

		data = load_data_from_file_pairs(file_pairs, max_label_len, is_grayscale)
	elif False:
		label_filepath = './text_line_samples_en_train_v11/labels.txt'
		data = load_data_from_label_file(label_filepath, max_label_len, is_grayscale, image_label_separator=',')
	elif False:
		data_dir_path = './text_line_samples_en_train/dic_h32_w1'
		image_filepaths = sorted(glob.glob(os.path.join(data_dir_path, '*.png'), recursive=False))
		data = load_data_from_image_files(image_filepaths, max_label_len, is_grayscale)

	labels = sorted(functools.reduce(lambda x, dat: x.union(dat[1]), data, set()))

	return data, labels

#--------------------------------------------------------------------

class Augment(object):
	def __init__(self, min_height=None, max_height=None):
		assert (min_height is None or isinstance(min_height, int)) and (max_height is None or isinstance(max_height, int))
		self.min_height, self.max_height = min_height, max_height
		self.augmenter = self._create_augmenter()

	def __call__(self, sample):
		inp, outp = sample['input'], sample['output']

		if self.min_height and self.max_height:
			inp = self._reduce_image(inp)

		inp, _ = self._augment(np.expand_dims(inp, axis=0), None)
		return {'input': np.squeeze(inp, axis=0), 'output': outp}

	def _augment(self, inputs, outputs, *args, **kwargs):
		if outputs is None:
			return self.augmenter.augment_images(inputs), None
		else:
			augmenter_det = self.augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

	def _reduce_image(self, image, *args, **kwargs):
		height = random.randint(self.min_height, self.max_height)
		interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4])
		return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height), interpolation=interpolation)

	def _create_augmenter(self):
		#import imgaug as ia
		from imgaug import augmenters as iaa
		np.random.bit_generator = np.random._bit_generator

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
					translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent along x-axis and -10 to +10 percent along y-axis.
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

class Resize(object):
	def __init__(self, height, width):
		assert isinstance(height, int) and isinstance(width, int)
		self.height, self.width = height, width

	def __call__(self, sample):
		inp, outp = sample['input'], sample['output']
		height, width = self.height, self.width

		"""
		hi, wi = inp.shape[:2]
		if wi >= width:
			return {'input': cv2.resize(inp, (width, height), interpolation=cv2.INTER_AREA), 'output': outp}
		else:
			aspect_ratio = height / hi
			min_width = min(width, int(wi * aspect_ratio))
			inp = cv2.resize(inp, (min_width, height), interpolation=cv2.INTER_AREA)
			if min_width < width:
				image_zeropadded = np.zeros((height, width) + inp.shape[2:], dtype=inp.dtype)
				image_zeropadded[:,:min_width] = inp[:,:min_width]
				return {'input': image_zeropadded, 'output': outp}
			else:
				return {'input': inp, 'output': outp}
		"""
		hi, wi = inp.shape[:2]
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = np.zeros((height, width) + inp.shape[2:], dtype=inp.dtype)
		zeropadded[:,:min_width] = cv2.resize(inp, (min_width, height), interpolation=cv2.INTER_AREA)
		return {'input': zeropadded, 'output': outp}
		"""
		return {'input': cv2.resize(inp, (width, height), interpolation=cv2.INTER_AREA), 'output': outp}
		"""

class Preprocess(object):
	def __call__(self, sample):
		sample['input'] = (sample['input'].astype(np.float32) / 255.0) * 2.0 - 1.0  # Normalization.
		return sample

class Reshape(object):
	def __init__(self, use_NWHC=True):
		assert isinstance(use_NWHC, bool)
		self.use_NWHC = use_NWHC

	def __call__(self, sample):
		inp, outp = sample['input'], sample['output']

		"""
		if 3 == inp.ndim:
			inp = inp.reshape(inp.shape + (-1,))  # Image channel = 1.
			#inp = np.reshape(inp, inp.shape + (-1,))  # Image channel = 1.

		if self.use_NWHC:
			# (examples, height, width, channels) -> (examples, width, height, channels).
			inp = np.swapaxes(inp, 1, 2)
			#inp = inp.transpose((0, 2, 1, 3))
		"""
		if 2 == inp.ndim:
			inp = inp.reshape(inp.shape + (-1,))  # Image channel = 1.
			#inp = np.reshape(inp, inp.shape + (-1,))  # Image channel = 1.

		if self.use_NWHC:
			# (height, width, channels) -> (width, height, channels).
			inp = np.swapaxes(inp, 0, 1)
			#inp = inp.transpose((1, 0, 2))

		return {'input': inp, 'output': outp}

class ToTensor(object):
	def __call__(self, sample):
		inp, outp = sample['input'], sample['output']

		# Swap channel axis:
		#	NumPy image: H x W x C.
		#	Torch image: C x H x W.
		inp = inp.transpose((2, 0, 1))
		return {'input': torch.from_numpy(inp), 'output': outp}

# REF [function] >> default_collate() in ${PyTorch_HOME}/utils/data/_utils/collate.py.
def my_collate_func(batch):
	batch_inp, batch_outp = list(zip(*list((el['input'], el['output']) for el in batch)))

	elem = batch_inp[0]
	out = None
	if torch.utils.data.get_worker_info() is not None:
		# If we're in a background process, concatenate directly into a
		# shared memory tensor to avoid an extra copy
		numel = sum([x.numel() for x in batch_inp])
		storage = elem.storage()._new_shared(numel)
		out = elem.new(storage)
	batch_inp = torch.stack(batch_inp, 0, out=out)
	
	batch_outp = list(zip(*batch_outp))

	return {'input': batch_inp, 'output': batch_outp}

#--------------------------------------------------------------------

def MyRunTimeTextLineDataset_test():
	image_height, image_width, image_channel = 64, 640, 1
	max_label_len = 80

	text_generator = create_printed_text_generator(image_height, image_channel)
	text_set, labels = generate_text_set(max_label_len)

	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	#--------------------
	num_examples = 100
	min_height, max_height = round(image_height * 0.5), image_height * 2
	#min_height, max_height = image_height, image_height * 2
	#min_height, max_height = None, None
	use_NWHC = True
	color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

	transform = torchvision.transforms.Compose([
		Augment(min_height, max_height),
		Resize(image_height, image_width),
		Preprocess(),
		Reshape(use_NWHC),
		ToTensor()
	])

	dataset = MyRunTimeTextLineDataset(num_examples, text_set, text_generator, color_functor, transform, labels, num_classes, default_value=-1)

	for idx in range(len(dataset)):
		sample = dataset[idx]

		print(idx, sample['input'].size(), sample['output'])

		if idx >= 5:
			break

	#--------------------
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=my_collate_func)

	for batch_step, batch_data in enumerate(dataloader):
		print(batch_step, batch_data['input'].size(), batch_data['output'])

		if batch_step >= 5:
			break

def MyFileBasedTextLineDataset_test():
	image_height, image_width, image_channel = 64, 640, 1
	max_label_len = 80

	data, labels = load_data(max_label_len, is_grayscale=(1 == image_channel))

	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	#--------------------
	#train_test_ratio = 0.8
	use_NWHC = True

	transform = torchvision.transforms.Compose([
		Augment(min_height=None, max_height=None),
		Resize(image_height, image_width),
		Preprocess(),
		Reshape(use_NWHC),
		ToTensor()
	])

	dataset = MyFileBasedTextLineDataset(data, transform, labels, num_classes, default_value=-1)

	for idx in range(len(dataset)):
		sample = dataset[idx]

		print(idx, sample['input'].size(), sample['output'])

		if idx >= 5:
			break

	#--------------------
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=my_collate_func)

	for batch_step, batch_data in enumerate(dataloader):
		print(batch_step, batch_data['input'].size(), batch_data['output'])

		if batch_step >= 5:
			break

def main():
	#MyRunTimeTextLineDataset_test()
	MyFileBasedTextLineDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
