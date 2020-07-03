#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, functools
import swl.language_processing.util as swl_langproc_util
import TextRecognitionDataGenerator_data

class MyEnglishTextLineDataset(TextRecognitionDataGenerator_data.EnglishTextRecognitionDataGeneratorTextLineDataset):
	def __init__(self, label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle=True):
		super().__init__(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle)

		#--------------------
		#import imgaug as ia
		from imgaug import augmenters as iaa

		self._augmenter = iaa.Sequential([
			iaa.Sometimes(0.5, iaa.OneOf([
				iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
				#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
				#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
			])),
			iaa.Sometimes(0.5, iaa.OneOf([
				iaa.Affine(
					scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
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
				iaa.PerspectiveTransform(scale=(0.01, 0.1)),
				iaa.ElasticTransformation(alpha=(15.0, 30.0), sigma=5.0),  # Move pixels locally around (with random strengths).
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

	def augment(self, inputs, outputs, *args, **kwargs):
		if outputs is None:
			return self._augmenter.augment_images(inputs), None
		else:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
# REF [script] >> generate_TextRecognitionDataGenerator_data.sh
def EnglishTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './text_line_samples_en_train/dic_h32_w1'

	image_height, image_width, image_channel = 32, 100, 1
	train_test_ratio = 0.8
	max_char_count = 50

	import string
	labels = \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))

	label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
	# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
	print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
	print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

	if True:
		print('Start creating an EnglishTextRecognitionDataGeneratorTextLineDataset...')
		start_time = time.time()
		dataset = TextRecognitionDataGenerator_data.EnglishTextRecognitionDataGeneratorTextLineDataset(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
		print('End creating an EnglishTextRecognitionDataGeneratorTextLineDataset: {} secs.'.format(time.time() - start_time))
	else:
		print('Start creating an MyEnglishTextLineDataset...')
		start_time = time.time()
		dataset = MyEnglishTextLineDataset(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count, shuffle=True)
		print('End creating an MyEnglishTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
# REF [script] >> generate_TextRecognitionDataGenerator_data.sh
def HangeulTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './text_line_samples_kr_train/dic_h32_w1'

	#image_height, image_width, image_channel = 32, 160, 1
	image_height, image_width, image_channel = 64, 320, 1
	train_test_ratio = 0.8
	max_char_count = 50

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

	import string
	labels = \
		hangeul_charset + \
		hangeul_jamo_charset + \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	# There are words of Unicode Hangeul letters besides KS X 1001.
	#labels = functools.reduce(lambda x, fpath: x.union(fpath.split('_')[0]), os.listdir(data_dir_path), set(labels))
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))

	label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
	# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
	print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
	print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

	print('Start creating a HangeulTextRecognitionDataGeneratorTextLineDataset...')
	start_time = time.time()
	dataset = TextRecognitionDataGenerator_data.HangeulTextRecognitionDataGeneratorTextLineDataset(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating a HangeulTextRecognitionDataGeneratorTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
# REF [script] >> generate_TextRecognitionDataGenerator_data.sh
def HangeulJamoTextRecognitionDataGeneratorTextLineDataset_test():
	import hangeul_util as hg_util

	data_dir_path = './text_line_samples_kr_train/dic_h32_w1'

	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	hangeul2jamo_functor = lambda hangeul_str: hg_util.hangeul2jamo(hangeul_str, eojc_str=swl_langproc_util.JamoTokenConverter.EOJ, use_separate_consonants=False, use_separate_vowels=True)
	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	jamo2hangeul_functor = lambda jamo_str: hg_util.jamo2hangeul(jamo_str, eojc_str=swl_langproc_util.JamoTokenConverter.EOJ, use_separate_consonants=False, use_separate_vowels=True)

	#image_height, image_width, image_channel = 32, 160, 1
	image_height, image_width, image_channel = 64, 320, 1
	train_test_ratio = 0.8
	max_char_count = 50

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	labels = \
		hangeul_jamo_charset + \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	# There are words of Unicode Hangeul letters besides KS X 1001.
	labels = functools.reduce(lambda x, fpath: x.union(hangeul2jamo_functor(fpath.split('_')[0])), os.listdir(data_dir_path), set(labels))
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))

	label_converter = swl_langproc_util.JamoTokenConverter(labels, hangeul2jamo_functor, jamo2hangeul_functor, pad_value=None)
	# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
	print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
	print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

	print('Start creating a HangeulJamoTextRecognitionDataGeneratorTextLineDataset...')
	start_time = time.time()
	dataset = TextRecognitionDataGenerator_data.HangeulJamoTextRecognitionDataGeneratorTextLineDataset(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating a HangeulJamoTextRecognitionDataGeneratorTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
# REF [script] >> generate_TextRecognitionDataGenerator_data.sh
def merge_generated_data_directories():
	if True:
		src_base_data_dir_path = './text_line_samples_en_train'
		#src_base_data_dir_path = './text_line_samples_en_test'
		dir_prefixes = ['dic', 'rs']
		font_sizes = [16, 24, 32, 40, 48]
		word_counts = [1, 2, 3]
		src_data_dir_paths = list()
		for dir_prefix in dir_prefixes:
			for font_size in font_sizes:
				for word_count in word_counts:
					src_data_dir_paths.append('{}/{}_h{}_w{}'.format(src_base_data_dir_path, dir_prefix, font_size, word_count))
		dst_data_dir_path = src_base_data_dir_path
	elif False:
		src_base_data_dir_path = './text_line_samples_kr_train'
		#src_base_data_dir_path = './text_line_samples_kr_test'
		dir_prefixes = ['dic', 'rs']
		font_sizes = [32, 48, 64, 80, 96]
		word_counts = [1, 2, 3]
		src_data_dir_paths = list()
		for dir_prefix in dir_prefixes:
			for font_size in font_sizes:
				for word_count in word_counts:
					src_data_dir_paths.append('{}/{}_h{}_w{}'.format(src_base_data_dir_path, dir_prefix, font_size, word_count))
		dst_data_dir_path = src_base_data_dir_path
	src_label_filename = 'labels.txt'
	dst_label_filename = src_label_filename

	os.makedirs(dst_data_dir_path, exist_ok=True)

	os.sep = '/'
	new_lines = list()
	for src_data_dir_path in src_data_dir_paths:
		try:
			with open(os.path.join(src_data_dir_path, src_label_filename), 'r') as fd:
				lines = fd.readlines()
		except FileNotFoundError:
			print('[SWL] Error: File not found: {}.'.format(os.path.join(src_data_dir_path, src_label_filename)))
			continue

		rel_path = os.path.relpath(src_data_dir_path, dst_data_dir_path)
		for line in lines:
			line = line.rstrip('\n')
			if not line:
				continue

			pos = line.find(' ')
			if -1 == pos:
				print('[SWL] Warning: Invalid image-label pair: {}.'.format(line))
				continue
			fname, label = line[:pos], line[pos+1:]

			new_line = '{} {}'.format(os.path.join(rel_path, fname), label)
			new_lines.append(new_line)

	with open(os.path.join(dst_data_dir_path, dst_label_filename), 'w', encoding='UTF8') as fd:
		for line in new_lines:
			fd.write('{}\n'.format(line))

def check_label_distribution():
	label_filepath = './text_line_samples_en_train/labels.txt'
	#label_filepath = './text_line_samples_en_test/labels.txt'

	try:
		with open(label_filepath, 'r', encoding='UTF8') as fd:
			#lines = fd.readlines()
			lines = fd.read().splitlines()
	except FileNotFoundError:
		print('[SWL] Error: File not found: {}.'.format(label_filepath))
		return
	except UnicodeDecodeError as ex:
		print('[SWL] Error: Unicode decode error: {}.'.format(label_filepath))
		return

	texts = list()
	for line in lines:
		pos = line.find(' ')
		if -1 == pos:
			print('[SWL] Warning: Invalid image-label pair: {}.'.format(line))
			continue
		fname, label = line[:pos], line[pos+1:]
		texts.append(label)

	#--------------------
	swl_langproc_util.draw_character_histogram(texts, charset=None)

def main():
	#EnglishTextRecognitionDataGeneratorTextLineDataset_test()
	#HangeulTextRecognitionDataGeneratorTextLineDataset_test()
	#HangeulJamoTextRecognitionDataGeneratorTextLineDataset_test()

	merge_generated_data_directories()

	#check_label_distribution()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
