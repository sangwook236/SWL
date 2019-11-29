#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, random, time, functools, glob, json
import numpy as np
import cv2
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util

def generate_random_word_set_test():
	hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangeul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	if False:
		print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
		print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
		print('Digit charset =', len(digit_charset), digit_charset)
		print('Symbol charset =', len(symbol_charset), symbol_charset)

	charsets = [
		hangeul_charset,
		alphabet_charset,
		digit_charset,
		#symbol_charset,
		hangeul_charset + digit_charset,
		hangeul_charset + symbol_charset,
		alphabet_charset + digit_charset,
		alphabet_charset + symbol_charset,
		hangeul_charset + alphabet_charset,
		hangeul_charset + alphabet_charset + digit_charset,
		hangeul_charset + alphabet_charset + symbol_charset,
		hangeul_charset + alphabet_charset + digit_charset + symbol_charset,
	]

	num_words = 10
	min_char_count, max_char_count = 2, 10
	word_set = set()
	for charset in charsets:
		word_set = word_set.union(tg_util.generate_random_word_set(num_words, charset, min_char_count, max_char_count))
	
	print('#generated word set =', len(word_set))
	print('Generated word set =', word_set)

def generate_repetitive_word_set_test():
	hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangeul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	if False:
		print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
		print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
		print('Digit charset =', len(digit_charset), digit_charset)
		print('Symbol charset =', len(symbol_charset), symbol_charset)

	charsets = [
		hangeul_charset,
		alphabet_charset,
		digit_charset,
		#symbol_charset,
		hangeul_charset + digit_charset,
		hangeul_charset + symbol_charset,
		alphabet_charset + digit_charset,
		alphabet_charset + symbol_charset,
		hangeul_charset + alphabet_charset,
		hangeul_charset + alphabet_charset + digit_charset,
		hangeul_charset + alphabet_charset + symbol_charset,
		hangeul_charset + alphabet_charset + digit_charset + symbol_charset,
	]

	num_char_repetitions = 2
	min_char_count, max_char_count = 2, 10
	word_set = set()
	for charset in charsets:
		word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, charset, min_char_count, max_char_count))
	
	print('#generated word set =', len(word_set))
	print('Generated word set =', word_set)

def generate_font_list_test():
	texts = [
		'ABCDEFGHIJKLMnopqrstuvwxyz',
		'abcdefghijklmNOPQRSTUVWXYZ',
		'0123456789',
		' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?',
	]

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/eng'

	#os.path.sep = '/'
	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	font_list = tg_util.generate_font_list(font_filepaths)

	#--------------------
	text_offset = (0, 0)
	crop_text_area = True
	draw_text_border = False

	for font_type, font_index in font_list:  # Font filepath, font index.
		print('Font: {}, font index: {}.'.format(font_type, font_index))
		for text in texts:
			font_size = random.randint(20, 40)
			#image_size = (math.ceil(font_size * 1.1) * len(text), math.ceil(font_size * 1.1))
			image_size = None
			#font_color = (255, 255, 255)
			#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
			#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
			font_color = None  # Uses a random font color.
			#bg_color = (0, 0, 0)
			#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
			#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
			bg_color = None  # Uses a random background color.
			alpha = swl_langproc_util.generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)

			cv2.imshow('Text', np.array(alpha))
			cv2.waitKey(0)

	cv2.destroyAllWindows()

def generate_hangeul_font_list_test():
	texts = [
		'가나다라마바사앙잦찿캌탙팦핳',
		'각난닫랄맘밥삿아자차카타파하',
		'ABCDEFGHIJKLMnopqrstuvwxyz',
		'abcdefghijklmNOPQRSTUVWXYZ',
		'0123456789',
		' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?',
	]

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'

	#os.path.sep = '/'
	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)

	#--------------------
	text_offset = (0, 0)
	crop_text_area = True
	draw_text_border = False

	for font_type, font_index in font_list:  # Font filepath, font index.
		print('Font: {}, font index: {}.'.format(font_type, font_index))
		for text in texts:
			font_size = random.randint(20, 40)
			#image_size = (math.ceil(font_size * 1.1) * len(text), math.ceil(font_size * 1.1))
			image_size = None
			#font_color = (255, 255, 255)
			#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
			#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
			font_color = None  # Uses a random font color.
			#bg_color = (0, 0, 0)
			#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
			#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
			bg_color = None  # Uses a random background color.
			alpha = swl_langproc_util.generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)

			cv2.imshow('Text', np.array(alpha))
			cv2.waitKey(0)

	cv2.destroyAllWindows()

def generate_font_colors(image_depth):
	import random
	#font_color = (255,) * image_depth
	#font_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * image_depth  # Uses a specific grayscale font color.
	gray_val = random.randrange(255)
	font_color = (gray_val,) * image_depth  # Uses a specific black font color.
	#font_color = (random.randrange(128, 256),) * image_depth  # Uses a specific white font color.
	#font_color = None  # Uses a random font color.
	#bg_color = (0,) * image_depth
	#bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * image_depth  # Uses a specific grayscale background color.
	#bg_color = (random.randrange(0, 128),) * image_depth  # Uses a specific black background color.
	bg_color = (random.randrange(gray_val + 1, 256),) * image_depth  # Uses a specific white background color.
	#bg_color = None  # Uses a random background color.
	return font_color, bg_color

def basic_printed_text_generator_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)

	#--------------------
	font_size = 32
	char_space_ratio = 0.5

	#textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (font_size, font_size), None, mode='RGB', mask_mode='1')
	textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (font_size, font_size), (char_space_ratio, char_space_ratio), mode='RGB', mask_mode='1')

	#text = '가나다라마바사아자차카타파하'
	text = 'abcdefghijklmNOPQRSTUVWXYZ'
	font_color, bg_color = generate_font_colors(image_depth=3)
	text_line, text_mask = textGenerator(text, font_color, bg_color)

	#--------------------
	# No background.
	text_mask[text_mask > 0] = 255
	if False:
		cv2.imwrite('./text_line.png', text_line)
		cv2.imwrite('./text_mask.png', text_mask)
	else:
		cv2.imshow('Text Line', text_line)
		cv2.imshow('Text Mask', text_mask)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def generate_basic_text_lines_test():
	word_set = set()
	word_set.add('abc')
	word_set.add('defg')
	word_set.add('hijklmn')
	word_set.add('가나')
	word_set.add('다라마바사')
	word_set.add('자차카타파하')
	word_set.add('123')
	word_set.add('45')
	word_set.add('67890')

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.2, 1.5
	color_functor = functools.partial(generate_font_colors, image_depth=3)

	#textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), None, mode='RGB', mask_mode='1')
	textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), mode='RGB', mask_mode='1')

	batch_size = 4
	generator = textGenerator.create_generator(word_set, batch_size, color_functor)

	#--------------------
	step = 1
	for text_list, image_list, mask_list in generator:
		for text, img, mask in zip(text_list, image_list, mask_list):
			mask[mask > 0] = 255

			print('Text = {}.'.format(text))
			if False:
				cv2.imwrite('./text.png', img)
				cv2.imwrite('./text_mask.png', mask)
			else:
				cv2.imshow('Text', img)
				cv2.imshow('Text Mask', mask)
				cv2.waitKey(0)

		if step >= 3:
			break
		step += 1

	cv2.destroyAllWindows()

def text_alpha_matte_generator_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	font_size = 32
	char_space_ratio = 1.2
	font_color, _ = generate_font_colors(image_depth=3)

	characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='1')
	#characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='L')
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()
	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, (font_size, font_size), (char_space_ratio, char_space_ratio))

	char_alpha_list, char_alpha_coordinate_list = textAlphaMatteGenerator('가나다라마바사아자차카타파하')
	text_line, text_line_alpha = tg_util.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)

	#--------------------
	# No background.
	if False:
		cv2.imwrite('./text_line.png', text_line)
	else:
		cv2.imshow('Text line', text_line)

		cv2.waitKey(0)
		#cv2.destroyAllWindows()

	#--------------------
	sceneTextGenerator = tg_util.MyAlphaMatteSceneTextGenerator(tg_util.IdentityTransformer())

	# Grayscale background.
	bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
	scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])

	if False:
		cv2.imwrite('./scene.png', scene)
		cv2.imwrite('./scene_text_mask.png', scene_text_mask)
	else:
		cv2.imshow('Scene', scene)
		#scene_text_mask[scene_text_mask > 0] = 255
		#scene_text_mask = scene_text_mask.astype(np.uint8)
		minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
		scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)
		cv2.imshow('Scene Mask', scene_text_mask)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def scene_text_alpha_matte_generator_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='1')
	#characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='L')
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()

	#--------------------
	texts, text_alphas = list(), list()

	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(32, 32), char_space_ratio_interval=(0.9, 0.9))
	char_alpha_list, char_alpha_coordinate_list = textAlphaMatteGenerator('가나다라마바사아자차카타파하')
	font_color, _ = generate_font_colors(image_depth=3)
	text_line, text_line_alpha = tg_util.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(24, 24), char_space_ratio_interval=(1.6, 1.6))
	#char_alpha_list, char_alpha_coordinate_list = textAlphaMatteGenerator('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
	char_alpha_list, char_alpha_coordinate_list = textAlphaMatteGenerator('ABCDEFGHIJKLM')
	font_color, _ = generate_font_colors(image_depth=3)
	text_line, text_line_alpha = tg_util.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(16, 16), char_space_ratio_interval=(2.0, 2.0))
	#char_alpha_list, char_alpha_coordinate_list = textAlphaMatteGenerator('abcdefghijklmnopqrstuvwxyz')
	char_alpha_list, char_alpha_coordinate_list = textAlphaMatteGenerator('abcdefghijklm')
	font_color, _ = generate_font_colors(image_depth=3)
	text_line, text_line_alpha = tg_util.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#--------------------
	if True:
		textTransformer = tg_util.PerspectiveTransformer()
	else:
		textTransformer = tg_util.ProjectiveTransformer()
	sceneTextGenerator = tg_util.MyAlphaMatteSceneTextGenerator(textTransformer)

	if True:
		sceneProvider = tg_util.MySceneProvider()
	else:
		# Grayscale background.
		scene_shape = (800, 1000, 3)  # Some handwritten characters have 3 channels.
		sceneProvider = tg_util.MyGrayscaleBackgroundProvider(scene_shape)

	#--------------------
	scene = sceneProvider()
	if 3 == scene.ndim and 3 != scene.shape[-1]:
		raise ValueError('Invalid image shape')

	scene, scene_text_mask, _ = sceneTextGenerator(scene, texts, text_alphas)

	#--------------------
	if False:
		cv2.imwrite('./scene.png', scene)
		cv2.imwrite('./scene_text_mask.png', scene_text_mask)
	else:
		cv2.imshow('Scene', scene)
		#scene_text_mask[scene_text_mask > 0] = 255
		#scene_text_mask = scene_text_mask.astype(np.uint8)
		minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
		scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)

		cv2.imshow('Scene Mask', scene_text_mask)
		cv2.waitKey(0)

		cv2.destroyAllWindows()

def generate_alpha_matte_text_lines_test_1():
	word_set = set()
	word_set.add('abc')
	word_set.add('defg')
	word_set.add('hijklmn')
	word_set.add('가나')
	word_set.add('다라마바사')
	word_set.add('자차카타파하')
	word_set.add('123')
	word_set.add('45')
	word_set.add('67890')

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2
	color_functor = functools.partial(generate_font_colors, image_depth=3)

	characterTransformer = tg_util.IdentityTransformer()
	#characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()
	textGenerator = tg_util.MySimpleTextAlphaMatteGenerator(characterTransformer, characterPositioner, font_list=font_list, handwriting_dict=handwriting_dict, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio), alpha_matte_mode='1')

	batch_size = 4
	generator = textGenerator.create_generator(word_set, batch_size, color_functor)

	#--------------------
	step = 1
	for text_list, scene_list, scene_text_mask_list in generator:
		for text, scene, scene_text_mask in zip(text_list, scene_list, scene_text_mask_list):
			print('Text = {}.'.format(text))
			if False:
				cv2.imwrite('./scene.png', scene)
				cv2.imwrite('./scene_text_mask.png', scene_text_mask)
			else:
				#scene_text_mask[scene_text_mask > 0] = 255
				#scene_text_mask = scene_text_mask.astype(np.uint8)
				minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
				scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)

				cv2.imshow('Scene', scene)
				cv2.imshow('Scene Mask', scene_text_mask)
				cv2.waitKey(0)

		if step >= 3:
			break
		step += 1

	cv2.destroyAllWindows()

def generate_alpha_matte_text_lines_test_2():
	word_set = set()
	word_set.add('abc')
	word_set.add('defg')
	word_set.add('hijklmn')
	word_set.add('가나')
	word_set.add('다라마바사')
	word_set.add('자차카타파하')
	word_set.add('123')
	word_set.add('45')
	word_set.add('67890')

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2
	color_functor = functools.partial(generate_font_colors, image_depth=3)

	characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='1')
	#characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='L')
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()
	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio))
	#textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, (min_font_size, max_font_size), None)

	batch_size = 4
	generator = textAlphaMatteGenerator.create_generator(word_set, batch_size, color_functor)

	#--------------------
	step = 1
	for text_list, scene_list, scene_text_mask_list in generator:
		for text, scene, scene_text_mask in zip(text_list, scene_list, scene_text_mask_list):
			print('Text = {}.'.format(text))
			if False:
				cv2.imwrite('./scene.png', scene)
				cv2.imwrite('./scene_text_mask.png', scene_text_mask)
			else:
				#scene_text_mask[scene_text_mask > 0] = 255
				#scene_text_mask = scene_text_mask.astype(np.uint8)
				minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
				scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)

				if False:
					if 'posix' == os.name:
						font_dir_path = '/home/sangwook/work/font'
					else:
						font_dir_path = 'D:/work/font'
					font_type, font_index = font_dir_path + '/gulim.ttf', 0
					font_size = random.randrange(min_font_size, max_font_size)
					cv2.imshow('Naive Font Image', np.array(swl_langproc_util.generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False)))

				cv2.imshow('Scene', scene)
				cv2.imshow('Scene (Gray)', cv2.cvtColor(scene, cv2.COLOR_BGRA2GRAY))
				cv2.imshow('Scene Mask', scene_text_mask)
				cv2.waitKey(0)

		if step >= 3:
			break
		step += 1

	cv2.destroyAllWindows()

def generate_alpha_matte_scene_texts_test():
	word_set = set()
	word_set.add('abc')
	word_set.add('defg')
	word_set.add('hijklmn')
	word_set.add('가나')
	word_set.add('다라마바사')
	word_set.add('자차카타파하')
	word_set.add('123')
	word_set.add('45')
	word_set.add('67890')

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2

	characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='1')
	#characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='L')
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()
	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio))
	#textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=None)

	#--------------------
	if True:
		textTransformer = tg_util.PerspectiveTransformer()
	else:
		textTransformer = tg_util.ProjectiveTransformer()
	sceneTextGenerator = tg_util.MyAlphaMatteSceneTextGenerator(textTransformer)

	if True:
		sceneProvider = tg_util.MySceneProvider()
	else:
		# Grayscale background.
		scene_shape = (800, 1000, 3)  # Some handwritten characters have 3 channels.
		sceneProvider = tg_util.MyGrayscaleBackgroundProvider(scene_shape)

	#--------------------
	min_text_count_per_image, max_text_count_per_image = 2, 10
	color_functor = functools.partial(generate_font_colors, image_depth=3)

	batch_size = 4
	generator = sceneTextGenerator.create_generator(textAlphaMatteGenerator, sceneProvider, list(word_set), batch_size, (min_text_count_per_image, max_text_count_per_image), color_functor)

	#--------------------
	step = 1
	for texts_list, scene_list, scene_text_mask_list, bboxes_list in generator:
		for texts, scene, scene_text_mask, bboxes in zip(texts_list, scene_list, scene_text_mask_list, bboxes_list):
			if False:
				cv2.imwrite('./scene.png', scene)
				cv2.imwrite('./scene_text_mask.png', scene_text_mask)
			else:
				#scene_text_mask[scene_text_mask > 0] = 255
				#scene_text_mask = scene_text_mask.astype(np.uint8)
				minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
				scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)

				# Draw bounding rectangles.
				for box in bboxes:
					scene = cv2.line(scene, (box[0,0], box[0,1]), (box[1,0], box[1,1]), (0, 0, 255), 2, cv2.LINE_8)
					scene = cv2.line(scene, (box[1,0], box[1,1]), (box[2,0], box[2,1]), (0, 255, 0), 2, cv2.LINE_8)
					scene = cv2.line(scene, (box[2,0], box[2,1]), (box[3,0], box[3,1]), (255, 0, 0), 2, cv2.LINE_8)
					scene = cv2.line(scene, (box[3,0], box[3,1]), (box[0,0], box[0,1]), (255, 0, 255), 2, cv2.LINE_8)

				print('Texts =', texts)
				cv2.imshow('Scene', scene)
				cv2.imshow('Scene Mask', scene_text_mask)
				cv2.waitKey(0)

		if step >= 3:
			break
		step += 1

	cv2.destroyAllWindows()

def generate_scene_text_dataset(dir_path, json_filename, sceneTextGenerator, sceneProvider, textGenerator, num_images):
	scene_subdir_name = 'scene'
	mask_subdir_name = 'mask'
	scene_dir_path = os.path.join(dir_path, scene_subdir_name)
	mask_dir_path = os.path.join(dir_path, mask_subdir_name)

	os.makedirs(dir_path, exist_ok=True)
	os.makedirs(scene_dir_path, exist_ok=True)
	os.makedirs(mask_dir_path, exist_ok=True)

	hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangeul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	#print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
	#print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
	#print('Digit charset =', len(digit_charset), digit_charset)
	#print('Symbol charset =', len(symbol_charset), symbol_charset)

	#charset_list = [ hangeul_charset, alphabet_charset, digit_charset, symbol_charset ]
	#charset_selection_ratios = [ 0.25, 0.5, 0.75, 1.0 ]
	charset_list = [ hangeul_charset, alphabet_charset, digit_charset, hangeul_charset + alphabet_charset, hangeul_charset + alphabet_charset + digit_charset ]
	charset_selection_ratios = [ 0.6, 0.8, 0.9, 0.95, 1.0 ]

	min_char_count_per_text, max_char_count_per_text = 1, 10
	min_text_count_per_image, max_text_count_per_image = 2, 10
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 3

	#--------------------
	data_list = list()
	for idx in range(num_images):
		num_texts_per_image = random.randint(min_text_count_per_image, max_text_count_per_image)

		texts, text_images, text_alphas = list(), list(), list()
		for ii in range(num_texts_per_image):
			font_size = random.randint(min_font_size, max_font_size)
			char_space_ratio = random.uniform(min_char_space_ratio, max_char_space_ratio)
			font_color, _ = generate_font_colors(image_depth=3)

			num_chars_per_text = random.randint(min_char_count_per_text, max_char_count_per_text)

			charset_selection_ratio = random.uniform(0.0, 1.0)
			for charset_idx, ratio in enumerate(charset_selection_ratios):
				if charset_selection_ratio < ratio:
					break

			charset = charset_list[charset_idx]
			charset_len = len(charset)
			text = ''.join(list(charset[random.randrange(charset_len)] for _ in range(num_chars_per_text)))

			char_alpha_list, char_alpha_coordinate_list = textGenerator(text)
			text_line_image, text_line_alpha = tg_util.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)

			texts.append(text)
			text_images.append(text_line_image)
			text_alphas.append(text_line_alpha)

		#--------------------
		scene = sceneProvider()
		if 3 == scene.ndim and 3 != scene.shape[-1]:
			#raise ValueError('Invalid image shape')
			print('Error: Invalid image shape.')
			continue

		scene, scene_text_mask, bboxes = sceneTextGenerator(scene, text_images, text_alphas)

		#--------------------
		if True:
			text_image_filepath = os.path.join(scene_subdir_name, 'scene_{:07}.png'.format(idx))
			mask_image_filepath = os.path.join(mask_subdir_name, 'mask_{:07}.png'.format(idx))
		elif False:
			# For MS-D-Net.
			#scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255.0
			scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR)
			text_image_filepath = os.path.join(scene_subdir_name, 'scene_{:07}.tiff'.format(idx))
			mask_image_filepath = os.path.join(mask_subdir_name, 'mask_{:07}.tiff'.format(idx))
		elif False:
			# For MS-D_Net_PyTorch.
			#scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255.0
			scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR)
			scene_text_mask[scene_text_mask > 0] = 1
			#scene_text_mask = scene_text_mask.astype(np.uint8)
			text_image_filepath = os.path.join(scene_subdir_name, 'img_{:07}.tif'.format(idx))
			mask_image_filepath = os.path.join(mask_subdir_name, 'img_{:07}.tif'.format(idx))
		"""
		# Draw bounding rectangles.
		for box in bboxes:
			scene = cv2.line(scene, (box[0,0], box[0,1]), (box[1,0], box[1,1]), (0, 0, 255, 255), 2, cv2.LINE_8)
			scene = cv2.line(scene, (box[1,0], box[1,1]), (box[2,0], box[2,1]), (0, 255, 0, 255), 2, cv2.LINE_8)
			scene = cv2.line(scene, (box[2,0], box[2,1]), (box[3,0], box[3,1]), (255, 0, 0, 255), 2, cv2.LINE_8)
			scene = cv2.line(scene, (box[3,0], box[3,1]), (box[0,0], box[0,1]), (255, 0, 255, 255), 2, cv2.LINE_8)
		"""
		cv2.imwrite(os.path.join(dir_path, text_image_filepath), scene)
		cv2.imwrite(os.path.join(dir_path, mask_image_filepath), scene_text_mask)

		datum = {
			'image': text_image_filepath,
			#'image': os.path.abspath(text_image_filepath),
			'mask': mask_image_filepath,
			#'mask': os.path.abspath(mask_image_filepath),
			'texts': texts,
			'bboxes': bboxes.tolist(),
		}
		data_list.append(datum)

	json_filepath = os.path.join(dir_path, json_filename)
	with open(json_filepath, 'w', encoding='UTF-8') as json_file:
		#json.dump(data_list, json_file)
		json.dump(data_list, json_file, ensure_ascii=False, indent='  ')

def load_scene_text_dataset(dir_path, json_filename):
	json_filepath = os.path.join(dir_path, json_filename)
	with open(json_filepath, 'r', encoding='UTF-8') as json_file:
		json_data = json.load(json_file)

	image_filepaths, mask_filepaths, gt_texts, gt_boxes = list(), list(), list(), list()
	for dat in json_data:
		image_filepaths.append(dat['image'])
		mask_filepaths.append(dat['mask'])
		gt_texts.append(dat['texts'])
		#gt_boxes.append(dat['bboxes'])
		gt_boxes.append(np.array(dat['bboxes']))

	return image_filepaths, mask_filepaths, gt_texts, gt_boxes

def generate_hangeul_synthetic_scene_text_dataset():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2

	characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='1')
	#characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode='L')
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()
	textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio))
	#textAlphaMatteGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=None)

	#--------------------
	if True:
		textTransformer = tg_util.PerspectiveTransformer()
	else:
		textTransformer = tg_util.ProjectiveTransformer()
	sceneTextGenerator = tg_util.MyAlphaMatteSceneTextGenerator(textTransformer)

	if True:
		sceneProvider = tg_util.MySimpleSceneProvider()
	else:
		# Grayscale background.
		scene_shape = (800, 1000, 3)  # Some handwritten characters have 4 channels.
		sceneProvider = tg_util.MyGrayscaleBackgroundProvider(scene_shape)

	# Generate a scene dataset.
	scene_text_dataset_dir_path = './scene_text_dataset'
	#scene_text_dataset_dir_path = './scene_text_dataset_for_ms_d_net'
	#scene_text_dataset_dir_path = './scene_text_dataset_for_ms_d_net_pytorch'
	scene_text_dataset_json_filename = 'scene_text_dataset.json'
	num_images = 100

	print('Start generating scene text data in {}...'.format(scene_text_dataset_dir_path))
	start_time = time.time()
	generate_scene_text_dataset(scene_text_dataset_dir_path, scene_text_dataset_json_filename, sceneTextGenerator, sceneProvider, textAlphaMatteGenerator, num_images)
	print('End generating scene text data: {} secs.'.format(time.time() - start_time))

	# Load a scene dataset.
	print('Start loading scene text data...')
	start_time = time.time()
	image_filepaths, mask_filepaths, gt_texts, gt_boxes = load_scene_text_dataset(scene_text_dataset_dir_path, scene_text_dataset_json_filename)
	print('End loading scene text data: {} secs.'.format(time.time() - start_time))
	print('Loaded scene dataset: #images = {}, #masks = {}, #texts = {}, #boxes = {}.'.format(len(image_filepaths), len(mask_filepaths), len(gt_texts), len(gt_boxes)))

def generate_single_letter_dataset():
	hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangeul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangeul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.
	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	charsets = [
		#hangeul_charset + alphabet_charset + digit_charset,
		hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset,
		#hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset + symbol_charset,
	]

	num_char_repetitions = 2
	min_char_count, max_char_count = 1, 1
	word_set = set()
	for charset in charsets:
		word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, charset, min_char_count, max_char_count))

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
	handwriting_dict = None

	#--------------------
	font_size = 64
	min_font_size, max_font_size = int(font_size * 0.8), int(font_size * 1.25)
	min_char_space_ratio, max_char_space_ratio = 0.8, 1.2
	color_functor = functools.partial(generate_font_colors, image_depth=3)

	characterTransformer = tg_util.IdentityTransformer()
	#characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterPositioner = tg_util.MyCharacterPositioner()
	textGenerator = tg_util.MySimpleTextAlphaMatteGenerator(characterTransformer, characterPositioner, font_list=font_list, handwriting_dict=handwriting_dict, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio), alpha_matte_mode='1')

	batch_size = 1024
	generator = textGenerator.create_generator(word_set, batch_size, color_functor)

	#--------------------
	num_texts = 10000
	data_dir_path = './single_letters_{}'.format(num_texts)
	os.makedirs(data_dir_path, exist_ok=True)

	print('Start generating single letter data in {}...'.format(data_dir_path))
	start_time = time.time()
	is_finished = False
	idx = 0
	for text_list, scene_list, scene_text_mask_list in generator:
		for text, scene, scene_text_mask in zip(text_list, scene_list, scene_text_mask_list):
			cv2.imwrite(os.path.join(data_dir_path, '{}_{}.jpg'.format(text, idx)), scene)
			#cv2.imwrite(os.path.join(data_dir_path, '{}_{}_mask.jpg'.format(text, idx)), scene_text_mask)
			idx += 1
			if idx >= num_texts:
				is_finished = True
				break
		if is_finished:
			break
	print('End generating scene text data: {} secs.'.format(time.time() - start_time))

def main():
	#generate_random_word_set_test()
	#generate_repetitive_word_set_test()

	#generate_font_list_test()
	#generate_hangeul_font_list_test()

	#--------------------
	basic_printed_text_generator_test()

	#generate_basic_text_lines_test()

	#--------------------
	#text_alpha_matte_generator_test()
	#scene_text_alpha_matte_generator_test()

	#generate_alpha_matte_text_lines_test_1()
	#generate_alpha_matte_text_lines_test_2()
	#generate_alpha_matte_scene_texts_test()

	#--------------------
	# Application.

	#generate_hangeul_synthetic_scene_text_dataset()

	#generate_single_letter_dataset()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
