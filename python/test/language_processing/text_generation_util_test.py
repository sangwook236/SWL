#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, json
import numpy as np
import cv2
from swl.util.util import make_dir
import text_generation_util as tg_util

def generate_random_word_set_test():
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#data = fd.readlines()  # A string.
		#data = fd.read().strip('\n')  # A list of strings.
		#data = fd.read().splitlines()  # A list of strings.
		data = fd.read().replace(' ', '').replace('\n', '')  # A string.
	count = 80
	hangeul_charset = str()
	for idx in range(0, len(data), count):
		txt = ''.join(data[idx:idx+count])
		#hangeul_charset += ('' if 0 == idx else '\n') + txt
		hangeul_charset += txt
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	#print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
	#print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
	#print('Digit charset =', len(digit_charset), digit_charset)
	#print('Symbol charset =', len(symbol_charset), symbol_charset)

	num_words = 10
	min_char_count, max_char_count = 2, 10
	word_set = tg_util.generate_random_word_set(num_words, hangeul_charset, min_char_count, max_char_count)
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, alphabet_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, digit_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_random_word_set(num_words, symbol_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, hangeul_charset + digit_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, hangeul_charset + symbol_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, alphabet_charset + digit_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, alphabet_charset + symbol_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, hangeul_charset + alphabet_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, hangeul_charset + alphabet_charset + digit_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_random_word_set(num_words, hangeul_charset + alphabet_charset + symbol_charset, min_char_count, max_char_count))
	
	print('#generated word set =', len(word_set))
	print('Generated word set =', word_set)

def generate_repetitive_word_set_test():
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#data = fd.readlines()  # A string.
		#data = fd.read().strip('\n')  # A list of strings.
		#data = fd.read().splitlines()  # A list of strings.
		data = fd.read().replace(' ', '').replace('\n', '')  # A string.
	count = 80
	hangeul_charset = str()
	for idx in range(0, len(data), count):
		txt = ''.join(data[idx:idx+count])
		#hangeul_charset += ('' if 0 == idx else '\n') + txt
		hangeul_charset += txt
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	#print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
	#print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
	#print('Digit charset =', len(digit_charset), digit_charset)
	#print('Symbol charset =', len(symbol_charset), symbol_charset)

	num_char_repetitions = 2
	min_char_count, max_char_count = 2, 10
	word_set = tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset, min_char_count, max_char_count)
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, alphabet_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, digit_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, symbol_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + digit_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + symbol_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, alphabet_charset + digit_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, alphabet_charset + symbol_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + alphabet_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + alphabet_charset + digit_charset, min_char_count, max_char_count))
	word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + alphabet_charset + symbol_charset, min_char_count, max_char_count))
	
	print('#generated word set =', len(word_set))
	print('Generated word set =', word_set)

def text_generator_test():
	font_size = 32
	#font_color = (random.randint(0, 255),) * 3  # Uses a random font color.
	font_color = None  # Uses random font colors.
	char_space_ratio = 1.2

	characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	char_alpha_list, char_alpha_coordinate_list = textGenerator('가나다라마바사아자차카타파하', char_space_ratio, font_size)
	text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=None)

	#--------------------
	# No background.
	if 'posix' == os.name:
		cv2.imwrite('./text_line.png', text_line)
	else:
		cv2.imshow('Text line', text_line)

		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	#--------------------
	sceneTextGenerator = tg_util.MySceneTextGenerator(tg_util.IdentityTransformer())

	# Grayscale background.
	bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
	scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])

	if 'posix' == os.name:
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

def scene_text_generator_test():
	#font_color = (random.randint(0, 255),) * 3  # Uses a random font color.
	font_color = None  # Uses random font colors.

	characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	#--------------------
	texts, text_alphas = list(), list()

	char_alpha_list, char_alpha_coordinate_list = textGenerator('가나다라마바사아자차카타파하', char_space_ratio=0.9, font_size=32)
	text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#char_alpha_list, char_alpha_coordinate_list = textGenerator('ABCDEFGHIJKLMNOPQRSTUVWXYZ', char_space_ratio=1.6, font_size=24)
	char_alpha_list, char_alpha_coordinate_list = textGenerator('ABCDEFGHIJKLM', char_space_ratio=1.6, font_size=24)
	text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#char_alpha_list, char_alpha_coordinate_list = textGenerator('abcdefghijklmnopqrstuvwxyz', char_space_ratio=2.0, font_size=16)
	char_alpha_list, char_alpha_coordinate_list = textGenerator('abcdefghijklm', char_space_ratio=2.0, font_size=16)
	text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#--------------------
	if True:
		textTransformer = tg_util.PerspectiveTransformer()
	else:
		textTransformer = tg_util.ProjectiveTransformer()
	sceneTextGenerator = tg_util.MySceneTextGenerator(textTransformer)

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
	if 'posix' == os.name:
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

def generate_simple_text_lines_test():
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
	characterTransformer = tg_util.IdentityTransformer()
	#characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MySimplePrintedHangeulTextGenerator(characterTransformer, characterAlphaMattePositioner)

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2

	#font_color = (255, 255, 255)
	#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
	font_color = None  # Uses random font colors.
	#bg_color = (0, 0, 0)
	bg_color = None  # Uses random colors.

	batch_size = 4
	generator = tg_util.generate_text_lines(word_set, textGenerator, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), batch_size, font_color, bg_color)

	#--------------------
	step = 1
	for scene_list, scene_text_mask_list in generator:
		for scene, scene_text_mask in zip(scene_list, scene_text_mask_list):
			if 'posix' == os.name:
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

def generate_text_lines_test():
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
	characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	#--------------------
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2

	#font_color = (255, 255, 255)
	#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
	font_color = None  # Uses random font colors.
	#bg_color = (0, 0, 0)
	bg_color = None  # Uses random colors.

	batch_size = 4
	generator = tg_util.generate_text_lines(word_set, textGenerator, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), batch_size, font_color, bg_color)

	#--------------------
	step = 1
	for scene_list, scene_text_mask_list in generator:
		for scene, scene_text_mask in zip(scene_list, scene_text_mask_list):
			if 'posix' == os.name:
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

def generate_scene_texts_test():
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
	characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	#--------------------
	if True:
		textTransformer = tg_util.PerspectiveTransformer()
	else:
		textTransformer = tg_util.ProjectiveTransformer()
	sceneTextGenerator = tg_util.MySceneTextGenerator(textTransformer)

	if True:
		sceneProvider = tg_util.MySceneProvider()
	else:
		# Grayscale background.
		scene_shape = (800, 1000, 3)  # Some handwritten characters have 3 channels.
		sceneProvider = tg_util.MyGrayscaleBackgroundProvider(scene_shape)

	#--------------------
	min_text_count_per_image, max_text_count_per_image = 2, 10
	min_font_size, max_font_size = 15, 30
	min_char_space_ratio, max_char_space_ratio = 0.8, 2

	#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
	font_color = None  # Uses random font colors.

	batch_size = 4
	generator = tg_util.generate_scene_texts(word_set, sceneTextGenerator, sceneProvider, textGenerator, (min_text_count_per_image, max_text_count_per_image), (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), batch_size, font_color)

	#--------------------
	step = 1
	for scene_list, scene_text_mask_list, bboxes_list in generator:
		for scene, scene_text_mask, bboxes in zip(scene_list, scene_text_mask_list, bboxes_list):
			if 'posix' == os.name:
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

	make_dir(dir_path)
	make_dir(scene_dir_path)
	make_dir(mask_dir_path)

	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#data = fd.readlines()  # A string.
		#data = fd.read().strip('\n')  # A list of strings.
		#data = fd.read().splitlines()  # A list of strings.
		data = fd.read().replace(' ', '').replace('\n', '')  # A string.
	count = 80
	hangeul_charset = str()
	for idx in range(0, len(data), count):
		txt = ''.join(data[idx:idx+count])
		#hangeul_charset += ('' if 0 == idx else '\n') + txt
		hangeul_charset += txt
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

			#font_color = (random.randint(0, 255),) * 3  # Uses a random text color.
			font_color = None  # Uses random font colors.

			num_chars_per_text = random.randint(min_char_count_per_text, max_char_count_per_text)

			charset_selection_ratio = random.uniform(0.0, 1.0)
			for charset_idx, ratio in enumerate(charset_selection_ratios):
				if charset_selection_ratio < ratio:
					break

			charset = charset_list[charset_idx]
			charset_len = len(charset)
			text = ''.join(list(charset[random.randrange(charset_len)] for _ in range(num_chars_per_text)))

			char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space_ratio=char_space_ratio, font_size=font_size)
			text_line_image, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)

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
	characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = tg_util.IdentityTransformer()
	characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	#--------------------
	if True:
		textTransformer = tg_util.PerspectiveTransformer()
	else:
		textTransformer = tg_util.ProjectiveTransformer()
	sceneTextGenerator = tg_util.MySceneTextGenerator(textTransformer)

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
	num_images = 50000
	generate_scene_text_dataset(scene_text_dataset_dir_path, scene_text_dataset_json_filename, sceneTextGenerator, sceneProvider, textGenerator, num_images)

	# Load a scene dataset.
	image_filepaths, mask_filepaths, gt_texts, gt_boxes = load_scene_text_dataset(scene_text_dataset_dir_path, scene_text_dataset_json_filename)
	print('Generated scene dataset: #images = {}, #masks = {}, #texts = {}, #boxes = {}.'.format(len(image_filepaths), len(mask_filepaths), len(gt_texts), len(gt_boxes)))

def main():
	#generate_random_word_set_test()
	#generate_repetitive_word_set_test()

	#text_generator_test()
	#scene_text_generator_test()

	# Application.
	#generate_hangeul_synthetic_scene_text_dataset()

	# Create a text generator.
	#generate_simple_text_lines_test()
	#generate_text_lines_test()
	generate_scene_texts_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
