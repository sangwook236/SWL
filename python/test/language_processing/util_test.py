#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, glob, time, string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import swl.util.util as swl_util
import swl.language_processing.util as swl_langproc_util

def compute_simple_text_recognition_accuracy_test():
	inferred_texts     = ['abc', 'df',  'ghijk', 'ab cde', 'fhijk lmno', 'pqrst uvwy', 'abc defg hijklmn opqr', 'abc deg hijklmn opqr', 'abc df hijklmn opqr', '',    'zyx']
	ground_truth_texts = ['abc', 'def', 'gijk',  'ab cde', 'fghijk lmo', 'pqst uvwxy', 'abc defg hijklmn opqr', 'abc defg hiklmn opqr', 'abc defg hijklmn pr', 'xyz', '']
	# #texts = 11, #words = 23, #characters = 103.

	print('[SWL] Info: Start computing text recognition accuracy...')
	start_time = time.time()
	correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = swl_langproc_util.compute_simple_text_recognition_accuracy(inferred_texts, ground_truth_texts)
	print('[SWL] Info: End computing text recognition accuracy: {} secs.'.format(time.time() - start_time))
	print('\tText accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count))
	print('\tWord accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
	print('\tChar accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

def compute_string_distance_test():
	inferred_texts     = ['abc', 'df',  'ghijk', 'ab cde', 'fhijk lmno', 'pqrst uvwy', 'abc defg hijklmn opqr', 'abc deg hijklmn opqr', 'abc df hijklmn opqr', '',    'zyx']
	ground_truth_texts = ['abc', 'def', 'gijk',  'ab cde', 'fghijk lmo', 'pqst uvwxy', 'abc defg hijklmn opqr', 'abc defg hiklmn opqr', 'abc defg hijklmn pr', 'xyz', '']
	# #texts = 11, #words = 23, #characters = 103.

	print('[SWL] Info: Start computing text recognition accuracy...')
	start_time = time.time()
	text_distance, word_distance, char_distance, total_text_count, total_word_count, total_char_count = swl_langproc_util.compute_string_distance(inferred_texts, ground_truth_texts)
	print('[SWL] Info: End computing text recognition accuracy: {} secs.'.format(time.time() - start_time))
	print('\tText: Distance = {0}, average distance = {0} / {1} = {2}.'.format(text_distance, total_text_count, text_distance / total_text_count))
	print('\tWord: Distance = {0}, average distance = {0} / {1} = {2}.'.format(word_distance, total_word_count, word_distance / total_word_count))
	print('\tChar: Distance = {0}, average distance = {0} / {1} = {2}.'.format(char_distance, total_char_count, char_distance / total_char_count))

def hangeul_example(need_mask=False):
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF8') as fd:
		#hangeul_str = fd.read().replace(' ', '').strip('\n')  # A string.
		hangeul_str = fd.read().replace(' ', '').rstrip()  # A string.
		#hangeul_str = fd.readlines()  # A list of strings.
		#hangeul_str = fd.read().replace(' ', '').splitlines()  # A list of strings.
	#print(hangeul_str)

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'

	font_type = os.path.join(font_dir_path, 'gulim.ttf')
	font_index = 1

	font_size = 30
	#font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	font_color = None  # Uses a random font color.
	#bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	bg_color = None  # Uses a random background color.
	#image_size = (1000, 3000)  # (width, height).
	image_size = None
	#char_space_ratio = 0.8
	char_space_ratio = None

	if need_mask:
		img, msk = swl_langproc_util.generate_text_image(hangeul_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mask=True)
		img.save('./hangeul_text.png')
		msk.save('./hangeul_mask.png')
	else:
		img = swl_langproc_util.generate_text_image(hangeul_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio)
		img.save('./hangeul.png')

def alphabet_example(need_mask=False):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/eng'

	font_type = os.path.join(font_dir_path, 'arial.ttf')
	font_index = 0

	font_size = 30
	#font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	font_color = None  # Uses a random font color.
	#bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	bg_color = None  # Uses a random background color.
	#image_size = (1000, 40)  # (width, height).
	image_size = None
	#char_space_ratio = 0.8
	char_space_ratio = None

	#alphabet_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	alphabet_str = string.ascii_letters
	#alphabet_str = string.ascii_lowercase
	#alphabet_str = string.ascii_uppercase
	if need_mask:
		img, msk = swl_langproc_util.generate_text_image(alphabet_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mask=True)
		img.save('./alphabet_text.png')
		msk.save('./alphabet_mask.png')
	else:
		img = swl_langproc_util.generate_text_image(alphabet_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio)
		img.save('./alphabet.png')

def digit_example(need_mask=False):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/eng'

	font_type = os.path.join(font_dir_path, 'arial.ttf')
	font_index = 0

	font_size = 30
	#font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	font_color = None  # Uses a random font color.
	#bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	bg_color = None  # Uses a random background color.
	#image_size = (200, 40)  # (width, height).
	image_size = None
	#char_space_ratio = 0.8
	char_space_ratio = None

	#digit_str = '0123456789'
	digit_str = string.digits
	#digit_str = string.hexdigits
	#digit_str = string.octdigits
	if need_mask:
		img, msk = swl_langproc_util.generate_text(digit_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mask=True)
		img.save('./digit_text.png')
		msk.save('./digit_mask.png')
	else:
		img = swl_langproc_util.generate_text_image(digit_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio)
		img.save('./digit.png')

def punctuation_example(need_mask=False):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/eng'

	font_type = os.path.join(font_dir_path, 'arial.ttf')
	font_index = 0

	font_size = 30
	#font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	font_color = None  # Uses a random font color.
	#bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	bg_color = None  # Uses a random background color.
	#image_size = (500, 40)  # (width, height).
	image_size = None
	#char_space_ratio = 0.8
	char_space_ratio = None

	#punctuation_str = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'
	punctuation_str = string.punctuation
	if need_mask:
		img, msk = swl_langproc_util.generate_text_image(punctuation_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mask=True)
		img.save('./punctuation_text.png')
		msk.save('./punctuation_mask.png')
	else:
		img = swl_langproc_util.generate_text_image(punctuation_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio)
		img.save('./punctuation.png')

def symbol_example(need_mask=False):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/eng'

	font_type = os.path.join(font_dir_path, 'arial.ttf')
	font_index = 0

	font_size = 30
	#font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	font_color = None  # Uses a random font color.
	#bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	bg_color = None  # Uses a random background color.
	#image_size = (500, 40)  # (width, height).
	image_size = None
	#char_space_ratio = 0.8
	char_space_ratio = None

	symbol_str = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'
	if need_mask:
		img, msk = swl_langproc_util.generate_text_image(symbol_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mask=True)
		img.save('./symbol_text.png')
		msk.save('./symbol_mask.png')
	else:
		img = swl_langproc_util.generate_text_image(symbol_str, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio)
		img.save('./symbol.png')

def mixed_example(need_mask=False):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'

	font_type = os.path.join(font_dir_path, 'gulim.ttf')
	font_index = 1

	font_size = 30
	#font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	font_color = None  # Uses a random font color.
	#bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	bg_color = None  # Uses a random background color.
	#image_size = (500, 40)  # (width, height).
	image_size = None
	#char_space_ratio = 0.8
	char_space_ratio = None

	mixed_text = """(학교] school is 178 좋34,지."""
	#mixed_text = string.printable
	#mixed_text = string.whitespace
	if need_mask:
		img, msk = swl_langproc_util.generate_text_image(mixed_text, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mask=True)
		img.save('./mixed_text.png')
		msk.save('./mixed_mask.png')
	else:
		img = swl_langproc_util.generate_text_image(mixed_text, font_type=font_type, font_index=font_index, font_size=font_size, font_color=font_color, bg_color=bg_color, image_size=image_size, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio)
		img.save('./mixed.png')

def generate_text_image_test():
	hangeul_example(need_mask=False)
	alphabet_example(need_mask=False)
	digit_example(need_mask=False)
	punctuation_example(need_mask=False)
	#symbol_example(need_mask=False)  # Not yet implemented.
	mixed_example(need_mask=False)

def generate_text_image_and_mask_test():
	hangeul_example(need_mask=True)
	alphabet_example(need_mask=True)
	digit_example(need_mask=True)
	punctuation_example(need_mask=True)
	#symbol_example(need_mask=True)  # Not yet implemented.
	mixed_example(need_mask=True)

def draw_text_on_image_test():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/gulim.ttf'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_type = '/usr/share/fonts/truetype/batang.ttf'  # 바탕, 바탕체, 궁서, 궁서체.
	else:
		font_type = 'C:/Windows/Fonts/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
	font_index = 0

	is_white_background = False  # Uses white or black background.
	font_size = 32
	font_color = (255, 255, 255)
	#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
	#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
	#font_color = None  # Uses a random font color.
	bg_color = (0, 0, 0)
	#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
	#bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
	#bg_color = None  # Uses a random background color.
	text_offset = (0, 0)
	crop_text_area = True
	draw_text_border = False
	image_size = (500, 500)

	text = swl_langproc_util.generate_text_image('가나다라마바사아자차카타파하', font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)
	text = np.asarray(text, dtype=np.uint8)

	mask = np.zeros_like(text)
	mask[text > 0] = 255

	pixels = np.where(text > 0)
	#pixels = np.where(mask > 0)
	scene = np.full((500, 500, 3), 128, dtype=np.uint8)
	scene[:,:][pixels] = text[pixels]
	
	cv2.imshow('Text', text)
	cv2.imshow('Text Mask', mask)
	cv2.imshow('Scene', scene)

	scene2 = np.full((1000, 1000, 3), 128, dtype=np.uint8)
	#scene2 = cv2.imread('./image1.jpg')
	text_offset = (100, 300)
	rotation_angle = None

	scene2, text_mask, text_bbox = swl_langproc_util.draw_text_on_image(scene2, '가나다라마바사아자차카타파하', font_type, font_index, font_size, font_color, text_offset=text_offset, rotation_angle=rotation_angle)

	text_bbox = np.round(text_bbox).astype(np.int)
	#cv2.drawContours(scene2, [text_bbox], 0, (0, 0, 255), 1, cv2.LINE_8)
	cv2.line(scene2, tuple(text_bbox[0]), tuple(text_bbox[1]), (255, 0, 0), 1, cv2.LINE_8)
	cv2.line(scene2, tuple(text_bbox[1]), tuple(text_bbox[2]), (0, 255, 0), 1, cv2.LINE_8)
	cv2.line(scene2, tuple(text_bbox[2]), tuple(text_bbox[3]), (0, 0, 255), 1, cv2.LINE_8)
	cv2.line(scene2, tuple(text_bbox[3]), tuple(text_bbox[0]), (255, 0, 255), 1, cv2.LINE_8)

	cv2.imshow('Scene2', scene2)
	cv2.imshow('Scene2 Text Mask', text_mask)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def generate_text_mask_and_distribution_test():
	if 'posix' == os.name:
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		font_base_dir_path = '/work/font'
	font_dir_path = font_base_dir_path + '/kor'

	font_type = font_dir_path + '/gulim.ttf'
	font_index = 0

	image_shape = (800, 1500)
	mask = np.zeros(image_shape, dtype=np.uint8)
	pdf = np.zeros(image_shape, dtype=np.float32)
	rgb = np.zeros(image_shape + (3,), dtype=np.uint8)

	import string
	text = string.ascii_letters
	font_size = 32
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_offset = (100, 200)  # (x, y).
	rotation_angle = -15  # [deg].
	text_mask, text_pdf = swl_langproc_util.generate_text_mask_and_distribution(text, font, rotation_angle)
	text_bbox = (text_offset[0], text_offset[1], text_offset[0] + text_mask.shape[1], text_offset[1] + text_mask.shape[0])
	mask = swl_util.add_mask(mask, text_mask, text_bbox)
	pdf = swl_util.add_pdf(pdf, text_pdf, text_bbox)
	font_color = (127, 63, 255)  # RGB.
	rgb = swl_util.draw_using_mask(rgb, text_mask, font_color, text_bbox)

	text = string.digits
	font_size = 64
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_offset = (800, 650)  # (x, y).
	rotation_angle = -5  # [deg].
	text_mask, text_pdf = swl_langproc_util.generate_text_mask_and_distribution(text, font, rotation_angle)
	text_bbox = (text_offset[0], text_offset[1], text_offset[0] + text_mask.shape[1], text_offset[1] + text_mask.shape[0])
	mask = swl_util.add_mask(mask, text_mask, text_bbox)
	pdf = swl_util.add_pdf(pdf, text_pdf, text_bbox)
	font_color = (255, 0, 0)  # RGB.
	rgb = swl_util.draw_using_mask(rgb, text_mask, font_color, text_bbox)

	text = string.punctuation
	font_size = 32
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_offset = (500, 250)  # (x, y).
	rotation_angle = 0  # [deg].
	text_mask, text_pdf = swl_langproc_util.generate_text_mask_and_distribution(text, font, rotation_angle)
	text_bbox = (text_offset[0], text_offset[1], text_offset[0] + text_mask.shape[1], text_offset[1] + text_mask.shape[0])
	mask = swl_util.add_mask(mask, text_mask, text_bbox)
	pdf = swl_util.add_pdf(pdf, text_pdf, text_bbox)
	font_color = (0, 255, 0)  # RGB.
	rgb = swl_util.draw_using_mask(rgb, text_mask, font_color, text_bbox)

	text = '가나다라마바사자차카타파하'
	font_size = 64
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_offset = (300, 400)  # (x, y).
	rotation_angle = 10  # [deg].
	text_mask, text_pdf = swl_langproc_util.generate_text_mask_and_distribution(text, font, rotation_angle)
	text_bbox = (text_offset[0], text_offset[1], text_offset[0] + text_mask.shape[1], text_offset[1] + text_mask.shape[0])
	mask = swl_util.add_mask(mask, text_mask, text_bbox)
	pdf = swl_util.add_pdf(pdf, text_pdf, text_bbox)
	font_color = (0, 0, 255)  # RGB.
	rgb = swl_util.draw_using_mask(rgb, text_mask, font_color, text_bbox)

	#--------------------
	fig = plt.figure(tight_layout=True)

	ax = fig.add_subplot(221)
	ax.imshow(mask, cmap='gray', aspect='equal')
	ax.set_title('Text Mask')
	ax.axis('off')

	ax = fig.add_subplot(222)
	ax.imshow(pdf, cmap='Reds', aspect='equal')
	ax.set_title('Text Distribution')
	ax.axis('off')

	ax = fig.add_subplot(223)
	text_pdf_blended = 0.5 * pdf / np.max(pdf) + 0.5 * mask / 255
	ax.imshow(text_pdf_blended, cmap='gray', aspect='equal')
	ax.set_title('Text and its Distribution Overlay')
	ax.axis('off')

	ax = fig.add_subplot(224)
	ax.imshow(rgb, aspect='equal')
	ax.set_title('Colored Text')
	ax.axis('off')

	plt.show()

def transform_text_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_dir_path = '/home/sangwook/work/font/eng'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_dir_path = 'D:/work/font/eng'

	#os.path.sep = '/'
	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))

	font_list = list()
	for fpath in font_filepaths:
		num_fonts = 4 if os.path.basename(fpath).lower() in ['gulim.ttf', 'batang.ttf'] else 1

		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	#--------------------
	texts = ['bd', 'ace', 'ABC', 'defghijklmn', 'opqrSTU']
	text = ' '.join(texts)
	tx, ty = 500, 500  # Translation. [pixel].
	rotation_angle = 15  # [deg].

	paper_height, paper_width, paper_channel = 1000, 1000, 3

	font_type, font_index = font_list[0]
	font_size = 32
	if True:
		font_color = (255,) * 3
		bg_color = 0
	else:
		# NOTE [warning] >> Invalid text masks are generated.
		font_color = (0,) * 3
		bg_color = 255

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	#--------------------
	img = np.full((paper_height, paper_width, paper_channel), bg_color, dtype=np.uint8)

	#text_bbox = swl_langproc_util.transform_text(text, tx, ty, rotation_angle, font)
	text_bbox, img, text_mask = swl_langproc_util.transform_text_on_image(text, tx, ty, rotation_angle, img, font, font_color, bg_color)

	#--------------------
	bbox_img = np.zeros_like(img)

	text_bbox = np.round(text_bbox).astype(np.int)
	if False:
		cv2.drawContours(bbox_img, [text_bbox], 0, (0, 0, 255), 1, cv2.LINE_8)
	else:
		cv2.drawContours(bbox_img, [text_bbox], 0, (0, 0, 255), cv2.FILLED)
	#if np.linalg.norm(text_bbox[0] - text_bbox[1]) > np.linalg.norm(text_bbox[2] - text_bbox[1]):
	if abs(text_bbox[0][0] - text_bbox[1][0]) > abs(text_bbox[2][0] - text_bbox[1][0]):
		cv2.line(bbox_img, tuple(text_bbox[0]), tuple(text_bbox[1]), (255, 0, 0), 2, cv2.LINE_8)
		cv2.line(bbox_img, tuple(text_bbox[2]), tuple(text_bbox[3]), (255, 0, 0), 2, cv2.LINE_8)
	else:
		cv2.line(bbox_img, tuple(text_bbox[1]), tuple(text_bbox[2]), (255, 0, 0), 2, cv2.LINE_8)
		cv2.line(bbox_img, tuple(text_bbox[3]), tuple(text_bbox[0]), (255, 0, 0), 2, cv2.LINE_8)

	cv2.circle(img, (tx, ty), 3, (255, 0, 255), cv2.FILLED)
	cv2.circle(text_mask, (tx, ty), 3, (255, 0, 255), cv2.FILLED)
	cv2.circle(bbox_img, (tx, ty), 3, (255, 0, 255), cv2.FILLED)

	cv2.imshow('Image (Text)', img)
	cv2.imshow('Text Mask (Text)', text_mask)
	cv2.imshow('Text Bounding Box Image (Text)', bbox_img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def transform_texts_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_dir_path = '/home/sangwook/work/font/eng'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_dir_path = 'D:/work/font/eng'

	#os.path.sep = '/'
	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))

	font_list = list()
	for fpath in font_filepaths:
		num_fonts = 4 if os.path.basename(fpath).lower() in ['gulim.ttf', 'batang.ttf'] else 1

		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	#--------------------
	texts = ['bd', 'ace', 'ABC', 'defghijklmn', 'opqrSTU']
	tx, ty = 500, 500  # Translation. [pixel].
	rotation_angle = 15  # [deg].

	paper_height, paper_width, paper_channel = 1000, 1000, 3

	font_type, font_index = font_list[0]
	font_size = 32
	if True:
		font_color = (255,) * 3
		bg_color = 0
	else:
		# NOTE [warning] >> Invalid text masks are generated.
		font_color = (0,) * 3
		bg_color = 255

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	#--------------------
	img = np.full((paper_height, paper_width, paper_channel), bg_color, dtype=np.uint8)

	#text_bboxes = swl_langproc_util.transform_texts(texts, tx, ty, rotation_angle, font)
	text_bboxes, img, text_mask = swl_langproc_util.transform_texts_on_image(texts, tx, ty, rotation_angle, img, font, font_color, bg_color)

	#--------------------
	bbox_img = np.zeros_like(img)

	pt1, pt4 = text_bboxes[0][0] + 0.25 * (text_bboxes[0][3] - text_bboxes[0][0]), text_bboxes[0][0] + 0.75 * (text_bboxes[0][3] - text_bboxes[0][0])
	pt2, pt3 = text_bboxes[-1][1] + 0.25 * (text_bboxes[-1][2] - text_bboxes[-1][1]), text_bboxes[-1][1] + 0.75 * (text_bboxes[-1][2] - text_bboxes[-1][1])
	cv2.drawContours(bbox_img, [np.round([pt1, pt2, pt3, pt4]).astype(np.int)], 0, (0, 255, 0), cv2.FILLED)

	text_bboxes = np.round(text_bboxes).astype(np.int)
	for text_bbox in text_bboxes:
		if False:
			cv2.drawContours(bbox_img, [text_bbox], 0, (0, 0, 255), 1, cv2.LINE_8)
		else:
			cv2.drawContours(bbox_img, [text_bbox], 0, (0, 0, 255), cv2.FILLED)
		#if np.linalg.norm(text_bbox[0] - text_bbox[1]) > np.linalg.norm(text_bbox[2] - text_bbox[1]):
		if abs(text_bbox[0][0] - text_bbox[1][0]) > abs(text_bbox[2][0] - text_bbox[1][0]):
			cv2.line(bbox_img, tuple(text_bbox[0]), tuple(text_bbox[1]), (255, 0, 0), 2, cv2.LINE_8)
			cv2.line(bbox_img, tuple(text_bbox[2]), tuple(text_bbox[3]), (255, 0, 0), 2, cv2.LINE_8)
		else:
			cv2.line(bbox_img, tuple(text_bbox[1]), tuple(text_bbox[2]), (255, 0, 0), 2, cv2.LINE_8)
			cv2.line(bbox_img, tuple(text_bbox[3]), tuple(text_bbox[0]), (255, 0, 0), 2, cv2.LINE_8)

	cv2.circle(img, (tx, ty), 3, (255, 0, 255), cv2.FILLED)
	cv2.circle(text_mask, (tx, ty), 3, (255, 0, 255), cv2.FILLED)
	cv2.circle(bbox_img, (tx, ty), 3, (255, 0, 255), cv2.FILLED)

	cv2.imshow('Image (Texts)', img)
	cv2.imshow('Text Mask (Texts)', text_mask)
	cv2.imshow('Text Bounding Box Image (Texts)', bbox_img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def transform_text_line_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_dir_path = '/home/sangwook/work/font/eng'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_dir_path = 'D:/work/font/eng'

	#os.path.sep = '/'
	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))

	font_list = list()
	for fpath in font_filepaths:
		num_fonts = 4 if os.path.basename(fpath).lower() in ['gulim.ttf', 'batang.ttf'] else 1

		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	#--------------------
	texts = ['bd', 'ace', 'ABC', 'defghijklmn', 'opqrSTU']
	text = ' '.join(texts)
	tx, ty = 500, 500  # Translation. [pixel].
	rotation_angle = 15  # [deg].

	paper_height, paper_width, paper_channel =  1000, 1000, 3

	font_type, font_index = font_list[0]
	font_size = 32
	if True:
		font_color = (255,) * 3
		bg_color = 0
	else:
		# NOTE [warning] >> Invalid text masks are generated.
		font_color = (0,) * 3
		bg_color = 255

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	#--------------------
	img = np.full((paper_height, paper_width, paper_channel), bg_color, dtype=np.uint8)
	bg_img = Image.fromarray(img)

	text_offset = (0, 0)  # (x, y).
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).
	text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

	#--------------------
	#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
	text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))
	sx0, sy0 = text_img.size

	text_draw = ImageDraw.Draw(text_img)
	text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

	text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.

	sx, sy = text_img.size
	bg_img.paste(text_img, (text_offset[0], text_offset[1], text_offset[0] + sx, text_offset[1] + sy), text_img)

	text_mask = Image.new('L', bg_img.size, (0,))
	text_mask.paste(text_img, (text_offset[0], text_offset[1], text_offset[0] + sx, text_offset[1] + sy), text_img)

	dx, dy = (sx0 - sx) / 2, (sy0 - sy) / 2
	x1, y1, x2, y2 = text_rect
	rect = (((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), -rotation_angle)
	text_bbox = cv2.boxPoints(rect)
	text_bbox = list(map(lambda xy: [xy[0] - dx, xy[1] - dy], text_bbox))

	#--------------------
	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)

	text_bbox = np.round(text_bbox).astype(np.int)
	
	bbox_img = np.zeros_like(img)
	if False:
		cv2.drawContours(bbox_img, [text_bbox], 0, (0, 0, 255), 1, cv2.LINE_8)
	else:
		cv2.drawContours(bbox_img, [text_bbox], 0, (0, 0, 255), cv2.FILLED)
	#if np.linalg.norm(text_bbox[0] - text_bbox[1]) > np.linalg.norm(text_bbox[2] - text_bbox[1]):
	if abs(text_bbox[0][0] - text_bbox[1][0]) > abs(text_bbox[2][0] - text_bbox[1][0]):
		cv2.line(bbox_img, tuple(text_bbox[0]), tuple(text_bbox[1]), (255, 0, 0), 2, cv2.LINE_8)
		cv2.line(bbox_img, tuple(text_bbox[2]), tuple(text_bbox[3]), (255, 0, 0), 2, cv2.LINE_8)
	else:
		cv2.line(bbox_img, tuple(text_bbox[1]), tuple(text_bbox[2]), (255, 0, 0), 2, cv2.LINE_8)
		cv2.line(bbox_img, tuple(text_bbox[3]), tuple(text_bbox[0]), (255, 0, 0), 2, cv2.LINE_8)
	
	cv2.imshow('Image (Text line)', img)
	cv2.imshow('Text Mask (Text line)', text_mask)
	cv2.imshow('Text Bounding Box Image (Text line)', bbox_img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	#compute_simple_text_recognition_accuracy_test()
	#compute_string_distance_test()

	#generate_text_image_test()
	#generate_text_image_and_mask_test()
	#draw_text_on_image_test()  # Not so good.

	generate_text_mask_and_distribution_test()

	# NOTE [info] >> Bounding boxes of transform_text() and transform_texts() are different.
	#	I don't know why.
	#transform_text_test()
	#transform_texts_test()

	#transform_text_line_test()  # Not so good.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
