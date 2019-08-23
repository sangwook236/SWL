#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os
import numpy as np
import cv2
from swl.language_processing.util import generate_text_image, draw_text_on_image

def hangul_example():
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF8') as fd:
		#hangul_str = fd.readlines()  # A string.
		#hangul_str = fd.read().strip('\n')  # A list of strings.
		#hangul_str = fd.read().splitlines()  # A list of strings.
		hangul_str = fd.read().replace(' ', '').replace('\n', '')  # A string.
	#print(hangul_str)

	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/gulim.ttf'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_type = '/usr/share/fonts/truetype/batang.ttf'  # 바탕, 바탕체, 궁서, 궁서체.
	else:
		font_type = 'C:/Windows/Fonts/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
	font_index = 1

	img = generate_text_image(hangul_str, font_type=font_type, font_index=font_index, font_size=30, font_color=(0, 0, 0), bg_color=(240, 240, 240), image_size=(1500, 1000), text_offset=None, crop_text_area=True, draw_text_border=False)
	img.save('./hangul.png')

def alphabet_example():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/Arial.ttf'
	else:
		font_type = 'C:/Windows/Fonts/Arial.ttf'
	font_index = 0

	alphabet_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	img = generate_text_image(alphabet_str, font_type=font_type, font_index=font_index, font_size=30, font_color=(0, 0, 0), bg_color=(240, 240, 240), image_size=(1000, 40), text_offset=None, crop_text_area=True, draw_text_border=False)
	img.save('./alphabet.png')

def digit_example():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/Arial.ttf'
	else:
		font_type = 'C:/Windows/Fonts/Arial.ttf'
	font_index = 0

	digit_str = '0123456789'
	img = generate_text_image(digit_str, font_type=font_type, font_index=font_index, font_size=30, font_color=(0, 0, 0), bg_color=(240, 240, 240), image_size=(200, 40), text_offset=None, crop_text_area=True, draw_text_border=False)
	img.save('./digit.png')

def punctuation_example():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/Arial.ttf'
	else:
		font_type = 'C:/Windows/Fonts/Arial.ttf'
	font_index = 0

	punctuation_str = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'
	img = generate_text_image(punctuation_str, font_type=font_type, font_index=font_index, font_size=30, font_color=(0, 0, 0), bg_color=(240, 240, 240), image_size=(500, 40), text_offset=None, crop_text_area=True, draw_text_border=False)
	img.save('./punctuation.png')

def symbol_example():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/Arial.ttf'
	else:
		font_type = 'C:/Windows/Fonts/Arial.ttf'
	font_index = 0

	symbol_str = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'
	img = generate_text_image(symbol_str, font_type=font_type, font_index=font_index, font_size=30, font_color=(0, 0, 0), bg_color=(240, 240, 240), image_size=(500, 40), text_offset=None, crop_text_area=True, draw_text_border=False)
	img.save('./symbol.png')

def all_font_example():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/gulim.ttf'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_type = '/usr/share/fonts/truetype/batang.ttf'  # 바탕, 바탕체, 궁서, 궁서체.
	else:
		font_type = 'C:/Windows/Fonts/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
	font_index = 1

	all_text = """(학교] school is 178 좋34,지."""
	img = generate_text_image(all_text, font_type=font_type, font_index=font_index, font_size=30, font_color=(0, 0, 0), bg_color=(240, 240, 240), image_size=(500, 40), text_offset=None, crop_text_area=True, draw_text_border=False)
	img.save('./all_text.png')

def generate_text_image_test():
	hangul_example()
	alphabet_example()
	digit_example()
	punctuation_example()
	#symbol_example()  # Not yet implemented.
	all_font_example()

def draw_text_on_image_test():
	if 'posix' == os.name:
		font_type = '/usr/share/fonts/truetype/gulim.ttf'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_type = '/usr/share/fonts/truetype/batang.ttf'  # 바탕, 바탕체, 궁서, 궁서체.
	else:
		font_type = 'C:/Windows/Fonts/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
	font_index = 0

	is_white_background = False  # Uses white or black background.
	font_size = 32
	#font_color = None  # Uses random colors.
	#font_color = (0, 0, 0) if is_white_background else (255, 255, 255)
	#font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	font_color = (255, 255, 255)
	bg_color = (0, 0, 0)
	text_offset = (0, 0)
	crop_text_area = True
	draw_text_border = False
	image_size = (500, 500)

	text = generate_text_image('가나다라마바사아자차카타파하', font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)
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

	scene2, text_mask, text_bbox = draw_text_on_image(scene2, '가나다라마바사아자차카타파하', font_type, font_index, font_size, font_color, text_offset=text_offset, rotation_angle=rotation_angle)

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

def main():
	#generate_text_image_test()
	draw_text_on_image_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
