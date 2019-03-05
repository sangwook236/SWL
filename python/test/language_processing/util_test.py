#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os
from swl.language_processing.util import generate_text_image

def hangul_example():
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF8') as fd:
		#data = fd.readlines()  # A string.
		#data = fd.read().strip('\n')  # A list of strings.
		#data = fd.read().splitlines()  # A list of strings.
		data = fd.read().replace(' ', '').replace('\n', '')  # A string.
	count = 80
	hangul_str = str()
	for idx in range(0, len(data), count):
		txt = ''.join(data[idx:idx+count])
		hangul_str += ('' if 0 == idx else '\n') + txt
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

def main():
	generate_text_image_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
