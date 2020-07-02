#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, math, random, glob, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import swl.language_processing.util as swl_langproc_util
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
		font_dir_paths.append(font_base_dir_path + '/kor_large')
		#font_dir_paths.append(font_base_dir_path + '/kor_receipt')
	if english:
		font_dir_paths.append(font_base_dir_path + '/eng_large')
		#font_dir_paths.append(font_base_dir_path + '/eng_receipt')

	return tg_util.construct_font(font_dir_paths)

def visualize_font_text(font_dir_path):
	if False:
		font_list = construct_font(korean=True, english=True)
	else:
		font_filepaths = glob.glob(font_dir_path + '/*.ttf')
		#font_list = tg_util.generate_font_list(font_filepaths)
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)

		"""
		gabia_solmee.ttf: 일부 자음 없음.
		godoRoundedL.ttf: ? 표시.
		godoRoundedR.ttf: ? 표시.
		HS여름물빛체.ttf: 위아래 글자 겹침.
		"""

	if True:
		text = tg_util.construct_charset(space=True)
		line_len = 27
		text = '\n'.join(text[i:i+line_len] for i in range(0, len(text), line_len))
	else:
		hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
		with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
			#hangeul_letters = fd.read().strip('\n')  # A string.
			hangeul_letters = fd.read().replace(' ', '')  # A string.
			#hangeul_letters = fd.readlines()  # A list of strings.
			#hangeul_letters = fd.read().splitlines()  # A list of strings.

		hangeul_consonants = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
		hangeul_vowels = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

		alphabet1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		alphabet2 = 'abcdefghijklmnopqrstuvwxyz'
		digits = '0123456789'
		symbols = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		text = alphabet1 + '\n' + \
			alphabet2 + '\n' + \
			digits + '\n' + \
			symbols + '\n' + \
			hangeul_consonants + '\n' + \
			hangeul_vowels + '\n' + \
			hangeul_letters

	print('Text =', text)

	text_offset = (0, 0)
	crop_text_area = False
	draw_text_border = False
	font_size = 32
	font_color = (255, 255, 255)
	#font_color = None
	bg_color = (0, 0, 0)
	#bg_color = None
	image_size = (1000, 4000)

	#--------------------
	output_dir_path = './font_test_results'
	os.makedirs(output_dir_path, exist_ok=True)

	invalid_fonts = list()
	for font_fpath, font_index in font_list:
		print('Font filepath: {}, font index: {}.'.format(os.path.basename(font_fpath), font_index))

		try:
			img = swl_langproc_util.generate_text_image(text, font_fpath, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)

			#img_fpath = os.path.join(output_dir_path, os.path.basename(font_fpath) + '_{}.png'.format(font_index))
			#print('\tSaved an font image to {}.'.format(img_fpath))
			#img.save(img_fpath)
			#cv2.imwrite(img_fpath, np.array(img))
			#cv2.imshow('Text', np.array(img))
			#cv2.waitKey(0)
		except:
			invalid_fonts.append((font_fpath, font_index))

	#cv2.destroyAllWindows()

	print('#invalid fonts = {}.'.format(len(invalid_fonts)))
	if len(invalid_fonts) > 0:
		print('Invalid fonts:')
		for font_fpath, font_index in invalid_fonts:
			print('\t{}: {}.'.format(font_fpath, font_index))
	else:
		print('No invalid fonts.')

def is_valid_font(font_filepath, char_pair, height=64, width=64):
	try:
		font = ImageFont.truetype(font_filepath, size=height - 6)
	except Exception as ex:
		print('Invalid font, {}: {}.'.format(font_filepath, ex))
		return False

	img1 = Image.new(mode='1', size=(width, height))
	draw = ImageDraw.Draw(img1)
	draw.text(xy=(3, 3), text=char_pair[0], font=font, fill=(1))
	img_data1 = list(img1.getdata())

	img2 = Image.new(mode='1', size=(width, height))
	draw = ImageDraw.Draw(img2)
	draw.text(xy=(3, 3), text=char_pair[1], font=font, fill=(1))
	img_data2 = list(img2.getdata())

	is_same = (img_data1 == img_data2)

	"""
	if is_same:
		img1.show(char_pair[0])
		img2.show(char_pair[1])
	"""

	return not is_same

def check_font_validity(font_dir_path, char_pair, dst_dir_path=None):
	print('Start checking font validity...')
	start_time = time.time()
	invalid_fonts = list()
	for font in os.listdir(font_dir_path):
		font_fpath = os.path.join(font_dir_path, font)
		#is_valid = is_valid_font(font_fpath)
		is_valid = is_valid_font(font_fpath, char_pair)
		if not is_valid:
			#os.remove(font_fpath)
			if dst_dir_path and os.path.isdir(dst_dir_path):
				os.rename(font_fpath, os.path.join(dst_dir_path, os.path.basename(font_fpath)))
			invalid_fonts.append(font_fpath)
	print('End checking font validity: {} secs.'.format(time.time() - start_time))

	print('#invalid fonts = {}.'.format(len(invalid_fonts)))
	if len(invalid_fonts) > 0:
		print('Invalid fonts:')
		for font_fpath in invalid_fonts:
			print('\t{}.'.format(font_fpath))
	else:
		print('No invalid fonts.')

def is_valid_font_by_area(font_filepath, ch, area_thresh=0.1, height=64, width=64):
	try:
		font = ImageFont.truetype(font_filepath, size=height - 6)
	except Exception as ex:
		print('Invalid font, {}: {}.'.format(font_filepath, ex))
		return False

	img = Image.new(mode='1', size=(width, height))
	draw = ImageDraw.Draw(img)
	draw.text(xy=(3, 3), text=ch, font=font, fill=(1))
	area = sum(img.getdata())  # #nonzero pixels.

	return area > (img.size[0] * img.size[1] * area_thresh)

def check_font_validity_by_area(font_dir_path, dst_dir_path=None):
	chars = tg_util.construct_charset(space=True)
	area_thresh = 0.05  # Area ratio threshold, [0, 1].

	print('Start checking font validity by area...')
	start_time = time.time()
	invalid_fonts = list()
	for font in os.listdir(font_dir_path):
		font_fpath = os.path.join(font_dir_path, font)
		num_invalid_chars = 0
		for ch in chars:
			if not is_valid_font_by_area(font_fpath, ch, area_thresh=area_thresh):
				num_invalid_chars += 1
		if num_invalid_chars > 0:
			#os.remove(font_fpath)
			if dst_dir_path and os.path.isdir(dst_dir_path):
				os.rename(font_fpath, os.path.join(dst_dir_path, os.path.basename(font_fpath)))
			invalid_fonts.append((font_fpath, num_invalid_chars))
	print('End checking font validity by area: {} secs.'.format(time.time() - start_time))

	print('#invalid fonts = {}.'.format(len(invalid_fonts)))
	if len(invalid_fonts) > 0:
		print('Invalid fonts:')
		for font_fpath, num_invalid_chars in invalid_fonts:
			print('\t{}: #invalid chars = {}.'.format(font_fpath, num_invalid_chars))
	else:
		print('No invalid fonts.')

def main():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng_large'
	#ont_dir_path = font_base_dir_path + '/kor_large'
	font_dir_path = font_base_dir_path + '/eng_receipt'
	#font_dir_path = font_base_dir_path + '/kor_receipt'

	#visualize_font_text(font_dir_path)

	#check_font_validity(font_dir_path, char_pair=('C', 'M'))  # For English.
	#check_font_validity(font_dir_path, char_pair=('가', '너'))  # For Korean.

	check_font_validity_by_area(font_dir_path)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
