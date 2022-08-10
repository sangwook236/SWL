#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, glob, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util

def visualize_font_text():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if False:
		#font_dir_path = font_base_dir_path + '/kor_small'
		font_dir_path = font_base_dir_path + '/kor_large'
		#font_dir_path = font_base_dir_path + '/kor_receipt'

		image_size = (1800, 2500)
		line_len = 54
		text = tg_util.construct_charset(hangeul=True, hangeul_jamo=True)
		text = '\n'.join(text[i:i+line_len] for i in range(0, len(text), line_len))
	elif False:
		#font_dir_path = font_base_dir_path + '/eng_small'
		font_dir_path = font_base_dir_path + '/eng_large'
		#font_dir_path = font_base_dir_path + '/eng_receipt'

		image_size = (1800, 2500)
		line_len = 54
		#text = tg_util.construct_charset(latin=True, greek_uppercase=True, greek_lowercase=True)
		text = tg_util.construct_charset(latin=True, greek_uppercase=True, greek_lowercase=True, unit=True, currency=True, symbol=True, math_symbol=True)
		text = '\n'.join(text[i:i+line_len] for i in range(0, len(text), line_len))
	elif False:
		font_dir_path = font_base_dir_path + '/chinese'

		line_len = 54
		image_size = (2000, 3000)
		text = tg_util.construct_charset(chinese=True)
		text = '\n'.join(text[i:i+line_len] for i in range(0, len(text), line_len))
	elif True:
		font_dir_path = font_base_dir_path + '/japanese'
		#font_dir_path = font_base_dir_path + '/japanese_no_kanji'

		image_size = (2000, 3000)
		line_len = 54
		text = tg_util.construct_charset(hiragana=True, katakana=True, kanji=True)
		text = '\n'.join(text[i:i+line_len] for i in range(0, len(text), line_len))

	print('Text =', text)

	font_filepaths = glob.glob(font_dir_path + '/*.*')
	#font_filepaths = glob.glob(font_dir_path + '/*.ttf')
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)

	"""
	gabia_solmee.ttf: 일부 자음 없음.
	godoRoundedL.ttf: ? 표시.
	godoRoundedR.ttf: ? 표시.
	HS여름물빛체.ttf: 위아래 글자 겹침.
	"""

	#--------------------
	text_offset = (0, 0)
	crop_text_area = False
	draw_text_border = False
	font_size = 32
	font_color = (255, 255, 255)
	#font_color = None
	bg_color = (0, 0, 0)
	#bg_color = None

	save_image = True
	output_dir_path = './font_check_results'
	os.makedirs(output_dir_path, exist_ok=True)

	invalid_fonts = list()
	for font_fpath, font_index in font_list:
		print('Font filepath: {}, font index: {}.'.format(os.path.basename(font_fpath), font_index))

		try:
			img = swl_langproc_util.generate_text_image(text, font_fpath, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)

			if save_image:
				img_fpath = os.path.join(output_dir_path, os.path.basename(font_fpath) + '_{}.png'.format(font_index))
				print('\tSaved a font image to {}.'.format(img_fpath))
				img.save(img_fpath)
				#cv2.imwrite(img_fpath, np.array(img))
				#cv2.imshow('Text', np.array(img))
				#cv2.waitKey(0)
		except Exception as ex:
			print('\tException raised in {}: {}.'.format(font_fpath, ex))
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

def check_font_validity():
	dst_dir_path = None

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if True:
		#font_dir_path = font_base_dir_path + '/kor_small'
		font_dir_path = font_base_dir_path + '/kor_large'
		#font_dir_path = font_base_dir_path + '/kor_receipt'
		char_pair = ('가', '너')
	elif False:
		#font_dir_path = font_base_dir_path + '/eng_small'
		font_dir_path = font_base_dir_path + '/eng_large'
		#font_dir_path = font_base_dir_path + '/eng_receipt'
		char_pair = ('C', 'M')
	elif False:
		font_dir_path = font_base_dir_path + '/chinese'
		char_pair = ('', '')  # FIXME [implement] >>
	elif False:
		font_dir_path = font_base_dir_path + '/japanese'
		#font_dir_path = font_base_dir_path + '/japanese_no_kanji'
		char_pair = ('', '')  # FIXME [implement] >>

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

def check_font_validity_by_area():
	chars = tg_util.construct_charset(hangeul=True, hangeul_jamo=True)
	area_thresh = 0.05  # Area ratio threshold, [0, 1].
	dst_dir_path = None

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if True:
		#font_dir_path = font_base_dir_path + '/kor_small'
		font_dir_path = font_base_dir_path + '/kor_large'
		#font_dir_path = font_base_dir_path + '/kor_receipt'
	elif False:
		#font_dir_path = font_base_dir_path + '/eng_small'
		font_dir_path = font_base_dir_path + '/eng_large'
		#font_dir_path = font_base_dir_path + '/eng_receipt'
	elif False:
		font_dir_path = font_base_dir_path + '/chinese'
	elif False:
		font_dir_path = font_base_dir_path + '/japanese'
		#font_dir_path = font_base_dir_path + '/japanese_no_kanji'

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

def check_PIL_ImageFont_API_support():
	if 'posix' == os.name:
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		font_base_dir_path = 'D:/work/font'

	if True:
		font_dir_paths = [
			#font_base_dir_path + '/kor_small',
			font_base_dir_path + '/kor_large',
			#font_base_dir_path + '/kor_receipt',
		]
		dst_dir_path = font_base_dir_path + '/kor_error'

		chars = tg_util.construct_charset(hangeul=True, hangeul_jamo=True)
	elif False:
		font_dir_paths = [
			#font_base_dir_path + '/eng_small'
			font_base_dir_path + '/eng_large',
			#font_base_dir_path + '/eng_receipt',
		]
		dst_dir_path = font_base_dir_path + '/eng_error'

		chars = tg_util.construct_charset()
	elif False:
		font_dir_paths = [
			font_base_dir_path + '/chinese',
		]
		dst_dir_path = font_base_dir_path + '/chinese_error'

		chars = tg_util.construct_charset(chinese=True)
	elif False:
		font_dir_paths = [
			font_base_dir_path + '/japanese',
			#font_base_dir_path + '/japanese_no_kanji'
		]
		dst_dir_path = font_base_dir_path + '/japanese_error'

		chars = tg_util.construct_charset(hiragana=True, katakana=True, kanji=True)

	#-----
	font_sizes = list(range(8, 48, 2))
	move_font_files = False

	print('Checking getlength() support...')
	start_time = time.time()
	for font_dir_path in font_dir_paths:
		font_filepaths = sorted(glob.glob(os.path.join(font_dir_path, '*.*'), recursive=True))
		#font_filepaths = sorted(glob.glob(os.path.join(font_dir_path, '*.ttf'), recursive=True))
		for font_filepath in font_filepaths:
			for font_size in font_sizes:
				try:
					font = ImageFont.truetype(font=font_filepath, size=font_size)
					text_length = font.getlength(chars)
					font_offset = font.getoffset(chars)  # (x, y).
				except Exception as ex:
					print('Exception raised in {} (font size = {}): {}.'.format(font_filepath, font_size, ex))
					if move_font_files:
						import shutil
						shutil.move(font_filepath, dst_dir_path)
						print('{} moved to {}.'.format(font_filepath, dst_dir_path))
	print('getlength() support checked: {} secs.'.format(time.time() - start_time))

def main():
	visualize_font_text()

	#--------------------
	#check_font_validity()
	#check_font_validity_by_area()

	#--------------------
	#check_PIL_ImageFont_API_support()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
