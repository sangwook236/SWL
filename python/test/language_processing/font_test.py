#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, random, glob
import numpy as np
import cv2
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util

def font_display_test():
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_letters = fd.readlines()  # A string.
		#hangeul_letters = fd.read().strip('\n')  # A list of strings.
		#hangeul_letters = fd.read().splitlines()  # A list of strings.
		hangeul_letters = fd.read().replace(' ', '')  # A string.

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
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'
	#font_dir_path = font_base_dir_path + '/receipt_eng'
	#font_dir_path = font_base_dir_path + '/receipt_kor'

	font_filepaths = glob.glob(font_dir_path + '/*.ttf')
	#font_list = tg_util.generate_font_list(font_filepaths)
	font_list = tg_util.generate_hangeul_font_list(font_filepaths)

	"""
	gabia_solmee.ttf: 일부 자음 없음.
	godoRoundedL.ttf: ? 표시.
	godoRoundedR.ttf: ? 표시.
	HS여름물빛체.ttf: 위아래 글자 겹침.
	"""

	output_dir_path = './font_test_results'
	os.makedirs(output_dir_path, exist_ok=True)
	for font_fpath, font_index in font_list:
		print('Font filepath: {}, font index: {}.'.format(os.path.basename(font_fpath), font_index))

		img = swl_langproc_util.generate_text_image(text, font_fpath, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border)

		img.save(os.path.join(output_dir_path, os.path.basename(font_fpath) + '_{}.png'.format(font_index)))
		#cv2.imwrite(os.path.join(output_dir_path, os.path.basename(font_fpath) + '_{}.png'.format(font_index)), np.array(img))
		#cv2.imshow('Text', np.array(img))
		#cv2.waitKey(0)

	#cv2.destroyAllWindows()

def main():
	font_display_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
