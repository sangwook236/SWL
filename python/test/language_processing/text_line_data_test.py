#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import time
import text_line_data

def create_charsets():
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

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	if False:
		print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
		print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
		print('Digit charset =', len(digit_charset), digit_charset)
		print('Symbol charset =', len(symbol_charset), symbol_charset)

		print('Hangeul jamo charset =', len(hangeul_jamo_charset), hangeul_jamo_charset)

	return hangeul_charset, alphabet_charset, digit_charset, symbol_charset, hangeul_jamo_charset

def TextLineDataset_test():
	hangeul_charset, alphabet_charset, digit_charset, symbol_charset, _ = create_charsets()

	char_labels = set(hangeul_charset)
	#char_labels = char_labels.union(alphabet_charset)
	#char_labels = char_labels.union(digit_charset)
	#char_labels = char_labels.union(symbol_charset)
	char_labels = sorted(char_labels)

	charsets = [
		hangeul_charset,
		"""
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
		"""
	]

	#--------------------
	image_height, image_width, image_channel = 64, 320, 1
	batch_size = 4

	min_font_size, max_font_size = 64, 64
	min_char_count, max_char_count = 5, 5
	min_char_space_ratio, max_char_space_ratio = 0.8, 1.25

	#font_color = (255, 255, 255)
	#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
	font_color = None  # Uses random font colors.
	#bg_color = (0, 0, 0)
	bg_color = None  # Uses random colors.

	# NOTE [caution] >>
	#	Font가 깨져 (한글) 문자가 물음표로 표시되는 경우 발생.
	#	생성된 (한글) 문자의 하단부가 일부 짤리는 경우 발생.
	#	Image resizing에 의해 얇은 획이 사라지는 경우 발생.

	dataset = text_line_data.TextLineDataset(char_labels)

	print('Start creating a train text line batch generator...')
	start_time = time.time()
	num_char_repetitions = 10
	train_batch_generator = dataset.create_batch_generator(image_height, image_width, batch_size, charsets, num_char_repetitions, min_char_count, max_char_count, min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color, bg_color)
	print('End creating a train text line batch generator: {} secs.'.format(time.time() - start_time))
	print('Start creating a test text line batch generator...')
	start_time = time.time()
	num_char_repetitions = 2
	test_batch_generator = dataset.create_batch_generator(image_height, image_width, batch_size, charsets, num_char_repetitions, min_char_count, max_char_count, min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color, bg_color)
	print('End creating a test text line batch generator: {} secs.'.format(time.time() - start_time))

	print('#classes =', dataset.num_classes)
	#print('SOS token =', dataset.sos_token)
	print('EOS token =', dataset.eos_token)

	dataset.display_data(train_batch_generator)
	#dataset.display_data(test_batch_generator)

def TextLineDatasetWithHangeulJamoLabel_test():
	hangeul_charset, alphabet_charset, digit_charset, symbol_charset, hangeul_jamo_charset = create_charsets()

	char_labels = set(hangeul_jamo_charset)
	#char_labels = char_labels.union(alphabet_charset)
	#char_labels = char_labels.union(digit_charset)
	#char_labels = char_labels.union(symbol_charset)
	char_labels = sorted(char_labels)

	charsets = [
		hangeul_charset,
		"""
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
		"""
	]

	#--------------------
	image_height, image_width, image_channel = 64, 320, 1
	batch_size = 4

	min_font_size, max_font_size = 64, 64
	min_char_count, max_char_count = 5, 5
	min_char_space_ratio, max_char_space_ratio = 0.8, 1.25

	#font_color = (255, 255, 255)
	#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
	font_color = None  # Uses random font colors.
	#bg_color = (0, 0, 0)
	bg_color = None  # Uses random colors.

	# NOTE [caution] >>
	#	Font가 깨져 (한글) 문자가 물음표로 표시되는 경우 발생.
	#	생성된 (한글) 문자의 하단부가 일부 짤리는 경우 발생.
	#	Image resizing에 의해 얇은 획이 사라지는 경우 발생.

	dataset = text_line_data.TextLineDatasetWithHangeulJamoLabel(char_labels)

	print('Start creating a train text line batch generator...')
	start_time = time.time()
	num_char_repetitions = 10
	train_batch_generator = dataset.create_batch_generator(image_height, image_width, batch_size, charsets, num_char_repetitions, min_char_count, max_char_count, min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color, bg_color)
	print('End creating a train text line batch generator: {} secs.'.format(time.time() - start_time))
	print('Start creating a test text line batch generator...')
	start_time = time.time()
	num_char_repetitions = 2
	test_batch_generator = dataset.create_batch_generator(image_height, image_width, batch_size, charsets, num_char_repetitions, min_char_count, max_char_count, min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color, bg_color)
	print('End creating a test text line batch generator: {} secs.'.format(time.time() - start_time))

	print('#classes =', dataset.num_classes)
	#print('SOJC token =', dataset.sojc_token)
	print('EOJC token =', dataset.eojc_token)
	#print('SOS token =', dataset.sos_token)
	print('EOS token =', dataset.eos_token)

	dataset.display_data(train_batch_generator)
	#dataset.display_data(test_batch_generator)

def main():
	#TextLineDataset_test()
	TextLineDatasetWithHangeulJamoLabel_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
