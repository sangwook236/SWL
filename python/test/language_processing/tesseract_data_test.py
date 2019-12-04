#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import time
import tesseract_data

# REF [site] >> https://github.com/tesseract-ocr
#	text2image --fonts_dir /path/to/font --font 'Arial' --text /path/to/eng_training.txt --outputbase ./eng_training
#	tesseract eng_training.tif eng_training --tessdata-dir /path/to/tessdata -l eng wordstrbox
def EnglishTesseractTextLineDataset_test():
	image_filepaths = ['./eng_training.tif']
	box_filepaths = ['./eng_training.box']

	image_height, image_width, image_channel = 64, 1600, 1
	train_test_ratio = 0.8
	max_char_count = 200

	import string
	labels = \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	labels = list(labels) + [tesseract_data.EnglishTesseractTextLineDataset.UNKNOWN]
	labels.sort()
	#labels = ''.join(sorted(labels))
	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	print('Start creating an EnglishTesseractTextLineDataset...')
	start_time = time.time()
	dataset = tesseract_data.EnglishTesseractTextLineDataset(image_filepaths, box_filepaths, image_height, image_width, image_channel, train_test_ratio, max_char_count, labels, num_classes)
	print('End creating an EnglishTesseractTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/tesseract-ocr
#	text2image --fonts_dir /path/to/font --font 'gulimche' --text /path/to/kor_training.txt --outputbase ./kor_training
#	tesseract kor_training.tif kor_training --tessdata-dir /path/to/tessdata -l kor+eng wordstrbox
def HangeulTesseractTextLineDataset_test():
	image_filepaths = ['./kor_training.tif']
	box_filepaths = ['./kor_training.box']

	image_height, image_width, image_channel = 64, 1600, 1
	train_test_ratio = 0.8
	max_char_count = 200

	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
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
	labels = list(labels) + [tesseract_data.HangeulTesseractTextLineDataset.UNKNOWN]
	labels.sort()
	#labels = ''.join(sorted(labels))
	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	print('Start creating a HangeulTesseractTextLineDataset...')
	start_time = time.time()
	dataset = tesseract_data.HangeulTesseractTextLineDataset(image_filepaths, box_filepaths, image_height, image_width, image_channel, train_test_ratio, max_char_count, labels, num_classes)
	print('End creating a HangeulTesseractTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/tesseract-ocr
#	text2image --fonts_dir /path/to/font --font 'gulimche' --text /path/to/kor_training.txt --outputbase ./kor_training
#	tesseract kor_training.tif kor_training --tessdata-dir /path/to/tessdata -l kor+eng wordstrbox
def HangeulJamoTesseractTextLineDataset_test():
	image_filepaths = ['./kor_training.tif']
	box_filepaths = ['./kor_training.box']

	image_height, image_width, image_channel = 64, 1600, 1
	train_test_ratio = 0.8
	max_char_count = 200

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
	labels = list(labels) + [tesseract_data.HangeulJamoTesseractTextLineDataset.UNKNOWN, tesseract_data.HangeulJamoTesseractTextLineDataset.EOJC]
	labels.sort()
	#labels = ''.join(sorted(labels))
	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	print('Start creating a HangeulJamoTesseractTextLineDataset...')
	start_time = time.time()
	dataset = tesseract_data.HangeulJamoTesseractTextLineDataset(image_filepaths, box_filepaths, image_height, image_width, image_channel, train_test_ratio, max_char_count, labels, num_classes)
	print('End creating a HangeulJamoTesseractTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def main():
	EnglishTesseractTextLineDataset_test()
	#HangeulTesseractTextLineDataset_test()
	#HangeulJamoTesseractTextLineDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
