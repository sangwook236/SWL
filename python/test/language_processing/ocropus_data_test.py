#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time
import ocropus_data

# REF [site] >> https://github.com/tmbdev/ocropy
#	ocropus-linegen -t tomsawyer.txt -F eng_font_list.txt
def EnglishOcropusTextLineDataset_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/ocropus/linegen_eng'

	image_height, image_width, image_channel = 64, 1000, 1
	train_test_ratio = 0.8
	max_char_count = 200

	import string
	labels = \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	labels = list(labels) + [ocropus_data.EnglishOcropusTextLineDataset.UNKNOWN]
	labels.sort()
	#labels = ''.join(sorted(labels))
	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	print('Start creating an EnglishOcropusTextLineDataset...')
	start_time = time.time()
	dataset = ocropus_data.EnglishOcropusTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count, labels, num_classes)
	print('End creating an EnglishOcropusTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/tmbdev/ocropy
#	ocropus-linegen -t korean_modern_novel_1.txt:korean_modern_novel_2.txt -F kor_font_list.txt
def HangeulOcropusTextLineDataset_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/ocropus/linegen_kor'

	image_height, image_width, image_channel = 64, 1000, 1
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
	labels = list(labels) + [ocropus_data.HangeulOcropusTextLineDataset.UNKNOWN]
	labels.sort()
	#labels = ''.join(sorted(labels))
	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	print('Start creating a HangeulOcropusTextLineDataset...')
	start_time = time.time()
	dataset = ocropus_data.HangeulOcropusTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count, labels, num_classes)
	print('End creating a HangeulOcropusTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/tmbdev/ocropy
#	ocropus-linegen -t korean_modern_novel_1.txt:korean_modern_novel_2.txt -F kor_font_list.txt
def HangeulJamoOcropusTextLineDataset_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/ocropus/linegen_kor'

	image_height, image_width, image_channel = 64, 1000, 1
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
	labels = list(labels) + [ocropus_data.HangeulJamoOcropusTextLineDataset.UNKNOWN, ocropus_data.HangeulJamoOcropusTextLineDataset.EOJC]
	labels.sort()
	#labels = ''.join(sorted(labels))
	print('[SWL] Info: Labels = {}.'.format(labels))
	print('[SWL] Info: #labels = {}.'.format(len(labels)))

	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = len(labels) + 1  # Labels + blank label.

	print('Start creating a HangeulJamoOcropusTextLineDataset...')
	start_time = time.time()
	dataset = ocropus_data.HangeulJamoOcropusTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count, labels, num_classes)
	print('End creating a HangeulJamoOcropusTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def main():
	EnglishOcropusTextLineDataset_test()
	#HangeulOcropusTextLineDataset_test()
	#HangeulJamoOcropusTextLineDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
