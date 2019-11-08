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
		#hangeul_charset = fd.readlines()  # A string.
		#hangeul_charset = fd.read().strip('\n')  # A list of strings.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.

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

def BasicRunTimeTextLineDataset_test():
	print('Start loading a Korean dictionary...')
	start_time = time.time()
	korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
	with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#korean_words = fd.readlines()
		#korean_words = fd.read().strip('\n')
		korean_words = fd.read().splitlines()
	print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('Start loading an English dictionary...')
	start_time = time.time()
	english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
	with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#english_words = fd.readlines()
		#english_words = fd.read().strip('\n')
		english_words = fd.read().splitlines()
	print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))

	korean_word_set = set(korean_words)
	english_word_set = set(english_words)
	all_word_set = set(korean_words + english_words)

	if False:
		print('Start creating a Korean dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.BasicRunTimeTextLineDataset(korean_word_set, image_height, image_width, image_channel)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		print('Start creating an English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 32, 320, 1
		dataset = text_line_data.BasicRunTimeTextLineDataset(english_word_set, image_height, image_width, image_channel)
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.BasicRunTimeTextLineDataset(all_word_set, image_height, image_width, image_channel)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def RunTimeTextLineDataset_test():
	print('Start loading a Korean dictionary...')
	start_time = time.time()
	korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
	with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#korean_words = fd.readlines()
		#korean_words = fd.read().strip('\n')
		korean_words = fd.read().splitlines()
	print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('Start loading an English dictionary...')
	start_time = time.time()
	english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
	with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#english_words = fd.readlines()
		#english_words = fd.read().strip('\n')
		english_words = fd.read().splitlines()
	print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))

	korean_word_set = set(korean_words)
	english_word_set = set(english_words)
	all_word_set = set(korean_words + english_words)

	if False:
		print('Start creating a Korean dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.RunTimeTextLineDataset(korean_word_set, image_height, image_width, image_channel)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		print('Start creating an English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 32, 320, 1
		dataset = text_line_data.RunTimeTextLineDataset(english_word_set, image_height, image_width, image_channel)
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.RunTimeTextLineDataset(all_word_set, image_height, image_width, image_channel)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def HangeulJamoRunTimeTextLineDataset_test():
	print('Start loading a Korean dictionary...')
	start_time = time.time()
	korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
	with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#korean_words = fd.readlines()
		#korean_words = fd.read().strip('\n')
		korean_words = fd.read().splitlines()
	print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('Start loading an English dictionary...')
	start_time = time.time()
	english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
	with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#english_words = fd.readlines()
		#english_words = fd.read().strip('\n')
		english_words = fd.read().splitlines()
	print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))

	korean_word_set = set(korean_words)
	all_word_set = set(korean_words + english_words)

	if False:
		print('Start creating a Korean dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.HangeulJamoRunTimeTextLineDataset(korean_word_set, image_height, image_width, image_channel)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.HangeulJamoRunTimeTextLineDataset(all_word_set, image_height, image_width, image_channel)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def JsonBasedTextLineDataset_test():
	# REF [function] >> generate_text_datasets() in ${DataAnalysis_HOME}/app/text_recognition/generate_text_dataset.py.
	train_json_filepath = './text_train_dataset/text_dataset.json'
	test_json_filepath = './text_test_dataset/text_dataset.json'

	print('Start creating a JsonBasedTextLineDataset...')
	start_time = time.time()
	image_height, image_width, image_channel = 64, 640, 1
	dataset = text_line_data.JsonBasedTextLineDataset(train_json_filepath, test_json_filepath, image_height, image_width, image_channel)
	print('End creating a JsonBasedTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def HangeulJamoJsonBasedTextLineDataset_test():
	# REF [function] >> generate_text_datasets() in ${DataAnalysis_HOME}/app/text_recognition/generate_text_dataset.py.
	train_json_filepath = './text_train_dataset/text_dataset.json'
	test_json_filepath = './text_test_dataset/text_dataset.json'

	print('Start creating a HangeulJamoJsonBasedTextLineDataset...')
	start_time = time.time()
	image_height, image_width, image_channel = 64, 640, 1
	dataset = text_line_data.HangeulJamoJsonBasedTextLineDataset(train_json_filepath, test_json_filepath, image_height, image_width, image_channel)
	print('End creating a HangeulJamoJsonBasedTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def RunTimePairedCorruptedTextLineDataset_test():
	print('Start loading a Korean dictionary...')
	start_time = time.time()
	korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
	with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#korean_words = fd.readlines()
		#korean_words = fd.read().strip('\n')
		korean_words = fd.read().splitlines()
	print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('Start loading an English dictionary...')
	start_time = time.time()
	english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
	with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#english_words = fd.readlines()
		#english_words = fd.read().strip('\n')
		english_words = fd.read().splitlines()
	print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))

	korean_word_set = set(korean_words)
	english_word_set = set(english_words)
	all_word_set = set(korean_words + english_words)

	if False:
		print('Start creating a Korean dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.RunTimePairedCorruptedTextLineDataset(korean_word_set, image_height, image_width, image_channel)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		print('Start creating an English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 32, 320, 1
		dataset = text_line_data.RunTimePairedCorruptedTextLineDataset(english_word_set, image_height, image_width, image_channel)
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		image_height, image_width, image_channel = 64, 640, 1
		dataset = text_line_data.RunTimePairedCorruptedTextLineDataset(all_word_set, image_height, image_width, image_channel)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def main():
	#hangeul_charset, alphabet_charset, digit_charset, symbol_charset, hangeul_jamo_charset = create_charsets()

	#--------------------
	#BasicRunTimeTextLineDataset_test()

	#--------------------
	#RunTimeTextLineDataset_test()
	#HangeulJamoRunTimeTextLineDataset_test()

	#JsonBasedTextLineDataset_test()
	#HangeulJamoJsonBasedTextLineDataset_test()

	#--------------------
	RunTimePairedCorruptedTextLineDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
