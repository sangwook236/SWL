#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time, functools, glob
import swl.language_processing.util as swl_langproc_util
import text_line_data
import text_generation_util as tg_util

def construct_charsets():
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A string.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of strings.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	alphabet_charset = string.ascii_uppercase + string.ascii_lowercase
	digit_charset = string.digits
	#symbol_charset = string.punctuation
	symbol_charset = string.punctuation + ' '

	if False:
		print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
		print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
		print('Digit charset =', len(digit_charset), digit_charset)
		print('Symbol charset =', len(symbol_charset), symbol_charset)

		print('Hangeul jamo charset =', len(hangeul_jamo_charset), hangeul_jamo_charset)

	return hangeul_charset, alphabet_charset, digit_charset, symbol_charset, hangeul_jamo_charset

def construct_word_sets():
	korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
	#english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
	english_dictionary_filepath = '../../data/language_processing/wordlist_mono_clean.txt'
	#english_dictionary_filepath = '../../data/language_processing/wordlist_bi_clean.txt'

	print('Start loading a Korean dictionary...')
	start_time = time.time()
	with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#korean_words = fd.readlines()
		#korean_words = fd.read().strip('\n')
		korean_words = fd.read().splitlines()
	print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('Start loading an English dictionary...')
	start_time = time.time()
	with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
		#english_words = fd.readlines()
		#english_words = fd.read().strip('\n')
		english_words = fd.read().splitlines()
	print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))

	return set(korean_words), set(english_words), set(korean_words + english_words)

def generate_font_colors(image_depth):
	import random
	#font_color = (255,) * image_depth  # White font color.
	#font_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB font color.
	#font_color = (random.randrange(256),) * image_depth  # A grayscale font color.
	#gray_val = random.randrange(255)
	#font_color = (gray_val,) * image_depth  # A lighter grayscale font color.
	font_color = (random.randrange(128, 256),) * image_depth  # A light grayscale font color.
	#bg_color = (0,) * image_depth  # Black background color.
	#bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB background color.
	#bg_color = (random.randrange(256),) * image_depth  # A grayscale background color.
	#bg_color = (random.randrange(gray_val + 1, 256),) * image_depth  # A darker grayscale background color.
	bg_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale background color.
	return font_color, bg_color

def BasicRunTimeTextLineDataset_test():
	korean_word_set, english_word_set, all_word_set = construct_word_sets()

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if False:
		font_dir_path = font_base_dir_path + '/kor'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), korean_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean dataset...')
		start_time = time.time()
		dataset = text_line_data.BasicRunTimeTextLineDataset(label_converter, korean_word_set, image_height, image_width, image_channel, font_list, color_functor=color_functor)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		font_dir_path = font_base_dir_path + '/eng'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_font_list(font_filepaths)

		image_height, image_width, image_channel = 32, 320, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), english_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating an English dataset...')
		start_time = time.time()
		dataset = text_line_data.BasicRunTimeTextLineDataset(label_converter, english_word_set, image_height, image_width, image_channel, font_list, color_functor=color_functor)
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		#font_dir_path = font_base_dir_path + '/eng'
		font_dir_path = font_base_dir_path + '/kor'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		#font_list = tg_util.generate_font_list(font_filepaths)
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), all_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		dataset = text_line_data.BasicRunTimeTextLineDataset(label_converter, all_word_set, image_height, image_width, image_channel, font_list, color_functor=color_functor)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def RunTimeAlphaMatteTextLineDataset_test():
	korean_word_set, english_word_set, all_word_set = construct_word_sets()

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if False:
		font_dir_path = font_base_dir_path + '/kor'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

		labels = functools.reduce(lambda x, txt: x.union(txt), korean_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean dataset...')
		start_time = time.time()
		dataset = text_line_data.RunTimeAlphaMatteTextLineDataset(label_converter, korean_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor, alpha_matte_mode='1')
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		font_dir_path = font_base_dir_path + '/eng'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 32, 320, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

		labels = functools.reduce(lambda x, txt: x.union(txt), english_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating an English dataset...')
		start_time = time.time()
		dataset = text_line_data.RunTimeAlphaMatteTextLineDataset(label_converter, english_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor, alpha_matte_mode='1')
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		#font_dir_path = font_base_dir_path + '/eng'
		font_dir_path = font_base_dir_path + '/kor'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		#font_list = tg_util.generate_font_list(font_filepaths)
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

		labels = functools.reduce(lambda x, txt: x.union(txt), all_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		dataset = text_line_data.RunTimeAlphaMatteTextLineDataset(label_converter, all_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor, alpha_matte_mode='1')
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def RunTimeHangeulJamoAlphaMatteTextLineDataset_test():
	import hangeul_util as hg_util

	korean_word_set, _, all_word_set = construct_word_sets()

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	hangeul2jamo_functor = lambda hangeul_str: hg_util.hangeul2jamo(hangeul_str, eojc_str=swl_langproc_util.JamoTokenConverter.EOJ, use_separate_consonants=False, use_separate_vowels=True)
	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	jamo2hangeul_functor = lambda jamo_str: hg_util.jamo2hangeul(jamo_str, eojc_str=swl_langproc_util.JamoTokenConverter.EOJ, use_separate_consonants=False, use_separate_vowels=True)

	if False:
		font_dir_path = font_base_dir_path + '/kor'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

		labels = functools.reduce(lambda x, txt: x.union(hangeul2jamo_functor(txt)), korean_word_set, set())
		labels.remove(swl_langproc_util.JamoTokenConverter.EOJ)
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.JamoTokenConverter(labels, hangeul2jamo_functor, jamo2hangeul_functor, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean dataset...')
		start_time = time.time()
		dataset = text_line_data.RunTimeHangeulJamoAlphaMatteTextLineDataset(label_converter, korean_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor, alpha_matte_mode='1')
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		#font_dir_path = font_base_dir_path + '/eng'
		font_dir_path = font_base_dir_path + '/kor'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		#font_list = tg_util.generate_font_list(font_filepaths)
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)

		labels = functools.reduce(lambda x, txt: x.union(hangeul2jamo_functor(txt)), all_word_set, set())
		labels.remove(swl_langproc_util.JamoTokenConverter.EOJ)
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.JamoTokenConverter(labels, hangeul2jamo_functor, jamo2hangeul_functor, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		dataset = text_line_data.RunTimeHangeulJamoAlphaMatteTextLineDataset(label_converter, all_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor, alpha_matte_mode='1')
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def JsonBasedTextLineDataset_test():
	# REF [function] >> generate_text_datasets() in ${DataAnalysis_HOME}/app/text_recognition/generate_text_dataset.py.
	train_json_filepath = './text_train_dataset/text_dataset.json'
	test_json_filepath = './text_test_dataset/text_dataset.json'

	image_height, image_width, image_channel = 64, 640, 1

	import string
	labels = \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))

	label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
	# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
	print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
	print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

	print('Start creating a JsonBasedTextLineDataset...')
	start_time = time.time()
	dataset = text_line_data.JsonBasedTextLineDataset(label_converter, train_json_filepath, test_json_filepath, image_height, image_width, image_channel)
	print('End creating a JsonBasedTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def JsonBasedHangeulJamoTextLineDataset_test():
	import hangeul_util as hg_util

	# REF [function] >> generate_text_datasets() in ${DataAnalysis_HOME}/app/text_recognition/generate_text_dataset.py.
	train_json_filepath = './text_train_dataset/text_dataset.json'
	test_json_filepath = './text_test_dataset/text_dataset.json'

	image_height, image_width, image_channel = 64, 640, 1

	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	hangeul2jamo_functor = lambda hangeul_str: hg_util.hangeul2jamo(hangeul_str, eojc_str=swl_langproc_util.JamoTokenConverter.EOJ, use_separate_consonants=False, use_separate_vowels=True)
	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	jamo2hangeul_functor = lambda jamo_str: hg_util.jamo2hangeul(jamo_str, eojc_str=swl_langproc_util.JamoTokenConverter.EOJ, use_separate_consonants=False, use_separate_vowels=True)

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	labels = \
		hangeul_jamo_charset + \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '
	labels = sorted(labels)
	#labels = ''.join(sorted(labels))  # Error.

	label_converter = swl_langproc_util.JamoTokenConverter(labels, hangeul2jamo_functor, jamo2hangeul_functor, pad_value=None)
	# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
	print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
	print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

	print('Start creating a JsonBasedHangeulJamoTextLineDataset...')
	start_time = time.time()
	dataset = text_line_data.JsonBasedHangeulJamoTextLineDataset(label_converter, train_json_filepath, test_json_filepath, image_height, image_width, image_channel)
	print('End creating a JsonBasedHangeulJamoTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def create_corrupter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	return iaa.Sequential([
		iaa.Sometimes(0.5, iaa.OneOf([
			#iaa.Affine(
			#	scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
			#	translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent (per axis).
			#	rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
			#	shear=(-5, 5),  # Shear by -5 to +5 degrees.
			#	#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
			#	order=0,  # Use nearest neighbour or bilinear interpolation (fast).
			#	#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
			#	#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			#	#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			#),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(10.0, 30.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
		])),
		iaa.OneOf([
			iaa.OneOf([
				iaa.GaussianBlur(sigma=(0.5, 1.5)),
				iaa.AverageBlur(k=(2, 4)),
				iaa.MedianBlur(k=(3, 3)),
				iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0), order=1),
			]),
			iaa.Sequential([
				iaa.OneOf([
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
					iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
				]),
				iaa.GaussianBlur(sigma=(0.7, 1.0)),
			]),
			#iaa.OneOf([
			#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
			#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
			#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

			#	iaa.Invert(p=1, per_channel=False),

			#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
			#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
			#]),
		])
	])

class MyRunTimeCorruptedTextLinePairDataset(text_line_data.RunTimeCorruptedTextLinePairDatasetBase):
	def __init__(self, label_converter, text_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=None, use_NWHC=True):
		super().__init__(label_converter, text_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor, use_NWHC)

		self._corrupter = create_corrupter()

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def corrupt(self, inputs, *args, **kwargs):
		return self._corrupter.augment_images(inputs)

	def corrupt2(self, inputs, outputs, *args, **kwargs):
		if outputs is None:
			return self._corrupter.augment_images(inputs), None
		else:
			augmenter_det = self._corrupter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

def RunTimeCorruptedTextLinePairDataset_test():
	korean_word_set, english_word_set, all_word_set = construct_word_sets()

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if False:
		font_dir_path = font_base_dir_path + '/kor'
		#font_dir_path = font_base_dir_path + '/kor_receipt'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), korean_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean dataset...')
		start_time = time.time()
		dataset = MyRunTimeCorruptedTextLinePairDataset(label_converter, korean_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		font_dir_path = font_base_dir_path + '/eng'
		#font_dir_path = font_base_dir_path + '/eng_receipt'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 32, 320, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), english_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating an English dataset...')
		start_time = time.time()
		dataset = MyRunTimeCorruptedTextLinePairDataset(label_converter, english_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor)
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		#font_dir_path = font_base_dir_path + '/eng'
		font_dir_path = font_base_dir_path + '/kor'
		#font_dir_path = font_base_dir_path + '/eng_receipt'
		#font_dir_path = font_base_dir_path + '/kor_receipt'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		#font_list = tg_util.generate_font_list(font_filepaths)
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		image_height, image_width, image_channel = 64, 640, 1
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), all_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean+English dataset...')
		start_time = time.time()
		dataset = MyRunTimeCorruptedTextLinePairDataset(label_converter, all_word_set, image_height, image_width, image_channel, font_list, char_images_dict, color_functor=color_functor)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

class MyRunTimeSuperResolvedTextLinePairDataset(text_line_data.RunTimeSuperResolvedTextLinePairDatasetBase):
	def __init__(self, label_converter, text_set, hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel, font_list, char_images_dict, color_functor=None, use_NWHC=True):
		super().__init__(label_converter, text_set, hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel, font_list, char_images_dict, color_functor, use_NWHC)

		self._corrupter = create_corrupter()

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def corrupt(self, inputs, *args, **kwargs):
		return self._corrupter.augment_images(inputs)

	def corrupt2(self, inputs, outputs, *args, **kwargs):
		if outputs is None:
			return self._corrupter.augment_images(inputs), None
		else:
			augmenter_det = self._corrupter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

def RunTimeSuperResolvedTextLinePairDataset_test():
	korean_word_set, english_word_set, all_word_set = construct_word_sets()

	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	if False:
		font_dir_path = font_base_dir_path + '/kor'
		#font_dir_path = font_base_dir_path + '/kor_receipt'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		sr_ratio = 4
		lr_image_height, lr_image_width, image_channel = 16, 160, 1
		hr_image_height, hr_image_width = lr_image_height * sr_ratio, lr_image_width * sr_ratio
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), korean_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean dataset...')
		start_time = time.time()
		dataset = MyRunTimeSuperResolvedTextLinePairDataset(label_converter, korean_word_set, hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel, font_list, char_images_dict, color_functor=color_functor)
		print('End creating a Korean dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if False:
		font_dir_path = font_base_dir_path + '/eng'
		#font_dir_path = font_base_dir_path + '/eng_receipt'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		sr_ratio = 4
		lr_image_height, lr_image_width, image_channel = 16, 160, 1
		hr_image_height, hr_image_width = lr_image_height * sr_ratio, lr_image_width * sr_ratio
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), english_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating an English dataset...')
		start_time = time.time()
		dataset = MyRunTimeSuperResolvedTextLinePairDataset(label_converter, english_word_set, hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel, font_list, char_images_dict, color_functor=color_functor)
		print('End creating an English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

	if True:
		#font_dir_path = font_base_dir_path + '/eng'
		font_dir_path = font_base_dir_path + '/kor'
		#font_dir_path = font_base_dir_path + '/eng_receipt'
		#font_dir_path = font_base_dir_path + '/kor_receipt'

		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		#font_list = tg_util.generate_font_list(font_filepaths)
		font_list = tg_util.generate_hangeul_font_list(font_filepaths)
		#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
		char_images_dict = None

		sr_ratio = 4
		lr_image_height, lr_image_width, image_channel = 16, 160, 1
		hr_image_height, hr_image_width = lr_image_height * sr_ratio, lr_image_width * sr_ratio
		color_functor = functools.partial(generate_font_colors, image_depth=image_channel)
		#color_functor = None

		labels = functools.reduce(lambda x, txt: x.union(txt), all_word_set, set())
		labels = sorted(labels)
		#labels = ''.join(sorted(labels))

		label_converter = swl_langproc_util.TokenConverter(labels, pad_value=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(label_converter.num_tokens))

		print('Start creating a Korean+English dataset...')
		dataset = MyRunTimeSuperResolvedTextLinePairDataset(label_converter, all_word_set, hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel, font_list, char_images_dict, color_functor=color_functor)
		print('End creating a Korean+English dataset: {} secs.'.format(time.time() - start_time))

		train_generator = dataset.create_train_batch_generator(batch_size=32)
		test_generator = dataset.create_test_batch_generator(batch_size=32)

		dataset.visualize(train_generator, num_examples=10)
		dataset.visualize(test_generator, num_examples=10)

def main():
	#hangeul_charset, alphabet_charset, digit_charset, symbol_charset, hangeul_jamo_charset = construct_charsets()

	#--------------------
	BasicRunTimeTextLineDataset_test()

	#--------------------
	#RunTimeAlphaMatteTextLineDataset_test()
	#RunTimeHangeulJamoAlphaMatteTextLineDataset_test()

	#JsonBasedTextLineDataset_test()
	#JsonBasedHangeulJamoTextLineDataset_test()

	#--------------------
	#RunTimeCorruptedTextLinePairDataset_test()
	#RunTimeSuperResolvedTextLinePairDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
