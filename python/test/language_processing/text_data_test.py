#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, glob, time
import numpy as np
import torch, torchvision
import cv2
import text_data
import text_generation_util as tg_util

def SingleCharacterDataset_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'
	#font_dir_path = font_base_dir_path + '/eng'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	font_list = tg_util.generate_font_list(font_filepaths)

	#--------------------
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	alphabet_charset = string.ascii_uppercase + string.ascii_lowercase
	digit_charset = string.digits
	symbol_charset = string.punctuation
	#symbol_charset = string.punctuation + ' '

	#charset = hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset + symbol_charset
	charset = hangeul_charset + alphabet_charset + digit_charset + symbol_charset

	#--------------------
	image_size = (32, 32)
	train_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor()
	])
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor()
	])

	#--------------------
	font_size_interval = (10, 50)
	num_train_examples, num_test_examples = int(1e6), int(1e4)

	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleCharacterDataset(charset, font_list, font_size_interval, num_train_examples, transform=train_transform)
	test_dataset = text_data.SingleCharacterDataset(charset, font_list, font_size_interval, num_test_examples, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	#--------------------
	batch_size = 64
	shuffle = True
	num_workers = 4

	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	for dataloader in [train_dataloader, test_dataloader]:
		data_iter = iter(dataloader)
		images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
		images, labels = images.numpy(), labels.numpy()
		for idx, (img, lbl) in enumerate(zip(images, labels)):
			print('Label: (int) = {}, (str) = {}.'.format(lbl, charset[lbl]))
			cv2.imshow('Image', img[0])
			cv2.waitKey(0)
			if idx >= 9: break
		cv2.destroyAllWindows()

def SingleWordDataset_test():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'
	#font_dir_path = font_base_dir_path + '/eng'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	font_list = tg_util.generate_font_list(font_filepaths)

	#--------------------
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	alphabet_charset = string.ascii_uppercase + string.ascii_lowercase
	digit_charset = string.digits
	symbol_charset = string.punctuation
	#symbol_charset = string.punctuation + ' '

	#charset = hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset + symbol_charset
	charset = hangeul_charset + alphabet_charset + digit_charset + symbol_charset

	#--------------------
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

	korean_word_set = set(korean_words)
	english_word_set = set(english_words)
	all_word_set = set(korean_words + english_words)

	word_set = all_word_set

	#--------------------
	image_size = (32, 160)
	train_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor()
	])
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor()
	])

	#--------------------
	font_size_interval = (10, 50)
	num_train_examples, num_test_examples = int(1e6), int(1e4)

	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleWordDataset(word_set, charset, font_list, font_size_interval, num_train_examples, transform=train_transform)
	test_dataset = text_data.SingleWordDataset(word_set, charset, font_list, font_size_interval, num_test_examples, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	#--------------------
	batch_size = 64
	shuffle = True
	num_workers = 4

	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	for dataloader in [train_dataloader, test_dataloader]:
		data_iter = iter(dataloader)
		images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
		images, labels = images.numpy(), labels.numpy()
		for idx, (img, lbl) in enumerate(zip(images, labels)):
			print('Label: (int) = {}, (str) = {}.'.format([ll for ll in lbl if ll >= 0], train_dataset.decode_label(lbl)))
			cv2.imshow('Image', img[0])
			cv2.waitKey(0)
			if idx >= 9: break
	cv2.destroyAllWindows()

def TextLineDataset_test():
	raise NotImplementedError

def main():
	#SingleCharacterDataset_test()
	SingleWordDataset_test()
	#TextLineDataset_test()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
