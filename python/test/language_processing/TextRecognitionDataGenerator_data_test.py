#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import time
import TextRecognitionDataGenerator_data

def EnglishTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './en_samples_100000'
	#data_dir_path = './en_samples_200000'

	image_height, image_width, image_channel = 32, 100, 1
	train_test_ratio = 0.8
	max_char_count = 50

	print('Start creating an EnglishTextRecognitionDataGeneratorTextLineDataset...')
	start_time = time.time()
	dataset = TextRecognitionDataGenerator_data.EnglishTextRecognitionDataGeneratorTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating an EnglishTextRecognitionDataGeneratorTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def HangeulTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './kr_samples_100000'
	#data_dir_path = './kr_samples_200000'

	#image_height, image_width, image_channel = 32, 160, 1
	image_height, image_width, image_channel = 64, 320, 1
	train_test_ratio = 0.8
	max_char_count = 50

	print('Start creating a HangeulTextRecognitionDataGeneratorTextLineDataset...')
	start_time = time.time()
	dataset = TextRecognitionDataGenerator_data.HangeulTextRecognitionDataGeneratorTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating a HangeulTextRecognitionDataGeneratorTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def HangeulJamoTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './kr_samples_100000'
	#data_dir_path = './kr_samples_200000'

	#image_height, image_width, image_channel = 32, 160, 1
	image_height, image_width, image_channel = 64, 320, 1
	train_test_ratio = 0.8
	max_char_count = 50

	print('Start creating a HangeulJamoTextRecognitionDataGeneratorTextLineDataset...')
	start_time = time.time()
	dataset = TextRecognitionDataGenerator_data.HangeulJamoTextRecognitionDataGeneratorTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating a HangeulJamoTextRecognitionDataGeneratorTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32)
	test_generator = dataset.create_test_batch_generator(batch_size=32)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def main():
	EnglishTextRecognitionDataGeneratorTextLineDataset_test()
	#HangeulTextRecognitionDataGeneratorTextLineDataset_test()
	#HangeulJamoTextRecognitionDataGeneratorTextLineDataset_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()