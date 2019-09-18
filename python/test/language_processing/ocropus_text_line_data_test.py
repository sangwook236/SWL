#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import time
import ocropus_text_line_data

# REF [site] >> https://github.com/tmbdev/ocropy
#	ocropus-linegen -t tomsawyer.txt -F eng_font_list.txt
def EnglishOcropusTextLineDataset_test():
	data_dir_path = './linegen_eng'

	image_height, image_width, image_channel = 64, 1000, 1
	train_test_ratio = 0.8
	max_char_count = 200

	print('Start creating an EnglishOcropusTextLineDataset...')
	start_time = time.time()
	dataset = ocropus_text_line_data.EnglishOcropusTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating an EnglishOcropusTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/tmbdev/ocropy
#	ocropus-linegen -t korean_modern_novel_1.txt:korean_modern_novel_2.txt -F kor_font_list.txt
def HangeulOcropusTextLineDataset_test():
	data_dir_path = './linegen_kor'

	image_height, image_width, image_channel = 64, 1000, 1
	train_test_ratio = 0.8
	max_char_count = 200

	print('Start creating a HangeulOcropusTextLineDataset...')
	start_time = time.time()
	dataset = ocropus_text_line_data.HangeulOcropusTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating a HangeulOcropusTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

# REF [site] >> https://github.com/tmbdev/ocropy
#	ocropus-linegen -t korean_modern_novel_1.txt:korean_modern_novel_2.txt -F kor_font_list.txt
def HangeulJamoOcropusTextLineDataset_test():
	data_dir_path = './linegen_kor'

	image_height, image_width, image_channel = 64, 1000, 1
	train_test_ratio = 0.8
	max_char_count = 200

	print('Start creating a HangeulJamoOcropusTextLineDataset...')
	start_time = time.time()
	dataset = ocropus_text_line_data.HangeulJamoOcropusTextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_char_count)
	print('End creating a HangeulJamoOcropusTextLineDataset: {} secs.'.format(time.time() - start_time))

	train_generator = dataset.create_train_batch_generator(batch_size=32, shuffle=True)
	test_generator = dataset.create_test_batch_generator(batch_size=32, shuffle=True)

	dataset.visualize(train_generator, num_examples=10)
	dataset.visualize(test_generator, num_examples=10)

def main():
	EnglishOcropusTextLineDataset_test()
	#HangeulOcropusTextLineDataset_test()
	#HangeulJamoOcropusTextLineDataset_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
