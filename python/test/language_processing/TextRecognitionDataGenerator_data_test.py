#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, time
import TextRecognitionDataGenerator_data

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
#	python run.py -c 100000 -w 1 -f 32 -b 1 -t 8 --name_format 2 --output_dir en_samples_100000_h32
#	python run.py -c 200000 -w 1 -f 32 -b 1 -t 8 --name_format 2 --output_dir en_samples_200000_h32
#	python run.py -c 1000 -w 1 -f 32 -b 1 -t 8 --name_format 2 --output_dir en_samples_1000_h32
#	python run.py -c 100000 -w 1 -f 32 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir en_samples_100000_h32
#	python run.py -c 200000 -w 1 -f 32 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir en_samples_200000_h32
#	python run.py -c 1000 -w 1 -f 32 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir en_samples_1000_h32
#	python run.py -c 100000 -w 1 -f 32 -rs -b 1 -t 8 --name_format 2 --output_dir en_samples_100000_h32
#	python run.py -c 200000 -w 1 -f 32 -rs -b 1 -t 8 --name_format 2 --output_dir en_samples_200000_h32
#	python run.py -c 1000 -w 1 -f 32 -rs -b 1 -t 8 --name_format 2 --output_dir en_samples_1000_h32
#	python run.py -c 100000 -w 1 -f 32 -rs -num -sym -b 1 -t 8 --name_format 2 --output_dir en_samples_100000_h32
#	python run.py -c 200000 -w 1 -f 32 -rs -num -sym -b 1 -t 8 --name_format 2 --output_dir en_samples_200000_h32
#	python run.py -c 1000 -w 1 -f 32 -rs -num -sym -b 1 -t 8 --name_format 2 --output_dir en_samples_1000_h32
def EnglishTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './en_samples_100000_h32'
	#data_dir_path = './en_samples_200000_h32'

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

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
#	python run_sangwook.py -l kr -c 100000 -w 1 -f 64 -b 1 -t 8 --name_format 2 --output_dir kr_samples_100000_h64
#	python run_sangwook.py -l kr -c 200000 -w 1 -f 64 -b 1 -t 8 --name_format 2 --output_dir kr_samples_200000_h64
#	python run_sangwook.py -l kr -c 1000 -w 1 -f 64 -b 1 -t 8 --name_format 2 --output_dir kr_samples_1000_h64
#	python run_sangwook.py -l kr -c 100000 -w 1 -f 64 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir kr_samples_100000_h64
#	python run_sangwook.py -l kr -c 200000 -w 1 -f 64 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir kr_samples_200000_h64
#	python run_sangwook.py -l kr -c 1000 -w 1 -f 64 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir kr_samples_1000_h64
def HangeulTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './kr_samples_100000_h64'
	#data_dir_path = './kr_samples_200000_h64'

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

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
#	python run_sangwook.py -l kr -c 100000 -w 1 -f 64 -b 1 -t 8 --name_format 2 --output_dir kr_samples_100000_h64
#	python run_sangwook.py -l kr -c 200000 -w 1 -f 64 -b 1 -t 8 --name_format 2 --output_dir kr_samples_200000_h64
#	python run_sangwook.py -l kr -c 1000 -w 1 -f 64 -b 1 -t 8 --name_format 2 --output_dir kr_samples_1000_h64
#	python run_sangwook.py -l kr -c 100000 -w 1 -f 64 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir kr_samples_100000_h64
#	python run_sangwook.py -l kr -c 200000 -w 1 -f 64 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir kr_samples_200000_h64
#	python run_sangwook.py -l kr -c 1000 -w 1 -f 64 -k 5 -rk -d 3 -do 2 -b 4 -rbl -b 1 -t 8 --name_format 2 --output_dir kr_samples_1000_h64
def HangeulJamoTextRecognitionDataGeneratorTextLineDataset_test():
	data_dir_path = './kr_samples_100000_h64'
	#data_dir_path = './kr_samples_200000_h64'

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

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator
#	python run.py -c 200000 -w 1 -f 32 -rs -b 1 -t 8 --name_format 2 --output_dir en_samples_train/h32_w1
#	python run.py -c 200000 -w 2 -f 32 -rs -b 1 -t 8 --name_format 2 --output_dir en_samples_train/h32_w2
#	python run.py -c 200000 -w 3 -f 32 -rs -b 1 -t 8 --name_format 2 --output_dir en_samples_train/h32_w3
def merge_generated_data_directories():
	if True:
		src_data_dir_paths = [
			'./en_samples_train/h32_w1',
			'./en_samples_train/h32_w2',
			'./en_samples_train/h32_w3',
			'./en_samples_train/h64_w1',
			'./en_samples_train/h64_w2',
			'./en_samples_train/h64_w3',
		]
		dst_data_dir_path = './en_samples_train'
	elif False:
		src_data_dir_paths = [
			'./en_samples_test/h32_w1',
			'./en_samples_test/h32_w2',
			'./en_samples_test/h32_w3',
			'./en_samples_test/h64_w1',
			'./en_samples_test/h64_w2',
			'./en_samples_test/h64_w3',
		]
		dst_data_dir_path = './en_samples_test'
	elif False:
		src_data_dir_paths = [
			'./kr_samples_train/h32_w1',
			'./kr_samples_train/h32_w2',
			'./kr_samples_train/h32_w3',
			'./kr_samples_train/h64_w1',
			'./kr_samples_train/h64_w2',
			'./kr_samples_train/h64_w3',
		]
		dst_data_dir_path = './kr_samples_train'
	elif False:
		src_data_dir_paths = [
			'./kr_samples_test/h32_w1',
			'./kr_samples_test/h32_w2',
			'./kr_samples_test/h32_w3',
			'./kr_samples_test/h64_w1',
			'./kr_samples_test/h64_w2',
			'./kr_samples_test/h64_w3',
		]
		dst_data_dir_path = './kr_samples_test'
	src_label_filename = 'labels.txt'
	dst_label_filename = src_label_filename

	os.makedirs(dst_data_dir_path, exist_ok=True)

	os.sep = '/'
	new_lines = list()
	for src_data_dir_path in src_data_dir_paths:
		try:
			with open(os.path.join(src_data_dir_path, src_label_filename), 'r') as fd:
				lines = fd.readlines()
		except FileNotFoundError:
			print('[SWL] Error: File not found: {}.'.format(os.path.join(src_data_dir_path, src_label_filename)))
			continue

		rel_path = os.path.relpath(src_data_dir_path, dst_data_dir_path)
		for line in lines:
			line = line.rstrip('\n')
			if not line:
				continue

			pos = line.find(' ')
			if -1 == pos:
				print('[SWL] Warning: Invalid image-label pair: {}.'.format(line))
				continue
			fname, label = line[:pos], line[pos+1:]

			new_line = '{} {}'.format(os.path.join(rel_path, fname), label)
			new_lines.append(new_line)

	with open(os.path.join(dst_data_dir_path, dst_label_filename), 'w', encoding='UTF8') as fd:
		for line in new_lines:
			fd.write('{}\n'.format(line))

def main():
	#EnglishTextRecognitionDataGeneratorTextLineDataset_test()
	#HangeulTextRecognitionDataGeneratorTextLineDataset_test()
	#HangeulJamoTextRecognitionDataGeneratorTextLineDataset_test()

	merge_generated_data_directories()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
