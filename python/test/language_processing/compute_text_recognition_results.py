#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, time, glob, csv
import numpy as np
import swl.language_processing.util as swl_langproc_util

# Generates Tesseract results:
#	cd ~/work/dataset/text/receipt_icdar2019_tesseract
#	magick mogrify -resize x70 -path ./image_70h ./image/*.jpg
#	tesseract --tessdata-dir ~/lib_repo/cpp/tesseract_tessdata_best_github file_list.txt tess_ocr_results -l eng --oem 3 --psm 3 --dpi 70
def load_tesseract_results():
	if 'posix' == os.name:
		receipt_icdar2019_base_dir_path = '/home/sangwook/work/dataset/text/receipt_icdar2019_tesseract'
	else:
		receipt_icdar2019_base_dir_path = 'D:/work/dataset/text/receipt_icdar2019_tesseract'

	filepath = receipt_icdar2019_base_dir_path + '/tess_ocr_results_70h.txt'
	try:
		with open(filepath, 'r', encoding='UTF8') as fd:
			data = fd.read()
	except FileNotFoundError as ex:
		print('[SWL] Error: Failed to load a file: {}.'.format(filepath))
		return None

	page_separator = '\f'  # Form feed: 0x0C.

	data = str(data)
	idx, start_pos, end_pos = 0, 0, 0
	text_dict = dict()
	while True:
		if 11496 == idx:
			idx += 1

		end_pos = data.find(page_separator, start_pos)
		if -1 == end_pos:
			break
		id = 'image/{:05}.jpg'.format(idx)
		if id in text_dict:
			print('[SWL] Warning: Duplicate ID {}.'.format(id))
			continue
		text_dict[id] = data[start_pos:end_pos].rstrip('\n')
		start_pos = end_pos + 1
		idx += 1

	return dict(sorted(text_dict.items()))

# Generates OCRopy results:
#	cd ~/work/dataset/text
#	conda activate ocropus
#	ocropus-nlbin receipt_icdar2019_ocropy/image/*.jpg -o receipt_icdar2019_ocropy/bin -n
#	ocropus-rpred -Q 4 -m ~/lib_repo/python/ocropy_github/models/en-default.pyrnn.gz 'receipt_icdar2019_ocropy/bin/?????.bin.png'
def load_ocropy_results():
	if 'posix' == os.name:
		receipt_icdar2019_base_dir_path = '/home/sangwook/work/dataset/text/receipt_icdar2019_ocropy'
	else:
		receipt_icdar2019_base_dir_path = 'D:/work/dataset/text/receipt_icdar2019_ocropy'

	filepaths = glob.glob(receipt_icdar2019_base_dir_path + '/bin/*.txt')
	if filepaths is None:
		print('[SWL] Error: Failed to load recognition results.')
		return None
	#filepaths.sort()

	text_dict = dict()
	for fpath in filepaths:
		try:
			with open(fpath, 'r', encoding='UTF8') as fd:
				lines = fd.readlines()
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load a file: {}.'.format(fpath))
			return None

		if 1 != len(lines):
			print('[SWL] Warning: Invalid number of texts: {} != 1 in {}.'.format(len(lines), fpath))
			continue

		fname = os.path.splitext(os.path.basename(fpath))[0]
		try:
			id = 'image/{:05}.jpg'.format(int(fname) - 1)
		except ValueError:
			print('[SWL] Warning: Invalid file path: {}.'.format(fpath))
			continue
		if id in text_dict:
			print('[SWL] Warning: Duplicate ID {}.'.format(fpath))
			continue
		text_dict[id] = lines[0].rstrip('\n')

	return dict(sorted(text_dict.items()))

# Generates ABBYY results:
#	REF [file] >> run_abbyy_cloud_sdk.bat
def load_abbyy_results():
	text_dict = dict()

	if 'posix' == os.name:
		receipt_icdar2019_base_dir_path = '/home/sangwook/work/dataset/text/receipt_icdar2019'
	else:
		receipt_icdar2019_base_dir_path = 'D:/work/dataset/text/receipt_icdar2019'

	#--------------------
	# The recognition results of ABBYY FineReaderOCR 15.
	filepath = receipt_icdar2019_base_dir_path + '/abbyy/abbyy_finereader_15_receipt_icdar2019_results.txt'
	try:
		with open(filepath, 'r', encoding='UTF8') as fd:
			lines = fd.readlines()
	except FileNotFoundError as ex:
		print('[SWL] Error: Failed to load a file: {}.'.format(filepath))
		return None

	for idx, line in enumerate(lines):
		id = 'image/{:05}.jpg'.format(idx)
		if id in text_dict:
			print('[SWL] Warning: Duplicate ID {}.'.format(id))
			continue
		#text_dict[id] = line.rstrip('\n')
		text_dict[id] = line.rstrip(' \n')

	#--------------------
	# The recognition results of ABBYY Cloud OCR SDK.
	filepaths = glob.glob(receipt_icdar2019_base_dir_path + '/abbyy/Python/results/*.txt')
	if filepaths is None:
		print('[SWL] Error: Failed to load recognition results.')
		return None
	#filepaths.sort()

	for fpath in filepaths:
		try:
			with open(fpath, 'r', encoding='UTF8') as fd:
				lines = fd.readlines()
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load a file: {}.'.format(fpath))
			return None

		if 0 == len(lines):
			print('[SWL] Warning: Invalid number of texts: {} == = in {}.'.format(len(lines), fpath))
			continue
		if len(lines) > 1:  # Multiple text lines.
			print('[SWL] Warning: Invalid number of texts: {} > 1 in {}.'.format(len(lines), fpath))

		fname = os.path.splitext(os.path.basename(fpath))[0]
		try:
			id = 'image/{:05}.jpg'.format(int(fname))
		except ValueError:
			print('[SWL] Warning: Invalid file path: {}.'.format(fpath))
			continue
		if id in text_dict:
			print('[SWL] Warning: Duplicate ID {}.'.format(fpath))
			continue
		#text_dict[id] = lines[0].rstrip('\n')
		text_dict[id] = lines[0].lstrip('\ufeff').rstrip('\n')

	return dict(sorted(text_dict.items()))

# Generates my recognition results:
#	REF [file] >> run_simple_english_crnn.py or run_simple_hangeul_crnn.py
def load_my_text_recognition_results():
	inference_filepath = 'D:/depot/download/inference_results_v17.csv'

	id_prefix = './receipt_icdar2019/'
	id_prefix_len = len(id_prefix)

	if True:
		text_dict = dict()
		try:
			with open(inference_filepath, 'r', newline='', encoding='UTF8') as fd:
				reader = csv.reader(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

				# ID(filepath),text.
				for row in reader:
					if 2 != len(row):
						print('[SWL] Error: Invalid line: {}.'.format(row))
						continue
					pos = row[0].find(id_prefix)
					if 0 != pos:
						print('[SWL] Warning: Invalid ID {}.'.format(row[0]))
						continue
					id = row[0][id_prefix_len:]
					if id in text_dict:
						print('[SWL] Warning: Duplicate inference ID {}.'.format(row[0]))
						continue
					text_dict[id] = row[1]
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load an inference file {}.'.format(inference_filepath))
			return None
	else:
		separator = ','  # ID(filepath),text.
		#separator = ' '  # ID(filepath) text.

		text_dict = dict()
		try:
			with open(inference_filepath, 'r', encoding='UTF8') as fd:
				lines = fd.readlines()
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load an inference file {}.'.format(inference_filepath))
			return None

		for line in lines:
			line = line.rstrip('\n')

			pos = line.find(separator)
			if -1 == pos:
				print('[SWL] Error: Invalid line: {}.'.format(line))
				continue
			id, text = line[:pos], line[pos+1:]

			pos = id.find(id_prefix)
			if 0 != pos:
				print('[SWL] Warning: Invalid ID {}.'.format(id))
				continue
			id = id[id_prefix_len:]

			if id in text_dict:
				print('[SWL] Warning: Duplicate inference ID {}.'.format(id))
				continue
			text_dict[id] = text

	return dict(sorted(text_dict.items()))

def load_ground_truth_of_receipt_icdar2019():
	if 'posix' == os.name:
		receipt_icdar2019_base_dir_path = '/home/sangwook/work/dataset/text/receipt_icdar2019'
	else:
		receipt_icdar2019_base_dir_path = 'D:/work/dataset/text/receipt_icdar2019'

	# ICDAR SROIE dataset.
	#	REF [site] >> https://rrc.cvc.uab.es/?ch=13
	ground_truth_filepath = receipt_icdar2019_base_dir_path + '/labels.txt'

	if False:
		text_dict = dict()
		try:
			with open(ground_truth_filepath, 'r', newline='', encoding='UTF8') as fd:
				reader = csv.reader(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

				# ID(filepath),text.
				for row in reader:
					if 2 != len(row):
						print('[SWL] Error: Invalid line: {}.'.format(row))
						continue
					if row[0] in text_dict:
						print('[SWL] Warning: Duplicate ground-truth ID {}.'.format(row[0]))
						continue
					text_dict[row[0]] = row[1]
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load a ground-truth file {}.'.format(ground_truth_filepath))
			return None
	else:
		separator = ','
		#separator = ' '

		text_dict = dict()
		try:
			with open(ground_truth_filepath, 'r', encoding='UTF8') as fd:
				lines = fd.readlines()
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load a ground-truth file {}.'.format(ground_truth_filepath))
			return None

		# ID(filepath),text.
		for line in lines:
			line = line.rstrip('\n')

			pos = line.find(separator)
			if -1 == pos:
				print('[SWL] Error: Invalid line: {}.'.format(line))
				continue
			id, text = line[:pos], line[pos+1:]

			if id in text_dict:
				print('[SWL] Warning: Duplicate ground-truth ID {}.'.format(id))
				continue
			text_dict[id] = text

	return dict(sorted(text_dict.items()))

def compute_text_recognition_results():
	print('[SWL] Info: Start loading text recognition results...')
	start_time = time.time()
	gt_text_dict = load_ground_truth_of_receipt_icdar2019()

	if False:
		inf_text_dict = load_tesseract_results()
	elif False:
		inf_text_dict = load_ocropy_results()
	elif False:
		inf_text_dict = load_abbyy_results()
	else:
		inf_text_dict = load_my_text_recognition_results()
	print('[SWL] Info: End loading text recognition results: {} secs.'.format(time.time() - start_time))
	if gt_text_dict is None:
		print('[SWL] Error: Failed to load the ground-truth info.')
		return
	if inf_text_dict is None:
		print('[SWL] Error: Failed to load the inference info.')
		return

	#--------------------
	print('[SWL] Info: Start checking ID validity...')
	start_time = time.time()
	text_pairs = list()
	for key in inf_text_dict:
		if key not in gt_text_dict:
			print('[SWL] Warning: Nonexistent inference ID {}.'.format(key))
		else:
			text_pairs.append((inf_text_dict[key], gt_text_dict[key]))

	if len(text_pairs) != len(gt_text_dict):
		print('[SWL] Warning: Unmatched lengths of text pairs and ground-truth texts: {} != {}.'.format(len(text_pairs), len(gt_text_dict)))	
	print('[SWL] Info: End checking ID validity: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('[SWL] Info: Start comparing text recognition results...')
	start_time = time.time()
	correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = swl_langproc_util.compute_text_recognition_accuracy(text_pairs)
	print('[SWL] Info: End comparing text recognition results: {} secs.'.format(time.time() - start_time))
	print('\tText accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count))
	print('\tWord accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
	print('\tChar accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

#--------------------------------------------------------------------

def main():
	compute_text_recognition_results()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
