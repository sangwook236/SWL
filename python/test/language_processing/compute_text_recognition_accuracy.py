#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, math, functools, time, glob, csv
import numpy as np
import swl.language_processing.util as swl_langproc_util

# Generates Tesseract results:
#	cd ~/work/dataset/text/icdar2019_sroie/task1_test_text_line/tesseract
#	magick mogrify -resize x70 -path ./image_70h ./image/*.jpg
#	tesseract --tessdata-dir ~/lib_repo/cpp/tesseract_tessdata_best_github file_list.txt tess_ocr_results -l eng --oem 3 --psm 3 --dpi 70
def load_tesseract_results():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset/text/icdar2019_sroie/task1_test_text_line/tesseract_70h'
	else:
		data_dir_path = 'D:/work/dataset/text/icdar2019_sroie/task1_test_text_line/tesseract_70h'

	filepath = data_dir_path + '/tess_ocr_results_70h.txt'
	try:
		with open(filepath, 'r', encoding='UTF8') as fd:
			data = fd.read()
	except FileNotFoundError as ex:
		print('[SWL] Error: Failed to load a file: {}.'.format(filepath))
		return None
	except UnicodeDecodeError as ex:
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
			#continue
		text_dict[id] = data[start_pos:end_pos].rstrip('\n')
		start_pos = end_pos + 1
		idx += 1

	return dict(sorted(text_dict.items()))

# Generates OCRopy results:
#	cd ~/work/dataset/text
#	conda activate ocropus
#	ocropus-nlbin icdar2019_sroie/task1_test_text_line/ocropy/image/*.jpg -o icdar2019_sroie/task1_test_text_line/ocropy/bin -n
#	ocropus-rpred -Q 4 -m ~/lib_repo/python/ocropy_github/models/en-default.pyrnn.gz 'icdar2019_sroie/task1_test_text_line/ocropy/bin/?????.bin.png'
def load_ocropy_results():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset/text/icdar2019_sroie/task1_test_text_line/ocropy'
	else:
		data_dir_path = 'D:/work/dataset/text/icdar2019_sroie/task1_test_text_line/ocropy'

	filepaths = glob.glob(data_dir_path + '/bin/*.txt')
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
			continue
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Failed to load a file: {}.'.format(fpath))
			continue

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
			#continue
		text_dict[id] = lines[0].rstrip('\n')

	return dict(sorted(text_dict.items()))

# Generates ABBYY results:
#	REF [file] >> run_abbyy_cloud_sdk.bat
def load_abbyy_results():
	text_dict = dict()

	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset/text/icdar2019_sroie/task1_test_text_line'
	else:
		data_dir_path = 'D:/work/dataset/text/icdar2019_sroie/task1_test_text_line'

	#--------------------
	# The recognition results of ABBYY FineReaderOCR 15.
	if False:
		filepath = data_dir_path + '/abbyy/abbyy_finereader_15_icdar2019_sroie_task1_test_results.txt'
		try:
			with open(filepath, 'r', encoding='UTF8') as fd:
				lines = fd.readlines()
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load a file: {}.'.format(filepath))
			return None
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Failed to load a file: {}.'.format(filepath))
			return None

		for idx, line in enumerate(lines):
			id = 'image/{:05}.jpg'.format(idx)
			if id in text_dict:
				print('[SWL] Warning: Duplicate ID {}.'.format(id))
				#continue
			#text_dict[id] = line.rstrip('\n')
			text_dict[id] = line.rstrip(' \n')

	#--------------------
	# The recognition results of ABBYY Cloud OCR SDK.
	#filepaths = glob.glob(data_dir_path + '/abbyy/Python/results/*.txt')
	filepaths = glob.glob(data_dir_path + '/abbyy/ICDAR2019_Recipt (Abbyy)/abbyy/*.txt')
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
			continue
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Failed to load a file: {}.'.format(fpath))
			continue

		if 0 == len(lines):
			print('[SWL] Warning: Invalid number of texts: {} == 0 in {}.'.format(len(lines), fpath))
			continue
		#if len(lines) > 1:  # Multiple text lines.
		#	print('[SWL] Warning: Invalid number of texts: {} > 1 in {}.'.format(len(lines), fpath))

		fname = os.path.splitext(os.path.basename(fpath))[0]
		try:
			id = 'image/{:05}.jpg'.format(int(fname))
		except ValueError:
			print('[SWL] Warning: Invalid file path: {}.'.format(fpath))
			continue
		if id in text_dict:
			print('[SWL] Warning: Duplicate ID {}.'.format(fpath))
			#continue
		#text_dict[id] = lines[0].rstrip('\n')
		text_dict[id] = lines[0].lstrip('\ufeff').rstrip('\n')

	return dict(sorted(text_dict.items()))

# Generates my recognition results:
#	REF [file] >> run_simple_english_crnn.py or run_simple_hangeul_crnn.py
def load_my_text_recognition_results():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset/text/icdar2019_sroie/task1_test_text_line'
	else:
		data_dir_path = 'D:/work/dataset/text/icdar2019_sroie/task1_test_text_line'

	#inference_filepath = data_dir_path + '/inference_results_17.csv'
	#inference_filepath = data_dir_path + '/inference_results_densenet_01.csv'
	#inference_filepath = data_dir_path + '/inference_results_densenet_02.csv'
	#inference_filepath = data_dir_path + '/inference_results_densenet_03.csv'
	inference_filepath = data_dir_path + '/inference_results_densenet_04.csv'
	#inference_filepath = data_dir_path + '/inference_results_resnetv2_04.csv'

	id_prefix = './icdar2019_sroie/task1_test_text_line/'
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
						#continue
					text_dict[id] = row[1]
		except FileNotFoundError as ex:
			print('[SWL] Error: Failed to load an inference file {}.'.format(inference_filepath))
			return None
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Failed to load an inference file: {}.'.format(inference_filepath))
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
				#continue
			text_dict[id] = text

	return dict(sorted(text_dict.items()))

def load_ground_truth_of_icdar2019_sroie_task1_test():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset/text/icdar2019_sroie/task1_test_text_line'
	else:
		data_dir_path = 'D:/work/dataset/text/icdar2019_sroie/task1_test_text_line'

	# ICDAR SROIE dataset.
	#	REF [site] >> https://rrc.cvc.uab.es/?ch=13
	ground_truth_filepath = data_dir_path + '/labels.txt'

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
				#continue
			text_dict[id] = text

	return dict(sorted(text_dict.items()))

def compute_simple_matching_accuracy_using_icdar2019_sroie():
	print('[SWL] Info: Start loading text recognition results...')
	start_time = time.time()
	gt_text_dict = load_ground_truth_of_icdar2019_sroie_task1_test()

	if False:
		inf_text_dict = load_tesseract_results()
	elif False:
		inf_text_dict = load_ocropy_results()
	elif False:
		inf_text_dict = load_abbyy_results()
	else:
		inf_text_dict = load_my_text_recognition_results()
	print('[SWL] Info: End loading text recognition results: {} secs.'.format(time.time() - start_time))

	if gt_text_dict is None or 0 == len(gt_text_dict):
		print('[SWL] Error: Failed to load the ground-truth info.')
		return
	if inf_text_dict is None or 0 == len(inf_text_dict):
		print('[SWL] Error: Failed to load the inference info.')
		return

	#--------------------
	print('[SWL] Info: Start checking ID validity...')
	start_time = time.time()
	text_pairs = list()  # Pairs of inferences and G/T's .
	for key in inf_text_dict:
		if key not in gt_text_dict:
			print('[SWL] Warning: Nonexistent inference ID {}.'.format(key))
		else:
			# Case sensitive.
			text_pairs.append((inf_text_dict[key], gt_text_dict[key]))
			# Case insensitive.
			#text_pairs.append((inf_text_dict[key].lower(), gt_text_dict[key].lower()))

	if 0 == len(text_pairs) or len(text_pairs) != len(gt_text_dict):
		print('[SWL] Warning: Unmatched lengths of text pairs and ground-truth texts: {} != {}.'.format(len(text_pairs), len(gt_text_dict)))	
	print('[SWL] Info: End checking ID validity: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('[SWL] Info: Start computing simple matching accuracy...')
	start_time = time.time()
	correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = swl_langproc_util.compute_simple_text_matching_accuracy(text_pairs)
	print('[SWL] Info: End computing simple matching accuracy: {} secs.'.format(time.time() - start_time))

	print('\tText: Simple matching accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count if total_text_count > 0 else -1))
	print('\tWord: Simple matching accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count if total_word_count > 0 else -1))
	print('\tChar: Simple matching accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count if total_char_count > 0 else -1))

def compute_string_distance_using_icdar2019_sroie():
	print('[SWL] Info: Start loading text recognition results...')
	start_time = time.time()
	gt_text_dict = load_ground_truth_of_icdar2019_sroie_task1_test()

	if False:
		inf_text_dict = load_tesseract_results()
	elif False:
		inf_text_dict = load_ocropy_results()
	elif False:
		inf_text_dict = load_abbyy_results()
	else:
		inf_text_dict = load_my_text_recognition_results()
	print('[SWL] Info: End loading text recognition results: {} secs.'.format(time.time() - start_time))

	if gt_text_dict is None or 0 == len(gt_text_dict):
		print('[SWL] Error: Failed to load the ground-truth info.')
		return
	if inf_text_dict is None or 0 == len(inf_text_dict):
		print('[SWL] Error: Failed to load the inference info.')
		return

	#--------------------
	print('[SWL] Info: Start checking ID validity...')
	start_time = time.time()
	text_pairs = list()  # Pairs of inferences and G/T's .
	for key in inf_text_dict:
		if key not in gt_text_dict:
			print('[SWL] Warning: Nonexistent inference ID {}.'.format(key))
		else:
			# Case sensitive.
			text_pairs.append((inf_text_dict[key], gt_text_dict[key]))
			# Case insensitive.
			#text_pairs.append((inf_text_dict[key].lower(), gt_text_dict[key].lower()))

	if 0 == len(text_pairs) or len(text_pairs) != len(gt_text_dict):
		print('[SWL] Warning: Unmatched lengths of text pairs and ground-truth texts: {} != {}.'.format(len(text_pairs), len(gt_text_dict)))	
	print('[SWL] Info: End checking ID validity: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('[SWL] Info: Start computing string distance...')
	start_time = time.time()
	text_distance, word_distance, char_distance, total_text_count, total_word_count, total_char_count = swl_langproc_util.compute_string_distance(text_pairs)
	print('[SWL] Info: End computing string distance: {} secs.'.format(time.time() - start_time))

	print('\tText: String distance = {0}, average string distance = {0} / {1} = {2}.'.format(text_distance, total_text_count, text_distance / total_text_count if total_text_count > 0 else -1))
	print('\tWord: String distance = {0}, average string distance = {0} / {1} = {2}.'.format(word_distance, total_word_count, word_distance / total_word_count if total_word_count > 0 else -1))
	print('\tChar: String distance = {0}, average string distance = {0} / {1} = {2}.'.format(char_distance, total_char_count, char_distance / total_char_count if total_char_count > 0 else -1))

#--------------------------------------------------------------------

def compute_accuracy_from_inference_result():
	text_pairs = list()  # Pairs of inferences and G/T's.
	if True:
		# For a CSV or TSV file.
		if False:
			# For a CSV file.
			delimiter = ','
			# index,filepath,G/T,inference.
			result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_aihub_printed_inference_results_20200807.csv'
		else:
			# For a TSV file.
			delimiter = '\t'
			# index\tfilepath\tG/T\tinference.
			#result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_aihub_printed_inference_results_20200807.tsv'
			result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_aihub_printed_inference_results_20200809.tsv'
			#result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1920x1_aihub_printed_inference_results_20200810.tsv'
			#result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_epapyrus_font_test_inference_results_20200807.tsv'
			#result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_epapyrus_font_test_inference_results_20200809.tsv'
			#result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_epapyrus_font_test_inference_results_20200810.tsv'

		quotechar = '"'
		escapechar = '\\'
		try:
			with open(result_filepath, 'r', newline='', encoding='UTF8') as fd:
				#reader = csv.reader(fd, delimiter=delimiter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
				#reader = csv.reader(fd, delimiter=delimiter, quotechar=quotechar, doublequote=False, escapechar=escapechar, quoting=csv.QUOTE_MINIMAL)
				#reader = csv.reader(fd, delimiter=delimiter, escapechar=escapechar, quoting=csv.QUOTE_NONE)
				reader = csv.reader(fd, delimiter=delimiter, quotechar=None, escapechar=escapechar, quoting=csv.QUOTE_NONE)
				for idx, row in enumerate(reader):
					# Case sensitive.
					#text_pairs.append((row[3], row[2]))
					#text_pairs.append((row[3].replace(' ', ''), row[2].replace(' ', '')))  # Ignore blank spaces.
					# Case insensitive.
					#text_pairs.append((row[3].lower(), row[2].lower()))
					text_pairs.append((row[3].lower().replace(' ', ''), row[2].lower().replace(' ', '')))  # Ignore blank spaces.

					if idx == 277149: break  # Words only.
		except csv.Error as ex:
			print('csv.Error in {}: {}.'.format(result_filepath, ex))
		except UnicodeDecodeError as ex:
			print('Unicode decode error in {} : {}.'.format(result_filepath, ex))
		except FileNotFoundError as ex:
			print('File not found, {} : {}.'.format(result_filepath, ex))
	else:
		# For a generic text file.
		delimiter = ','
		#delimiter = '\t'

		# index,filepath,G/T,inference.
		result_filepath = './simple_hangeul_crnn_densenet_large_rt_ch99_64x1280x1_aihub_printed_inference_results_20200806.txt'

		try:
			with open(result_filepath, 'r', encoding='UTF8') as fd:
				lines = fd.read().splitlines()  # A list of strings.

				for idx, line in enumerate(lines):
					if line.count(',') != 3:
						#print('Invalid comma count: {}.'.format(line))
						continue

					line = line.split(delimiter)

					# Case sensitive.
					text_pairs.append((line[3], line[2]))
					#text_pairs.append((line[3].replace(' ', ''), line[2].replace(' ', '')))  # Ignore blank spaces.
					# Case insensitive.
					#text_pairs.append((line[3].lower(), line[2].lower()))
					#text_pairs.append((line[3].lower().replace(' ', ''), line[2].lower().replace(' ', '')))  # Ignore blank spaces.
					
					#if idx == 277149: break  # Words only.
		except UnicodeDecodeError as ex:
			print('Unicode decode error in {}: {}.'.format(result_filepath, ex))
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(result_filepath, ex))

	#--------------------
	print('[SWL] Info: Start computing simple matching accuracy...')
	start_time = time.time()
	correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = swl_langproc_util.compute_simple_text_matching_accuracy(text_pairs)
	print('[SWL] Info: End computing simple matching accuracy: {} secs.'.format(time.time() - start_time))

	print('\tText: Simple matching accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count if total_text_count > 0 else -1))
	print('\tWord: Simple matching accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count if total_word_count > 0 else -1))
	print('\tChar: Simple matching accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count if total_char_count > 0 else -1))

	#--------------------
	print('[SWL] Info: Start computing average sequence matching ratio...')
	start_time = time.time()
	if text_pairs:
		ave_matching_ratio = swl_langproc_util.compute_sequence_matching_ratio(text_pairs, isjunk=None)  # [0, 1].
		#ave_matching_ratio = swl_langproc_util.compute_sequence_matching_ratio(text_pairs, isjunk=lambda x: x == '\n\r')  # [0, 1].
		#ave_matching_ratio = swl_langproc_util.compute_sequence_matching_ratio(text_pairs, isjunk=lambda x: x == ' \t\n\r')  # [0, 1].
	else: ave_matching_ratio = -1
	print('[SWL] Info: End computing average sequence matching ratio: {} secs.'.format(time.time() - start_time))

	print('\tAverage sequence matching ratio = {}.'.format(ave_matching_ratio))

	#--------------------
	print('[SWL] Info: Start computing string distance...')
	start_time = time.time()
	text_distance, word_distance, char_distance, total_text_count, total_word_count, total_char_count = swl_langproc_util.compute_string_distance(text_pairs)
	print('[SWL] Info: End computing string distance: {} secs.'.format(time.time() - start_time))

	print('\tText: String distance = {0}, average string distance = {0} / {1} = {2}.'.format(text_distance, total_text_count, text_distance / total_text_count if total_text_count > 0 else -1))
	print('\tWord: String distance = {0}, average string distance = {0} / {1} = {2}.'.format(word_distance, total_word_count, word_distance / total_word_count if total_word_count > 0 else -1))
	print('\tChar: String distance = {0}, average string distance = {0} / {1} = {2}.'.format(char_distance, total_char_count, char_distance / total_char_count if total_char_count > 0 else -1))

	#--------------------
	print('[SWL] Info: Start computing sequence precision and recall...')
	start_time = time.time()
	metrics, classes = swl_langproc_util.compute_sequence_precision_and_recall(text_pairs, classes=None, isjunk=None)  # A list of (TP + FP, TP + FN, TP)'s.
	print('[SWL] Info: End computing sequence precision and recall: {} secs.'.format(time.time() - start_time))

	print('Classes = {}.'.format(classes))
	print('#classes = {}.'.format(len(classes)))

	precisions, recalls, precisions_for_nonzero_tp, recalls_for_nonzero_tp = list(), list(), list(), list()
	for idx, (cls, (TP_FP, TP_FN, TP)) in enumerate(zip(classes, metrics)):
		prec = (TP / TP_FP) if TP_FP != 0 else None
		recall = (TP / TP_FN) if TP_FN != 0 else None
		if prec is not None:
			precisions.append(prec)
			if TP != 0: precisions_for_nonzero_tp.append(prec)
		if recall is not None:
			recalls.append(recall)
			if TP != 0: recalls_for_nonzero_tp.append(recall)
		print('{}: {}: Precision = {}, recall = {}.'.format(idx, cls, prec, recall))
	avg_precision, avg_recall = sum(precisions) / len(precisions) if len(precisions) > 0 else -1, sum(recalls) / len(recalls) if len(recalls) > 0 else -1
	f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
	avg_precision_for_nonzero_tp, avg_recall_for_nonzero_tp = sum(precisions_for_nonzero_tp) / len(precisions_for_nonzero_tp) if len(precisions_for_nonzero_tp) > 0 else -1, sum(recalls_for_nonzero_tp) / len(recalls_for_nonzero_tp) if len(recalls_for_nonzero_tp) > 0 else -1
	f1_for_nonzero_tp = 2 * avg_precision_for_nonzero_tp * avg_recall_for_nonzero_tp / (avg_precision_for_nonzero_tp + avg_recall_for_nonzero_tp)
	print('Average precision = {}, average recall = {}, F1 = {}.'.format(avg_precision, avg_recall, f1))
	print('Average precision (for non-zero TP) = {}, average recall (for non-zero TP) = {}, F1 (for non-zero TP) = {}.'.format(avg_precision_for_nonzero_tp, avg_recall_for_nonzero_tp, f1_for_nonzero_tp))

#--------------------------------------------------------------------

def main():
	#compute_simple_matching_accuracy_using_icdar2019_sroie()
	#compute_string_distance_using_icdar2019_sroie()

	compute_accuracy_from_inference_result()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
