#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, time, csv
import numpy as np

def compare_text_recognition_results_in_csv(inference_filepath, ground_truth_filepath):
	print('[SWL] Info: Start loading labels...')
	start_time = time.time()
	try:
		inf_filepaths, inf_labels = list(), list()
		with open(inference_filepath, 'r', newline='', encoding='UTF8') as fd:
			reader = csv.reader(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			# Filepath,label.
			inf_lines = list()
			for row in reader:
				if 2 != len(row):
					print('[SWL] Error: Invalid line: {}.'.format(row))
					continue
				inf_lines.append(row)
			inf_lines.sort()
	except FileNotFoundError as ex:
		print('[SWL] Error: Failed to load an inference label file {}.'.format(inference_filepath))
		return

	try:
		gt_filepaths, gt_labels = list(), list()
		with open(ground_truth_filepath, 'r', newline='', encoding='UTF8') as fd:
			reader = csv.reader(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			# Filepath,label.
			gt_lines = list()
			for row in reader:
				if 2 != len(row):
					print('[SWL] Error: Invalid line: {}.'.format(row))
					continue
				gt_lines.append(row)
			gt_lines.sort()
	except FileNotFoundError as ex:
		print('[SWL] Error: Failed to load a ground-truth label file {}.'.format(ground_truth_filepath))
		return
	print('[SWL] Info: End loading labels: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('[SWL] Info: Start checking label validity...')
	start_time = time.time()
	unmatched_lines = list()
	for gt_line in gt_lines:
		exist = False
		for inf_line in inf_lines:
			if gt_line[0] in inf_line[0]:
				exist = True
				break
		if not exist:
			unmatched_lines.append(gt_line)
	for line in unmatched_lines:
		gt_lines.remove(line)

	inf_labels, gt_labels = list(), list()
	for inf, gt in zip(inf_lines, gt_lines):
		if gt[0] in inf[0]:
			inf_labels.append(inf[1])
			gt_labels.append(gt[1])
		else:
			print('[SWL] Warning: Unmatched filepath {} != {}.'.format(inf[0], gt[0]))
	print('[SWL] Info: End checking label validity: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('[SWL] Info: Start comparing labels...')
	start_time = time.time()
	correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0
	for pred, gt in zip(inf_labels, gt_labels):
		#correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(pred, gt))))
		correct_word_count += len(list(filter(lambda x: x[0].lower() == x[1].lower(), zip(pred, gt))))
		total_word_count += len(gt)
		for ps, gs in zip(pred, gt):
			#correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(ps, gs))))
			correct_char_count += len(list(filter(lambda x: x[0].lower() == x[1].lower(), zip(ps, gs))))
			total_char_count += max(len(ps), len(gs))
		#correct_char_count += functools.reduce(lambda l, pgs: l + len(list(filter(lambda pg: pg[0] == pg[1], zip(pgs[0], pgs[1])))), zip(pred, gt), 0)
		#total_char_count += functools.reduce(lambda l, pg: l + max(len(pg[0]), len(pg[1])), zip(pred, gt), 0)
	print('[SWL] Info: End comparing labels: {} secs.'.format(time.time() - start_time))
	print('\tTest: Word accuracy      = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
	print('\tTest: Character accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

#--------------------------------------------------------------------

def main():
	# Inference results by run_simple_english_crnn.py or run_simple_hangeul_crnn.py.
	inference_filepath = 'D:/depot/download/inference_results_v15_revised.csv'
	# ICDAR SROIE dataset.
	#	REF [site] >> https://rrc.cvc.uab.es/?ch=13
	ground_truth_filepath = 'D:/work/dataset/text/receipt_icdar2019/labels.txt'

	compare_text_recognition_results_in_csv(inference_filepath, ground_truth_filepath)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
