#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os
import numpy as np
import editdistance

# REF [site] >> https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py
def ibm_top_k_decoding_test():
	raise NotImplementedError

# REF [site] >> https://github.com/githubharald/CTCWordBeamSearch
def word_beam_search_test():
	import ctc_word_beam_search.WordBeamSearch, ctc_word_beam_search.Metrics

	beamWidth = 10
	useNGrams = True

	data_dir_path = './ctc_word_beam_search/data/bentham'
	corpus_filepath = os.path.join(data_dir_path, 'corpus.txt')
	data_filepaths = [
		(os.path.join(data_dir_path, 'mat_0.csv'), os.path.join(data_dir_path, 'gt_0.txt')),
		(os.path.join(data_dir_path, 'mat_1.csv'), os.path.join(data_dir_path, 'gt_1.txt')),
		(os.path.join(data_dir_path, 'mat_2.csv'), os.path.join(data_dir_path, 'gt_2.txt')),
	]

	# #classes = #chars + blank label.
	chars = r''' !"#&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz|£§àâèéê⊥'''  # 93 chars.
	wordChars = "'ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz§àâèéê"

	#--------------------
	# Create a language model.
	try:
		with open(corpus_filepath, 'r', encoding='utf8') as fd:
			langModel = ctc_word_beam_search.LanguageModel.LanguageModel(fd.read(), chars, wordChars)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(corpus_filepath))
		raise
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(corpus_filepath))
		raise

	# Metrics calculates CER and WER for dataset.
	metric = ctc_word_beam_search.Metrics.Metrics(langModel.getWordChars())

	#--------------------
	def softmax(mat):
		# mat: [time steps, #classes].
		maxT, _ = mat.shape
		res = np.zeros(mat.shape)
		for t in range(maxT):
			y = mat[t, :]
			#maxValue = np.max(y)
			#e = np.exp(y - maxValue)
			e = np.exp(y)
			s = np.sum(e)
			res[t, :] = e / s

		return res

	for idx, (mat_fpath, gt_fpath) in enumerate(data_filepaths):
		try:
			mat = np.genfromtxt(mat_fpath, delimiter=';')[:, :-1]  # [time steps, #classes].
			mat = softmax(mat)
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(mat_fpath))
			continue
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(mat_fpath))
			continue

		try:
			with open(gt_fpath, 'r', encoding='utf8') as fd:
				gt = fd.read()
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(gt_fpath))
			continue
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(gt_fpath))
			continue

		# Decode matrix.
		decoded = ctc_word_beam_search.WordBeamSearch.wordBeamSearch(mat, beamWidth, langModel, useNGrams)
		print('Sample: {}.'.format(idx + 1))
		print('Decoded: {}.'.format(decoded))
		print('G/T:     {}.'.format(gt))

		dist = editdistance.eval(decoded, gt)
		print('Edit distance = {}.'.format(dist))

		# Output CER and WER.
		metric.addSample(gt, decoded)
		print('Accumulated CER and WER so far: CER = {}, WER = {}.'.format(metric.getCER(), metric.getWER()))
		print('')

def main():
	#ibm_top_k_decoding_test()  # Not yet implemented.

	# OpenNMT.
	#	- Greedy search.
	#	- Beam search.

	word_beam_search_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
