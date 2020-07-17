#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, time

# REF [site] >> https://github.com/konlpy/sejong-sanitizer
# REF [text] >> sejong_corpus_usage_guide.txt
def sejong_sanitizer_test():
	corpus_dir_path = '../../data/language_processing/sejong_corpus'

	# File format.
	#	Word\tPOS-results.
	corpus_filepaths = [
		#corpus_dir_path + '/colloquial_word_to_morph.txt',
		corpus_dir_path + '/colloquial_word_to_morphpos.txt',
		#corpus_dir_path + '/written_word_to_morph.txt',
		corpus_dir_path + '/written_word_to_morphpos.txt'
	]

	print('Start loading Sejong corpus...')
	start_time = time.time()
	lines = list()
	for fpath in corpus_filepaths:
		try:
			with open(fpath, 'r', encoding='utf8') as fd:
				lines.extend(fd.read().splitlines())  # A list of strings.
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(fpath))
			raise
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(fpath))
			raise
	print('End loading Sejong corpus: {} secs.'.format(time.time() - start_time))

	words = list()
	#pos_results = list()
	for line in lines:
		pos = line.find('\t')
		words.append(line[:pos])  # Word.
		#pos_results.append(line[pos+1:])  # POS results.
	del lines

	print('#words = {}.'.format(len(words)))
	#print('#POS-results = {}.'.format(len(pos_results)))

# REF [site] >> https://github.com/lovit/sejong_corpus_cleaner
# REF [text] >> sejong_corpus_usage_guide.txt
def sejong_corpus_cleaner_test():
	import sejong_corpus_cleaner

	paths = sejong_corpus_cleaner.get_data_paths()
	#paths = sejong_corpus_cleaner.get_data_paths(corpus_types='written')
	#paths = sejong_corpus_cleaner.get_data_paths(corpus_types='colloquial')

	print('paths =', paths)

	sejong_corpus_cleaner.check_encoding(paths)

	#--------------------
	# Load a list of sentences.
	sents, n_errors = sejong_corpus_cleaner.load_a_file(paths[0])

	print('sents[0] =', sents[0])

	print('sents[0][0] =', sents[0][0])  # ('프랑스의', [프랑스/NNP, 의/JKG]).

	for eojeol, morphtags in sents[0]:
		print('{} has {} morphemes'.format(eojeol, len(morphtags)))

	eojeol, morphtags = sents[0][0]

	print('morphtags[0].morph =', morphtags[0].morph)  # 프랑스.
	print('morphtags[0].tag =', morphtags[0].tag)  # NNP.

	#--------------------
	# Load Sentences.
	sents = sejong_corpus_cleaner.Sentences(paths)
	#sents = sejong_corpus_cleaner.Sentences(paths[0], verbose=False)  # Setting verbose to True outputs the iteration process.
	#sents = sejong_corpus_cleaner.Sentences()  # Load all Sejong corpus files.

	print('len(sents) =', len(sents))

	for sent in sents:
		# Do something.
		pass

	sents = [sent for sent in sejong_corpus_cleaner.Sentences(paths, num_sents=100)]

	#--------------------
	# Save cleansed sentences.
	sejong_corpus_cleaner.write_sentences(sents, 'sejong_corpus.txt')

	#--------------------
	# Load cleansed sentences.
	sents = sejong_corpus_cleaner.load_a_sentences_file('sejong_corpus.txt')
	sents = sejong_corpus_cleaner.Sentences('sejong_corpus.txt', processed=True)

	#--------------------
	# Simplify POS tags.
	for tag in 'NNB NNG NNP XR XSN NR EC EF JC JKB SH NNNG'.split():
		print('{} -> {}'.format(tag, sejong_corpus_cleaner.to_simple_tag(tag)))

	for eojeol, morphtags in sents[0]:
		simple_morphtags = sejong_corpus_cleaner.to_simple_morphtags(morphtags)
		print('{} = {} -> {}'.format(eojeol, morphtags, simple_morphtags))

	#--------------------
	# L-R corpus.
	sent_lr_type1 = sejong_corpus_cleaner.make_lr_corpus(sents[:10], noun_xsv_as_verb=False)[0]
	print('sent_lr_type1 =', sent_lr_type1)

	sent_lr_type2 = sejong_corpus_cleaner.make_lr_corpus(sents[:10], noun_xsv_as_verb=True)[0]
	print('sent_lr_type2 =', sent_lr_type2)

	sent_lr_type3 = sejong_corpus_cleaner.make_lr_corpus(sents[:10], xsv_as_root=True)[0]
	print('sent_lr_type3 =', sent_lr_type3)

	# Save to a file.
	sejong_corpus_cleaner.make_lr_corpus(sents, filepath='lr_corpus_type1.txt')
	sejong_corpus_cleaner.make_lr_corpus(sents, noun_xsv_as_verb=True, filepath='lr_corpus_type2.txt')
	sejong_corpus_cleaner.make_lr_corpus(sents, xsv_as_root=True, filepath='lr_corpus_type3.txt')

	# Load a file.
	corpus_type1 = sejong_corpus_cleaner.Sentences('lr_corpus_type1.txt', processed=True)

	#--------------------
	# Frequency.
	counter = sejong_corpus_cleaner.make_counter(sents)
	sorted(counter.items(), key=lambda x: -x[1])[:5]

	counter = sejong_corpus_cleaner.make_counter(sents, eojeol_morpheme_pair=False)
	sorted(counter.items(), key=lambda x: -x[1])[:5]

	counter = sejong_corpus_cleaner.make_counter(sents, convert_lr=True)
	counter = sejong_corpus_cleaner.make_counter(sents, convert_lr=True, xsv_as_root=True)

def main():
	sejong_sanitizer_test()
	#sejong_corpus_cleaner_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
