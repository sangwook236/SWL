#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time

def load_sejong_corpus():
	corpus_dir_path = '../../data/language_processing/sejong_corpus'

	# File format.
	#	Word\tPOS-results.
	corpus_filepaths = [
		#corpus_dir_path + '/colloquial_word_to_morph.txt',
		corpus_dir_path + '/colloquial_word_to_morphpos.txt',
		#corpus_dir_path + '/written_word_to_morph.txt',
		corpus_dir_path + '/written_word_to_morphpos.txt'
	]

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

	words = list()
	#pos_tags = list()
	for line in lines:
		pos = line.find('\t')
		words.append(line[:pos])  # Word.
		#pos_tags.append(line[pos+1:])  # POS tag.
	del lines

	return words
	#return words, pos_tags

# REF [site] >> https://github.com/konlpy/sejong-sanitizer
# REF [text] >> sejong_corpus_usage_guide.txt
def sejong_sanitizer_test():
	print('Start loading Sejong corpus...')
	start_time = time.time()
	words = load_sejong_corpus()
	#words, pos_tags = load_sejong_corpus()
	print('End loading Sejong corpus: {} secs.'.format(time.time() - start_time))

	print('#words = {}.'.format(len(words)))
	#print('#POS tags = {}.'.format(len(pos_tags)))

# REF [site] >> https://github.com/lovit/sejong_corpus_cleaner
# REF [text] >> sejong_corpus_usage_guide.txt
def sejong_corpus_cleaner_test():
	lib_dir_path = '/home/sangwook/lib_repo/etc/sejong_corpus_cleaner_github'
	sys.path.append(lib_dir_path)

	import sejong_corpus_cleaner

	paths = sejong_corpus_cleaner.get_data_paths()
	#paths = sejong_corpus_cleaner.get_data_paths(corpus_types='written')
	#paths = sejong_corpus_cleaner.get_data_paths(corpus_types='colloquial')

	print('Corpus file paths:\n{}.'.format(paths))

	if False:
		print('Start checking encoding...')
		start_time = time.time()
		encodings = sejong_corpus_cleaner.check_encoding(paths)  # Too slow.
		print('End checking encoding: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Load a list of sentences.
	print('Start loading sentences...')
	start_time = time.time()
	#sents, n_errors = sejong_corpus_cleaner.load_a_sejong_file(paths[0])  # No sentence.
	sents, n_errors = sejong_corpus_cleaner.load_a_sejong_file(lib_dir_path + '/data/raw/written/BTAA0001.txt')
	print('End loading sentences: {} secs.'.format(time.time() - start_time))

	print('sents[0]:\n{}.'.format(sents[0]))  # Show the first sentence.
	print('sents[0][0] = {}.'.format(sents[0][0]))  # Show the first term, ('프랑스의', [프랑스/NNP, 의/JKG]).

	for eojeol, morphtags in sents[0]:
		print('{} has {} morphemes.'.format(eojeol, len(morphtags)))

	eojeol, morphtags = sents[0][0]
	print('morphtags[0].morph = {}.'.format(morphtags[0].morph))  # 프랑스.
	print('morphtags[0].tag = {}.'.format(morphtags[0].tag))  # NNP.

	#--------------------
	#cleansed_sentences_filepath = './sejong_corpus_cleansed_sentences.txt'
	cleansed_sentences_filepath = './sejong_corpus_cleansed_sentences_100.txt'
	if False:
		# Load sentences.
		print('Start loading sentences...')
		start_time = time.time()
		#sents = sejong_corpus_cleaner.Sentences(paths)
		#sents = sejong_corpus_cleaner.Sentences(paths[0], verbose=False)  # Setting verbose to True outputs the iteration process.
		#sents = sejong_corpus_cleaner.Sentences()  # Load all Sejong corpus files.
		sents = sejong_corpus_cleaner.Sentences(paths, num_sents=100)
		print('End loading sentences: {} secs.'.format(time.time() - start_time))

		sents = [sent for sent in sents]

		# Save cleansed sentences.
		print('Start saving preprocessed sentences to {}...'.format(cleansed_sentences_filepath))
		start_time = time.time()
		sejong_corpus_cleaner.write_sentences(sents, cleansed_sentences_filepath)
		print('End saving preprocessed sentences: {} secs.'.format(time.time() - start_time))
	else:
		# Load cleansed sentences.
		print('Start loading preprocessed sentences from {}...'.format(cleansed_sentences_filepath))
		start_time = time.time()
		if True:
			sents = sejong_corpus_cleaner.load_a_sentences_file(cleansed_sentences_filepath)  # Create a list.
		else:
			sents = sejong_corpus_cleaner.Sentences(cleansed_sentences_filepath, processed=True)  # Create a generator.

			sents = [sent for sent in sents]
		print('End loading preprocessed sentences: {} secs.'.format(time.time() - start_time))

	print('len(sents) = {}.'.format(len(sents)))

	#--------------------
	# Simplify POS tags.
	for tag in 'NNB NNG NNP XR XSN NR EC EF JC JKB SH NNNG'.split():
		print('{} -> {}'.format(tag, sejong_corpus_cleaner.to_simple_tag(tag)))

	for eojeol, morphtags in sents[0]:
		simple_morphtags = sejong_corpus_cleaner.to_simple_morphtags(morphtags)
		print('{} = {} -> {}'.format(eojeol, morphtags, simple_morphtags))

	#--------------------
	# FIXME [fix] >> Error.
	if False:
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
	print('sorted(counter.items(), key=lambda x: -x[1])[:5] = {}.'.format(sorted(counter.items(), key=lambda x: -x[1])[:5]))

	counter = sejong_corpus_cleaner.make_counter(sents, eojeol_morpheme_pair=False)
	print('sorted(counter.items(), key=lambda x: -x[1])[:5] = {}.'.format(sorted(counter.items(), key=lambda x: -x[1])[:5]))

	counter = sejong_corpus_cleaner.make_counter(sents, convert_lr=True)
	print('sorted(counter.items(), key=lambda x: -x[1])[:5] = {}.'.format(sorted(counter.items(), key=lambda x: -x[1])[:5]))

	counter = sejong_corpus_cleaner.make_counter(sents, convert_lr=True, xsv_as_root=True)
	print('sorted(counter.items(), key=lambda x: -x[1])[:5] = {}.'.format(sorted(counter.items(), key=lambda x: -x[1])[:5]))

def construct_dictionary_from_sejong_corpus():
	import spellchecker

	dictionary_filepath = './sejong_corpus_dictionary.json'
	is_dictionary_constructed = False

	if is_dictionary_constructed:
		# REF [function] >> construct_korean_dictionary_example() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/pyspellchecker_test.py.

		print('Start loading Sejong corpus...')
		start_time = time.time()
		words = load_sejong_corpus()
		#words, pos_tags = load_sejong_corpus()
		print('End loading Sejong corpus: {} secs.'.format(time.time() - start_time))

		print('#words = {}.'.format(len(words)))
		#print('#POS tags = {}.'.format(len(pos_tags)))

		text_data = ' '.join(words)
		del words

		if False:
			# NOTE [error] >> Out-of-memory.

			import konlpy
			#import nltk

			# Initialize the Java virtual machine (JVM).
			konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=4096)

			# TODO [check] >> Is it good to extract nouns or do POS tagging?
			print('Start preprocessing Sejong corpus...')
			start_time = time.time()
			kkma = konlpy.tag.Kkma()
			text_data = kkma.nouns(text_data)
			#okt = konlpy.tag.Okt()
			#text_data = okt.nouns(text_data)
			print('End preprocessing Sejong corpus: {} secs.'.format(time.time() - start_time))

			text_data = ' '.join(text_data)

		print('Start saving a dictionary to {}...'.format(dictionary_filepath))
		start_time = time.time()
		spell = spellchecker.SpellChecker(language=None)
		spell.word_frequency.load_text(text_data)
		spell.export(dictionary_filepath, encoding='UTF-8', gzipped=True)
		print('End saving a dictionary: {} secs.'.format(time.time() - start_time))
	else:
		# REF [function] >> simple_korean_example() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/pyspellchecker_test.py.

		print('Start loading a dictionary from {}...'.format(dictionary_filepath))
		start_time = time.time()
		spell = spellchecker.SpellChecker(language=None)
		spell.word_frequency.load_dictionary(dictionary_filepath, encoding='UTF-8')
		print('End loading a dictionary: {} secs.'.format(time.time() - start_time))

	print('type(spell.word_frequency.dictionary) = {}.'.format(type(spell.word_frequency.dictionary)))
	print('type(spell.word_frequency.letters) = {}.'.format(type(spell.word_frequency.letters)))

	print('len(spell.word_frequency.dictionary) = {}.'.format(len(spell.word_frequency.dictionary)))
	print('len(spell.word_frequency.letters) = {}.'.format(len(spell.word_frequency.letters)))
	print('spell.word_frequency.total_words = {}.'.format(spell.word_frequency.total_words))
	print('spell.word_frequency.unique_words = {}.'.format(spell.word_frequency.unique_words))
	print('spell.word_frequency.longest_word_length = {}.'.format(spell.word_frequency.longest_word_length))

def main():
	#sejong_sanitizer_test()
	#sejong_corpus_cleaner_test()

	#--------------------
	construct_dictionary_from_sejong_corpus()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
