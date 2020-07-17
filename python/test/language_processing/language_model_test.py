#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
#import pandas as pd

# REF [function] >> sejong_sanitizer_test() in corpus_test.py.
def load_sejong_corpus():
	corpus_dir_path = '../../data/language_processing/sejong_corpus'

	corpus_filepaths = [
		#corpus_dir_path + '/colloquial_word_to_morph.txt',
		corpus_dir_path + '/colloquial_word_to_morphpos.txt',
		#corpus_dir_path + '/written_word_to_morph.txt',
		corpus_dir_path + '/written_word_to_morphpos.txt'
	]

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
	for line in lines:
		pos = line.find('\t')
		words.append(line[:pos])
	del lines

	return words

# REF [site] >> https://github.com/githubharald/CTCWordBeamSearch/blob/master/py/LanguageModel.py
def n_gram_language_model_with_prefix_tree_test():
	import ctc_word_beam_search.LanguageModel
	import text_generation_util as tg_util

	beamWidth = 10
	useNGrams = True

	# Sejong corpus.
	print('Start loading Sejong corpus...')
	start_time = time.time()
	words = load_sejong_corpus()
	words = '\n'.join(words)
	print('End loading Sejong corpus: {} secs.'.format(time.time() - start_time))

	# #classes = #chars + blank label.
	chars = tg_util.construct_charset(space=False)
	wordChars = tg_util.construct_charset(digit=False, punctuation=False, space=False)
	#chars += "£§àâèéê⊥"
	#wordChars += "'§àâèéê"
	wordChars += "'"

	#--------------------
	# Create a language model.
	print('Start creating a language model...')
	start_time = time.time()
	langModel = ctc_word_beam_search.LanguageModel.LanguageModel(words, chars, wordChars)
	print('End creating a language model: {} secs.'.format(time.time() - start_time))
	del words

	print("langModel.getNextWords('대한') =", langModel.getNextWords('대한'))
	print("langModel.getNextChars('석가') =", langModel.getNextChars('석가'))
	print("langModel.isWord('장난감') =", langModel.isWord('장난감'))
	print("langModel.isWord('가몹') =", langModel.isWord('가몹'))
	print("langModel.getUnigramProb('아빠') =", langModel.getUnigramProb('아빠'))
	print("langModel.getUnigramProb('아세') =", langModel.getUnigramProb('아세'))
	print("langModel.getBigramProb('대한', '민국') =", langModel.getBigramProb('대한', '민국'))
	print("langModel.getBigramProb('대한', '원소') =", langModel.getBigramProb('대한', '원소'))

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def simple_neural_language_model_example():
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import LSTM, Dense, GRU, Embedding
	from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
	from tensorflow.keras.utils import to_categorical
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	from sklearn.model_selection import train_test_split

	# Read the dataset.
	data_filepath = '../../data/language_processing/us_declaration_of_independance.txt'
	try:
		with open(data_filepath, 'r', encoding='UTF8') as fd:
			data_text = fd.read()
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(data_filepath))
		return
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(data_filepath))
		return

	def clean_text(text):
		import re
		# Lower case text.
		newString = text.lower()
		newString = re.sub(r"'s\b", '', newString)
		# Remove punctuations.
		newString = re.sub('[^a-zA-Z]', ' ', newString)
		long_words = []
		# Remove short word.
		for i in newString.split():
			if len(i) >= 3:
				long_words.append(i)
		return (' '.join(long_words)).strip()

	# Preprocess the text.
	data_new = clean_text(data_text)

	def create_seq(text):
		length = 30
		sequences = list()
		for i in range(length, len(text)):
			# Select sequence of tokens.
			seq = text[i-length:i+1]
			sequences.append(seq)
		print('Total Sequences: %d' % len(sequences))
		return sequences

	# Create sequences.
	sequences = create_seq(data_new)

	# Create a character mapping index.
	chars = sorted(list(set(data_new)))
	mapping = dict((c, i) for i, c in enumerate(chars))

	def encode_seq(seq):
		sequences = list()
		for line in seq:
			# Integer encode line.
			encoded_seq = [mapping[char] for char in line]
			# Store.
			sequences.append(encoded_seq)
		return sequences

	# Encode the sequences.
	sequences = encode_seq(sequences)

	# Vocabulary size.
	vocab = len(mapping)
	sequences = np.array(sequences)
	# Create X and y.
	X, y = sequences[:,:-1], sequences[:,-1]
	# One hot encode y.
	y = to_categorical(y, num_classes=vocab)
	# Create train and validation sets.
	X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

	print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

	#--------------------
	# Define model.
	model = Sequential()
	model.add(Embedding(vocab, 50, input_length=30, trainable=True))
	model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
	model.add(Dense(vocab, activation='softmax'))
	print(model.summary())

	# Compile the model.
	model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
	# Fit the model.
	model.fit(X_tr, y_tr, epochs=100, verbose=2, validation_data=(X_val, y_val))

	#--------------------
	# Generate a sequence of characters with a language model.
	def generate_seq(model, mapping, seq_length, seed_text, n_chars):
		in_text = seed_text
		# Generate a fixed number of characters.
		for _ in range(n_chars):
			# Encode the characters as integers.
			encoded = [mapping[char] for char in in_text]
			# Truncate sequences to a fixed length.
			encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
			# Predict character.
			yhat = model.predict_classes(encoded, verbose=0)
			# Reverse map integer to character.
			out_char = ''
			for char, index in mapping.items():
				if index == yhat:
					out_char = char
					break
			# Append to input.
			in_text += char
		return in_text

	inp = 'large armies'
	#inp = 'large armies of'
	#inp = 'large armies of '
	#inp = 'large armies for'
	#inp = 'large armies for '
	print('len(inp) = ', len(inp))
	print('Generated sequence =', generate_seq(model, mapping, 30, inp.lower(), 15))

def main():
	#--------------------
	# n-gram language model.

	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/nltk_test.py

	n_gram_language_model_with_prefix_tree_test()

	#--------------------
	# Neural language model.

	#simple_neural_language_model_example()

	#--------------------
	# Transformer.

	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/gpt2_test.py
	# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/language_processing/transformers_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
