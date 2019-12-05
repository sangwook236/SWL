#!/usr/bin/env python
# coding: UTF-8

import nltk
from nltk.corpus import treebank, reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def n_gram_language_model_example():
	nltk.download('punkt')
	#nltk.download('averaged_perceptron_tagger')
	#nltk.download('maxent_ne_chunker')
	#nltk.download('words')
	nltk.download('reuters')

	# Create a placeholder for model.
	model = defaultdict(lambda: defaultdict(lambda: 0))

	# Count frequency of co-occurance  .
	for sentence in reuters.sents():
		for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
			model[(w1, w2)][w3] += 1

	# Transform the counts to probabilities.
	for w1_w2 in model:
		total_count = float(sum(model[w1_w2].values()))
		for w3 in model[w1_w2]:
			model[w1_w2][w3] /= total_count

	#--------------------
	# Predict the next word.

	# Words which start with two simple words, 'today the'.
	print("model['today', 'the'] =", dict(model['today', 'the']))

	# Words which start with two simple words, 'the price'.
	print("model['the', 'price'] =", dict(model['the', 'price']))

	#--------------------
	# Generate a random piece of text using our n-gram model.

	# Starting words.
	text = ['today', 'the']
	sentence_finished = False

	import random
	while not sentence_finished:
		# Select a random probability threshold.
		r = random.random()
		accumulator = 0.0

		for word in model[tuple(text[-2:])].keys():
			accumulator += model[tuple(text[-2:])][word]
			# Select words that are above the probability threshold.
			if accumulator >= r:
				text.append(word)
				break

		if text[-2:] == [None, None]:
			sentence_finished = True

	print ('Generated text =', ' '.join([t for t in text if t]))

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def simple_neural_language_model_example():
	# Read the dataset.
	data_filepath = '../../data/language_processing/us_declaration of_independance.txt'
	try:
		with open(data_filepath, 'r', encoding='UTF8') as fd:
			data_text = fd.read()
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(data_filepath))
		return
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(data_filepath))
		return

	def text_cleaner(text):
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
	data_new = text_cleaner(data_text)

	def create_seq(text):
		length = 30
		sequences = list()
		for i in range(length, len(text)):
			# Select sequence of tokens.
			seq = text[i-length:i+1]
			# Store.
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
	#n_gram_language_model_example()
	simple_neural_language_model_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
