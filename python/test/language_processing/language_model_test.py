#!/usr/bin/env python
# coding: UTF-8

import nltk
from nltk.corpus import treebank, reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

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

	# Let's transform the counts to probabilities.
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

def main():
	n_gram_language_model_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
