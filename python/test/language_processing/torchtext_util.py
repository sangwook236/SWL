import collections
import torchtext

# REF [function] >>
# 	Vocab.__init__() in https://github.com/pytorch/text/blob/master/torchtext/vocab.py.
#	Field.build_vocab() in https://github.com/pytorch/text/blob/master/torchtext/data/field.py.
def build_vocab_from_lexicon(lexicon, max_size=None, min_freq=1, specials=('<unk>', '<pad>'), specials_first=True, sort=True):
	counter = collections.Counter(lexicon)
	vocab = torchtext.vocab.Vocab(counter.copy())

	vocab.itos = list()
	vocab.unk_index = None
	if specials_first:
		vocab.itos = list(specials)
		# only extend max size if specials are prepended
		max_size = None if max_size is None else max_size + len(specials)

	# frequencies of special tokens are not counted when building vocabulary
	# in frequency order
	for tok in specials:
		del counter[tok]

	if sort:
		# sort by frequency, then alphabetically
		words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
		words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
	else:
		words_and_frequencies = counter.items()

	for word, freq in words_and_frequencies:
		if freq < min_freq or len(vocab.itos) == max_size:
			break
		vocab.itos.append(word)

	if torchtext.vocab.Vocab.UNK in specials:  # hard-coded for now
		unk_index = specials.index(torchtext.vocab.Vocab.UNK)  # position in list
		# account for ordering of specials, set variable
		vocab.unk_index = unk_index if specials_first else len(vocab.itos) + unk_index
		vocab.stoi = collections.defaultdict(vocab._default_unk_index)
	else:
		vocab.stoi = collections.defaultdict()

	if not specials_first:
		vocab.itos.extend(list(specials))

	# stoi is simply a reverse dict for itos
	vocab.stoi.update({tok: i for i, tok in enumerate(vocab.itos)})

	"""
	vocab.vectors = None
	if vectors is not None:
		vocab.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
	else:
		assert unk_init is None and vectors_cache is None
	"""

	return vocab
