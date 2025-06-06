import math, random, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

#--------------------------------------------------------------------

# NOTE [info] >> In order to deal with "Can't pickle local object" error.
def decorate_token(x, prefix_ids, suffix_ids):
	return prefix_ids + x + suffix_ids

class TokenConverterBase(object):
	def __init__(self, tokens, unknown='<UNK>', sos=None, eos=None, pad=None, prefixes=None, suffixes=None, additional_tokens=None):
		"""
		Inputs:
			tokens (list of tokens): Tokens to be regarded as individual units.
			unknown (token): Unknown token.
			sos (token or None): A special token to use as <SOS> token. If None, <SOS> token is not used.
				All token sequences may start with the Start-Of-Sequence (SOS) token.
			eos (token or None): A special token to use as <EOS> token. If None, <EOS> token is not used.
				All token sequences may end with the End-Of-Sequence (EOS) token.
			pad (token, int, or None): A special token or integer token ID for padding, which may be not an actual token. If None, the pad token is not used.
			prefixes (list of tokens): Special tokens to be used as prefix.
			suffixes (list of tokens): Special tokens to be used as suffix.
		"""

		assert unknown is not None

		self.unknown, self.sos, self.eos = unknown, sos, eos
		if prefixes is None: prefixes = list()
		if suffixes is None: suffixes = list()
		if self.sos: prefixes = [self.sos] + prefixes
		if self.eos: suffixes += [self.eos]
		self._num_affixes = len(prefixes + suffixes)

		if additional_tokens:
			extended_tokens = tokens + additional_tokens + prefixes + suffixes + [self.unknown]
		else:
			extended_tokens = tokens + prefixes + suffixes + [self.unknown]
		#self._tokens = tokens
		self._tokens = extended_tokens

		self.unknown_id = extended_tokens.index(self.unknown)
		prefix_ids, suffix_ids = [extended_tokens.index(tok) for tok in prefixes], [extended_tokens.index(tok) for tok in suffixes]

		default_pad_id = -1 #len(extended_tokens)
		if pad is None:
			self._pad_id = default_pad_id
			self.pad = None
		elif isinstance(pad, int):
			self._pad_id = pad
			try:
				self.pad = extended_tokens[pad]
			except IndexError:
				self.pad = None
		else:
			try:
				self._pad_id = extended_tokens.index(pad)
				self.pad = pad
			except ValueError:
				self._pad_id = default_pad_id
				self.pad = None

		self.auxiliary_token_ids = [self._pad_id] + prefix_ids + suffix_ids
		self.decoration_functor = functools.partial(decorate_token, prefix_ids=prefix_ids, suffix_ids=suffix_ids)

		if self.eos:
			eos_id = extended_tokens.index(self.eos)
			#self.auxiliary_token_ids.remove(self.eos)  # TODO [decide] >>
			self.decode_functor = functools.partial(self._decode_with_eos, eos_id=eos_id)
		else:
			self.decode_functor = self._decode

	@property
	def num_tokens(self):
		return len(self._tokens)

	@property
	def tokens(self):
		return self._tokens

	@property
	def UNKNOWN(self):
		return self.unknown

	@property
	def SOS(self):
		return self.sos

	@property
	def EOS(self):
		return self.eos

	@property
	def PAD(self):
		return self.pad

	@property
	def pad_id(self):
		return self._pad_id

	@property
	def num_affixes(self):
		return self._num_affixes

	# Token sequence -> token ID sequence.
	def encode(self, seq, is_bare_output=False, *args, **kwargs):
		"""
		Inputs:
			seq (list of tokens): A sequence of tokens to encode.
			is_bare_output (bool): Specifies whether an encoded token ID sequence without prefixes and suffixes is returned or not.
		"""
		def tok2id(tok):
			try:
				return self._tokens.index(tok)
			except ValueError:
				#print('[SWL] Error: Failed to encode a token, {} in {}.'.format(tok, seq))
				return self.unknown_id
		id_seq = [tok2id(tok) for tok in seq]
		return id_seq if is_bare_output else self.decoration_functor(id_seq)

	# Token ID sequence -> token sequence.
	def decode(self, id_seq, is_string_output=True, *args, **kwargs):
		"""
		Inputs:
			id_seq (list of token IDs): A sequence of integer token IDs to decode.
			is_string_output (bool): Specifies whether the decoded output is a string or not.
		"""

		return self.decode_functor(id_seq, is_string_output, *args, **kwargs)

	# Token ID sequence -> token sequence.
	def _decode(self, id_seq, is_string_output=True, *args, **kwargs):
		def id2tok(tok):
			try:
				return self._tokens[tok]
			except IndexError:
				#print('[SWL] Error: Failed to decode a token ID, {} in {}.'.format(tok, id_seq))
				return self.unknown  # TODO [check] >> Is it correct?
		seq = [id2tok(tok) for tok in id_seq if tok not in self.auxiliary_token_ids]
		return ''.join(seq) if is_string_output else seq

	# Token ID sequence -> token sequence.
	def _decode_with_eos(self, id_seq, is_string_output=True, eos_id=None, *args, **kwargs):
		def id2tok(tok):
			try:
				return self._tokens[tok]
			except IndexError:
				#print('[SWL] Error: Failed to decode a token ID, {} in {}.'.format(tok, id_seq))
				return self.unknown  # TODO [check] >> Is it correct?
		"""
		try:
			id_seq = id_seq[:id_seq.index(eos_id)]  # NOTE [info] >> It is applied to list only.
		except ValueError:
			pass
		return self._decode(id_seq, is_string_output, *args, **kwargs)
		"""
		tokens = list()
		for tok in id_seq:
			if tok == eos_id: break
			elif tok in self.auxiliary_token_ids: continue
			else: tokens.append(id2tok(tok))
		return ''.join(tokens) if is_string_output else tokens
		"""
		def ff(tok):
			if tok == eos_id: raise StopIteration
			elif tok in self.auxiliary_token_ids: pass  # Error: return None.
			else: return id2tok(tok)
		try:
			tokens = map(ff, id_seq)
			#tokens = map(ff, filter(lambda tok: tok in self.auxiliary_token_ids, id_seq))
		except StopIteration:
			pass
		"""

class TokenConverter(TokenConverterBase):
	def __init__(self, tokens, unknown='<UNK>', sos=None, eos=None, pad=None, prefixes=None, suffixes=None):
		super().__init__(tokens, unknown, sos, eos, pad, prefixes, suffixes)

class JamoTokenConverter(TokenConverterBase):
	#def __init__(self, tokens, hangeul2jamo_functor, jamo2hangeul_functor, unknown='<UNK>', sos=None, eos=None, soj=None, eoj='<EOJ>', pad=None, prefixes=None, suffixes=None):
	def __init__(self, tokens, hangeul2jamo_functor, jamo2hangeul_functor, unknown='<UNK>', sos=None, eos=None, eoj='<EOJ>', pad=None, prefixes=None, suffixes=None):
		"""
		Inputs:
			tokens (list of tokens): Tokens to be regarded as individual units.
			hangeul2jamo_functor (functor): A functor to convert a Hangeul letter to a sequence of Jamos.
			jamo2hangeul_functor (functor): A functor to convert a sequence of Jamos to a Hangeul letter.
			unknown (token): Unknown token.
			sos (token or None): A special token to use as <SOS> token. If None, <SOS> token is not used.
				All token sequences may start with the Start-Of-Sequence (SOS) token.
			eos (token or None): A special token to use as <EOS> token. If None, <EOS> token is not used.
				All token sequences may end with the End-Of-Sequence (EOS) token.
			soj (token or None): A special token to use as <SOJ> token. If None, <SOJ> token is not used.
				All Hangeul jamo sequences may start with the Start-Of-Jamo-Sequence (SOJ) token.
			eoj (token or None): A special token to use as <EOJ> token. If None, <EOJ> token is not used.
				All Hangeul jamo sequences may end with the End-Of-Jamo-Sequence (EOJ) token.
			pad (token, int, or None): A special token or integer token ID for padding, which may be not an actual token. If None, the pad token is not used.
			prefixes (list of tokens): Special tokens to be used as prefix.
			suffixes (list of tokens): Special tokens to be used as suffix.
		"""

		#assert soj is not None and eoj is not None
		assert eoj is not None

		#super().__init__(tokens, unknown, sos, eos, pad, prefixes, suffixes, additional_tokens=[soj, eoj])
		super().__init__(tokens, unknown, sos, eos, pad, prefixes, suffixes, additional_tokens=[eoj])

		#self.soj, self.eoj = soj, eoj
		self.soj, self.eoj = None, eoj

		# TODO [check] >> This implementation using itertools.chain() may be slow.
		import itertools
		#self.hangeul2jamo_functor = hangeul2jamo_functor
		self.hangeul2jamo_functor = lambda hgstr: list(itertools.chain(*[[tt] if len(tt) > 1 else hangeul2jamo_functor(tt) for tt in hgstr]))
		self.jamo2hangeul_functor = jamo2hangeul_functor
		#self.jamo2hangeul_functor = lambda jmstr: list(itertools.chain(*[[tt] if len(tt) > 1 else jamo2hangeul_functor(tt) for tt in jmstr]))

	@property
	def SOJ(self):
		return self.soj

	@property
	def EOJ(self):
		return self.eoj

	# Token sequence -> token ID sequence.
	def encode(self, seq, is_bare_output=False, *args, **kwargs):
		"""
		Inputs:
			seq (list of tokens): A sequence of tokens to encode.
			is_bare_output (bool): Specifies whether an encoded token ID sequence without prefixes and suffixes is returned or not.
		"""

		try:
			return super().encode(self.hangeul2jamo_functor(seq), is_bare_output, *args, **kwargs)
		except Exception as ex:
			print('[SWL] Error: Failed to encode a token sequence: {}.'.format(seq))
			raise

	# Token ID sequence -> token sequence.
	def decode(self, id_seq, is_string_output=True, *args, **kwargs):
		"""
		Inputs:
			id_seq (list of token IDs): A sequence of integer token IDs to decode.
			is_string_output (bool): Specifies whether the decoded output is a string or not.
		"""

		try:
			return self.jamo2hangeul_functor(super().decode(id_seq, is_string_output, *args, **kwargs))
		except Exception as ex:
			print('[SWL] Error: Failed to decode a token ID sequence: {}.'.format(id_seq))
			raise

#--------------------------------------------------------------------

def compute_simple_text_matching_accuracy(text_pairs):
	total_text_count = len(text_pairs)
	correct_text_count = len(list(filter(lambda x: x[0] == x[1], text_pairs)))
	correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0
	for inf_text, gt_text in text_pairs:
		inf_words, gt_words = inf_text.split(' '), gt_text.split(' ')
		total_word_count += max(len(inf_words), len(gt_words))
		correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(inf_words, gt_words))))

		total_char_count += max(len(inf_text), len(gt_text))
		correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(inf_text, gt_text))))

	return correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count

def compute_sequence_matching_ratio(seq_pairs, isjunk=None):
	import difflib
	return functools.reduce(lambda total_ratio, pair: total_ratio + difflib.SequenceMatcher(isjunk, pair[0], pair[1]).ratio(), seq_pairs, 0) / len(seq_pairs)
	"""
	total_ratio = 0
	for inf, gt in seq_pairs:
		matcher = difflib.SequenceMatcher(isjunk, inf, gt)
		# sum(matched sequence lengths) / len(G/T).
		total_ratio += functools.reduce(lambda matched_len, mth: matched_len + mth.size, matcher.get_matching_blocks(), 0) / len(gt) if len(gt) > 0 else 0
	return total_ratio / len(seq_pairs)
	"""

def compute_string_distance(text_pairs):
	import jellyfish

	#string_distance_functor = jellyfish.hamming_distance
	string_distance_functor = jellyfish.levenshtein_distance
	#string_distance_functor = jellyfish.damerau_levenshtein_distance
	#string_distance_functor = jellyfish.jaro_distance
	#string_distance_functor = functools.partial(jellyfish.jaro_winkler, long_tolerance=False)
	#string_distance_functor = jellyfish.match_rating_comparison

	total_text_count = len(text_pairs)
	text_distance = functools.reduce(lambda ss, x: ss + string_distance_functor(x[0], x[1]), text_pairs, 0)
	word_distance, total_word_count, char_distance, total_char_count = 0, 0, 0, 0
	for inf_text, gt_text in text_pairs:
		inf_words, gt_words = inf_text.split(' '), gt_text.split(' ')
		total_word_count += max(len(inf_words), len(gt_words))
		word_distance += functools.reduce(lambda ss, x: ss + string_distance_functor(x[0], x[1]), zip(inf_words, gt_words), 0)

		total_char_count += max(len(inf_text), len(gt_text))
		char_distance += functools.reduce(lambda ss, x: ss + string_distance_functor(x[0], x[1]), zip(inf_text, gt_text), 0)

	return text_distance, word_distance, char_distance, total_text_count, total_word_count, total_char_count

def compute_sequence_precision_and_recall(seq_pairs, classes=None, isjunk=None):
	import difflib

	if classes is None:
		classes = list(zip(*seq_pairs))
		classes = sorted(functools.reduce(lambda x, txt: x.union(txt), classes[0] + classes[1], set()))

	"""
	# Too slow.
	def compute_metric(seq_pairs, cls):
		TP_FP, TP_FN, TP = 0, 0, 0
		for inf, gt in seq_pairs:
			TP_FP += inf.count(cls)  # Retrieved examples. TP + FP.
			TP_FN += gt.count(cls)  # Relevant examples. TP + FN.
			#TP += len(list(filter(lambda ig: ig[0] == ig[1] == cls, zip(inf, gt))))  # Too simple.
			#TP += sum([inf[mth.a:mth.a+mth.size].count(cls) for mth in difflib.SequenceMatcher(isjunk, inf, gt).get_matching_blocks() if mth.size > 0])
			TP = functools.reduce(lambda tot, mth: tot + inf[mth.a:mth.a+mth.size].count(cls) if mth.size > 0 else tot, difflib.SequenceMatcher(isjunk, inf, gt).get_matching_blocks(), TP)
		return TP_FP, TP_FN, TP

	# A list of (TP + FP, TP + FN, TP)'s.
	#return list(map(lambda cls: compute_metric(seq_pairs, cls), classes)), classes
	# A dictionary of {class: (TP + FP, TP + FN, TP)} pairs.
	return {cls: metric for cls, metric in zip(classes, map(lambda cls: compute_metric(seq_pairs, cls), classes))}
	"""
	metrics = {cls: [0, 0, 0] for cls in classes}  # A dictionary of {class: (TP + FP, TP + FN, TP)} pairs.
	for inf, gt in seq_pairs:
		#for cls in set(inf): metrics[cls][0] += inf.count(cls)  # Retrieved examples. TP + FP.
		#for cls in set(gt): metrics[cls][1] += gt.count(cls)  # Relevant examples. TP + FN.
		for cls in inf: metrics[cls][0] += 1  # Retrieved examples. TP + FP.
		for cls in gt: metrics[cls][1] += 1  # Relevant examples. TP + FN.
		matches = difflib.SequenceMatcher(isjunk, inf, gt).get_matching_blocks()
		for mth in matches:
			if mth.size > 0:
				#for cls in set(inf[mth.a:mth.a+mth.size]): metrics[cls][2] += inf[mth.a:mth.a+mth.size].count(cls)
				for cls in inf[mth.a:mth.a+mth.size]: metrics[cls][2] += 1
	return metrics

#--------------------------------------------------------------------

def compute_text_size(text, font_type, font_index, font_size):
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	'''
	font_offset = font.getoffset(text)  # (x, y).
	text_size = font.getsize(text)  # (width, height).
	return font_offset[0] + text_size[0], font_offset[1] + text_size[1]
	'''
	return font.getbbox(text)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).

def generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=None, mode='RGB', mask=False, mask_mode='1'):
	if char_space_ratio is None or 1 == char_space_ratio:
		if mask:
			return generate_simple_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, mode, mask_mode)
		else:
			return generate_simple_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, mode)
	else:
		if mask:
			return generate_per_character_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, char_space_ratio, mode, mask_mode)
		else:
			return generate_per_character_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, char_space_ratio, mode)

def generate_simple_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode='RGB'):
	if image_size is None:
		image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if font_color is None:
		#font_color = (random.randrange(256),) * image_depth  # Uses a random grayscale font color.
		font_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB font color.
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	# Draws text.
	draw.text(xy=text_offset, text=text, font=font, fill=font_color)

	if draw_text_border or crop_text_area:
		'''
		font_offset = font.getoffset(text)  # (x, y).
		#text_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
		text_size = draw.textsize(text, font=font)  # (width, height).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + font_offset[0] + text_size[0], text_offset[1] + font_offset[1] + text_size[1])
		'''
		text_bbox = font.getbbox(text)  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])
		#text_rect = (text_offset[0] + text_bbox[0], text_offset[1] + text_bbox[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)

	return img

def generate_simple_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode='RGB', mask_mode='1'):
	if image_size is None:
		image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if font_color is None:
		#font_color = (random.randrange(256),) * image_depth  # Uses a random grayscale font color.
		font_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB font color.
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw_img = ImageDraw.Draw(img)

	msk = Image.new(mode=mask_mode, size=image_size, color=0)
	#msk = Image.new(mode='1', size=image_size, color=0)  # {0, 1}, bool.
	#msk = Image.new(mode='L', size=image_size, color=0)  # [0, 255], uint8.
	draw_msk = ImageDraw.Draw(msk)

	# Draws text.
	draw_img.text(xy=text_offset, text=text, font=font, fill=font_color)
	draw_msk.text(xy=text_offset, text=text, font=font, fill=255)

	if draw_text_border or crop_text_area:
		'''
		font_offset = font.getoffset(text)  # (x, y).
		#text_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
		text_size = draw_img.textsize(text, font=font)  # (width, height).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + font_offset[0] + text_size[0], text_offset[1] + font_offset[1] + text_size[1])
		'''
		text_bbox = font.getbbox(text)  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])
		#text_rect = (text_offset[0] + text_bbox[0], text_offset[1] + text_bbox[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw_img.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)
			msk = msk.crop(text_rect)

	return img, msk

def generate_per_character_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=None, mode='RGB'):
	num_chars, num_newlines = len(text), text.count('\n')
	if image_size is None:
		image_size = (math.ceil(num_chars * font_size * char_space_ratio * 1.1), math.ceil((num_newlines + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	# Draws text.
	char_offset = list(text_offset)
	char_space = math.ceil(font_size * char_space_ratio)
	if font_color is None:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw.text(xy=char_offset, text=ch, font=font, fill=tuple(random.randrange(256) for _ in range(image_depth)))  # Random font color.
			char_offset[0] += char_space
	#elif len(font_colors) == num_chars:
	#	for idx, (ch, fcolor) in enumerate(zip(text, font_colors)):
	#		char_offset[0] = text_offset[0] + char_space * idx
	#		draw.text(xy=char_offset, text=ch, font=font, fill=fcolor)
	else:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw.text(xy=char_offset, text=ch, font=font, fill=font_color)
			char_offset[0] += char_space

	if draw_text_border or crop_text_area:
		'''
		font_offset = font.getoffset(text)  # (x, y).
		#text_size = list(font.getsize(text))  # (width, height). This is erroneous for multiline text.
		text_size = list(draw.textsize(text, font=font))  # (width, height).
		if num_chars > 1:
			max_chars_in_line = functools.reduce(lambda ll, line: max(ll, len(line)), text.splitlines(), 0)
			#text_size[0] = char_space * (max_chars_in_line - 1) + font_size
			text_size[0] = char_space * (max_chars_in_line - 1) + font.getsize(text[-1])[0]
			text_size[1] = (num_newlines + 1) * font_size
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + font_offset[0] + text_size[0], text_offset[1] + font_offset[1] + text_size[1])
		'''
		text_bbox = font.getbbox(text)  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])
		#text_rect = (text_offset[0] + text_bbox[0], text_offset[1] + text_bbox[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)

	return img

def generate_per_character_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=None, mode='RGB', mask_mode='1'):
	num_chars, num_newlines = len(text), text.count('\n')
	if image_size is None:
		image_size = (math.ceil(num_chars * font_size * char_space_ratio * 1.1), math.ceil((num_newlines + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw_img = ImageDraw.Draw(img)

	msk = Image.new(mode=mask_mode, size=image_size, color=0)
	#msk = Image.new(mode='1', size=image_size, color=0)  # {0, 1}, bool.
	#msk = Image.new(mode='L', size=image_size, color=0)  # [0, 255], uint8.
	draw_msk = ImageDraw.Draw(msk)

	# Draws text.
	char_offset = list(text_offset)
	char_space = math.ceil(font_size * char_space_ratio)
	if font_color is None:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw_img.text(xy=char_offset, text=ch, font=font, fill=tuple(random.randrange(256) for _ in range(image_depth)))  # Random font color.
			draw_msk.text(xy=char_offset, text=ch, font=font, fill=255)
			char_offset[0] += char_space
	#elif len(font_colors) == num_chars:
	#	for idx, (ch, fcolor) in enumerate(zip(text, font_colors)):
	#		char_offset[0] = text_offset[0] + char_space * idx
	#		draw_img.text(xy=char_offset, text=ch, font=font, fill=fcolor)
	#		draw_msk.text(xy=char_offset, text=ch, font=font, fill=255)
	else:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw_img.text(xy=char_offset, text=ch, font=font, fill=font_color)
			draw_msk.text(xy=char_offset, text=ch, font=font, fill=255)
			char_offset[0] += char_space

	if draw_text_border or crop_text_area:
		'''
		font_offset = font.getoffset(text)  # (x, y).
		#text_size = list(font.getsize(text))  # (width, height). This is erroneous for multiline text.
		text_size = list(draw_img.textsize(text, font=font))  # (width, height).
		if num_chars > 1:
			max_chars_in_line = functools.reduce(lambda ll, line: max(ll, len(line)), text.splitlines(), 0)
			#text_size[0] = char_space * (max_chars_in_line - 1) + font_size
			text_size[0] = char_space * (max_chars_in_line - 1) + font.getsize(text[-1])[0]
			text_size[1] = (num_newlines + 1) * font_size
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + font_offset[0] + text_size[0], text_offset[1] + font_offset[1] + text_size[1])
		'''
		text_bbox = font.getbbox(text)  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])
		#text_rect = (text_offset[0] + text_bbox[0], text_offset[1] + text_bbox[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw_img.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)
			msk = msk.crop(text_rect)

	return img, msk

def generate_text_mask_and_distribution(text, font, rotation_angle=None):
	import scipy.stats

	#text_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
	#text_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	text_size = font.getbbox(text)[2:]

	# Draw a distribution of character centers.
	text_pil = Image.new('L', text_size, 0)
	text_draw = ImageDraw.Draw(text_pil)
	text_draw.text(xy=(0, 0), text=text, font=font, fill=255)

	x, y = np.mgrid[0:text_pil.size[0], 0:text_pil.size[1]]
	#x, y = np.mgrid[0:text_pil.size[0]:0.5, 0:text_pil.size[1]:0.5]
	pos = np.dstack((x, y))
	text_pdf = np.zeros(x.shape, dtype=np.float32)
	offset = [0, 0]
	for ch in text:
		'''
		font_offset = font.getoffset(ch)  # (x, y).
		#char_size = font.getsize(ch)  # (width, height). This is erroneous for multiline text.
		char_size = text_draw.textsize(ch, font=font)  # (width, height).
		char_rect = (offset[0], offset[1], offset[0] + font_offset[0] + char_size[0], offset[1] + font_offset[1] + char_size[1])
		'''
		char_bbox = font.getbbox(ch)  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
		char_rect = (offset[0], offset[1], offset[0] + char_bbox[2], offset[1] + char_bbox[3])
		#char_rect = (offset[0] + char_bbox[0], offset[1] + char_bbox[1], offset[0] + char_bbox[2], offset[1] + char_bbox[3])

		if not ch.isspace():
			# TODO [decide] >> Which one is better?
			pts = cv2.findNonZero(np.array(text_pil)[char_rect[1]:char_rect[3],char_rect[0]:char_rect[2]])
			if pts is not None:
				pts += offset
				center, axis, angle = cv2.minAreaRect(pts)
				angle = math.radians(angle)
				"""
				try:
					pts = np.squeeze(pts, axis=1)
					center = np.mean(pts, axis=0)
					size = np.max(pts, axis=0) - np.min(pts, axis=0)
					pts = pts - center  # Centering.

					u, s, vh = np.linalg.svd(pts, full_matrices=True)
					center = center + offset
					#axis = s * max(size) / max(s)
					axis = s * math.sqrt((size[0] * size[0] + size[1] * size[1]) / (s[0] * s[0] + s[1] * s[1]))
					angle = math.atan2(vh[0,1], vh[0,0])
				except np.linalg.LinAlgError:
					print('np.linalg.LinAlgError raised.')
					raise
				"""

				cos_theta, sin_theta = math.cos(angle), math.sin(angle)
				R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
				# TODO [decide] >> Which one is better?
				#cov = np.diag(np.array(axis))  # 1 * sigma.
				cov = np.diag(np.array(axis) * 2)  # 2 * sigma.
				cov = np.matmul(R, np.matmul(cov, R.T))

				try:
					char_pdf = scipy.stats.multivariate_normal(center, cov, allow_singular=False).pdf(pos)
					#char_pdf = scipy.stats.multivariate_normal(center, cov, allow_singular=True).pdf(pos)

					# TODO [decide] >>
					char_pdf /= np.max(char_pdf)

					text_pdf = np.where(text_pdf >= char_pdf, text_pdf, char_pdf)
					#text_pdf += char_pdf
				except np.linalg.LinAlgError:
					print('[SWL] Warning: Singular covariance, {} of {}.'.format(ch, text))
			else:
				print('[SWL] Warning: No non-zero point in {} of {}.'.format(ch, text))

		offset[0] += char_size[0] + font_offset[0]

	#text_pdf /= np.sum(text_pdf)  # sum(text_pdf) = 1.
	#text_pdf /= np.max(text_pdf)
	text_pdf_pil = Image.fromarray(text_pdf.T)

	text_mask_pil = Image.new('L', text_size, 0)
	text_mask_draw = ImageDraw.Draw(text_mask_pil)
	text_mask_draw.text(xy=(0, 0), text=text, font=font, fill=255)
	if rotation_angle is not None:
		# Rotates the image around the top-left corner point.
		text_mask_pil = text_mask_pil.rotate(rotation_angle, expand=1)
		text_pdf_pil = text_pdf_pil.rotate(rotation_angle, expand=1)

	return np.asarray(text_mask_pil), np.asarray(text_pdf_pil)

#--------------------------------------------------------------------

def draw_text_on_image(img, text, font_type, font_index, font_size, font_color, text_offset=(0, 0), rotation_angle=None):
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	'''
	font_offset = font.getoffset(text)  # (x, y).
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	text_rect = (text_offset[0], text_offset[1], text_offset[0] + font_offset[0] + text_size[0], text_offset[1] + font_offset[1] + text_size[1])
	'''
	text_bbox = font.getbbox(text)  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
	text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])
	#text_rect = (text_offset[0] + text_bbox[0], text_offset[1] + text_bbox[1], text_offset[0] + text_bbox[2], text_offset[1] + text_bbox[3])

	bg_img = Image.fromarray(img)

	# Draws text.
	if rotation_angle is None:
		bg_draw = ImageDraw.Draw(bg_img)
		bg_draw.text(xy=text_offset, text=text, font=font, fill=font_color)

		text_mask = Image.new('L', bg_img.size, (0,))
		mask_draw = ImageDraw.Draw(text_mask)
		mask_draw.text(xy=text_offset, text=text, font=font, fill=(255,))

		x1, y1, x2, y2 = text_rect
		text_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
	else:
		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))
		sx0, sy0 = text_img.size

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.

		sx, sy = text_img.size
		bg_img.paste(text_img, (text_offset[0], text_offset[1], text_offset[0] + sx, text_offset[1] + sy), text_img)

		text_mask = Image.new('L', bg_img.size, (0,))
		text_mask.paste(text_img, (text_offset[0], text_offset[1], text_offset[0] + sx, text_offset[1] + sy), text_img)

		dx, dy = (sx0 - sx) / 2, (sy0 - sy) / 2
		x1, y1, x2, y2 = text_rect
		rect = (((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), -rotation_angle)
		text_bbox = cv2.boxPoints(rect)
		text_bbox = list(map(lambda xy: [xy[0] - dx, xy[1] - dy], text_bbox))

	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)
	return img, text_mask, text_bbox

def transform_text(text, tx, ty, rotation_angle, font, text_offset=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offset is None:
		text_offset = (0, 0)  # The coordinates (x, y) before transformation.
	'''
	font_offset = font.getoffset(text)  # (x, y).
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	'''
	text_size = font.getbbox(text)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).

	# z = -y.
	#	xy: left-handed, xz: right-handed.
	x1, z1 = transform(text_offset[0], -text_offset[1])
	x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
	x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
	x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
	xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])

	##x0, y0 = xmin, -zmax
	#text_bbox = np.array([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	dx, dy = xmin - tx, -zmax - ty
	#x0, y0 = xmin - dx, -zmax - dy
	text_bbox = np.array([[x1 - dx, -z1 - dy], [x2 - dx, -z2 - dy], [x3 - dx, -z3 - dy], [x4 - dx, -z4 - dy]])

	return text_bbox

def transform_text_on_image(text, tx, ty, rotation_angle, img, font, font_color, bg_color, text_offset=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offset is None:
		text_offset = (0, 0)  # The coordinates (x, y) before transformation.
	'''
	font_offset = font.getoffset(text)  # (x, y).
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	'''
	text_size = font.getbbox(text)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).

	# z = -y.
	#	xy: left-handed, xz: right-handed.
	x1, z1 = transform(text_offset[0], -text_offset[1])
	x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
	x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
	x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
	xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])

	#x0, y0 = xmin, -zmax
	#text_bbox = np.array([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	dx, dy = xmin - tx, -zmax - ty
	x0, y0 = xmin - dx, -zmax - dy
	text_bbox = np.array([[x1 - dx, -z1 - dy], [x2 - dx, -z2 - dy], [x3 - dx, -z3 - dy], [x4 - dx, -z4 - dy]])

	#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
	text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

	text_draw = ImageDraw.Draw(text_img)
	text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

	text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
	text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

	bg_img = Image.fromarray(img)
	bg_img.paste(text_img, text_rect, text_img)

	text_mask = Image.new('L', bg_img.size, (0,))
	text_mask.paste(text_img, text_rect, text_img)

	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)

	return text_bbox, img, text_mask

def transform_texts(texts, tx, ty, rotation_angle, font, text_offsets=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offsets is None:
		text_offsets, text_sizes = list(), list()
		max_text_height = 0
		for idx, text in enumerate(texts):
			'''
			if 0 == idx:
				text_offset = (0, 0)  # The coordinates (x, y) before transformation.
			else:
				prev_texts = ' '.join(texts[:idx]) + ' '
				text_size = font.getsize(prev_texts)  # (width, height).
				text_offset = (text_size[0], 0)  # (x, y).
			text_offsets.append(text_offset)

			font_offset = font.getoffset(text)  # (x, y).
			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			sx, sy = font_offset[0] + text_size[0], font_offset[1] + text_size[1]
			'''
			if 0 == idx:
				text_offset = (0, 0)  # The coordinates (x, y) before transformation.
			else:
				prev_texts = ' '.join(texts[:idx]) + ' '
				text_size = font.getbbox(prev_texts)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
				text_offset = (text_size[0], 0)  # (x, y).
			text_offsets.append(text_offset)

			sx, sy = font.getbbox(text)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
			text_sizes.append((sx, sy))
			if sy > max_text_height:
				max_text_height = sy
		tmp_text_offsets = list()
		for offset, sz in zip(text_offsets, text_sizes):
			dy = int(round((max_text_height - sz[1]) / 2))
			tmp_text_offsets.append((offset[0], offset[1] + dy))
		text_offsets = tmp_text_offsets
	else:
		if len(texts) != len(text_offsets):
			print('[SWL] Error: Unmatched lengths of texts and text offsets {} != {}.'.format(len(texts), len(text_offsets)))
			return None

		text_sizes = list()
		for text in texts:
			'''
			font_offset = font.getoffset(text)  # (x, y).
			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			text_sizes.append((font_offset[0] + text_size[0], font_offset[1] + text_size[1]))
			'''
			text_sizes.append(font.getbbox(text)[2:])  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).

	text_bboxes = list()
	"""
	for text, text_offset, text_size in zip(texts, text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])
		#x0, y0 = xmin, -zmax

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])

	return np.array(text_bboxes)
	"""
	xy0_list = list()
	for text_offset, text_size in zip(text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xy0_list.append((min([x1, x2, x3, x4]), -max([z1, z2, z3, z4])))

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	text_bboxes = np.array(text_bboxes)

	dxy = functools.reduce(lambda xym, xy0: (min(xym[0], xy0[0] - tx), min(xym[1], xy0[1] - ty)), xy0_list, (0, 0))
	text_bboxes[:,:] -= dxy

	return text_bboxes

def transform_texts_on_image(texts, tx, ty, rotation_angle, img, font, font_color, bg_color, text_offsets=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offsets is None:
		text_offsets, text_sizes = list(), list()
		max_text_height = 0
		for idx, text in enumerate(texts):
			'''
			if 0 == idx:
				text_offset = (0, 0)  # The coordinates (x, y) before transformation.
			else:
				prev_texts = ' '.join(texts[:idx]) + ' '
				text_size = font.getsize(prev_texts)  # (width, height).
				text_offset = (text_size[0], 0)  # (x, y).
			text_offsets.append(text_offset)

			font_offset = font.getoffset(text)  # (x, y).
			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			sx, sy = font_offset[0] + text_size[0], font_offset[1] + text_size[1]
			'''
			if 0 == idx:
				text_offset = (0, 0)  # The coordinates (x, y) before transformation.
			else:
				prev_texts = ' '.join(texts[:idx]) + ' '
				text_size = font.getbbox(prev_texts)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
				text_offset = (text_size[0], 0)  # (x, y).
			text_offsets.append(text_offset)

			sx, sy = font.getbbox(text)[2:]  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).
			text_sizes.append((sx, sy))
			if sy > max_text_height:
				max_text_height = sy
		tmp_text_offsets = list()
		for offset, sz in zip(text_offsets, text_sizes):
			dy = int(round((max_text_height - sz[1]) / 2))
			tmp_text_offsets.append((offset[0], offset[1] + dy))
		text_offsets = tmp_text_offsets
	else:
		if len(texts) != len(text_offsets):
			print('[SWL] Error: Unmatched lengths of texts and text offsets {} != {}.'.format(len(texts), len(text_offsets)))
			return None, None, None

		text_sizes = list()
		for text in texts:
			'''
			font_offset = font.getoffset(text)  # (x, y).
			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			text_sizes.append((font_offset[0] + text_size[0], font_offset[1] + text_size[1]))
			'''
			text_sizes.append(font.getbbox(text)[2:])  # (left, top, right, bottom). (x offset, y offset) = (left, top), (text width, text height) = (right, bottom).

	bg_img = Image.fromarray(img)
	text_mask = Image.new('L', bg_img.size, (0,))
	text_bboxes = list()
	"""
	for text, text_offset, text_size in zip(texts, text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])
		x0, y0 = xmin, -zmax

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])

		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
		text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

		bg_img.paste(text_img, text_rect, text_img)
		text_mask.paste(text_img, text_rect, text_img)
	"""
	xy0_list = list()
	for text_offset, text_size in zip(text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xy0_list.append((min([x1, x2, x3, x4]), -max([z1, z2, z3, z4])))

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	text_bboxes = np.array(text_bboxes)

	dxy = functools.reduce(lambda xym, xy0: (min(xym[0], xy0[0] - tx), min(xym[1], xy0[1] - ty)), xy0_list, (0, 0))
	text_bboxes[:,:] -= dxy

	for text, text_size, xy0 in zip(texts, text_sizes, xy0_list):
		x0, y0 = xy0[0] - dxy[0], xy0[1] - dxy[1]

		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
		text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

		bg_img.paste(text_img, text_rect, text_img)
		text_mask.paste(text_img, text_rect, text_img)

	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)

	return text_bboxes, img, text_mask

#--------------------------------------------------------------------

def draw_character_histogram(texts, charset=None):
	if charset is None:
		import string
		if True:
			charset = \
				string.ascii_uppercase + \
				string.ascii_lowercase + \
				string.digits + \
				string.punctuation + \
				' '
		else:
			hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
			#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
			#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
			with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
				#hangeul_charset = fd.read().strip('\n')  # A strings.
				hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
				#hangeul_charset = fd.readlines()  # A list of string.
				#hangeul_charset = fd.read().splitlines()  # A list of strings.
			#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
			#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
			hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

			charset = \
				hangeul_charset + \
				hangeul_jamo_charset + \
				string.ascii_uppercase + \
				string.ascii_lowercase + \
				string.digits + \
				string.punctuation + \
				' '

	charset = sorted(charset)
	#charset = ''.join(sorted(charset))

	#--------------------
	char_dict = dict()
	for ch in charset:
		char_dict[ch] = 0

	for txt in texts:
		if not txt:
			continue

		for ch in txt:
			try:
				char_dict[ch] += 1
			except KeyError:
				print('[SWL] Warning: Invalid character, {} in {}.'.format(ch, txt))

	#--------------------
	import numpy as np
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(10, 6))
	x_label = np.arange(len(char_dict.keys()))
	plt.bar(x_label, char_dict.values(), align='center', alpha=0.5)
	plt.xticks(x_label, char_dict.keys())
	plt.show()

	fig.savefig('./character_frequency.png')
	plt.close(fig)
