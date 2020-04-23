import random
import numpy as np
import torch
import swl.language_processing.util as swl_langproc_util

#--------------------------------------------------------------------

class SingleCharacterDataset(torch.utils.data.Dataset):
	def __init__(self, charset, fonts, font_size_interval, num_examples, transform=None, target_transform=None):
		self.charset = charset
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.num_examples = num_examples
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		#ch = random.choice(self.charset)
		#target = self.charset.index(ch)
		target = random.randrange(len(self.charset))
		ch = self.charset[target]
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		image_depth = 1
		font_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale font color.
		bg_color = (random.randrange(128, 256),) * image_depth  # A light grayscale background color.
		#image, mask = swl_langproc_util.generate_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode='L', mask=False, mask_mode='1')
		image = swl_langproc_util.generate_simple_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode='L')
		#image = image.convert('RGB')
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return (image, target)

	@property
	def num_classes(self):
		return len(self.charset)

#--------------------------------------------------------------------

class SingleWordDataset(torch.utils.data.Dataset):
	UNKNOWN = '<UNK>'  # Unknown label token.

	def __init__(self, words, charset, fonts, font_size_interval, num_examples, transform=None, target_transform=None):
		self.words = words
		self.charset = list(charset) + [SingleWordDataset.UNKNOWN]
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.num_examples = num_examples
		self.transform = transform
		self.target_transform = target_transform

		self.max_word_len = len(max(self.words, key=len))
		self._default_value = -1

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		#word = random.choice(self.words)
		word = random.sample(self.words, 1)[0]
		target = [self.default_value,] * self.max_word_len
		target[:len(word)] = self.encode_label(word)
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		image_depth = 1
		font_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale font color.
		bg_color = (random.randrange(128, 256),) * image_depth  # A light grayscale background color.
		#image, mask = swl_langproc_util.generate_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode='L', mask=False, mask_mode='1')
		image = swl_langproc_util.generate_simple_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode='L')
		#image = image.convert('RGB')
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return (image, target)

	@property
	def num_classes(self):
		return len(self.charset)

	@property
	def default_value(self):
		return self._default_value

	# String label -> integer label.
	# REF [function] >> TextLineDatasetBase.encode_label() in text_line_data.py.
	def encode_label(self, label_str, *args, **kwargs):
		def label2index(ch):
			try:
				return self.charset.index(ch)
			except ValueError:
				print('[SWL] Error: Failed to encode a character, {} in {}.'.format(ch, label_str))
				return self.charset.index(SingleWordDataset.UNKNOWN)
		return list(label2index(ch) for ch in label_str)

	# Integer label -> string label.
	# REF [function] >> TextLineDatasetBase.decode_label() in text_line_data.py.
	def decode_label(self, label_int, *args, **kwargs):
		def index2label(id):
			try:
				return self.charset[id]
			except IndexError:
				print('[SWL] Error: Failed to decode an identifier, {} in {}.'.format(id, label_int))
				return SingleWordDataset.UNKNOWN  # TODO [check] >> Is it correct?
		return ''.join(list(index2label(id) for id in label_int if id != self._default_value))

#--------------------------------------------------------------------

class TextLineDataset(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

	def __getitem__(self, idx):
		raise NotImplementedError
