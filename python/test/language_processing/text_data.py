import os, random, functools
import numpy as np
import torch
import cv2
import swl.language_processing.util as swl_langproc_util

#--------------------------------------------------------------------

class TextDatasetBase(torch.utils.data.Dataset):
	def __init__(self, label_converter):
		super().__init__()

		self._label_converter = label_converter

	@property
	def label_converter(self):
		return self._label_converter

#--------------------------------------------------------------------

class FileBasedTextDatasetBase(TextDatasetBase):
	def __init__(self, label_converter):
		super().__init__(label_converter)

	# REF [function] >> FileBasedTextLineDatasetBase._load_data_from_image_label_info() in text_line_data.py
	def _load_data_from_image_label_info(self, image_label_info_filepath, image_height, image_width, image_channel, max_label_len, image_label_separator=' ', is_image_used=True):
		# In a image-label info file:
		#	Each line consists of 'image-filepath + image-label-separator + label'.

		try:
			with open(image_label_info_filepath, 'r', encoding='UTF8') as fd:
				#lines = fd.readlines()  # A list of strings.
				lines = fd.read().splitlines()  # A list of strings.
		except FileNotFoundError as ex:
			print('[SWL] Error: File not found: {}.'.format(image_label_info_filepath))
			raise
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error: {}.'.format(image_label_info_filepath))
			raise

		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		dir_path = os.path.dirname(image_label_info_filepath)
		images, labels_str, labels_int = list(), list(), list()
		for line in lines:
			img_fpath, label_str = line.split(image_label_separator, 1)

			if len(label_str) > max_label_len:
				print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
				continue
			fpath = os.path.join(dir_path, img_fpath)
			img = cv2.imread(fpath, flag)
			if img is None:
				print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
				continue

			#img = self.resize(img, None, image_height, image_width)
			try:
				label_int = self.label_converter.encode(label_str)
			except Exception:
				#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.label_converter.decode(label_int):
				print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.label_converter.decode(label_int)))
				continue

			images.append(img if is_image_used else img_fpath)
			labels_str.append(label_str)
			labels_int.append(label_int)

		##images = list(map(lambda image: self.resize(image), images))
		#images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)
		#images, _ = self.preprocess(images, None)

		return images, labels_str, labels_int

	# REF [function] >> FileBasedTextLineDatasetBase._load_data_from_image_and_label_files() in text_line_data.py
	def _load_data_from_image_and_label_files(self, image_filepaths, label_filepaths, image_height, image_width, image_channel, max_label_len, is_image_used=True):
		if len(image_filepaths) != len(label_filepaths):
			print('[SWL] Error: Different lengths of image and label files, {} != {}.'.format(len(image_filepaths), len(label_filepaths)))
			return
		for img_fpath, lbl_fpath in zip(image_filepaths, label_filepaths):
			img_fname, lbl_fname = os.path.splitext(os.path.basename(img_fpath))[0], os.path.splitext(os.path.basename(lbl_fpath))[0]
			if img_fname != lbl_fname:
				print('[SWL] Warning: Different file names of image and label pair, {} != {}.'.format(img_fname, lbl_fname))
				continue

		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		images, labels_str, labels_int = list(), list(), list()
		for img_fpath, lbl_fpath in zip(image_filepaths, label_filepaths):
			try:
				with open(lbl_fpath, 'r', encoding='UTF8') as fd:
					#label_str = fd.read()
					#label_str = fd.read().rstrip()
					label_str = fd.read().rstrip('\n')
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found: {}.'.format(lbl_fpath))
				continue
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error: {}.'.format(lbl_fpath))
				continue
			if len(label_str) > max_label_len:
				print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
				continue
			img = cv2.imread(img_fpath, flag)
			if img is None:
				print('[SWL] Error: Failed to load an image: {}.'.format(img_fpath))
				continue

			#img = self.resize(img, None, image_height, image_width)
			try:
				label_int = self.label_converter.encode(label_str)
			except Exception:
				#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.label_converter.decode(label_int):
				print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.label_converter.decode(label_int)))
				continue

			images.append(img if is_image_used else img_fpath)
			labels_str.append(label_str)
			labels_int.append(label_int)

		##images = list(map(lambda image: self.resize(image), images))
		#images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)
		#images, _ = self.preprocess(images, None)

		return images, labels_str, labels_int

#--------------------------------------------------------------------

class SimpleCharacterDataset(TextDatasetBase):
	def __init__(self, label_converter, chars, image_channel, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.image_channel = image_channel
		self.chars = chars
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * self.image_channel, (0,) * self.image_channel)

	def __len__(self):
		return len(self.chars)

	def __getitem__(self, idx):
		ch = self.chars[idx]
		target = self.label_converter.encode([ch])[0]
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		font_color, bg_color = self.color_functor()

		#image, mask = swl_langproc_util.generate_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
		image = swl_langproc_util.generate_simple_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)

		#if image and image.mode != self.mode:
		#	image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class NoisyCharacterDataset(TextDatasetBase):
	def __init__(self, label_converter, chars, image_channel, fonts, font_size_interval, char_clipping_ratio_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.image_channel = image_channel
		self.chars = chars
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.char_clipping_ratio_interval = char_clipping_ratio_interval
		self.transform = transform
		self.target_transform = target_transform

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * self.image_channel, (0,) * self.image_channel)

	def __len__(self):
		return len(self.chars)

	def __getitem__(self, idx):
		ch = self.chars[idx]
		ch2 = random.sample(self.label_converter.tokens, 2)
		#ch2 = [random.choice(self.label_converter.tokens) for _ in range(2)]
		ch3 = ch2[0] + ch + ch2[1]
		target = self.label_converter.encode([ch])[0]
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		font_color, bg_color = self.color_functor()

		#image, mask = swl_langproc_util.generate_text_image(ch3, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
		image = swl_langproc_util.generate_simple_text_image(ch3, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)

		# FIXME [modify] >> It's an experimental implementation.
		alpha, beta = 0.75, 0.5  # Min. character width ratio and min. font width ratio.
		if True:
			import math
			from PIL import Image, ImageDraw, ImageFont

			image_size = (math.ceil(len(ch3) * font_size * 1.1), math.ceil((ch3.count('\n') + 1) * font_size * 1.1))
			draw_img = Image.new(mode=self.mode, size=image_size, color=bg_color)
			draw = ImageDraw.Draw(draw_img)
			font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

			ch_widths = [draw.textsize(ch, font=font)[0] for ch in ch3]
			ch_width = max(alpha * ch_widths[1], beta * font_size)
			left_margin, right_margin = ch_widths[0] * random.uniform(*self.char_clipping_ratio_interval), ch_widths[2] * random.uniform(*self.char_clipping_ratio_interval)

			if image.size[0] - (left_margin + right_margin) < ch_width:
				ratio = (image.size[0] - ch_width) / (left_margin + right_margin)
				left_margin, right_margin = math.floor(ratio * left_margin), math.floor(ratio * right_margin)
		else:
			import math

			ch_width = alpha * font_size #max(alpha, beta) * font_size
			left_margin, right_margin = font_size * random.uniform(*self.char_clipping_ratio_interval), font_size * random.uniform(*self.char_clipping_ratio_interval)

			if image.size[0] - (left_margin + right_margin) < ch_width:
				ratio = (image.size[0] - ch_width) / (left_margin + right_margin)
				left_margin, right_margin = math.floor(ratio * left_margin), math.floor(ratio * right_margin)
		image = image.crop((left_margin, 0, image.size[0] - right_margin, image.size[1]))
		assert image.size[0] > 0 and image.size[1] > 0

		#if image and image.mode != self.mode:
		#	image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class FileBasedCharacterDataset(FileBasedTextDatasetBase):
	def __init__(self, label_converter, image_label_info_filepath, image_channel, is_image_used=True, transform=None, target_transform=None):
	#def __init__(self, label_converter, image_filepaths, label_filepaths, image_channel, is_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.image_channel = image_channel
		self.is_image_used = is_image_used
		self.transform = transform
		self.target_transform = target_transform

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		image_label_separator = ','
		self.data_dir_path = os.path.dirname(image_label_info_filepath)
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_label_info(image_label_info_filepath, None, None, self.image_channel, max_label_len=1, image_label_separator=image_label_separator, is_image_used=self.is_image_used)
		#self.images, self.labels_str, self.labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, None, None, self.image_channel, max_label_len=1, is_image_used=self.is_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		from PIL import Image

		if self.is_image_used:
			image = Image.fromarray(self.images[idx])
		else:
			fpath = os.path.join(self.data_dir_path, self.images[idx])
			try:
				image = Image.open(fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
				image = None
		target = self.labels_int[idx][0]

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class SimpleWordDataset(TextDatasetBase):
	def __init__(self, label_converter, words, num_examples, image_channel, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.words = words
		self.num_examples = num_examples
		self.image_channel = image_channel
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform
		self.max_word_len = len(max(self.words, key=len))

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * self.image_channel, (0,) * self.image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		#word = random.choice(self.words)
		word = random.sample(self.words, 1)[0]
		target = [self.label_converter.nil_token] * self.max_word_len
		target[:len(word)] = self.label_converter.encode(word)
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		font_color, bg_color = self.color_functor()

		#image, mask = swl_langproc_util.generate_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
		image = swl_langproc_util.generate_simple_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)

		#if image and image.mode != self.mode:
		#	image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class RandomWordDataset(TextDatasetBase):
	def __init__(self, label_converter, chars, num_examples, image_channel, char_len_interval, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.chars = chars
		self.num_examples = num_examples
		self.image_channel = image_channel
		self.char_len_interval = char_len_interval
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform
		self.max_word_len = char_len_interval[1]

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * self.image_channel, (0,) * self.image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		char_len = random.randint(*self.char_len_interval)
		#word = ''.join(random.sample(self.chars, char_len))
		word = ''.join(random.choice(self.chars) for _ in range(char_len))
		target = [self.label_converter.nil_token] * self.max_word_len
		target[:len(word)] = self.label_converter.encode(word)
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		font_color, bg_color = self.color_functor()

		#image, mask = swl_langproc_util.generate_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
		image = swl_langproc_util.generate_simple_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)

		#if image and image.mode != self.mode:
		#	image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class FileBasedWordDataset(FileBasedTextDatasetBase):
	def __init__(self, label_converter, image_label_info_filepath, image_channel, max_word_len, is_image_used=True, transform=None, target_transform=None):
	#def __init__(self, label_converter, image_filepaths, label_filepaths, image_channel, max_word_len, is_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.image_channel = image_channel
		self.max_word_len = max_word_len
		self.is_image_used = is_image_used
		self.transform = transform
		self.target_transform = target_transform

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		image_label_separator = ','
		self.data_dir_path = os.path.dirname(image_label_info_filepath)
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_label_info(image_label_info_filepath, None, None, self.image_channel, max_label_len=self.max_word_len, image_label_separator=image_label_separator, is_image_used=self.is_image_used)
		#self.images, self.labels_str, self.labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, None, None, self.image_channel, max_label_len=self.max_word_len, is_image_used=self.is_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		from PIL import Image

		if self.is_image_used:
			image = Image.fromarray(self.images[idx])
		else:
			fpath = os.path.join(self.data_dir_path, self.images[idx])
			try:
				image = Image.open(fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image: {}.'.format(fpath))
				image = None
		label = self.labels_int[idx]
		target = [self.label_converter.nil_token] * self.max_word_len
		target[:len(label)] = label

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class SimpleTextLineDataset(TextDatasetBase):
	def __init__(self, label_converter, words, num_examples, image_height, image_width, image_channel, max_text_len, fonts, font_size_interval, char_space_ratio_interval, word_count_interval, space_count_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.words = words
		self.num_examples = num_examples
		self.image_height, self.image_width, self.image_channel = image_height, image_width, image_channel
		self.max_text_len = max_text_len
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.char_space_ratio_interval = char_space_ratio_interval
		self.word_count_interval = word_count_interval
		self.space_count_interval = space_count_interval
		self.transform = transform
		self.target_transform = target_transform

		if self.image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif self.image_channel == 3:
			self.mode = 'RGB'
		elif self.image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(self.image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * self.image_channel, (0,) * self.image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		words = random.sample(self.words, random.randint(*self.word_count_interval))	
		textline = functools.reduce(lambda t, w: t + ' ' * random.randint(*self.space_count_interval) + w, words[1:], words[0])[:self.max_text_len]
		target = [self.label_converter.nil_token] * self.max_text_len
		target[:len(textline)] = self.label_converter.encode(textline)
		font_type, font_index = random.choice(self.fonts)
		font_size = random.randint(*self.font_size_interval)
		char_space_ratio = None if self.char_space_ratio_interval is None else random.uniform(*self.char_space_ratio_interval)
		font_color, bg_color = self.color_functor()

		#image, mask = swl_langproc_util.generate_text_image(textline, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mode=self.mode, mask=True, mask_mode='1')
		image = swl_langproc_util.generate_text_image(textline, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mode=self.mode)

		#if image and image.mode != self.mode:
		#	image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

	@property
	def shape(self):
		return self.image_height, self.image_width, self.image_channel
