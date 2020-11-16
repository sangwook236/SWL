import os, math, random, functools
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
	def _load_data_from_image_label_info(self, image_label_info_filepath, image_height, image_width, image_channel, max_label_len, image_label_separator=' ', is_preloaded_image_used=True):
		# In a image-label info file:
		#	Each line consists of 'image-filepath + image-label-separator + label'.

		try:
			with open(image_label_info_filepath, 'r', encoding='UTF8') as fd:
				#lines = fd.readlines()  # A list of strings.
				lines = fd.read().splitlines()  # A list of strings.
		except UnicodeDecodeError as ex:
			print('[SWL] Error: Unicode decode error, {}: {}.'.format(image_label_info_filepath, ex))
			raise
		except FileNotFoundError as ex:
			print('[SWL] Error: File not found, {}: {}.'.format(image_label_info_filepath, ex))
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
				label_int = self.label_converter.encode(label_str)  # Decorated/undecorated label ID.
			except Exception:
				print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.label_converter.decode(label_int):
				print('[SWL] Error: Mismatched original and decoded labels: {} != {}.'.format(label_str, self.label_converter.decode(label_int)))
				# TODO [check] >> I think such data should be used to deal with unknown characters (as negative data) in real data.
				#continue

			images.append(img if is_preloaded_image_used else img_fpath)
			labels_str.append(label_str)
			labels_int.append(label_int)

		##images = list(map(lambda image: self.resize(image), images))
		#images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)
		#images, _ = self.preprocess(images, None)

		return images, labels_str, labels_int

	# REF [function] >> FileBasedTextLineDatasetBase._load_data_from_image_and_label_files() in text_line_data.py
	def _load_data_from_image_and_label_files(self, image_filepaths, label_filepaths, image_height, image_width, image_channel, max_label_len, is_preloaded_image_used=True):
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
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error, {}: {}.'.format(lbl_fpath, ex))
				continue
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found, {}: {}.'.format(lbl_fpath, ex))
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
				label_int = self.label_converter.encode(label_str)  # Decorated/undecorated label ID.
			except Exception:
				print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.label_converter.decode(label_int):
				print('[SWL] Error: Mismatched original and decoded labels: {} != {}.'.format(label_str, self.label_converter.decode(label_int)))
				# TODO [check] >> I think such data should be used to deal with unknown characters (as negative data) in real data.
				#continue

			images.append(img if is_preloaded_image_used else img_fpath)
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

		self.chars = chars
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * image_channel, (0,) * image_channel)

	def __len__(self):
		return len(self.chars)

	def __getitem__(self, idx):
		while True:
			ch = self.chars[idx]
			target = self.label_converter.encode([ch])[0]  # Undecorated label ID.
			font_type, font_index = random.choice(self.fonts)
			font_size = random.randint(*self.font_size_interval)
			font_color, bg_color = self.color_functor()

			try:
				#image, mask = swl_langproc_util.generate_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
				image = swl_langproc_util.generate_simple_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)
			except OSError as ex:
				print('[SWL] Error: font_type = {}, font_index = {}, font_size = {}, char = {}: {}.'.format(font_type, font_index, font_size, ch, ex))
				continue

			#if image and image.mode != self.mode:
			#	image = image.convert(self.mode)
			#image = np.array(image, np.uint8)

			#if image: break
			if image.height * image.width > 0: break
			else:
				print('[SWL] Warning: Char generation failed, font: {}, font index: {}.'.format(font_type, font_index))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class NoisyCharacterDataset(TextDatasetBase):
	def __init__(self, label_converter, chars, image_channel, char_clipping_ratio_interval, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.chars = chars
		self.char_clipping_ratio_interval = char_clipping_ratio_interval
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * image_channel, (0,) * image_channel)

	def __len__(self):
		return len(self.chars)

	def __getitem__(self, idx):
		while True:
			ch = self.chars[idx]
			ch2 = random.sample(self.label_converter.tokens, 2)
			#ch2 = [random.choice(self.label_converter.tokens) for _ in range(2)]
			ch3 = ch2[0] + ch + ch2[1]
			target = self.label_converter.encode([ch])[0]  # Undecorated label ID.
			font_type, font_index = random.choice(self.fonts)
			font_size = random.randint(*self.font_size_interval)
			font_color, bg_color = self.color_functor()

			try:
				#image, mask = swl_langproc_util.generate_text_image(ch3, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
				image = swl_langproc_util.generate_simple_text_image(ch3, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)
			except OSError as ex:
				print('[SWL] Error: font_type = {}, font_index = {}, font_size = {}, char = {}: {}.'.format(font_type, font_index, font_size, ch3, ex))
				continue

			# FIXME [modify] >> It's an experimental implementation.
			alpha, beta = 0.75, 0.5  # Min. character width ratio and min. font width ratio.
			if True:
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

			#if image: break
			if image.height * image.width > 0: break
			else:
				print('[SWL] Warning: Char generation failed, font: {}, font index: {}.'.format(font_type, font_index))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)

		return image, target

#--------------------------------------------------------------------

class FileBasedCharacterDataset(FileBasedTextDatasetBase):
	def __init__(self, label_converter, image_label_info_filepath, image_channel, is_preloaded_image_used=True, transform=None, target_transform=None):
	#def __init__(self, label_converter, image_filepaths, label_filepaths, image_channel, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.target_transform = target_transform

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		image_label_separator = ','
		self.data_dir_path = os.path.dirname(image_label_info_filepath)
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_label_info(image_label_info_filepath, None, None, image_channel, max_label_len=1, image_label_separator=image_label_separator, is_preloaded_image_used=self.is_preloaded_image_used)
		#self.images, self.labels_str, self.labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, None, None, image_channel, max_label_len=1, is_preloaded_image_used=self.is_preloaded_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if self.is_preloaded_image_used:
			image = Image.fromarray(self.images[idx])
		else:
			fpath = os.path.join(self.data_dir_path, self.images[idx])
			try:
				image = Image.open(fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image, {}: {}.'.format(fpath, ex))
				image = None
		target = self.labels_int[idx][0]  # Undecorated label ID.

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
	def __init__(self, label_converter, words, num_examples, image_channel, max_word_len, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.words = words
		self.num_examples = num_examples
		#self.max_word_len = min(max_word_len, len(max(self.words, key=len))) if max_word_len else len(max(self.words, key=len))
		self.max_word_len = max_word_len if max_word_len else len(max(self.words, key=len))
		#assert self.max_word_len == max_word_len, 'Unmatched max. word length, {} != {}'.format(self.max_word_len, max_word_len)
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_word_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * image_channel, (0,) * image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		while True:
			#word = random.choice(self.words)
			word = random.sample(self.words, 1)[0][:self.max_word_len]
			target = [self.pad_id] * self.max_time_steps
			#target_len = len(word)
			#target[:target_len] = self.label_converter.encode(word)  # Undecorated label ID.
			word_id = self.label_converter.encode(word)  # Decorated/undecorated label ID.
			target_len = len(word_id)
			target[:target_len] = word_id
			font_type, font_index = random.choice(self.fonts)
			font_size = random.randint(*self.font_size_interval)
			font_color, bg_color = self.color_functor()

			try:
				#image, mask = swl_langproc_util.generate_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
				image = swl_langproc_util.generate_simple_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)
			except OSError as ex:
				print('[SWL] Error: font_type = {}, font_index = {}, font_size = {}, word = {}: {}.'.format(font_type, font_index, font_size, word, ex))
				continue

			#if image and image.mode != self.mode:
			#	image = image.convert(self.mode)
			#image = np.array(image, np.uint8)

			#if image: break
			if image.height * image.width > 0: break
			else:
				print('[SWL] Warning: Word generation failed, font: {}, font index: {}.'.format(font_type, font_index))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

#--------------------------------------------------------------------

class RandomWordDataset(TextDatasetBase):
	def __init__(self, label_converter, chars, num_examples, image_channel, max_word_len, word_len_interval, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.chars = chars
		self.num_examples = num_examples
		#self.max_word_len = min(max_word_len, word_len_interval[1]) if max_word_len else word_len_interval[1]
		self.max_word_len = max_word_len if max_word_len else word_len_interval[1]
		#assert self.max_word_len == max_word_len, 'Unmatched max. word length, {} != {}'.format(self.max_word_len, max_word_len)
		self.word_len_interval = word_len_interval
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_word_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * image_channel, (0,) * image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		while True:
			word_len = random.randint(*self.word_len_interval)
			#word = ''.join(random.sample(self.chars, word_len))[:self.max_word_len]
			word = ''.join(random.choice(self.chars) for _ in range(word_len))[:self.max_word_len]
			target = [self.pad_id] * self.max_time_steps
			#target_len = len(word)
			#target[:target_len] = self.label_converter.encode(word)  # Undecorated label ID.
			word_id = self.label_converter.encode(word)  # Decorated/undecorated label ID.
			target_len = len(word_id)
			target[:target_len] = word_id
			font_type, font_index = random.choice(self.fonts)
			font_size = random.randint(*self.font_size_interval)
			font_color, bg_color = self.color_functor()

			try:
				#image, mask = swl_langproc_util.generate_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size, image_size=None, text_offset=None, crop_text_area=True, char_space_ratio=None, mode=self.mode, mask=False, mask_mode='1')
				image = swl_langproc_util.generate_simple_text_image(word, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode=self.mode)
			except OSError as ex:
				print('[SWL] Error: font_type = {}, font_index = {}, font_size = {}, word = {}: {}.'.format(font_type, font_index, font_size, word, ex))
				continue

			#if image and image.mode != self.mode:
			#	image = image.convert(self.mode)
			#image = np.array(image, np.uint8)

			#if image: break
			if image.height * image.width > 0: break
			else:
				print('[SWL] Warning: Word generation failed, font: {}, font index: {}.'.format(font_type, font_index))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

#--------------------------------------------------------------------

class FileBasedWordDatasetBase(FileBasedTextDatasetBase):
	def __init__(self, label_converter, image_channel, max_word_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.max_word_len = max_word_len
		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_word_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.images = None

	def __len__(self):
		return len(self.images) if self.images is not None else 0

	def __getitem__(self, idx):
		if self.is_preloaded_image_used:
			image = Image.fromarray(self.images[idx])
		else:
			fpath = self.images[idx] if self.data_dir_path is None else os.path.join(self.data_dir_path, self.images[idx])
			try:
				image = Image.open(fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image, {}: {}.'.format(fpath, ex))
				image = None
		target = [self.pad_id] * self.max_time_steps
		word_id = self.labels_int[idx]  # Decorated/undecorated label ID.
		target_len = len(word_id)
		target[:target_len] = word_id

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

#--------------------------------------------------------------------

class InfoFileBasedWordDataset(FileBasedWordDatasetBase):
	def __init__(self, label_converter, image_label_info_filepath, image_channel, max_word_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter, image_channel, max_word_len, is_preloaded_image_used, transform, target_transform)

		image_label_separator = ','
		self.data_dir_path = os.path.dirname(image_label_info_filepath)
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_label_info(image_label_info_filepath, None, None, image_channel, max_label_len=max_word_len, image_label_separator=image_label_separator, is_preloaded_image_used=self.is_preloaded_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

#--------------------------------------------------------------------

class ImageLabelFileBasedWordDataset(FileBasedWordDatasetBase):
	def __init__(self, label_converter, image_filepaths, label_filepaths, image_channel, max_word_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter, image_channel, max_word_len, is_preloaded_image_used, transform, target_transform)

		self.data_dir_path = None
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, None, None, image_channel, max_label_len=max_word_len, is_preloaded_image_used=self.is_preloaded_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

#--------------------------------------------------------------------

class SimpleTextLineDataset(TextDatasetBase):
	def __init__(self, label_converter, words, num_examples, image_channel, max_textline_len, word_count_interval, space_count_interval, char_space_ratio_interval, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.words = words
		self.num_examples = num_examples
		self.max_textline_len = max_textline_len
		self.word_count_interval = word_count_interval
		self.space_count_interval = space_count_interval
		self.char_space_ratio_interval = char_space_ratio_interval
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_textline_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * image_channel, (0,) * image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		while True:
			words = random.sample(self.words, random.randint(*self.word_count_interval))	
			textline = functools.reduce(lambda t, w: t + ' ' * random.randint(*self.space_count_interval) + w, words[1:], words[0])[:self.max_textline_len]
			target = [self.pad_id] * self.max_time_steps
			#target_len = len(textline)
			#target[:len(textline)] = self.label_converter.encode(textline)  # Undecorated label ID.
			textline_id = self.label_converter.encode(textline)  # Decorated/undecorated label ID.
			target_len = len(textline_id)
			target[:target_len] = textline_id
			font_type, font_index = random.choice(self.fonts)
			font_size = random.randint(*self.font_size_interval)
			char_space_ratio = None if self.char_space_ratio_interval is None else random.uniform(*self.char_space_ratio_interval)
			font_color, bg_color = self.color_functor()

			try:
				#image, mask = swl_langproc_util.generate_text_image(textline, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mode=self.mode, mask=True, mask_mode='1')
				image = swl_langproc_util.generate_text_image(textline, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mode=self.mode)
			except OSError as ex:
				print('[SWL] Error: font_type = {}, font_index = {}, font_size = {}, textline = {}: {}.'.format(font_type, font_index, font_size, textline, ex))
				continue

			#if image and image.mode != self.mode:
			#	image = image.convert(self.mode)
			#image = np.array(image, np.uint8)

			#if image: break
			if image.height * image.width > 0: break
			else:
				print('[SWL] Warning: Text line generation failed, font: {}, font index: {}.'.format(font_type, font_index))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

#--------------------------------------------------------------------

class RandomTextLineDataset(TextDatasetBase):
	def __init__(self, label_converter, chars, num_examples, image_channel, max_textline_len, word_len_interval, word_count_interval, space_count_interval, char_space_ratio_interval, fonts, font_size_interval, color_functor=None, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.chars = chars
		self.num_examples = num_examples
		self.max_textline_len = max_textline_len
		self.word_len_interval = word_len_interval
		self.word_count_interval = word_count_interval
		self.space_count_interval = space_count_interval
		self.char_space_ratio_interval = char_space_ratio_interval
		self.fonts = fonts
		self.font_size_interval = font_size_interval
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_textline_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.color_functor = color_functor if color_functor else lambda: ((255,) * image_channel, (0,) * image_channel)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		while True:
			word_count = random.randint(*self.word_count_interval)
			#words = [''.join(random.sample(self.chars, random.randint(*self.word_len_interval))) for _ in range(word_count)]
			words = [''.join(random.choice(self.chars) for _ in range(random.randint(*self.word_len_interval))) for _ in range(word_count)]
			textline = functools.reduce(lambda t, w: t + ' ' * random.randint(*self.space_count_interval) + w, words[1:], words[0])[:self.max_textline_len]
			target = [self.pad_id] * self.max_time_steps
			#target_len = len(textline)
			#target[:target_len] = self.label_converter.encode(textline)  # Undecorated label ID.
			textline_id = self.label_converter.encode(textline)  # Decorated/undecorated label ID.
			target_len = len(textline_id)
			target[:target_len] = textline_id
			font_type, font_index = random.choice(self.fonts)
			font_size = random.randint(*self.font_size_interval)
			char_space_ratio = None if self.char_space_ratio_interval is None else random.uniform(*self.char_space_ratio_interval)
			font_color, bg_color = self.color_functor()

			try:
				#image, mask = swl_langproc_util.generate_text_image(textline, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mode=self.mode, mask=True, mask_mode='1')
				image = swl_langproc_util.generate_text_image(textline, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=char_space_ratio, mode=self.mode)
			except OSError as ex:
				print('[SWL] Error: font_type = {}, font_index = {}, font_size = {}, textline = {}: {}.'.format(font_type, font_index, font_size, textline, ex))
				continue

			#if image and image.mode != self.mode:
			#	image = image.convert(self.mode)
			#image = np.array(image, np.uint8)

			#if image: break
			if image.height * image.width > 0: break
			else:
				print('[SWL] Warning: Text line generation failed, font: {}, font index: {}.'.format(font_type, font_index))

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

#--------------------------------------------------------------------

class TextRecognitionDataGeneratorTextLineDataset(TextDatasetBase):
	def __init__(self, label_converter, lang, num_examples, image_channel, max_textline_len, font_filepaths, font_size, num_words, is_variable_length, is_randomly_generated=False, transform=None, target_transform=None, **kwargs):
		super().__init__(label_converter)

		self.num_examples = num_examples
		self.max_textline_len = max_textline_len
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_textline_len + label_converter.num_affixes

		import trdg.generators, trdg.string_generator, trdg.utils

		if lang == 'kr':
			# REF [function] >> korean_example() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/TextRecognitionDataGenerator_test.py

			num_strings_to_generate = num_examples

			if is_randomly_generated:
				use_letters, use_numbers, use_symbols = True, True, True

				# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_random.py
				# NOTE [warning] >> trdg.string_generator.create_strings_randomly() does not support Korean.
				#	In order to support Korean in the function, we have to change it.
				#strings = trdg.string_generator.create_strings_randomly(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, let=use_letters, num=use_numbers, sym=use_symbols, lang=lang)
				strings = self._create_strings_randomly(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, let=use_letters, num=use_numbers, sym=use_symbols, lang=lang)
			else:
				import text_generation_util as tg_util

				dictionary = list(tg_util.construct_word_set(korean=True, english=False))

				# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_dict.py
				strings = trdg.string_generator.create_strings_from_dict(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, lang_dict=dictionary)

			self.generator = trdg.generators.GeneratorFromStrings(
				strings=strings,
				language=lang,
				count=num_examples,
				fonts=font_filepaths, size=font_size,
				**kwargs
			)
		else:
			# REF [function] >> basic_example() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/TextRecognitionDataGenerator_test.py

			if is_randomly_generated:
				use_letters, use_numbers, use_symbols = True, True, True
				self.generator = trdg.generators.GeneratorFromRandom(
					length=num_words,
					allow_variable=is_variable_length,
					use_letters=use_letters, use_numbers=use_numbers, use_symbols=use_symbols,
					language=lang,
					count=num_examples,
					fonts=font_filepaths, size=font_size,
					**kwargs
				)
			else:
				self.generator = trdg.generators.GeneratorFromDict(
					length=num_words,
					allow_variable=is_variable_length,
					language=lang,
					count=num_examples,
					fonts=font_filepaths, size=font_size,
					**kwargs
				)

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

	def __len__(self):
		return self.num_examples

	def __getitem__(self, idx):
		while True:
			while True:
				try:
					image, text = next(self.generator)
					#(image, mask), text = next(self.generator)
					break
				except Exception as ex:
					print('[SWL] Warning: TRDG exception, {}.'.format(ex))
					continue

			if len(text) > self.max_textline_len:
				continue

			target = [self.pad_id] * self.max_time_steps
			#target_len = len(text)
			#target[:target_len] = self.label_converter.encode(text)  # Undecorated label ID.
			textline_id = self.label_converter.encode(text)  # Decorated/undecorated label ID.
			target_len = len(textline_id)
			target[:target_len] = textline_id

			if image and image.mode != self.mode:
				image = image.convert(self.mode)
			#image = np.array(image, np.uint8)

			#if image: break
			if image.height * image.width > 0: break
			else: print('[SWL] Warning: Text line generation failed.')

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

	# REF [function] >> create_strings_randomly() in https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/string_generator.py.
	@staticmethod
	def _create_strings_randomly(length, allow_variable, count, let, num, sym, lang):
		"""
			Create all strings by randomly sampling from a pool of characters.
		"""

		import string
		import text_generation_util as tg_util

		# If none specified, use all three
		if True not in (let, num, sym):
			let, num, sym = True, True, True

		pool = ""
		if let:
			if lang == 'kr':
				pool += tg_util.construct_charset(digit=False, alphabet_uppercase=False, alphabet_lowercase=False, punctuation=False, space=False, hangeul=True)
				#pool += tg_util.construct_charset(digit=False, alphabet_uppercase=True, alphabet_lowercase=True, punctuation=False, space=False, hangeul=True)
			elif lang == 'cn':
				pool += ''.join(
					[chr(i) for i in range(19968, 40908)]
				)  # Unicode range of CHK characters
			else:
				pool += string.ascii_letters
		if num:
			pool += '0123456789'
		if sym:
			pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"

		if lang == 'kr':
			min_seq_len = 2
			max_seq_len = 10
		elif lang == 'cn':
			min_seq_len = 1
			max_seq_len = 2
		else:
			min_seq_len = 2
			max_seq_len = 10

		strings = []
		for _ in range(0, count):
			current_string = ''
			for _ in range(0, random.randint(1, length) if allow_variable else length):
				seq_len = random.randint(min_seq_len, max_seq_len)
				current_string += ''.join([random.choice(pool) for _ in range(seq_len)])
				current_string += ' '
			strings.append(current_string[:-1])
		return strings

#--------------------------------------------------------------------

class FileBasedTextLineDatasetBase(FileBasedTextDatasetBase):
	def __init__(self, label_converter, image_channel, max_textline_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter)

		self.max_textline_len = max_textline_len
		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_textline_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.images = None

	def __len__(self):
		return len(self.images) if self.images is not None else 0

	def __getitem__(self, idx):
		if self.is_preloaded_image_used:
			image = Image.fromarray(self.images[idx])
		else:
			fpath = self.images[idx] if self.data_dir_path is None else os.path.join(self.data_dir_path, self.images[idx])
			try:
				image = Image.open(fpath)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image, {}: {}.'.format(fpath, ex))
				image = None
		target = [self.pad_id] * self.max_time_steps
		textline_id = self.labels_int[idx]  # Decorated/undecorated label ID.
		target_len = len(textline_id)
		target[:target_len] = textline_id

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return image, target, target_len

#--------------------------------------------------------------------

class InfoFileBasedTextLineDataset(FileBasedTextLineDatasetBase):
	def __init__(self, label_converter, image_label_info_filepath, image_channel, max_textline_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter, image_channel, max_textline_len, is_preloaded_image_used, transform, target_transform)

		image_label_separator = ','
		self.data_dir_path = os.path.dirname(image_label_info_filepath)
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_label_info(image_label_info_filepath, None, None, image_channel, max_label_len=max_textline_len, image_label_separator=image_label_separator, is_preloaded_image_used=self.is_preloaded_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)

#--------------------------------------------------------------------

class ImageLabelFileBasedTextLineDataset(FileBasedTextLineDatasetBase):
	def __init__(self, label_converter, image_filepaths, label_filepaths, image_channel, max_textline_len, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(label_converter, image_channel, max_textline_len, is_preloaded_image_used, transform, target_transform)

		self.data_dir_path = None
		self.images, self.labels_str, self.labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, None, None, image_channel, max_label_len=max_textline_len, is_preloaded_image_used=self.is_preloaded_image_used)
		assert len(self.images) == len(self.labels_str) == len(self.labels_int)
