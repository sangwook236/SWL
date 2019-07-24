import os, random, functools
import numpy as np
import cv2
import text_generation_util as tg_util
import hangeul_util as hg_util

#--------------------------------------------------------------------

class TextLineDataset(object):
	def __init__(self, char_labels):
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.

		num_labels = len(char_labels)

		char_labels = list(char_labels) + [self._EOS]
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
		self._num_classes = len(char_labels) + 1  # #labels + EOS + blank label.
		#self._eos_token_label = self._num_classes - 2
		self._blank_label = self._num_classes - 1

		self._label_int2char = char_labels
		self._label_char2int = {c:i for i, c in enumerate(char_labels)}

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def eos_token(self):
		return self._label_char2int[self._EOS]

	def create_batch_generator(self, image_height, image_width, batch_size, charsets, num_char_repetitions, min_char_count, max_char_count, min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color=None, bg_color=None):
		word_set = set()
		for charset in charsets:
			#word_set = word_set.union(tg_util.generate_random_word_set(num_chars, charset, min_char_count, max_char_count))
			word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, charset, min_char_count, max_char_count))

		characterTransformer = tg_util.IdentityTransformer()
		#characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
		textGenerator = tg_util.MySimplePrintedHangeulTextGenerator(characterTransformer, characterAlphaMattePositioner)

		batch_generator = self._generate_text_lines(word_set, textGenerator, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), image_height, image_width, batch_size, font_color, bg_color)

		return batch_generator

	# String data -> numeric data.
	def to_numeric(self, str_data):
		max_label_len = functools.reduce(lambda x, y: max(x, len(y)), str_data, 0)
		num_data = np.full((len(str_data), max_label_len), self._label_char2int[self._EOS], dtype=np.int16)
		for (idx, st) in enumerate(str_data):
			num_data[idx,:len(st)] = np.array(list(self._label_char2int[ch] for ch in st))
		return num_data

	# Numeric data -> string data.
	def to_string(self, num_data):
		def num2str(num):
			#label = list(self._label_int2char[n] for n in num)
			label = list(self._label_int2char[n] for n in num if n < self._blank_label)
			try:
				label = label[:label.index(self._EOS)]
			except ValueError:
				pass  # Uses the whole label.
			return ''.join(label)
		return list(map(num2str, num_data))

	# REF [function] >> generate_text_lines() in text_generation_util.py.
	def _generate_text_lines(self, word_set, textGenerator, font_size_interval, char_space_ratio_interval, image_height, image_width, batch_size, font_color=None, bg_color=None):
		sceneTextGenerator = tg_util.MySceneTextGenerator(tg_util.IdentityTransformer())

		scene_list, scene_text_mask_list, text_list = list(), list(), list()
		step = 0
		while True:
			font_size = random.randint(*font_size_interval)
			char_space_ratio = random.uniform(*char_space_ratio_interval)

			text = random.sample(word_set, 1)[0]

			char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space_ratio, font_size)
			text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

			if bg_color is None:
				# Grayscale background.
				bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
			else:
				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

			scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
			scene = cv2.resize(scene, (image_width, image_height), interpolation=cv2.INTER_AREA)
			scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
			scene = scene.reshape(scene.shape + (-1,))
			#scene_text_mask = cv2.resize(scene_text_mask, (image_width, image_height), interpolation=cv2.INTER_AREA)
			scene_list.append(scene)
			#scene_text_mask_list.append(scene_text_mask)
			text_list.append(text)

			step += 1
			if 0 == step % batch_size:
				#yield scene_list, scene_text_mask_list, text_list
				yield self._preprocess(np.array(scene_list, dtype=np.float32), self.to_numeric(text_list))
				scene_list, scene_text_mask_list, text_list = list(), list(), list()
				step = 0

	def _preprocess(self, data, labels):
		data = (data / 255.0) * 2 - 1  # [-1, 1].

		#labels = tf.keras.utils.to_categorical(labels, self._num_classes, np.int16)
		#labels = labels.astype(np.int16)

		# (samples, height, width, channels) -> (samples, width, height, channels).
		#data = data.transpose((0, 2, 1, 3))

		return data, labels

	def display_data(self, batch_generator):
		for data_batch, label_batch in batch_generator:
			idx = 0
			label_batch = self.to_string(label_batch)
			for img, lbl in zip(data_batch, label_batch):
				minval, maxval = np.min(img), np.max(img)
				img = ((img - minval) * (255 / (maxval - minval))).astype(np.uint8)
				if 'posix' == os.name:
					print('Label =', lbl)
					cv2.imwrite('./text_{}.png'.format(idx), img)
					#cv2.imwrite('./text_mask.png', mask)
				else:
					##mask[mask > 0] = 255
					##mask = mask.astype(np.uint8)
					#minval, maxval = np.min(mask), np.max(mask)
					#mask = (mask.astype(np.float32) - minval) / (maxval - minval)

					print('Label =', lbl)
					cv2.imshow('Text', img)
					#cv2.imshow('Text Mask', mask)
					cv2.waitKey(0)
				idx += 1
				if idx >= 10:
					break
			break  # For a single batch.

#--------------------------------------------------------------------

class TextLineDatasetWithHangeulJamoLabel(TextLineDataset):
	def __init__(self, char_labels):
		#self._SOJC = '<SOJC>'  # All Hangeul jamo strings will start with the Start-Of-Jamo-Character token.
		self._EOJC = '<EOJC>'  # All Hangeul jamo strings will end with the End-Of-Jamo-Character token.
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.

		num_labels = len(char_labels)

		char_labels = list(char_labels) + [self._EOJC] + [self._EOS]
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
		self._num_classes = len(char_labels) + 1  # #labels + EOJC + EOS + blank label.
		#self._eoc_token_label = self._num_classes - 3
		#self._eos_token_label = self._num_classes - 2
		self._blank_label = self._num_classes - 1

		self._label_int2char = char_labels
		self._label_char2int = {c:i for i, c in enumerate(char_labels)}

	@property
	def eojc_token(self):
		return self._label_char2int[self._EOJC]

	# REF [function] >> generate_text_lines() in text_generation_util.py.
	def _generate_text_lines(self, word_set, textGenerator, font_size_interval, char_space_ratio_interval, image_height, image_width, batch_size, font_color=None, bg_color=None):
		sceneTextGenerator = tg_util.MySceneTextGenerator(tg_util.IdentityTransformer())

		scene_list, scene_text_mask_list, text_list = list(), list(), list()
		step = 0
		while True:
			font_size = random.randint(*font_size_interval)
			char_space_ratio = random.uniform(*char_space_ratio_interval)

			text = random.sample(word_set, 1)[0]

			char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space_ratio, font_size)
			text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

			if bg_color is None:
				# Grayscale background.
				bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
			else:
				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

			scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
			scene = cv2.resize(scene, (image_width, image_height), interpolation=cv2.INTER_AREA)
			scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
			scene = scene.reshape(scene.shape + (-1,))
			#scene_text_mask = cv2.resize(scene_text_mask, (image_width, image_height), interpolation=cv2.INTER_AREA)
			scene_list.append(scene)
			#scene_text_mask_list.append(scene_text_mask)
			text = hg_util.hangeul2jamo(text, self._EOJC, use_separate_consonants=False, use_separate_vowels=True)  # Hangeul letters -> Hangeul jamos.
			text_list.append(text)

			step += 1
			if 0 == step % batch_size:
				#yield scene_list, scene_text_mask_list, text_list
				yield self._preprocess(np.array(scene_list, dtype=np.float32), self.to_numeric(text_list))
				scene_list, scene_text_mask_list, text_list = list(), list(), list()
				step = 0

	def _preprocess(self, data, labels):
		data = (data / 255.0) * 2 - 1  # [-1, 1].

		#labels = tf.keras.utils.to_categorical(labels, self._num_classes, np.int16)
		#labels = labels.astype(np.int16)

		# (samples, height, width, channels) -> (samples, width, height, channels).
		#data = data.transpose((0, 2, 1, 3))

		return data, labels
