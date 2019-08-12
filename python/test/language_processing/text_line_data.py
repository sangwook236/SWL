import os, abc, random, functools, time, json
import numpy as np
import cv2
import swl.machine_learning.util as swl_ml_util
import text_generation_util as tg_util
import hangeul_util as hg_util

#--------------------------------------------------------------------

class TextLineDatasetBase(abc.ABC):
	def __init__(self, labels=None, default_value=-1):
		super().__init__()

		self._labels = labels
		self._default_value = default_value

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			return [self._labels.index(ch) for ch in label_str]
		except Exception as ex:
			print('[SWL] Error: Failed to encode a label {}.'.format(label_str))
			raise

	# Integer label -> string label.
	def decode_label(self, label_int, *args, **kwargs):
		try:
			return ''.join(list(self._labels[id] for id in label_int if id != self._default_value))
		except Exception as ex:
			print('[SWL] Error: Failed to decode a label {}.'.format(label_int))
			raise

	# String labels -> Integer labels.
	def encode_labels(self, labels_str, dtype=np.int16, *args, **kwargs):
		max_label_len = functools.reduce(lambda x, y: max(x, len(y)), labels_str, 0)
		labels_int = np.full((len(labels_str), max_label_len), self._default_value, dtype=dtype)
		for (idx, lbl) in enumerate(labels_str):
			try:
				labels_int[idx,:len(lbl)] = np.array(list(self._labels.index(ch) for ch in lbl))
			except ValueError:
				pass
		return labels_int

	# Integer labels -> string labels.
	def decode_labels(self, labels_int, *args, **kwargs):
		def int2str(label):
			try:
				label = list(self._labels[id] for id in label if id != self._default_value)
				return ''.join(label)
			except ValueError:
				return None
		return list(map(int2str, labels_int))

	@abc.abstractmethod
	def preprocess(self, inputs, outputs, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def resize(self, input, output=None, height=None, width=None, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def create_train_batch_generator(self, batch_size, shuffle=True, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def create_test_batch_generator(self, batch_size, shuffle=False, *args, **kwargs):
		raise NotImplementedError

	def visualize(self, batch_generator, num_examples=10):
		for batch_data, num_batch_examples in batch_generator:
			batch_images, batch_labels_str, batch_sparse_labels_int = batch_data

			print('Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_images.shape, batch_images.dtype, np.min(batch_images), np.max(batch_images)))
			print('Label (str): shape = {}, dtype = {}.'.format(batch_labels_str.shape, batch_labels_str.dtype))
			print('Label (int): shape = {}, type = {}.'.format(batch_sparse_labels_int[2], type(batch_sparse_labels_int)))

			# (examples, width, height, channels) -> (examples, height, width, channels).
			batch_images = batch_images.transpose((0, 2, 1, 3))
			batch_labels_int = swl_ml_util.sparse_to_sequences(*batch_sparse_labels_int, dtype=np.int32)

			minval, maxval = np.min(batch_images), np.max(batch_images)
			for idx, (img, lbl_str, lbl_int) in enumerate(zip(batch_images, batch_labels_str, batch_labels_int)):
				print('Label (str) = {}, Label (int) = {}({}).'.format(lbl_str, lbl_int, self.decode_label(lbl_int)))

				#img = ((img - minval) * (255 / (maxval - minval))).astype(np.uint8)
				img = ((img - minval) / (maxval - minval)).astype(np.float32)
				#cv2.imwrite('./text_{}.png'.format(idx), img)
				cv2.imshow('Text', img)
				ch = cv2.waitKey(2000)
				if 27 == ch:  # ESC.
					break
				if (idx + 1) >= num_examples:
					break
			break  # For a single batch.
		cv2.destroyAllWindows()

	def load_images_from_files(self, image_filepaths):
		images = list()
		for fpath in image_filepaths:
			img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
			img = self.resize(img)
			img, _ = self.preprocess(img, None)
			images.append(img)

		# (examples, height, width) -> (examples, width, height).
		images = np.swapaxes(np.array(images), 1, 2)
		images = np.reshape(images, images.shape + (-1,))  # Image channel = 1.
		return images

#--------------------------------------------------------------------

class RunTimeTextLineDatasetBase(TextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, default_value=-1):
		super().__init__(labels=None, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._word_set = word_set

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def default_value(self):
		return self._default_value

	def preprocess(self, inputs, outputs, *args, **kwargs):
		"""
		if inputs is not None:
			# Contrast limited adaptive histogram equalization (CLAHE).
			#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			#inputs = np.array([clahe.apply(inp) for inp in inputs])

			# TODO [check] >> Preprocessing has influence on recognition rate.

			# Normalization, standardization, etc.
			#inputs = inputs.astype(np.float32)

			if False:
				inputs = preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
			elif False:
				# NOTE [info] >> Not good.
				inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
			elif False:
				# NOTE [info] >> Not bad.
				in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
				out_min, out_max = 0, 1 #-1, 1
				inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
			elif False:
				inputs /= 255.0  # Normalization.

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs
		"""
		inputs = (inputs / 255.0) * 2 - 1  # [-1, 1].

		#outputs = tf.keras.utils.to_categorical(outputs, self._num_classes, np.int16)
		#outputs = outputs.astype(np.int16)

		# (examples, height, width, channels) -> (examples, width, height, channels).
		#inputs = inputs.transpose((0, 2, 1, 3))

		return inputs, outputs

	def resize(self, input, output=None, height=None, width=None, *args, **kwargs):
		if height is None:
			height = self._image_height
		if width is None:
			width = self._image_width

		hh, ww, cc = input.shape
		if ww >= width:
			return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		else:
			ratio = height / hh
			min_width = min(width, int(ww * ratio))
			input = cv2.resize(input, (min_width, height), interpolation=cv2.INTER_AREA)
			if min_width < width:
				image_zeropadded = np.zeros((height, width, cc), dtype=input.dtype)
				image_zeropadded[:,0:min_width] = input[:,0:min_width]
				return image_zeropadded
			else:
				return input
		"""
		return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		"""

	def create_train_batch_generator(self, batch_size, shuffle=True, *args, **kwargs):
		return self._create_batch_generator(self._word_set, self._textGenerator, (self._min_font_size, self._max_font_size), (self._min_char_space_ratio, self._max_char_space_ratio), batch_size, self._font_color, self._bg_color)

	def create_test_batch_generator(self, batch_size, shuffle=False, *args, **kwargs):
		return self._create_batch_generator(self._word_set, self._textGenerator, (self._min_font_size, self._max_font_size), (self._min_char_space_ratio, self._max_char_space_ratio), batch_size, self._font_color, self._bg_color)

	def _create_batch_generator(self, word_set, textGenerator, font_size_interval, char_space_ratio_interval, batch_size, font_color, bg_color):
		for text_list, scene_list, _ in tg_util.generate_text_lines(word_set, textGenerator, font_size_interval, char_space_ratio_interval, batch_size, font_color, bg_color):
			scene_list = list(map(lambda image: cv2.cvtColor(self.resize(image), cv2.COLOR_BGR2GRAY), scene_list))
			#scene_list, scene_text_mask_list = list(zip(*list(map(lambda image, mask: (cv2.cvtColor(self.resize(image), cv2.COLOR_BGR2GRAY), self.resize(mask)), scene_list, scene_text_mask_list))))
			scenes = np.array(scene_list, dtype=np.float32)
			scenes = scenes.reshape(scenes.shape + (-1,))  # Image channel = 1.
			scenes, _ = self.preprocess(scenes, None)
			# (examples, height, width, channels) -> (examples, width, height, channels).
			scenes = scenes.transpose((0, 2, 1, 3))
			texts_int = list(map(lambda txt: self.encode_label(txt), text_list))
			texts_int = swl_ml_util.sequences_to_sparse(texts_int, dtype=np.int32)  # Sparse tensor.
			yield (scenes, text_list, texts_int), batch_size

# This class is independent of language.
class RunTimeTextLineDataset(RunTimeTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel

		#--------------------
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.

		self._word_set = word_set

		#--------------------
		label_set = functools.reduce(lambda x, word: x.union(word), self._word_set, set())
		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		self._min_font_size, self._max_font_size = int(image_height * 0.8), int(image_height * 1.25)
		self._min_char_space_ratio, self._max_char_space_ratio = 0.8, 1.2

		#self._font_color = (255, 255, 255)
		#self._font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
		self._font_color = None  # Uses random font colors.
		#self._bg_color = (0, 0, 0)
		self._bg_color = None  # Uses random colors.

		characterTransformer = tg_util.IdentityTransformer()
		#characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
		self._textGenerator = tg_util.MySimplePrintedHangeulTextGenerator(characterTransformer, characterAlphaMattePositioner)
		"""
		characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
		#characterTransformer = tg_util.IdentityTransformer()
		characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
		self._textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)
		"""

# This class is independent of language.
class HangeulJamoRunTimeTextLineDataset(RunTimeTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, default_value)

		#--------------------
		#self._SOJC = '<SOJC>'  # All Hangeul jamo strings will start with the Start-Of-Jamo-Character token.
		self._EOJC = '<EOJC>'  # All Hangeul jamo strings will end with the End-Of-Jamo-Character token.
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.
		#self._UNKNOWN = '<UNK>'  # Unknown label token.

		self._hangeul2jamo_functor = functools.partial(hg_util.hangeul2jamo, eojc_str=self._EOJC, use_separate_consonants=False, use_separate_vowels=True)
		self._jamo2hangeul_functor = functools.partial(hg_util.jamo2hangeul, eojc_str=self._EOJC, use_separate_consonants=False, use_separate_vowels=True)

		#--------------------
		label_set = functools.reduce(lambda x, word: x.union(self._hangeul2jamo_functor(word)), self._word_set, set())
		self._labels = sorted(label_set)
		#self._labels = ''.join(sorted(label_set))  # Error.
		print('[SWL] Info: labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		self._min_font_size, self._max_font_size = int(image_height * 0.8), int(image_height * 1.25)
		self._min_char_space_ratio, self._max_char_space_ratio = 0.8, 1.2

		#self._font_color = (255, 255, 255)
		#self._font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
		self._font_color = None  # Uses random font colors.
		#self._bg_color = (0, 0, 0)
		self._bg_color = None  # Uses random colors.

		characterTransformer = tg_util.IdentityTransformer()
		#characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
		self._textGenerator = tg_util.MySimplePrintedHangeulTextGenerator(characterTransformer, characterAlphaMattePositioner)
		"""
		characterAlphaMatteGenerator = tg_util.MyHangeulCharacterAlphaMatteGenerator()
		#characterTransformer = tg_util.IdentityTransformer()
		characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
		self._textGenerator = tg_util.MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)
		"""

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			label_str = self._hangeul2jamo_functor(label_str)
			return [self._labels.index(ch) for ch in label_str]
		except Exception as ex:
			print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
			raise

	# Integer label -> string label.
	def decode_label(self, label_int, *args, **kwargs):
		try:
			label_str = ''.join(list(self._labels[id] for id in label_int if id != self._default_value))
			return self._jamo2hangeul_functor(label_str)
		except Exception as ex:
			print('[SWL] Error: Failed to decode a label: {}.'.format(label_int))
			raise

	# String labels -> Integer labels.
	def encode_labels(self, labels_str, dtype=np.int16, *args, **kwargs):
		labels_str = list(map(lambda label: self._hangeul2jamo_functor(label), labels_str))

		max_label_len = functools.reduce(lambda x, y: max(x, len(y)), labels_str, 0)
		labels_int = np.full((len(labels_str), max_label_len), self._default_value, dtype=dtype)
		for (idx, lbl) in enumerate(labels_str):
			try:
				labels_int[idx,:len(lbl)] = np.array(list(self._labels.index(ch) for ch in lbl))
			except ValueError:
				pass
		return labels_int

	# Integer labels -> string labels.
	def decode_labels(self, labels_int, *args, **kwargs):
		def int2str(label):
			try:
				label = ''.join(list(self._labels[id] for id in label if id != self._default_value))
				return self._jamo2hangeul_functor(label)
			except ValueError:
				return None
		return list(map(int2str, labels_int))

#--------------------------------------------------------------------

class JsonBasedTextLineDatasetBase(TextLineDatasetBase):
	def __init__(self, image_height, image_width, image_channel, default_value=-1):
		super().__init__(labels=None, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def default_value(self):
		return self._default_value

	def preprocess(self, inputs, outputs, *args, **kwargs):
		"""
		if inputs is not None:
			# Contrast limited adaptive histogram equalization (CLAHE).
			#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			#inputs = np.array([clahe.apply(inp) for inp in inputs])

			# TODO [check] >> Preprocessing has influence on recognition rate.

			# Normalization, standardization, etc.
			#inputs = inputs.astype(np.float32)

			if False:
				inputs = preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
			elif False:
				# NOTE [info] >> Not good.
				inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
			elif False:
				# NOTE [info] >> Not bad.
				in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
				out_min, out_max = 0, 1 #-1, 1
				inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
			elif False:
				inputs /= 255.0  # Normalization.

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs
		"""
		inputs = (inputs / 255.0) * 2 - 1  # [-1, 1].

		#outputs = tf.keras.utils.to_categorical(outputs, self._num_classes, np.int16)
		#outputs = outputs.astype(np.int16)

		# (examples, height, width, channels) -> (examples, width, height, channels).
		#inputs = inputs.transpose((0, 2, 1, 3))

		return inputs, outputs

	def resize(self, input, output=None, height=None, width=None, *args, **kwargs):
		if height is None:
			height = self._image_height
		if width is None:
			width = self._image_width

		hh, ww = input.shape
		if ww >= width:
			return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		else:
			ratio = height / hh
			min_width = min(width, int(ww * ratio))
			input = cv2.resize(input, (min_width, height), interpolation=cv2.INTER_AREA)
			if min_width < width:
				image_zeropadded = np.zeros((height, width), dtype=input.dtype)
				image_zeropadded[:,0:min_width] = input[:,0:min_width]
				return image_zeropadded
			else:
				return input
		"""
		return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		"""

	def create_train_batch_generator(self, batch_size, shuffle=True, *args, **kwargs):
		return self._create_batch_generator(self._train_images, self._train_labels, batch_size, shuffle, self._default_value)

	def create_test_batch_generator(self, batch_size, shuffle=False, *args, **kwargs):
		return self._create_batch_generator(self._test_images, self._test_labels, batch_size, shuffle, self._default_value)

	def _create_batch_generator(self, images, labels_str, batch_size, shuffle, default_value):
		num_examples = len(images)
		if len(labels_str) != num_examples:
			raise ValueError('Invalid data length: {} != {}'.format(num_examples, len(labels_str)))
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		indices = np.arange(num_examples)
		if shuffle:
			np.random.shuffle(indices)

		start_idx = 0
		while True:
			end_idx = start_idx + batch_size
			batch_indices = indices[start_idx:end_idx]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				# FIXME [fix] >> Does not work correctly in time-major data.
				batch_images, batch_labels_str = images[batch_indices], labels_str[batch_indices]
				if batch_images.size > 0 and batch_labels_str.size > 0:  # If batch_images and batch_labels_str are non-empty.
					batch_sparse_labels_int = list(map(lambda lbl: self.encode_label(lbl), batch_labels_str))
					batch_sparse_labels_int = swl_ml_util.sequences_to_sparse(batch_sparse_labels_int, dtype=np.int32)  # Sparse tensor.
					yield (batch_images, batch_labels_str, batch_sparse_labels_int), batch_indices.size
				else:
					yield (None, None, None), 0
			else:
				yield (None, None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	def _text_dataset_to_numpy(self, dataset_json_filepath):
		with open(dataset_json_filepath, 'r', encoding='UTF8') as json_file:
			dataset = json.load(json_file)

		"""
		print(dataset['charset'])
		for datum in dataset['data']:
			print('file =', datum['file'])
			print('size =', datum['size'])
			print('text =', datum['text'])
			print('char IDs =', datum['char_id'])
		"""

		num_examples = len(dataset['data'])
		max_height, max_width, max_channel, max_label_len = 0, 0, 0, 0
		for datum in dataset['data']:
			sz = datum['size']
			if len(sz) != 3:
				print('[SWL] Warning: Invalid data size: {}.'.format(datum['file']))
				continue

			if sz[0] > max_height:
				max_height = sz[0]
			if sz[1] > max_width:
				max_width = sz[1]
			if sz[2] > max_channel:
				max_channel = sz[2]
			if len(datum['char_id']) > max_label_len:
				max_label_len = len(datum['char_id'])

		if 0 == max_height or 0 == max_width or 0 == max_channel or 0 == max_label_len:
			raise ValueError('Invalid dataset size')

		charset = list(dataset['charset'].values())
		#charset = sorted(charset)

		image_list, label_list = list(), list()
		for idx, datum in enumerate(dataset['data']):
			img, _ = self.preprocess(self.resize(cv2.imread(datum['file'], cv2.IMREAD_GRAYSCALE)), None)
			image_list.append(img)
			if False:  # Char ID.
				label_list.append(datum['char_id'])
			else:  # Unicode -> char ID.
				label_list.append(''.join(list(chr(id) for id in datum['char_id'])))

		return np.array(image_list), label_list, charset

# REF [function] >> generate_simple_text_lines_test() and generate_text_lines_test() in text_generation_util_test.py.
# This class is independent of language.
class JsonBasedTextLineDataset(JsonBasedTextLineDatasetBase):
	def __init__(self, train_json_filepath, test_json_filepath, image_height, image_width, image_channel, default_value=-1):
		super().__init__(image_height, image_width, image_channel, default_value=-1)

		#--------------------
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, train_charset = self._text_dataset_to_numpy(train_json_filepath)
		self._test_images, self._test_labels, test_charset = self._text_dataset_to_numpy(test_json_filepath)
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

		#self._labels = sorted(set(train_charset + test_charset))
		self._labels = ''.join(sorted(set(train_charset + test_charset)))
		print('[SWL] Info: labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		self._train_images = self._train_images.reshape(self._train_images.shape + (-1,))  # Image channel = 1.
		self._test_images = self._test_images.reshape(self._test_images.shape + (-1,))  # Image channel = 1.

		# (examples, height, width, channels) -> (examples, width, height, channels).
		self._train_images = self._train_images.transpose((0, 2, 1, 3))
		self._test_images = self._test_images.transpose((0, 2, 1, 3))

		self._train_labels = np.reshape(np.array(self._train_labels), (-1))
		self._test_labels = np.reshape(np.array(self._test_labels), (-1))

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		print('Train label: shape = {}, dtype = {}.'.format(self._train_labels.shape, self._train_labels.dtype))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		print('Test label: shape = {}, dtype = {}.'.format(self._test_labels.shape, self._test_labels.dtype))

# This class is independent of language.
class HangeulJamoJsonBasedTextLineDataset(JsonBasedTextLineDatasetBase):
	def __init__(self, train_json_filepath, test_json_filepath, image_height, image_width, image_channel, default_value=-1):
		super().__init__(image_height, image_width, image_channel, default_value=-1)

		#--------------------
		#self._SOJC = '<SOJC>'  # All Hangeul jamo strings will start with the Start-Of-Jamo-Character token.
		self._EOJC = '<EOJC>'  # All Hangeul jamo strings will end with the End-Of-Jamo-Character token.
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.
		#self._UNKNOWN = '<UNK>'  # Unknown label token.

		self._hangeul2jamo_functor = functools.partial(hg_util.hangeul2jamo, eojc_str=self._EOJC, use_separate_consonants=False, use_separate_vowels=True)
		self._jamo2hangeul_functor = functools.partial(hg_util.jamo2hangeul, eojc_str=self._EOJC, use_separate_consonants=False, use_separate_vowels=True)

		#--------------------
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, train_charset = self._text_dataset_to_numpy(train_json_filepath)
		self._test_images, self._test_labels, test_charset = self._text_dataset_to_numpy(test_json_filepath)
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

		label_set = set(self._hangeul2jamo_functor(list(set(train_charset + test_charset))))
		self._labels = sorted(label_set)
		#self._labels = ''.join(sorted(label_set))  # Error.
		print('[SWL] Info: labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		self._train_images = self._train_images.reshape(self._train_images.shape + (-1,))  # Image channel = 1.
		self._test_images = self._test_images.reshape(self._test_images.shape + (-1,))  # Image channel = 1.

		# (examples, height, width, channels) -> (examples, width, height, channels).
		self._train_images = self._train_images.transpose((0, 2, 1, 3))
		self._test_images = self._test_images.transpose((0, 2, 1, 3))

		self._train_labels = np.reshape(np.array(self._train_labels), (-1))
		self._test_labels = np.reshape(np.array(self._test_labels), (-1))

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		print('Train label: shape = {}, dtype = {}.'.format(self._train_labels.shape, self._train_labels.dtype))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		print('Test label: shape = {}, dtype = {}.'.format(self._test_images.shape, self._test_images.dtype))

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			label_str = self._hangeul2jamo_functor(label_str)
			return [self._labels.index(ch) for ch in label_str]
		except Exception as ex:
			print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
			raise

	# Integer label -> string label.
	def decode_label(self, label_int, *args, **kwargs):
		try:
			label_str = ''.join(list(self._labels[id] for id in label_int if id != self._default_value))
			return self._jamo2hangeul_functor(label_str)
		except Exception as ex:
			print('[SWL] Error: Failed to decode a label: {}.'.format(label_int))
			raise

	# String labels -> Integer labels.
	def encode_labels(self, labels_str, dtype=np.int16, *args, **kwargs):
		labels_str = list(map(lambda label: self._hangeul2jamo_functor(label), labels_str))

		max_label_len = functools.reduce(lambda x, y: max(x, len(y)), labels_str, 0)
		labels_int = np.full((len(labels_str), max_label_len), self._default_value, dtype=dtype)
		for (idx, lbl) in enumerate(labels_str):
			try:
				labels_int[idx,:len(lbl)] = np.array(list(self._labels.index(ch) for ch in lbl))
			except ValueError:
				pass
		return labels_int

	# Integer labels -> string labels.
	def decode_labels(self, labels_int, *args, **kwargs):
		def int2str(label):
			try:
				label = ''.join(list(self._labels[id] for id in label if id != self._default_value))
				return self._jamo2hangeul_functor(label)
			except ValueError:
				return None
		return list(map(int2str, labels_int))
