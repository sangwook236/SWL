import os, abc, random, functools, time, glob, json
import numpy as np
import cv2
import swl.machine_learning.util as swl_ml_util
import text_generation_util as tg_util
import hangeul_util as hg_util

#--------------------------------------------------------------------

class TextLineDatasetBase(abc.ABC):
	def __init__(self, labels=None, use_NWHC=True, default_value=-1):
		super().__init__()

		self._labels = labels
		self._use_NWHC = use_NWHC
		self._default_value = default_value

	@property
	def default_value(self):
		return self._default_value

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			return list(self._labels.index(ch) for ch in label_str)
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

	"""
	@abc.abstractmethod
	def augment(self, inputs, outputs, *args, **kwargs):
		raise NotImplementedError
	"""
	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	@abc.abstractmethod
	def preprocess(self, inputs, outputs, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def resize(self, input, output=None, height=None, width=None, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def create_train_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=True, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def create_test_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=False, *args, **kwargs):
		raise NotImplementedError

	def visualize(self, batch_generator, num_examples=10):
		for batch_data, num_batch_examples in batch_generator:
			batch_images, batch_labels_str, batch_labels_int = batch_data

			print('Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_images.shape, batch_images.dtype, np.min(batch_images), np.max(batch_images)))
			#print('Label (str): shape = {}, dtype = {}.'.format(batch_labels_str.shape, batch_labels_str.dtype))
			print('Label (str): shape = {}, dtype = {}, element type = {}.'.format(len(batch_labels_str), type(batch_labels_str), type(batch_labels_str[0])))
			#print('Label (int): shape = {}, type = {}.'.format(batch_labels_int[2], type(batch_labels_int)))  # Sparse tensor.
			print('Label (int): length = {}, type = {}, element type = {}.'.format(len(batch_labels_int), type(batch_labels_int), type(batch_labels_int[0])))

			if self._use_NWHC:
				# (examples, width, height, channels) -> (examples, height, width, channels).
				batch_images = batch_images.transpose((0, 2, 1, 3))
				#batch_labels_int = swl_ml_util.sparse_to_sequences(*batch_labels_int, dtype=np.int32)  # Sparse tensor.

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

	def load_images_from_files(self, image_filepaths, is_grayscale=True):
		flags = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR

		images = list()
		error_filepaths = list()
		for fpath in image_filepaths:
			img = cv2.imread(fpath, flags)
			if img is None:
				print('[SWL] Warning: Failed to load an image: {}.'.format(fpath))
				error_filepaths.append(fpath)
				continue
			img = self.resize(img)
			images.append(img)
		for fpath in error_filepaths:
			image_filepaths.remove(fpath)
		images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)

		images, _ = self.preprocess(images, None)
		return images, image_filepaths

	def _transform_images(self, images, use_NWHC=True):
		if 3 == images.ndim:
			images = images.reshape(images.shape + (-1,))  # Image channel = 1.
			#images = np.reshape(images, images.shape + (-1,))  # Image channel = 1.

		if use_NWHC:
			# (examples, height, width, channels) -> (examples, width, height, channels).
			images = np.swapaxes(images, 1, 2)
			#images = images.transpose((0, 2, 1, 3))

		return images

#--------------------------------------------------------------------

class RunTimeTextLineDatasetBase(TextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(labels=None, use_NWHC=use_NWHC, default_value=default_value)

		self._textGenerator = None

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes
		if max_label_len > 0:
			self._word_set = set(filter(lambda word: len(word) <= max_label_len, word_set))
		else:
			self._word_set = word_set

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

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
			elif True:
				inputs = (inputs / 255.0) * 2.0 - 1.0  # Normalization.

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes, np.uint16)
			pass
		"""
		inputs = (inputs.astype(np.float32) / 255.0) * 2.0 - 1.0  # Normalization.

		return inputs, outputs

	def resize(self, input, output=None, height=None, width=None, *args, **kwargs):
		if height is None:
			height = self._image_height
		if width is None:
			width = self._image_width

		"""
		hi, wi = input.shape[:2]
		if wi >= width:
			return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		else:
			aspect_ratio = height / hi
			min_width = min(width, int(wi * aspect_ratio))
			input = cv2.resize(input, (min_width, height), interpolation=cv2.INTER_AREA)
			if min_width < width:
				image_zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
				image_zeropadded[:,:min_width] = input[:,:min_width]
				return image_zeropadded
			else:
				return input
		"""
		hi, wi = input.shape[:2]
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
		zeropadded[:,:min_width] = cv2.resize(input, (min_width, height), interpolation=cv2.INTER_AREA)
		return zeropadded
		"""
		return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		"""

	def create_train_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=True, *args, **kwargs):
		return self._create_batch_generator(self._textGenerator, self._word_set, batch_size, steps_per_epoch, is_data_augmented=True)

	def create_test_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=False, *args, **kwargs):
		return self._create_batch_generator(self._textGenerator, self._word_set, batch_size, steps_per_epoch, is_data_augmented=False)

	def _create_batch_generator(self, textGenerator, word_set, batch_size, steps_per_epoch, is_data_augmented=False):
		generator = textGenerator.create_generator(word_set, batch_size)
		if is_data_augmented and hasattr(self, 'augment'):
			for step, (texts, scenes, _) in enumerate(generator):
				# For using RGB images.
				#scene_text_masks = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), scene_text_masks))
				# For using grayscale images.
				#scenes = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scenes))

				scenes = list(map(lambda image: self.resize(image), scenes))
				scenes, _ = self.augment(np.array(scenes), None)
				scenes = self._transform_images(scenes.astype(np.float32), use_NWHC=self._use_NWHC)
				#scene_text_masks = list(map(lambda image: self.resize(image), scene_text_masks))
				#scene_text_masks = self._transform_images(np.array(scene_text_masks, dtype=np.float32), use_NWHC=self._use_NWHC)

				scenes, _ = self.preprocess(scenes, None)
				#scene_text_masks, _ = self.preprocess(scene_text_masks, None)
				texts_int = list(map(lambda txt: self.encode_label(txt), texts))
				#texts_int = swl_ml_util.sequences_to_sparse(texts_int, dtype=np.int32)  # Sparse tensor.
				yield (scenes, texts, texts_int), batch_size
				if steps_per_epoch and (step + 1) >= steps_per_epoch:
					break
		else:
			for step, (texts, scenes, _) in enumerate(generator):
				# For using RGB images.
				#scene_text_masks = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), scene_text_masks))
				# For using grayscale images.
				#scenes = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scenes))

				scenes = list(map(lambda image: self.resize(image), scenes))
				scenes = self._transform_images(np.array(scenes, dtype=np.float32), use_NWHC=self._use_NWHC)
				#scene_text_masks = list(map(lambda image: self.resize(image), scene_text_masks))
				#scene_text_masks = self._transform_images(np.array(scene_text_masks, dtype=np.float32), use_NWHC=self._use_NWHC)

				scenes, _ = self.preprocess(scenes, None)
				#scene_text_masks, _ = self.preprocess(scene_text_masks, None)
				texts_int = list(map(lambda txt: self.encode_label(txt), texts))
				#texts_int = swl_ml_util.sequences_to_sparse(texts_int, dtype=np.int32)  # Sparse tensor.
				yield (scenes, texts, texts_int), batch_size
				if steps_per_epoch and (step + 1) >= steps_per_epoch:
					break

# This class is independent of language.
class BasicRunTimeTextLineDataset(RunTimeTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, font_list, handwriting_dict, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=max_label_len, use_NWHC=use_NWHC, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel

		#--------------------
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.

		#--------------------
		label_set = functools.reduce(lambda x, word: x.union(word), self._word_set, set())
		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		min_font_size, max_font_size = int(image_height * 0.8), int(image_height * 1.25)
		min_char_space_ratio, max_char_space_ratio = 0.8, 1.25

		#self._textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), None)
		self._textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio))

#--------------------------------------------------------------------

class RunTimeAlphaMatteTextLineDatasetBase(RunTimeTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, num_classes, max_label_len, use_NWHC, default_value)

	def _create_batch_generator(self, textGenerator, word_set, batch_size, steps_per_epoch, is_data_augmented=False):
		generator = textGenerator.create_generator(word_set, batch_size)
		if is_data_augmented and hasattr(self, 'augment'):
			for step, (texts, scenes, _) in enumerate(generator):
				#scenes = list(map(lambda image: self.resize(image), scenes))
				scenes = list(map(lambda image: cv2.cvtColor(self.resize(image), cv2.COLOR_BGR2GRAY), scenes))
				#scenes, scene_text_masks = list(zip(*list(map(lambda image, mask: (self.resize(image), self.resize(mask)), scenes, scene_text_masks))))
				#scenes, scene_text_masks = list(zip(*list(map(lambda image, mask: (cv2.cvtColor(self.resize(image), cv2.COLOR_BGR2GRAY), self.resize(mask)), scenes, scene_text_masks))))

				scenes, _ = self.augment(np.array(scenes), None)
				#scenes, scene_text_masks = self.augment(np.array(scenes), np.array(scene_text_masks))

				scenes = self._transform_images(scenes.astype(np.float32), use_NWHC=self._use_NWHC)
				#scene_text_masks = self._transform_images(scene_text_masks.astype(np.float32), use_NWHC=self._use_NWHC)

				scenes, _ = self.preprocess(scenes, None)
				#scene_text_masks = scene_text_masks.astype(np.float32) / 255
				texts_int = list(map(lambda txt: self.encode_label(txt), texts))
				#texts_int = swl_ml_util.sequences_to_sparse(texts_int, dtype=np.int32)  # Sparse tensor.
				yield (scenes, texts, texts_int), batch_size
				if steps_per_epoch and (step + 1) >= steps_per_epoch:
					break
		else:
			for step, (texts, scenes, _) in enumerate(generator):
				#scenes = list(map(lambda image: self.resize(image), scenes))
				scenes = list(map(lambda image: cv2.cvtColor(self.resize(image), cv2.COLOR_BGR2GRAY), scenes))
				#scenes, scene_text_masks = list(zip(*list(map(lambda image, mask: (self.resize(image), self.resize(mask)), scenes, scene_text_masks))))
				#scenes, scene_text_masks = list(zip(*list(map(lambda image, mask: (cv2.cvtColor(self.resize(image), cv2.COLOR_BGR2GRAY), self.resize(mask)), scenes, scene_text_masks))))
				scenes = self._transform_images(np.array(scenes, dtype=np.float32), use_NWHC=self._use_NWHC)
				#scene_text_masks = self._transform_images(np.array(scene_text_masks, dtype=np.float32), use_NWHC=self._use_NWHC)

				scenes, _ = self.preprocess(scenes, None)
				#scene_text_masks = scene_text_masks.astype(np.float32) / 255
				texts_int = list(map(lambda txt: self.encode_label(txt), texts))
				#texts_int = swl_ml_util.sequences_to_sparse(texts_int, dtype=np.int32)  # Sparse tensor.
				yield (scenes, texts, texts_int), batch_size
				if steps_per_epoch and (step + 1) >= steps_per_epoch:
					break

# This class is independent of language.
class RunTimeAlphaMatteTextLineDataset(RunTimeAlphaMatteTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, font_list, handwriting_dict, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=max_label_len, use_NWHC=use_NWHC, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel

		#--------------------
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.

		#--------------------
		label_set = functools.reduce(lambda x, word: x.union(word), self._word_set, set())
		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		min_font_size, max_font_size = int(image_height * 0.8), int(image_height * 1.25)
		min_char_space_ratio, max_char_space_ratio = 0.8, 1.25
		alpha_matte_mode = '1' #'L'

		characterTransformer = tg_util.IdentityTransformer()
		#characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterPositioner = tg_util.MyCharacterPositioner()
		self._textGenerator = tg_util.MySimpleTextAlphaMatteGenerator(characterTransformer, characterPositioner, font_list=font_list, handwriting_dict=handwriting_dict, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio), mode=alpha_matte_mode)
		"""
		characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode=alpha_matte_mode)
		#characterTransformer = tg_util.IdentityTransformer()
		characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterPositioner = tg_util.MyCharacterPositioner()
		self._textGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio))
		"""

# This class is independent of language.
class RunTimeHangeulJamoAlphaMatteTextLineDataset(RunTimeAlphaMatteTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, font_list, handwriting_dict, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=max_label_len, use_NWHC=use_NWHC, default_value=default_value)

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
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		min_font_size, max_font_size = int(image_height * 0.8), int(image_height * 1.25)
		min_char_space_ratio, max_char_space_ratio = 0.8, 1.25
		alpha_matte_mode = '1' #'L'

		characterTransformer = tg_util.IdentityTransformer()
		#characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterPositioner = tg_util.MyCharacterPositioner()
		self._textGenerator = tg_util.MySimpleTextAlphaMatteGenerator(characterTransformer, characterPositioner, font_list=font_list, handwriting_dict=handwriting_dict, font_size_interval=(min_font_size, max_font_size), char_space_ratio_interval=(min_char_space_ratio, max_char_space_ratio), mode=alpha_matte_mode)
		"""
		characterAlphaMatteGenerator = tg_util.MyCharacterAlphaMatteGenerator(font_list, handwriting_dict, mode=alpha_matte_mode)
		#characterTransformer = tg_util.IdentityTransformer()
		characterTransformer = tg_util.RotationTransformer(-30, 30)
		#characterTransformer = tg_util.ImgaugAffineTransformer()
		characterPositioner = tg_util.MyCharacterPositioner()
		self._textGenerator = tg_util.MyTextAlphaMatteGenerator(characterAlphaMatteGenerator, characterTransformer, characterPositioner, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio))
		"""

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			label_str = self._hangeul2jamo_functor(label_str)
			return list(self._labels.index(ch) for ch in label_str)
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
	def __init__(self, image_height, image_width, image_channel, num_classes=0, use_NWHC=True, default_value=-1):
		super().__init__(labels=None, use_NWHC=use_NWHC, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

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
			elif True:
				inputs = (inputs / 255.0) * 2.0 - 1.0  # Normalization.

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes, np.uint16)
			pass
		"""
		inputs = (inputs.astype(np.float32) / 255.0) * 2.0 - 1.0  # Normalization.

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

	def create_train_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=True, *args, **kwargs):
		return self._create_batch_generator(self._train_images, self._train_labels, batch_size, shuffle, self._default_value)

	def create_test_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=False, *args, **kwargs):
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
					batch_labels_int = list(map(lambda lbl: self.encode_label(lbl), batch_labels_str))
					#batch_labels_int = swl_ml_util.sequences_to_sparse(batch_labels_int, dtype=np.int32)  # Sparse tensor.
					yield (batch_images, batch_labels_str, batch_labels_int), batch_indices.size
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

		images, label_list = list(), list()
		for idx, datum in enumerate(dataset['data']):
			img = cv2.imread(datum['file'], cv2.IMREAD_GRAYSCALE)
			if img is None:
				print('[SWL] Warning: Failed to load an image: {}.'.format(datum['file']))
				continue
			img = self.resize(img)
			images.append(img)
			if False:  # Char ID.
				label_list.append(datum['char_id'])
			else:  # Unicode -> char ID.
				label_list.append(''.join(list(chr(id) for id in datum['char_id'])))
		images = self._transform_images(images = np.array(images), use_NWHC=self._use_NWHC)

		images, _ = self.preprocess(images, None)
		return images, label_list, charset

# REF [function] >> generate_simple_text_lines_test() and generate_text_lines_test() in text_generation_util_test.py.
# This class is independent of language.
class JsonBasedTextLineDataset(JsonBasedTextLineDatasetBase):
	def __init__(self, train_json_filepath, test_json_filepath, image_height, image_width, image_channel, use_NWHC=True, default_value=-1):
		super().__init__(image_height, image_width, image_channel, num_classes=0, use_NWHC=use_NWHC, default_value=default_value)

		#--------------------
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		self._train_images, self._train_labels, train_charset = self._text_dataset_to_numpy(train_json_filepath)
		self._test_images, self._test_labels, test_charset = self._text_dataset_to_numpy(test_json_filepath)
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

		#self._labels = sorted(set(train_charset + test_charset))
		self._labels = ''.join(sorted(set(train_charset + test_charset)))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		#self._train_labels = np.array(self._train_labels).flatten()
		#self._test_labels = np.array(self._test_labels).flatten()
		self._train_labels = np.array(self._train_labels)
		self._test_labels = np.array(self._test_labels)

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		print('Train label: shape = {}, dtype = {}.'.format(self._train_labels.shape, self._train_labels.dtype))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		print('Test label: shape = {}, dtype = {}.'.format(self._test_labels.shape, self._test_labels.dtype))

# This class is independent of language.
class JsonBasedHangeulJamoTextLineDataset(JsonBasedTextLineDatasetBase):
	def __init__(self, train_json_filepath, test_json_filepath, image_height, image_width, image_channel, use_NWHC=True, default_value=-1):
		super().__init__(image_height, image_width, image_channel, num_classes=0, use_NWHC=use_NWHC, default_value=default_value)

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
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		#self._train_labels = np.array(self._train_labels).flatten()
		#self._test_labels = np.array(self._test_labels).flatten()
		self._train_labels = np.array(self._train_labels)
		self._test_labels = np.array(self._test_labels)

		#--------------------
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_images.shape, self._train_images.dtype, np.min(self._train_images), np.max(self._train_images)))
		print('Train label: shape = {}, dtype = {}.'.format(self._train_labels.shape, self._train_labels.dtype))
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_images.shape, self._test_images.dtype, np.min(self._test_images), np.max(self._test_images)))
		print('Test label: shape = {}, dtype = {}.'.format(self._test_images.shape, self._test_images.dtype))

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			label_str = self._hangeul2jamo_functor(label_str)
			return list(self._labels.index(ch) for ch in label_str)
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

class PairedTextLineDatasetBase(TextLineDatasetBase):
	"""A base dataset for paired text lines, input & output text line images.
	"""

	def __init__(self, labels=None, use_NWHC=True, default_value=-1):
		super().__init__(labels, use_NWHC=use_NWHC, default_value=default_value)

		self._textGenerator = None

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
			elif True:
				inputs = (inputs / 255.0) * 2.0 - 1.0  # Normalization.

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes, np.uint16)
			pass
		"""
		inputs = inputs.astype(np.float32) / 255.0  # Normalization.

		return inputs, outputs

	def resize(self, input, output=None, height=None, width=None, *args, **kwargs):
		if height is None:
			height = self._image_height
		if width is None:
			width = self._image_width

		"""
		hi, wi = input.shape[:2]
		if wi >= width:
			return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		else:
			aspect_ratio = height / hi
			min_width = min(width, int(wi * aspect_ratio))
			input = cv2.resize(input, (min_width, height), interpolation=cv2.INTER_AREA)
			if min_width < width:
				image_zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
				image_zeropadded[:,:min_width] = input[:,:min_width]
				return image_zeropadded
			else:
				return input
		"""
		hi, wi = input.shape[:2]
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
		zeropadded[:,:min_width] = cv2.resize(input, (min_width, height), interpolation=cv2.INTER_AREA)
		return zeropadded
		"""
		return cv2.resize(input, (width, height), interpolation=cv2.INTER_AREA)
		"""

	def create_train_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=True, *args, **kwargs):
		return self._create_batch_generator(self._textGenerator, self._word_set, batch_size, steps_per_epoch)

	def create_test_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=False, *args, **kwargs):
		return self._create_batch_generator(self._textGenerator, self._word_set, batch_size, steps_per_epoch)

	def _create_batch_generator(self, textGenerator, word_set, batch_size, steps_per_epoch):
		raise NotImplementedError

	def visualize(self, batch_generator, num_examples=10):
		for batch_data, num_batch_examples in batch_generator:
			batch_input_images, batch_output_images, batch_labels_str, batch_labels_int = batch_data

			print('Input Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_input_images.shape, batch_input_images.dtype, np.min(batch_input_images), np.max(batch_input_images)))
			print('Output Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_output_images.shape, batch_output_images.dtype, np.min(batch_output_images), np.max(batch_output_images)))
			#print('Label (str): shape = {}, dtype = {}.'.format(batch_labels_str.shape, batch_labels_str.dtype))
			print('Label (str): shape = {}, dtype = {}, element type = {}.'.format(len(batch_labels_str), type(batch_labels_str), type(batch_labels_str[0])))
			#print('Label (int): shape = {}, type = {}.'.format(batch_labels_int[2], type(batch_labels_int)))  # Sparse tensor.
			print('Label (int): length = {}, type = {}, element type = {}.'.format(len(batch_labels_int), type(batch_labels_int), type(batch_labels_int[0])))

			if self._use_NWHC:
				# (examples, width, height, channels) -> (examples, height, width, channels).
				batch_input_images = batch_input_images.transpose((0, 2, 1, 3))
				batch_output_images = batch_output_images.transpose((0, 2, 1, 3))
				#batch_labels_int = swl_ml_util.sparse_to_sequences(*batch_labels_int, dtype=np.int32)  # Sparse tensor.

			inp_minval, inp_maxval = np.min(batch_input_images), np.max(batch_input_images)
			outp_minval, outp_maxval = np.min(batch_output_images), np.max(batch_output_images)
			for idx, (inp, outp, lbl_str, lbl_int) in enumerate(zip(batch_input_images, batch_output_images, batch_labels_str, batch_labels_int)):
				print('Label (str) = {}, Label (int) = {}({}).'.format(lbl_str, lbl_int, self.decode_label(lbl_int)))

				#inp = ((inp - inp_minval) * (255 / (inp_maxval - inp_minval))).astype(np.uint8)
				inp = ((inp - inp_minval) / (inp_maxval - inp_minval)).astype(np.float32)
				#outp = ((outp - outp_minval) * (255 / (outp_maxval - outp_minval))).astype(np.uint8)
				outp = ((outp - outp_minval) / (outp_maxval - outp_minval)).astype(np.float32)
				#cv2.imwrite('./input_text_{}.png'.format(idx), inp)
				#cv2.imwrite('./output_text_{}.png'.format(idx), outp)
				cv2.imshow('Input Text', inp)
				cv2.imshow('Output Text', outp)
				ch = cv2.waitKey(2000)
				if 27 == ch:  # ESC.
					break
				if (idx + 1) >= num_examples:
					break
			break  # For a single batch.
		cv2.destroyAllWindows()

#--------------------------------------------------------------------

class RunTimePairedTextLineDatasetBase(PairedTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(labels=None, use_NWHC=use_NWHC, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes
		if max_label_len > 0:
			self._word_set = set(filter(lambda word: len(word) <= max_label_len, word_set))
		else:
			self._word_set = word_set

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

# This class is independent of language.
class RunTimePairedCorruptedTextLineDataset(RunTimePairedTextLineDatasetBase):
	def __init__(self, word_set, image_height, image_width, image_channel, font_list, handwriting_dict, corrupt_functor, max_label_len=0, use_NWHC=True, default_value=-1):
		super().__init__(word_set, image_height, image_width, image_channel, num_classes=0, max_label_len=max_label_len, use_NWHC=use_NWHC, default_value=default_value)

		#--------------------
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.

		#--------------------
		label_set = functools.reduce(lambda x, word: x.union(word), self._word_set, set())
		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		self._corrupt_functor = corrupt_functor

		#--------------------
		min_font_size, max_font_size = int(image_height * 0.8), int(image_height * 1.25)
		min_char_space_ratio, max_char_space_ratio = 0.8, 1.25

		#self._textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), None, mask_mode='L')
		self._textGenerator = tg_util.MyBasicPrintedTextGenerator(font_list, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), mask_mode='L')

	def _create_batch_generator(self, textGenerator, word_set, batch_size, steps_per_epoch):
		def reduce_image(image, min_height, max_height):
			height = random.randint(min_height, max_height)
			interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4])
			return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height), interpolation=interpolation)

		min_height, max_height = 16, 32
		generator = textGenerator.create_generator(word_set, batch_size)
		for step, (texts, scenes, scene_text_masks) in enumerate(generator):
			# For using RGB images.
			#scene_text_masks = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), scene_text_masks))
			# For using grayscale images.
			scenes = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scenes))

			corrupted_scenes = scenes
			#corrupted_scenes = list(map(lambda image: cv2.pyrDown(cv2.pyrDown(image)), corrupted_scenes))
			corrupted_scenes = list(map(lambda image: reduce_image(image, min_height, max_height), corrupted_scenes))
			corrupted_scenes = list(map(lambda image: self.resize(np.squeeze(self._corrupt_functor(np.expand_dims(image, axis=0)), axis=0)), corrupted_scenes))
			corrupted_scenes = self._transform_images(np.array(corrupted_scenes, dtype=np.float32), use_NWHC=self._use_NWHC)
			"""
			corrupted_scenes = scene_text_masks
			corrupted_scenes = list(map(lambda image: self.resize(np.squeeze(self._corrupt_functor(np.expand_dims(image, axis=0)))), corrupted_scenes))
			corrupted_scenes = self._transform_images(np.array(corrupted_scenes, dtype=np.float32), use_NWHC=self._use_NWHC)
			#corrupted_scenes = self._transform_images(np.array(corrupted_scenes, dtype=np.float32) * 255, use_NWHC=self._use_NWHC)
			#corrupted_scenes = 255 - corrupted_scenes  # Invert.
			"""

			"""
			clean_scenes = scenes
			clean_scenes = list(map(lambda image: self.resize(image), clean_scenes))
			clean_scenes = self._transform_images(np.array(clean_scenes, dtype=np.float32), use_NWHC=self._use_NWHC)
			"""
			clean_scenes = scene_text_masks
			# FIXME [enhance] >> Resizing clean images is not a good idea.
			clean_scenes = list(map(lambda image: self.resize(image), clean_scenes))
			clean_scenes = self._transform_images(np.array(clean_scenes, dtype=np.float32), use_NWHC=self._use_NWHC)
			#clean_scenes = self._transform_images(np.array(clean_scenes, dtype=np.float32) * 255, use_NWHC=self._use_NWHC)
			clean_scenes = 255 - clean_scenes  # Invert.

			corrupted_scenes, _ = self.preprocess(corrupted_scenes, None)
			clean_scenes, _ = self.preprocess(clean_scenes, None)
			texts_int = list(map(lambda txt: self.encode_label(txt), texts))
			#texts_int = swl_ml_util.sequences_to_sparse(texts_int, dtype=np.int32)  # Sparse tensor.
			yield (corrupted_scenes, clean_scenes, texts, texts_int), batch_size
			if steps_per_epoch and (step + 1) >= steps_per_epoch:
				break
