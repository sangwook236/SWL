import os, random, functools, time, glob
import numpy as np
import cv2
#import sklearn
#import swl.machine_learning.util as swl_ml_util
import hangeul_util as hg_util
import text_line_data

# REF [site] >> https://rrc.cvc.uab.es/?ch=13
class Icdar2019SroieTextLineDatasetBase(text_line_data.TextLineDatasetBase):
	def __init__(self, image_height, image_width, image_channel, num_classes=0, default_value=-1, use_NWHC=True):
		super().__init__(labels=None, default_value=default_value)

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes
		self._use_NWHC = use_NWHC

		self._train_data, self._test_data = None, None

	@property
	def shape(self):
		return self._image_height, self._image_width, self._image_channel

	@property
	def num_classes(self):
		return self._num_classes

	@property
	def default_value(self):
		return self._default_value

	@property
	def train_examples(self):
		return self._train_data

	@property
	def test_examples(self):
		return self._test_data

	@property
	def num_train_examples(self):
		return 0 if self._train_data is None else len(self._train_data)

	@property
	def num_test_examples(self):
		return 0 if self._test_data is None else len(self._test_data)

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
		return self._create_batch_generator(self._train_data, batch_size, shuffle)

	def create_test_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=False, *args, **kwargs):
		return self._create_batch_generator(self._test_data, batch_size, shuffle)

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

	def _load_data(self, data_dir_path, image_height, image_width, image_channel, max_label_len):
		image_filepaths, label_filepaths = glob.glob(os.path.join(data_dir_path, '*.jpg'), recursive=False), glob.glob(os.path.join(data_dir_path, '*.txt'), recursive=False)
		image_filepaths.sort()
		label_filepaths.sort()

		images, labels_str, labels_int = list(), list(), list()
		for img_fpath, lbl_fpath in zip(image_filepaths, label_filepaths):
			with open(lbl_fpath, 'r', encoding='UTF8') as fd:
				#label_str = fd.read()
				#label_str = fd.read().rstrip()
				label_str = fd.read().rstrip('\n')
			if len(label_str) > max_label_len:
				print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
				continue
			img = cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE if 1 == image_channel else cv2.IMREAD_COLOR)
			if img is None:
				print('[SWL] Error: Failed to load an image: {}.'.format(img_fpath))
				continue

			img = self.resize(img, None, image_height, image_width)
			try:
				label_int = self.encode_label(label_str)
			except Exception:
				#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.decode_label(label_int):
				print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.decode_label(label_int)))
				continue

			images.append(img)
			labels_str.append(label_str)
			labels_int.append(label_int)

		#images = list(map(lambda image: self.resize(image), images))
		images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)
		images, _ = self.preprocess(images, None)

		return images, labels_str, labels_int

	def _create_batch_generator(self, data, batch_size, shuffle):
		images, labels_str, labels_int = data

		num_examples = len(images)
		if len(labels_str) != num_examples or len(labels_int) != num_examples:
			raise ValueError('Invalid data length: {} != {} != {}'.format(num_examples, len(labels_str), len(labels_int)))
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
				batch_data1, batch_data2, batch_data3 = images[batch_indices], labels_str[batch_indices], labels_int[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0 and batch_data3.size > 0:  # If batch_data1, batch_data2, and batch_data3 are non-empty.
				#batch_data3 = swl_ml_util.sequences_to_sparse(batch_data3, dtype=np.int32)  # Sparse tensor.
				#if batch_data1.size > 0 and batch_data2.size > 0 and batch_data3[2][0] > 0:  # If batch_data1, batch_data2, and batch_data3 are non-empty.
					yield (batch_data1, batch_data2, batch_data3), batch_indices.size
				else:
					yield (None, None, None), 0
			else:
				yield (None, None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

class Icdar2019SroieTextLineDataset(Icdar2019SroieTextLineDatasetBase):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len):
		super().__init__(image_height, image_width, image_channel, num_classes=0, default_value=-1, use_NWHC=True)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		digit_charset = '0123456789'
		symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		#label_set = set(alphabet_charset + digit_charset)
		label_set = set(alphabet_charset + digit_charset + symbol_charset)

		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		# Load data.
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		images, labels_str, labels_int = self._load_data(data_dir_path, self._image_height, self._image_width, self._image_channel, max_label_len)
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
		labels_str, labels_int = np.array(labels_str), np.array(labels_int)

		num_examples = len(images)
		indices = np.arange(num_examples)
		np.random.shuffle(indices)
		test_offset = round(train_test_ratio * num_examples)
		train_indices, test_indices = indices[:test_offset], indices[test_offset:]
		self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])

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
