import os, random, functools, time
import numpy as np
import cv2
#import sklearn
#import swl.machine_learning.util as swl_ml_util
import hangeul_util as hg_util
import text_line_data

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator

class TextRecognitionDataGeneratorTextLineDatasetBase(text_line_data.FileBasedTextLineDatasetBase):
	def __init__(self, image_height, image_width, image_channel, labels=None, num_classes=0, use_NWHC=True, default_value=-1):
		super().__init__(image_height, image_width, image_channel, labels, num_classes, use_NWHC, default_value)

	def _load_data(self, data_dir_path, image_height, image_width, image_channel, max_label_len, label_filename=None, use_NWHC=True):
		if label_filename is None:
			images, labels_str, labels_int = self._load_data_with_label_in_filename(data_dir_path, image_height, image_width, image_channel, max_label_len)
		else:
			images, labels_str, labels_int = self._load_data_with_label_file(data_dir_path, label_filename, image_height, image_width, image_channel, max_label_len)

		images = np.array(images)
		#labels_str = np.array(labels_str).flatten()
		#labels_int = np.array(labels_int).flatten()
		labels_str = np.array(labels_str)
		labels_int = np.array(labels_int)

		images = self._transform_images(images, use_NWHC=use_NWHC)

		num_examples = len(images)
		if len(labels_str) != num_examples or len(labels_int) != num_examples:
			raise ValueError('Unmatched data sizes, {0} != {1} or {0} != {2}'.format(num_examples, len(labels_str), len(labels_int)))

		return images, labels_str, labels_int, num_examples

	def _load_data_with_label_in_filename(self, data_dir_path, image_height, image_width, image_channel, max_label_len):
		if 1 == image_channel:
			flags = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flags = cv2.IMREAD_COLOR
		else:
			raise ValueError('Invalid channels {}'.format(image_channel))

		images, labels_str, labels_int = list(), list(), list()
		#images, labels_str, labels_int = None, list(), list()
		for fname in os.listdir(data_dir_path):
			label_str = fname.split('_')
			if 2 != len(label_str):
				print('[SWL] Warning: Invalid file name: {}.'.format(fname))
				continue
			label_str = label_str[0]

			if len(label_str) > max_label_len:
				print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
				continue
			img = cv2.imread(os.path.join(data_dir_path, fname), flags)
			if img is None:
				print('[SWL] Error: Failed to load an image: {}.'.format(os.path.join(data_dir_path, fname)))
				continue

			img = self.resize(img, None, image_height, image_width)
			try:
				#img, label_int = self.preprocess(img, self.encode_label(label_str))
				label_int = self.encode_label(label_str)
			except Exception:
				#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.decode_label(label_int):
				print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.decode_label(label_int)))
				continue

			images.append(img)
			#images = np.expand_dims(img, axis=0) if images is None else np.append(images, np.expand_dims(img, axis=0), axis=0)  # Too much slow.
			#images = np.expand_dims(img, axis=0) if images is None else np.vstack([images, np.expand_dims(img, axis=0)])  # Too much slow.
			labels_str.append(label_str)
			labels_int.append(label_int)

		return images, labels_str, labels_int

	def _load_data_with_label_file(self, data_dir_path, label_filename, image_height, image_width, image_channel, max_label_len):
		if 1 == image_channel:
			flags = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flags = cv2.IMREAD_COLOR
		else:
			raise ValueError('Invalid channels {}'.format(image_channel))

		try:
			with open(os.path.join(data_dir_path, label_filename), 'r') as fd:
				lines = fd.readlines()
		except FileNotFoundError:
			print('[SWL] Error: File not found: {}.'.format(os.path.join(data_dir_path, label_filename)))
			return None

		images, labels_str, labels_int = list(), list(), list()
		#images, labels_str, labels_int = None, list(), list()
		for line in lines:
			line = line.rstrip('\n')
			if not line:
				continue

			pos = line.find(' ')
			if -1 == pos:
				print('[SWL] Warning: Invalid image-label pair: {}.'.format(line))
				continue
			fname, label_str = line[:pos], line[pos+1:]

			if len(label_str) > max_label_len:
				print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
				continue
			img = cv2.imread(os.path.join(data_dir_path, fname), flags)
			if img is None:
				print('[SWL] Error: Failed to load an image: {}.'.format(os.path.join(data_dir_path, fname)))
				continue

			img = self.resize(img, None, image_height, image_width)
			try:
				#img, label_int = self.preprocess(img, self.encode_label(label_str))
				label_int = self.encode_label(label_str)
			except Exception:
				#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
				continue
			if label_str != self.decode_label(label_int):
				print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.decode_label(label_int)))
				continue

			images.append(img)
			#images = np.expand_dims(img, axis=0) if images is None else np.append(images, np.expand_dims(img, axis=0), axis=0)  # Too much slow.
			#images = np.expand_dims(img, axis=0) if images is None else np.vstack([images, np.expand_dims(img, axis=0)])  # Too much slow.
			labels_str.append(label_str)
			labels_int.append(label_int)

		return images, labels_str, labels_int

	def _create_batch_generator(self, data, batch_size, shuffle, is_training=False):
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

		if is_training and hasattr(self, 'augment'):
			apply_preprocessing = lambda images: self.preprocess(self.augment(images, None)[0], None)[0]
		else:
			apply_preprocessing = lambda images: self.preprocess(images, None)[0]

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
					batch_data1 = apply_preprocessing(batch_data1)
					yield (batch_data1, batch_data2, batch_data3), batch_indices.size
				else:
					yield (None, None, None), 0
			else:
				yield (None, None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

class EnglishTextRecognitionDataGeneratorTextLineDataset(TextRecognitionDataGeneratorTextLineDatasetBase):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, labels, num_classes, shuffle=True, use_NWHC=True, default_value=-1):
		super().__init__(image_height, image_width, image_channel, labels, num_classes, use_NWHC, default_value)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		if data_dir_path:
			# Load data.
			print('[SWL] Info: Start loading dataset...')
			start_time = time.time()
			label_filename = 'labels.txt'
			#label_filename = None
			images, labels_str, labels_int, num_examples = self._load_data(data_dir_path, self._image_height, self._image_width, self._image_channel, max_label_len, label_filename, use_NWHC=self._use_NWHC)
			print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

			test_offset = round(train_test_ratio * num_examples)
			indices = np.arange(num_examples)
			if shuffle:
				np.random.shuffle(indices)
			train_indices = indices[:test_offset]
			test_indices = indices[test_offset:]
			self._train_data = images[train_indices], labels_str[train_indices], labels_int[train_indices]
			self._test_data = images[test_indices], labels_str[test_indices], labels_int[test_indices]

			#--------------------
			print('[SWL] Info: Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_data[0].shape, self._train_data[0].dtype, np.min(self._train_data[0]), np.max(self._train_data[0])))
			print('[SWL] Info: Train string label: shape = {}, dtype = {}.'.format(self._train_data[1].shape, self._train_data[1].dtype))
			print('[SWL] Info: Train integer label: shape = {}, dtype = {}, element type = {}.'.format(self._train_data[2].shape, self._train_data[2].dtype, type(self._test_data[2][0]) if len(self._test_data[2]) > 0 else '?'))
			print('[SWL] Info: Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_data[0].shape, self._test_data[0].dtype, np.min(self._test_data[0]), np.max(self._test_data[0])))
			print('[SWL] Info: Test string label: shape = {}, dtype = {}.'.format(self._test_data[1].shape, self._test_data[1].dtype))
			print('[SWL] Info: Test integer label: shape = {}, dtype = {}, element type = {}.'.format(self._test_data[2].shape, self._test_data[2].dtype, type(self._test_data[2][0]) if len(self._test_data[2]) > 0 else '?'))
		else:
			print('[SWL] Info: Dataset were not loaded.')
			self._train_data, self._test_data = None, None
			num_examples = 0

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

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
				inputs = sklearn.preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = sklearn.preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = sklearn.preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = sklearn.preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
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

class HangeulTextRecognitionDataGeneratorTextLineDataset(TextRecognitionDataGeneratorTextLineDatasetBase):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, labels, num_classes, shuffle=True, use_NWHC=True, default_value=-1):
		super().__init__(image_height, image_width, image_channel, labels, num_classes, use_NWHC, default_value)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		if data_dir_path:
			# Load data.
			print('[SWL] Info: Start loading dataset...')
			start_time = time.time()
			label_filename = 'labels.txt'
			#label_filename = None
			images, labels_str, labels_int, num_examples = self._load_data(data_dir_path, self._image_height, self._image_width, self._image_channel, max_label_len, label_filename, use_NWHC=self._use_NWHC)
			print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

			test_offset = round(train_test_ratio * num_examples)
			indices = np.arange(num_examples)
			if shuffle:
				np.random.shuffle(indices)
			train_indices = indices[:test_offset]
			test_indices = indices[test_offset:]
			self._train_data = images[train_indices], labels_str[train_indices], labels_int[train_indices]
			self._test_data = images[test_indices], labels_str[test_indices], labels_int[test_indices]

			#--------------------
			print('[SWL] Info: Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_data[0].shape, self._train_data[0].dtype, np.min(self._train_data[0]), np.max(self._train_data[0])))
			print('[SWL] Info: Train string label: shape = {}, dtype = {}.'.format(self._train_data[1].shape, self._train_data[1].dtype))
			print('[SWL] Info: Train integer label: shape = {}, dtype = {}, element type = {}.'.format(self._train_data[2].shape, self._train_data[2].dtype, type(self._test_data[2][0]) if len(self._test_data[2]) > 0 else '?'))
			print('[SWL] Info: Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_data[0].shape, self._test_data[0].dtype, np.min(self._test_data[0]), np.max(self._test_data[0])))
			print('[SWL] Info: Test string label: shape = {}, dtype = {}.'.format(self._test_data[1].shape, self._test_data[1].dtype))
			print('[SWL] Info: Test integer label: shape = {}, dtype = {}, element type = {}.'.format(self._test_data[2].shape, self._test_data[2].dtype, type(self._test_data[2][0]) if len(self._test_data[2]) > 0 else '?'))
		else:
			print('[SWL] Info: Dataset were not loaded.')
			self._train_data, self._test_data = None, None
			num_examples = 0

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def preprocess(self, inputs, outputs, *args, **kwargs):
		"""
		if inputs is not None:
			# Contrast limited adaptive histogram equalization (CLAHE).
			#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			#inputs = np.array([clahe.apply(inp) for inp in inputs])

			# TODO [check] >> Preprocessing has influence on recognition rate.

			# Normalization, standardization, etc.
			inputs = inputs.astype(np.float32)

			if False:
				inputs = sklearn.preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = sklearn.preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = sklearn.preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = sklearn.preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
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

class HangeulJamoTextRecognitionDataGeneratorTextLineDataset(TextRecognitionDataGeneratorTextLineDatasetBase):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, labels, num_classes, shuffle=True, use_NWHC=True, default_value=-1):
		super().__init__(image_height, image_width, image_channel, labels, num_classes, use_NWHC, default_value)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		if data_dir_path:
			# Load data.
			print('[SWL] Info: Start loading dataset...')
			start_time = time.time()
			label_filename = 'labels.txt'
			#label_filename = None
			images, labels_str, labels_int, num_examples = self._load_data(data_dir_path, self._image_height, self._image_width, self._image_channel, max_label_len, label_filename, use_NWHC=self._use_NWHC)
			print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))

			test_offset = round(train_test_ratio * num_examples)
			indices = np.arange(num_examples)
			if shuffle:
				np.random.shuffle(indices)
			train_indices = indices[:test_offset]
			test_indices = indices[test_offset:]
			self._train_data = images[train_indices], labels_str[train_indices], labels_int[train_indices]
			self._test_data = images[test_indices], labels_str[test_indices], labels_int[test_indices]

			#--------------------
			print('[SWL] Info: Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._train_data[0].shape, self._train_data[0].dtype, np.min(self._train_data[0]), np.max(self._train_data[0])))
			print('[SWL] Info: Train string label: shape = {}, dtype = {}.'.format(self._train_data[1].shape, self._train_data[1].dtype))
			print('[SWL] Info: Train integer label: shape = {}, dtype = {}, element type = {}.'.format(self._train_data[2].shape, self._train_data[2].dtype, type(self._test_data[2][0]) if len(self._test_data[2]) > 0 else '?'))
			print('[SWL] Info: Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(self._test_data[0].shape, self._test_data[0].dtype, np.min(self._test_data[0]), np.max(self._test_data[0])))
			print('[SWL] Info: Test string label: shape = {}, dtype = {}.'.format(self._test_data[1].shape, self._test_data[1].dtype))
			print('[SWL] Info: Test integer label: shape = {}, dtype = {}, element type = {}.'.format(self._test_data[2].shape, self._test_data[2].dtype, type(self._test_data[2][0]) if len(self._test_data[2]) > 0 else '?'))
		else:
			print('[SWL] Info: Dataset were not loaded.')
			self._train_data, self._test_data = None, None
			num_examples = 0

	# String label -> integer label.
	def encode_label(self, label_str, *args, **kwargs):
		try:
			return list(self._labels.index(ch) for ch in HangeulJamoTextRecognitionDataGeneratorTextLineDataset.hangeul2jamo(label_str))
		except Exception as ex:
			print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
			raise

	# Integer label -> string label.
	def decode_label(self, label_int, *args, **kwargs):
		try:
			label_str = ''.join(list(self._labels[id] for id in label_int if id != self._default_value))
			return HangeulJamoTextRecognitionDataGeneratorTextLineDataset.jamo2hangeul(label_str)
		except Exception as ex:
			print('[SWL] Error: Failed to decode a label: {}.'.format(label_int))
			raise

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def preprocess(self, inputs, outputs, *args, **kwargs):
		"""
		if inputs is not None:
			# Contrast limited adaptive histogram equalization (CLAHE).
			#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			#inputs = np.array([clahe.apply(inp) for inp in inputs])

			# TODO [check] >> Preprocessing has influence on recognition rate.

			# Normalization, standardization, etc.
			inputs = inputs.astype(np.float32)

			if False:
				inputs = sklearn.preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
				#inputs = sklearn.preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
				#inputs = sklearn.preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
				#inputs = sklearn.preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
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
