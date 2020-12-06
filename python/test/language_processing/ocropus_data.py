import time, glob
import numpy as np
import cv2
#import sklearn
#import swl.machine_learning.util as swl_ml_util
#import hangeul_util as hg_util
import text_line_data

# REF [site] >> https://github.com/tmbdev/ocropy

class OcropusTextLineDatasetBase(text_line_data.FileBasedTextLineDatasetBase):
	def __init__(self, label_converter, image_height, image_width, image_channel, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

class EnglishOcropusTextLineDataset(OcropusTextLineDatasetBase):
	def __init__(self, label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		if data_dir_path:
			# Load data.
			print('[SWL] Info: Start loading dataset...')
			start_time = time.time()
			image_filepaths, label_filepaths = sorted(glob.glob(data_dir_path + '/**/*.bin.png', recursive=False)), sorted(glob.glob(data_dir_path + '/**/*.gt.txt', recursive=False))
			if not image_filepaths or not label_filepaths:
				raise IOError('Failed to load data from {}.'.format(data_dir_path))
			images, labels_str, labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, self._image_height, self._image_width, self._image_channel, max_label_len)
			print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
			labels_str, labels_int = np.array(labels_str), np.array(labels_int)

			num_examples = len(images)
			indices = np.arange(num_examples)
			np.random.shuffle(indices)
			test_offset = round(train_test_ratio * num_examples)
			train_indices, test_indices = indices[:test_offset], indices[test_offset:]
			self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])
		else:
			print('[SWL] Info: Dataset were not loaded.')
			self._train_data, self._test_data = None, None
			num_examples = 0

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def preprocess(self, inputs, outputs, *args, **kwargs):
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

		if outputs is not None:
			# One-hot encoding.
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs

class HangeulOcropusTextLineDataset(OcropusTextLineDatasetBase):
	def __init__(self, label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		if data_dir_path:
			# Load data.
			print('[SWL] Info: Start loading dataset...')
			start_time = time.time()
			image_filepaths, label_filepaths = sorted(glob.glob(data_dir_path + '/**/*.bin.png', recursive=False)), sorted(glob.glob(data_dir_path + '/**/*.gt.txt', recursive=False))
			if not image_filepaths or not label_filepaths:
				raise IOError('Failed to load data from {}.'.format(data_dir_path))
			images, labels_str, labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, self._image_height, self._image_width, self._image_channel, max_label_len)
			print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
			labels_str, labels_int = np.array(labels_str), np.array(labels_int)

			num_examples = len(images)
			indices = np.arange(num_examples)
			np.random.shuffle(indices)
			test_offset = round(train_test_ratio * num_examples)
			train_indices, test_indices = indices[:test_offset], indices[test_offset:]
			self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])
		else:
			print('[SWL] Info: Dataset were not loaded.')
			self._train_data, self._test_data = None, None
			num_examples = 0

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def preprocess(self, inputs, outputs, *args, **kwargs):
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
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs

class HangeulJamoOcropusTextLineDataset(OcropusTextLineDatasetBase):
	def __init__(self, label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		if data_dir_path:
			# Load data.
			print('[SWL] Info: Start loading dataset...')
			start_time = time.time()
			image_filepaths, label_filepaths = sorted(glob.glob(data_dir_path + '/**/*.bin.png', recursive=False)), sorted(glob.glob(data_dir_path + '/**/*.gt.txt', recursive=False))
			if not image_filepaths or not label_filepaths:
				raise IOError('Failed to load data from {}.'.format(data_dir_path))
			images, labels_str, labels_int = self._load_data_from_image_and_label_files(image_filepaths, label_filepaths, self._image_height, self._image_width, self._image_channel, max_label_len)
			print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
			labels_str, labels_int = np.array(labels_str), np.array(labels_int)

			num_examples = len(images)
			indices = np.arange(num_examples)
			np.random.shuffle(indices)
			test_offset = round(train_test_ratio * num_examples)
			train_indices, test_indices = indices[:test_offset], indices[test_offset:]
			self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])
		else:
			print('[SWL] Info: Dataset were not loaded.')
			self._train_data, self._test_data = None, None
			num_examples = 0

	def augment(self, inputs, outputs, *args, **kwargs):
		return inputs, outputs

	def preprocess(self, inputs, outputs, *args, **kwargs):
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
			#outputs = tf.keras.utils.to_categorical(outputs, num_classes).astype(np.uint8)
			pass

		return inputs, outputs
