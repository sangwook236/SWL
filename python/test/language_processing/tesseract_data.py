import os, random, functools, time
import numpy as np
import cv2
#import sklearn
#import swl.machine_learning.util as swl_ml_util
import hangeul_util as hg_util
import text_line_data

# REF [site] >> https://github.com/tesseract-ocr

class TesseractTextLineDatasetBase(text_line_data.FileBasedTextLineDatasetBase):
	def __init__(self, label_converter, image_height, image_width, image_channel, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

	def _load_data_from_image_and_label_files(self, image_filepaths, box_filepaths, image_height, image_width, image_channel, max_label_len):
		images, labels_str, labels_int = list(), list(), list()
		for img_fpath, box_fpath in zip(image_filepaths, box_filepaths):
			retval, images = cv2.imreadmulti(img_fpath, flags=cv2.IMREAD_GRAYSCALE if 1 == image_channel else cv2.IMREAD_COLOR)
			if not retval or images is None:
				print('[SWL] Error: Failed to load an image: {}.'.format(img_fpath))
				continue

			with open(box_fpath, 'r', encoding='UTF8') as fd:
				lines = fd.readlines()

			for line in lines:
				#line = line
				#line = line.rstrip()
				line = line.rstrip('\n')

				pos = line.find('#')
				if -1 == pos:  # Not found.
					# End-of-textline.
					box_info = line.split(' ')
					if 6 != len(box_info) or '\t' != box_info[0]:
						print('[SWL] Warning: Invalid box info: {}.'.format(box_info))
					continue
				else:
					box_info = line[:pos-1].split()
					if 6 != len(box_info) or 'WordStr' != box_info[0]:
						print('[SWL] Warning: Invalid box info: {}.'.format(box_info))
						continue
					left, bottom, right, top, page = list(map(int, box_info[1:]))

					img = images[page]
					if 2 != img.ndim:
						print('[SWL] Error: Invalid image dimension: {}.'.format(img.ndim))
						continue

					bottom, top = img.shape[0] - bottom, img.shape[0] - top
					label_str = line[pos+1:]

				if len(label_str) > max_label_len:
					print('[SWL] Warning: Too long label: {} > {}.'.format(len(label_str), max_label_len))
					continue

				if bottom < 0 or top < 0 or left < 0 or right < 0 or \
					bottom >= img.shape[0] or top >= img.shape[0] or left >= img.shape[1] or right >= img.shape[1]:
					print('[SWL] Warning: Invalid text line size: {}, {}, {}, {}.'.format(left, bottom, right, top))
					continue

				textline = img[top:bottom+1,left:right+1]
				textline = self.resize(textline, None, image_height, image_width)
				try:
					label_int = self.encode_label(label_str)
				except Exception:
					#print('[SWL] Error: Failed to encode a label: {}.'.format(label_str))
					continue
				if label_str != self.decode_label(label_int):
					print('[SWL] Error: Mismatched encoded and decoded labels: {} != {}.'.format(label_str, self.decode_label(label_int)))
					continue

				images.append(textline)
				labels_str.append(label_str)
				labels_int.append(label_int)

		if images:
			#images = list(map(lambda image: self.resize(image), images))
			images = self._transform_images(np.array(images), use_NWHC=self._use_NWHC)
			images, _ = self.preprocess(images, None)

		return images, labels_str, labels_int

class EnglishTesseractTextLineDataset(TesseractTextLineDatasetBase):
	def __init__(self, label_converter, image_filepaths, box_filepaths, image_height, image_width, image_channel, train_test_ratio, max_label_len, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		# Load data.
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		images, labels_str, labels_int = self._load_data_from_image_and_label_files(image_filepaths, box_filepaths, self._image_height, self._image_width, self._image_channel, max_label_len)
		if not images or not labels_str or not labels_int:
			raise IOError('Failed to load data from {} and {}.'.format(image_filepaths, box_filepaths))
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
		labels_str, labels_int = np.array(labels_str), np.array(labels_int)

		num_examples = len(images)
		indices = np.arange(num_examples)
		np.random.shuffle(indices)
		test_offset = round(train_test_ratio * num_examples)
		train_indices, test_indices = indices[:test_offset], indices[test_offset:]
		self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])

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

class HangeulTesseractTextLineDataset(TesseractTextLineDatasetBase):
	def __init__(self, label_converter, image_filepaths, box_filepaths, image_height, image_width, image_channel, train_test_ratio, max_label_len, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		# Load data.
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		images, labels_str, labels_int = self._load_data_from_image_and_label_files(image_filepaths, box_filepaths, self._image_height, self._image_width, self._image_channel, max_label_len)
		if not images or not labels_str or not labels_int:
			raise IOError('Failed to load data from {} and {}.'.format(image_filepaths, box_filepaths))
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
		labels_str, labels_int = np.array(labels_str), np.array(labels_int)

		num_examples = len(images)
		indices = np.arange(num_examples)
		np.random.shuffle(indices)
		test_offset = round(train_test_ratio * num_examples)
		train_indices, test_indices = indices[:test_offset], indices[test_offset:]
		self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])

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

class HangeulJamoTesseractTextLineDataset(TesseractTextLineDatasetBase):
	def __init__(self, label_converter, image_filepaths, box_filepaths, image_height, image_width, image_channel, train_test_ratio, max_label_len, use_NWHC=True):
		super().__init__(label_converter, image_height, image_width, image_channel, use_NWHC)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		# Load data.
		print('[SWL] Info: Start loading dataset...')
		start_time = time.time()
		images, labels_str, labels_int = self._load_data_from_image_and_label_files(image_filepaths, box_filepaths, self._image_height, self._image_width, self._image_channel, max_label_len)
		if not images or not labels_str or not labels_int:
			raise IOError('Failed to load data from {} and {}.'.format(image_filepaths, box_filepaths))
		print('[SWL] Info: End loading dataset: {} secs.'.format(time.time() - start_time))
		labels_str, labels_int = np.array(labels_str), np.array(labels_int)

		num_examples = len(images)
		indices = np.arange(num_examples)
		np.random.shuffle(indices)
		test_offset = round(train_test_ratio * num_examples)
		train_indices, test_indices = indices[:test_offset], indices[test_offset:]
		self._train_data, self._test_data = (images[train_indices], labels_str[train_indices], labels_int[train_indices]), (images[test_indices], labels_str[test_indices], labels_int[test_indices])

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
