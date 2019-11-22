import os, random, functools, time
import numpy as np
import cv2
#import sklearn
#import swl.machine_learning.util as swl_ml_util
import hangeul_util as hg_util
import text_line_data

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator

class TextRecognitionDataGeneratorTextLineDatasetBase(text_line_data.TextLineDatasetBase):
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

	def create_train_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=True, *args, **kwargs):
		return self._create_batch_generator(self._train_data, batch_size, shuffle, is_data_augmented=True)

	def create_test_batch_generator(self, batch_size, steps_per_epoch=None, shuffle=False, *args, **kwargs):
		return self._create_batch_generator(self._test_data, batch_size, shuffle, is_data_augmented=False)

	def visualize(self, batch_generator, num_examples=10):
		for batch_data, num_batch_examples in batch_generator:
			batch_images, batch_labels_str, batch_labels_int = batch_data

			print('Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_images.shape, batch_images.dtype, np.min(batch_images), np.max(batch_images)))
			print('Label (str): shape = {}, dtype = {}.'.format(batch_labels_str.shape, batch_labels_str.dtype))
			#print('Label (int): shape = {}, type = {}.'.format(batch_labels_int[2], type(batch_labels_int)))  # Sparse tensor.
			print('Label (int): length = {}, type = {}.'.format(len(batch_labels_int), type(batch_labels_int)))

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
			raise ValueError('[SWL] Error: Unmatched data sizes, {0} != {1} or {0} != {2}'.format(num_examples, len(labels_str), len(labels_int)))

		return images, labels_str, labels_int, num_examples

	def _load_data_with_label_in_filename(self, data_dir_path, image_height, image_width, image_channel, max_label_len):
		if 1 == image_channel:
			flags = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flags = cv2.IMREAD_COLOR
		else:
			raise ValueError('[SWL] Error: Invalid channels {}'.format(image_channel))

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
			raise ValueError('[SWL] Error: Invalid channels {}'.format(image_channel))

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

	def _create_batch_generator(self, data, batch_size, shuffle, is_data_augmented=False):
		images, labels_str, labels_int = data

		num_examples = len(images)
		if len(labels_str) != num_examples or len(labels_int) != num_examples:
			raise ValueError('[SWL] Error: Invalid data length: {} != {} != {}'.format(num_examples, len(labels_str), len(labels_int)))
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('[SWL] Error: Invalid batch size: {}'.format(batch_size))

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
					if is_data_augmented:
						batch_data1, _ = self.augment(batch_data1, None)
					batch_data1, _ = self.preprocess(batch_data1, None)
					yield (batch_data1, batch_data2, batch_data3), batch_indices.size
				else:
					yield (None, None, None), 0
			else:
				yield (None, None, None), 0

			if end_idx >= num_examples:
				break
			start_idx = end_idx

class EnglishTextRecognitionDataGeneratorTextLineDataset(TextRecognitionDataGeneratorTextLineDatasetBase):
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle=True):
		super().__init__(image_height, image_width, image_channel, num_classes=0, default_value=-1, use_NWHC=True)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		digit_charset = '0123456789'
		symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		#label_set = set(alphabet_charset + digit_charset)
		label_set = set(alphabet_charset + digit_charset + symbol_charset)

		# There are words of Unicode Hangeul letters besides KS X 1001.
		#label_set = functools.reduce(lambda x, fpath: x.union(fpath.split('_')[0]), os.listdir(data_dir_path), label_set)
		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

		#--------------------
		self._augmenter = iaa.Sequential([
			iaa.Sometimes(0.5, iaa.OneOf([
				iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
				#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
				#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
			])),
			iaa.Sometimes(0.5, iaa.OneOf([
				iaa.Affine(
					scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent (per axis).
					rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
					shear=(-5, 5),  # Shear by -5 to +5 degrees.
					#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
					order=0,  # Use nearest neighbour or bilinear interpolation (fast).
					#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
					#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
					#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				),
				#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
				iaa.PerspectiveTransform(scale=(0.01, 0.1)),
				iaa.ElasticTransformation(alpha=(15.0, 30.0), sigma=5.0),  # Move pixels locally around (with random strengths).
			])),
			iaa.Sometimes(0.5, iaa.OneOf([
				iaa.OneOf([
					iaa.GaussianBlur(sigma=(1.5, 2.5)),
					iaa.AverageBlur(k=(3, 6)),
					iaa.MedianBlur(k=(3, 5)),
					iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0), order=1),
				]),
				iaa.OneOf([
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.1 * 255, 0.3 * 255), per_channel=False),
					#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.1 * 255, 0.3 * 255), per_channel=False),
					#iaa.AdditivePoissonNoise(lam=(32, 64), per_channel=False),
					iaa.CoarseSaltAndPepper(p=(0.1, 0.3), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarseSalt(p=(0.1, 0.3), size_percent=(0.2, 0.9), per_channel=False),
					#iaa.CoarsePepper(p=(0.1, 0.3), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.05, 0.3), per_channel=False),
				]),
				#iaa.OneOf([
				#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
				#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
				#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

				#	iaa.Invert(p=1, per_channel=False),

				#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
				#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
				#]),
			])),
			#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
		])

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
		if outputs is None:
			return self._augmenter.augment_images(inputs), None
		else:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

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
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle=True):
		super().__init__(image_height, image_width, image_channel, num_classes=0, default_value=-1, use_NWHC=True)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
		#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
		with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
			#hangeul_charset = fd.read().strip('\n')  # A strings.
			hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
			#hangeul_charset = fd.readlines()  # A list of string.
			#hangeul_charset = fd.read().splitlines()  # A list of strings.
		#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
		alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		digit_charset = '0123456789'
		symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		#label_set = set(hangeul_charset + hangeul_jamo_charset)
		label_set = set(hangeul_charset + hangeul_jamo_charset + alphabet_charset + digit_charset + symbol_charset)

		# There are words of Unicode Hangeul letters besides KS X 1001.
		#label_set = functools.reduce(lambda x, fpath: x.union(fpath.split('_')[0]), os.listdir(data_dir_path), label_set)
		#self._labels = sorted(label_set)
		self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

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
		raise NotImplementedError

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
	def __init__(self, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle=True):
		super().__init__(image_height, image_width, image_channel, num_classes=0, default_value=-1, use_NWHC=False)

		if train_test_ratio < 0.0 or train_test_ratio > 1.0:
			raise ValueError('Invalid train-test ratio: {}'.format(train_test_ratio))

		#--------------------
		#self._SOJC = '<SOJC>'  # All Hangeul jamo strings will start with the Start-Of-Jamo-Character token.
		self._EOJC = '<EOJC>'  # All Hangeul jamo strings will end with the End-Of-Jamo-Character token.
		#self._SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
		#self._EOS = '<EOS>'  # All strings will end with the End-Of-String token.
		#self._UNKNOWN = '<UNK>'  # Unknown label token.

        # NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
		self._hangeul2jamo_functor = functools.partial(hg_util.hangeul2jamo, eojc_str=self._EOJC, use_separate_consonants=False, use_separate_vowels=True)
		self._jamo2hangeul_functor = functools.partial(hg_util.jamo2hangeul, eojc_str=self._EOJC, use_separate_consonants=False, use_separate_vowels=True)

		#--------------------
		#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
		alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
		digit_charset = '0123456789'
		symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

		#label_set = set(hangeul_jamo_charset + alphabet_charset + digit_charset)
		label_set = set(hangeul_jamo_charset + alphabet_charset + digit_charset + symbol_charset)
		label_set.add(self._EOJC)

		# There are words of Unicode Hangeul letters besides KS X 1001.
		label_set = functools.reduce(lambda x, fpath: x.union(self._hangeul2jamo_functor(fpath.split('_')[0])), os.listdir(data_dir_path), label_set)
		self._labels = sorted(label_set)
		#self._labels = ''.join(sorted(label_set))
		print('[SWL] Info: Labels = {}.'.format(self._labels))
		print('[SWL] Info: #labels = {}.'.format(len(self._labels)))

		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
		self._num_classes = len(self._labels) + 1  # Labels + blank label.

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
			return list(self._labels.index(ch) for ch in self._hangeul2jamo_functor(label_str))
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

	def augment(self, inputs, outputs, *args, **kwargs):
		raise NotImplementedError

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
