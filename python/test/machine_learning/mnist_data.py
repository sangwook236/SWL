from functools import partial
import numpy as np
from sklearn import preprocessing
import cv2 as cv
import tensorflow as tf
#import imgaug as ia
from imgaug import augmenters as iaa
from swl.machine_learning.data_generator import Data2Generator
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.imgaug_util as imgaug_util

#--------------------------------------------------------------------
# ImgaugDataAugmenter.

class ImgaugDataAugmenter(object):
	def __init__(self, is_output_augmented=False):
		self._augment_functor = self._augmentInputAndOutput if is_output_augmented else self._augmentInput
		self._augmenter = iaa.Sequential([
			iaa.OneOf([
				#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))), # Crop images by 0-10% of their height/width.
				#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
				#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
			]),
			iaa.Sometimes(0.5, iaa.SomeOf(1, [
				iaa.Affine(
					scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
					rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
					shear=(-16, 16),  # Shear by -16 to +16 degrees.
					#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
					order=0,  # Use nearest neighbour or bilinear interpolation (fast).
					#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
					#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
					#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				),
				#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
				iaa.PerspectiveTransform(scale=(0.01, 0.1)),
				iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),  # Move pixels locally around (with random strengths).
			])),
			iaa.Sometimes(0.5, iaa.OneOf([
				iaa.GaussianBlur(sigma=(0, 3.0)),  # Blur images with a sigma between 0 and 3.0
				iaa.AverageBlur(k=(2, 7)),  # Blur image using local means with kernel sizes between 2 and 7
				iaa.MedianBlur(k=(3, 11)),  # Blur image using local medians with kernel sizes between 2 and 7

				iaa.Invert(0.05, per_channel=True),  # Invert color channels.
				iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # Improve or worsen the contrast.

				#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # Sharpen images.
				#iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # Emboss images.

				# Search either for all edges or for directed edges, blend the result with the original image using a blobby mask.
				#iaa.SimplexNoiseAlpha(iaa.OneOf([
				#	iaa.EdgeDetect(alpha=(0.5, 1.0)),
				#	iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
				#])),
				iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),  # Add gaussian noise to images.
			])),
			#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
		])

	def __call__(self, inputs, outputs, *args, **kwargs):
		return self._augment_functor(inputs, outputs, *args, **kwargs)

	def _augmentInput(self, inputs, outputs, *args, **kwargs):
		return self._augmenter.augment_images(inputs), outputs

	def _augmentInputAndOutput(self, inputs, outputs, *args, **kwargs):
		augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

#--------------------------------------------------------------------
# MnistDataPreprocessor.

class MnistDataPreprocessor(object):
	def __init__(self, input_shape, num_classes):
		super().__init__()

		self._input_shape = input_shape
		self._num_classes = num_classes

		# Contrast limited adaptive histogram equalization (CLAHE).
		self._clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

	def __call__(self, inputs, outputs, *args, **kwargs):
		if inputs is not None:
			# Preprocessing.
			inputs = np.array([self._clahe.apply(inp) for inp in inputs])

			# Normalization, standardization, etc.
			inputs = inputs.astype(np.float32)

			"""
			inputs = preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
			#inputs = preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
			#inputs = preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
			#inputs = preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
			"""

			inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
			#in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
			#out_min, out_max = 0, 1 #-1, 1
			#inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
			#inputs /= 255.0  # Normalization.

			# Reshaping.
			#inputs = np.reshape(inputs, inputs.shape + (1,))
			inputs = np.reshape(inputs, (-1,) + self._input_shape)

		if outputs is not None:
			# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
			outputs = swl_ml_util.to_one_hot_encoding(outputs, self._num_classes).astype(np.uint8)
			#outputs = tf.keras.utils.to_categorical(outputs).astype(np.uint8)

		return inputs, outputs

#--------------------------------------------------------------------
# MnistDataVisualizer.

class MnistDataVisualizer(object):
	def __init__(self, start_index=0, end_index=5):
		"""
		Inputs:
			start_index (int): The start index of example to show.
			end_index (int): The end index of example to show.
				Shows examples between start_index and end_index, (start_index, end_index).
		"""

		self._start_index, self._end_index = start_index, end_index

	def __call__(self, data, *args, **kwargs):
		import types
		if isinstance(data, types.GeneratorType):
			start_example_index = 0
			for datum in data:
				self._visualize(datum, start_example_index, *args, **kwargs)
				start_example_index += num_examples
		else:
			self._visualize(data, 0, *args, **kwargs)
		cv.destroyAllWindows()

	def _visualize(self, data, start_example_index, *args, **kwargs):
		(inputs, outputs), num_examples = data
		if isinstance(inputs, np.ndarray):
			print('\tInput: shape = {}, dtype = {}.'.format(inputs.shape, inputs.dtype))
			print('\tInput: min = {}, max = {}.'.format(np.min(inputs), np.max(inputs)))
		else:
			print('\tInput: type = {}.'.format(type(inputs)))
		if isinstance(outputs, np.ndarray):
			print('\tOutput: shape = {}, dtype = {}.'.format(outputs.shape, outputs.dtype))
			print('\tOutput: min = {}, max = {}.'.format(np.min(outputs), np.max(outputs)))
		else:
			print('\tOutput type = {}.'.format(type(outputs)))

		if len(inputs) != num_examples or len(outputs) != num_examples:
			raise ValueError('The lengths of inputs and outputs are different: {} != {}'.format(len(inputs), len(outputs)))

		for idx, (inp, outp) in enumerate(zip(inputs, outputs)):
			idx += start_example_index
			if idx >= self._start_index and idx < self._end_index:
				print('\tLabel #{} = {} ({}).'.format(idx, outp, np.argmax(outp, axis=-1)))
				cv.imshow('Image', inp)
				ch = cv.waitKey(2000)
				if 27 == ch:  # ESC.
					break

#--------------------------------------------------------------------
# MnistDataGenerator.

class MnistDataGenerator(Data2Generator):
	def __init__(self, is_output_augmented=False, is_augmented_in_parallel=True):
		super().__init__()

		self._is_augmented_in_parallel = is_augmented_in_parallel

		self._num_classes = 10
		self._input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
		self._output_shape = (None, self._num_classes)

		self._train_inputs, self._train_outputs, self._test_inputs, self._test_outputs = (None,) * 4

		#--------------------
		self._preprocessor = MnistDataPreprocessor(self._input_shape[1:], self._num_classes)
		self._augmenter = ImgaugDataAugmenter(is_output_augmented)
		#self._augmenter = None

		if self._augmenter is None:
			self._batch_generator = MnistDataGenerator._generateBatchesWithoutAugmentation
		else:
			if self._is_augmented_in_parallel:
				num_processes, chunksize = 4, 5
				self._batch_generator = partial(imgaug_util.generateBatchesInParallelWithOutputAugmentation, num_processes, chunksize, self._augmenter._augmenter) if is_output_augmented else partial(imgaug_util.generateBatchesInParallelWithoutOutputAugmentation, num_processes, chunksize, self._augmenter._augmenter)
			else:
				self._batch_generator = partial(MnistDataGenerator._generateBatchesWithAugmentation, self._augmenter)
		self._batch_generator_without_aug = MnistDataGenerator._generateBatchesWithoutAugmentation

	@property
	def dataset(self):
		raise NotImplementedError

	@property
	def shapes(self):
		return self._input_shape, self._output_shape, self._num_classes

	def initialize(self, batch_size=None, *args, **kwargs):
		# Pixel value: [0, 255].
		(self._train_inputs, self._train_outputs), (self._test_inputs, self._test_outputs) = tf.keras.datasets.mnist.load_data()

		if self._preprocessor is not None:
			"""
			# NOTE [info] >> 'Data augmentation followd by preprocessing' is better.
			self._train_inputs, self._train_outputs = self._preprocessor(self._train_inputs, self._train_outputs)
			self._test_inputs, self._test_outputs = self._preprocessor(self._test_inputs, self._test_outputs)
			"""
			self._test_inputs, self._test_outputs = self._preprocessor(self._test_inputs, self._test_outputs)

		#--------------------
		if self._train_inputs is None or self._train_outputs is None:
			raise ValueError('At least one of train input or output data is None')
		if len(self._train_inputs) != len(self._train_outputs):
			raise ValueError('The lengths of train input and output data are different: {} != {}'.format(len(self._train_inputs), len(self._train_outputs)))
		if self._test_inputs is None or self._test_outputs is None:
			raise ValueError('At least one of test input or output data is None')
		if len(self._test_inputs) != len(self._test_outputs):
			raise ValueError('The lengths of test input and output data are different: {} != {}'.format(len(self._test_inputs), len(self._test_outputs)))

		#--------------------
		# Visualizes data to check data itself, as well as data preprocessing and augmentation.
		if True:
			visualizer = MnistDataVisualizer(start_index=0, end_index=5)
			print('[SWL] Train data which is augmented (optional) and preprocessed.')
			# Data augmentation (optional) + data preprocessing.
			visualizer(self._batch_generator(self._preprocessor, self._train_inputs, self._train_outputs, batch_size=None, shuffle=False))
			print('[SWL] Train data which is preprocessed but not augmented.')
			# No data augmentation + data preprocessing.
			visualizer(self._batch_generator_without_aug(self._preprocessor, self._train_inputs, self._train_outputs, batch_size=None, shuffle=False))
			#visualizer(self._batch_generator_without_aug(self._preprocessor, self._test_inputs, self._test_outputs, batch_size=None, shuffle=False))
			print('[SWL] Test data which is preprocessed but not augmented.')
			# No data augmentation + no data preprocessing (has already been processed).
			visualizer(self._batch_generator_without_aug(None, self._test_inputs, self._test_outputs, batch_size=None, shuffle=False))

	def getTrainBatches(self, batch_size, shuffle=True, *args, **kwargs):
		if self._train_inputs is None or self._train_outputs is None:
			raise ValueError('At least one of train input or output data is None')

		# Data augmentation (optional) + data preprocessing.
		return self._batch_generator(self._preprocessor, self._train_inputs, self._train_outputs, batch_size, shuffle, *args, **kwargs)

	def getTrainBatchesForEvaluation(self, batch_size, shuffle=False, *args, **kwargs):
		"""Gets train batches for evaluation such as loss and accuracy, etc.
		"""

		# Data augmentation (optional) + data preprocessing.
		#return self.getTrainBatches(batch_size, shuffle, *args, **kwargs)
		# No data augmentation + data preprocessing.
		#yield self._preprocessor(self._train_inputs, self._train_outputs), (0 if self._train_inputs is None else len(self._train_inputs))  # ResourceExhaustedError is raised.
		return self._batch_generator_without_aug(self._preprocessor, self._train_inputs, self._train_outputs, batch_size, shuffle, *args, **kwargs)

	def hasValidationBatches(self):
		return self.hasTestBatches()

	def getValidationBatches(self, batch_size=None, shuffle=False, *args, **kwargs):
		return self.getTestBatches(batch_size, shuffle, *args, **kwargs)

	def hasTestBatches(self):
		return self._test_inputs is not None and self._test_outputs is not None and len(self._test_inputs) > 0

	def getTestBatches(self, batch_size=None, shuffle=False, *args, **kwargs):
		if self._test_inputs is None or self._test_outputs is None:
			raise ValueError('At least one of test input or output data is None')

		#if batch_size is None:
		if True:  # Wants to infer all data at a time.
			# No data augmentation + data preprocessing.
			#yield ((self._test_inputs, self._test_outputs) if self._preprocessor is None else self._preprocessor(self._test_inputs, self._test_outputs)), (0 if self._test_inputs is None else len(self._test_inputs))
			# No data augmentation + no data preprocessing.
			yield (self._test_inputs, self._test_outputs), (0 if self._test_inputs is None else len(self._test_inputs))
		else:
			# Data augmentation (optional) + data preprocessing.
			#return self._batch_generator(self._preprocessor, self._test_inputs, self._test_outputs, batch_size, shuffle, *args, **kwargs)
			# Data augmentation (optional) + no data preprocessing.
			#return self._batch_generator(None, self._test_inputs, self._test_outputs, batch_size, shuffle, *args, **kwargs)
			# No data augmentation + data preprocessing.
			#return self._batch_generator_without_aug(self._preprocessor, self._test_inputs, self._test_outputs, batch_size, shuffle, *args, **kwargs)
			# No data augmentation + no data preprocessing.
			return self._batch_generator_without_aug(None, self._test_inputs, self._test_outputs, batch_size, shuffle, *args, **kwargs)
