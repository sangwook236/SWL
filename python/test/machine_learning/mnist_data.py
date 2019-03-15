from functools import partial
import numpy as np
import tensorflow as tf
#import imgaug as ia
from imgaug import augmenters as iaa
from swl.machine_learning.imgaug_data_generator import ImgaugDataGenerator
import swl.machine_learning.util as swl_ml_util

#%%------------------------------------------------------------------
# ImgaugDataAugmenter.

class ImgaugDataAugmenter(object):
	def __init__(self, is_output_augmented=False):
		self._augment_functor = self._augmentWithOutputAugmentation if is_output_augmented else self._augmentWithoutOutputAugmentation
		self._augmenter = iaa.Sequential([
			iaa.SomeOf(1, [
				#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))), # Crop images by 0-10% of their height/width.
				iaa.Fliplr(0.1),  # Horizontally flip 10% of the images.
				iaa.Flipud(0.1),  # Vertically flip 10% of the images.
				iaa.Sometimes(0.5, iaa.Affine(
					scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
					rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
					shear=(-16, 16),  # Shear by -16 to +16 degrees.
					#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
					order=0,  # Use nearest neighbour or bilinear interpolation (fast).
					#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
					#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
					#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				)),
				iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
			]),
			#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
		])

	def __call__(self, inputs, outputs, *args, **kwargs):
		return self._augment_functor(inputs, outputs, *args, **kwargs)

	def _augmentWithoutOutputAugmentation(self, inputs, outputs, *args, **kwargs):
		return self._augmenter.augment_images(inputs), outputs

	def _augmentWithOutputAugmentation(self, inputs, outputs, *args, **kwargs):
		augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

#%%------------------------------------------------------------------
# MnistDataPreprocessor.

class MnistDataPreprocessor(object):
	def __init__(self, input_shape, num_classes):
		super().__init__()

		self._input_shape = input_shape
		self._num_classes = num_classes

	def __call__(self, inputs, outputs, *args, **kwargs):
		if inputs is not None:
			# Preprocessing (normalization, standardization, etc.).
			#inputs = inputs.astype(np.float32)
			#inputs /= 255.0
			#inputs = (inputs - np.mean(inputs, axis=axis)) / np.std(inputs, axis=axis)
			#inputs = np.reshape(inputs, inputs.shape + (1,))
			inputs = inputs / 255.0
			inputs = np.reshape(inputs, (-1,) + self._input_shape)

		if outputs is not None:
			# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
			outputs = swl_ml_util.to_one_hot_encoding(outputs, self._num_classes).astype(np.uint8)
			#outputs = tf.keras.utils.to_categorical(outputs).astype(np.uint8)

		return inputs, outputs

#%%------------------------------------------------------------------
# MnistDataGenerator.

class MnistDataGenerator(ImgaugDataGenerator):
	def __init__(self, is_output_augmented=False, is_augmented_in_parallel=True):
		super().__init__()

		self._num_classes = 10
		self._input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
		self._output_shape = (None, self._num_classes)

		self._train_inputs, self._train_outputs, self._test_inputs, self._test_outputs = (None,) * 4

		#--------------------
		self._preprocessor = MnistDataPreprocessor(self._input_shape[1:], self._num_classes)
		self._augmenter = ImgaugDataAugmenter(is_output_augmented)
		#self._augmenter = None
		self._is_augmented_in_parallel = is_augmented_in_parallel

		if self._augmenter is None:
			self._batch_generator = MnistDataGenerator._generateBatchesWithoutAugmentation
		else:
			if self._is_augmented_in_parallel:
				self._batch_generator = partial(MnistDataGenerator._generateBatchesInParallelWithOutputAugmentation, self._augmenter._augmenter) if is_output_augmented else partial(MnistDataGenerator._generateBatchesInParallelWithoutOutputAugmentation, self._augmenter._augmenter)
			else:
				self._batch_generator = partial(MnistDataGenerator._generateBatchesWithAugmentation, self._augmenter)

	@property
	def dataset(self):
		raise NotImplementedError

	@property
	def shapes(self):
		if self._dataset is None:
			raise ValueError('Dataset is None')
		return self._input_shape, self._output_shape, self._num_classes

	def initialize(self):
		# Pixel value: [0, 255].
		(self._train_inputs, self._train_outputs), (self._test_inputs, self._test_outputs) = tf.keras.datasets.mnist.load_data()

		#--------------------
		if self._preprocessor is not None:
			self._train_inputs, self._train_outputs = self._preprocessor(self._train_inputs, self._train_outputs)
			self._test_inputs, self._test_outputs = self._preprocessor(self._test_inputs, self._test_outputs)

		if self._train_inputs is None or self._train_outputs is None:
			raise ValueError('At least one of train input or output data is None')
		if len(self._train_inputs) != len(self._train_outputs):
			raise ValueError('The lengths of train input and output data are different: {} != {}'.format(len(self._train_inputs), len(self._train_outputs)))
		if self._test_inputs is None or self._test_outputs is None:
			raise ValueError('At least one of test input or output data is None')
		if len(self._test_inputs) != len(self._test_outputs):
			raise ValueError('The lengths of test input and output data are different: {} != {}'.format(len(self._test_inputs), len(self._test_outputs)))

	def getTrainBatches(self, batch_size, shuffle=True, *args, **kwargs):
		if self._train_inputs is None or self._train_outputs is None:
			raise ValueError('At least one of train input or output data is None')

		return self._generateBatches(self._train_inputs, self._train_outputs, batch_size, shuffle)

	def hasValidationData(self):
		return self.hasTestData()

	def getValidationData(self, *args, **kwargs):
		return self.getTestData(*args, **kwargs)

	def getValidationBatches(self, batch_size=None, shuffle=False, *args, **kwargs):
		return self.getTestBatches(batch_size, shuffle, *args, **kwargs)

	def hasTestData(self):
		return self._test_inputs is not None and self._test_outputs is not None and len(self._test_inputs) > 0

	def getTestData(self, *args, **kwargs):
		return (self._test_inputs, self._test_outputs), (0 if self._test_inputs is None else len(self._test_inputs))

	def getTestBatches(self, batch_size=None, shuffle=False, *args, **kwargs):
		if self._test_inputs is None or self._test_outputs is None:
			raise ValueError('At least one of test input or output data is None')

		return self._generateBatches(self._test_inputs, self._test_outputs, batch_size, shuffle=False)

	def _generateBatches(self, inputs, outputs, batch_size, shuffle=True, *args, **kwargs):
		return self._batch_generator(inputs, outputs, batch_size, shuffle, *args, **kwargs)
