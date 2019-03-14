import abc
import numpy as np
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
from swl.machine_learning.data_generator import DataGenerator
import swl.machine_learning.util as swl_ml_util

#%%------------------------------------------------------------------
# ImgaugDataAugmenter.

class ImgaugDataAugmenter(object):
	def __init__(self, is_output_augmented=False, is_augmented_in_parallel=False):
		super().__init__()

		self._is_output_augmented = is_output_augmented
		self._is_augmented_in_parallel = is_augmented_in_parallel

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

	@property
	def isOutputAugmented(self):
		if self._is_output_augmented is None:
			raise TypeError
		return self._is_output_augmented

	@property
	def isAugmentedInParallel(self):
		if self._is_augmented_in_parallel is None:
			raise TypeError
		return self._is_augmented_in_parallel

	# Augments in sequence.
	def augment(self, inputs, outputs, *args, **kwargs):
		if self._is_output_augmented:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)
		else:
			return self._augmenter.augment_images(inputs), outputs

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

#class MnistDataGenerator(abc.ABC):
class MnistDataGenerator(DataGenerator):
	def __init__(self, preprocessor, augmenter):
		super().__init__()

		self._preprocessor = preprocessor
		self._augmenter = augmenter

		self._train_inputs, self._train_outputs, self._test_inputs, self._test_outputs = (None,) * 4

	def initialize(self):
		# Pixel value: [0, 255].
		(self._train_inputs, self._train_outputs), (self._test_inputs, self._test_outputs) = tf.keras.datasets.mnist.load_data()
		if self._preprocessor is not None:
			self._train_inputs, self._train_outputs = self._preprocessor(self._train_inputs, self._train_outputs)
			self._test_inputs, self._test_outputs = self._preprocessor(self._test_inputs, self._test_outputs)

		if self._train_inputs is None or self._train_outputs is None:
			raise ValueError('Train inputs or outputs is None')
		if len(self._train_inputs) != len(self._train_outputs):
			raise ValueError('The lengths of train inputs and outputs are different: {} != {}'.format(len(self._train_inputs), len(self._train_outputs)))
		if self._test_inputs is None or self._test_outputs is None:
			raise ValueError('Test inputs or outputs is None')
		if len(self._test_inputs) != len(self._test_outputs):
			raise ValueError('The lengths of test inputs and outputs are different: {} != {}'.format(len(self._test_inputs), len(self._test_outputs)))

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
		if self._augmenter is not None and self._augmenter.isAugmentedInParallel:
			return self._generateBatchesInParallel(inputs, outputs, batch_size, shuffle, *args, **kwargs)
		else:
			return self._generateBatchesInSequence(inputs, outputs, batch_size, shuffle, *args, **kwargs)

	def _generateBatchesInSequence(self, inputs, outputs, batch_size, shuffle=True, *args, **kwargs):
		num_examples = len(inputs)
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
				batch_inputs, batch_outputs = inputs[batch_indices], outputs[batch_indices]
				if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
					if self._augmenter is not None:
						batch_inputs, batch_outputs = self._augmenter.augment(batch_inputs, batch_outputs)
					yield (batch_inputs, batch_outputs), batch_indices.size

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	def _generateBatchesInParallel(self, inputs, outputs, batch_size, shuffle=True, *args, **kwargs):
		if not isinstance(self._augmenter._augmenter, iaa.Sequential):
			raise ValueError('The augmenter has to be an instance of imgaug.augmenters.Sequential to augment in parallel')

		# TODO [enhance] >> To use self._augmenter._augmenter is not so good.
		# Start a pool to augment on multiple CPU cores.
		#	processes=-1 means that all CPU cores except one are used for the augmentation, so one is kept free to move data to the GPU.
		#	maxtasksperchild=20 restarts child workers every 20 tasks.
		#		Only use this if you encounter problems such as memory leaks.
		#		Restarting child workers decreases performance.
		#	seed=123 makes the result of the whole augmentation process deterministic between runs of this script, i.e. reproducible results.
		#with self._augmenter._augmenter.pool(processes=-1, maxtasksperchild=20, seed=123) as pool:
		with self._augmenter._augmenter.pool(processes=4) as pool:
			batch_gen = self._createBatchGeneratorInParallel(inputs, outputs, batch_size, shuffle)

			# Augment on multiple CPU cores.
			#	The result of imap_batches() is also a generator.
			#	Use map_batches() if your input is a list.
			#	chunksize=10 controls how much data to send to each child worker per transfer, set it higher for better performance.
			batch_aug_gen = pool.imap_batches(batch_gen, chunksize=5)

			for batch in batch_aug_gen:
				yield (batch.images_aug, (batch.segmentation_maps_aug if self._augmenter.isOutputAugmented else batch.data)), len(batch.images_aug)

	def _createBatchGeneratorInParallel(self, inputs, outputs, batch_size, shuffle):
		num_examples = len(inputs)
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
				batch_inputs, batch_outputs = inputs[batch_indices], outputs[batch_indices]
				if batch_inputs.size > 0 and batch_outputs.size > 0:  # If batch_inputs and batch_outputs are non-empty.
					# Add e.g. keypoints=... or bounding_boxes=... here to also augment keypoints / bounding boxes on these images.
					if self._augmenter.isOutputAugmented:
						yield ia.Batch(images=batch_inputs, segmentation_maps=batch_outputs)
					else:
						yield ia.Batch(images=batch_inputs, data=batch_outputs)

			if end_idx >= num_examples:
				break
			start_idx = end_idx
