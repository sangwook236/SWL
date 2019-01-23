import os
from functools import partial
import numpy as np
from swl.machine_learning.batch_manager import BatchManager, FileBatchManager

#%%------------------------------------------------------------------
# AugmentationBatchManager.
#	Generates and augments batches.
#	An augmenter has to support a function, augment(images, labels, is_label_augmented=False). 
class AugmentationBatchManager(BatchManager):
	def __init__(self, augmenter, images, labels, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False, process_pool=None):
		super().__init__()

		self._augmenter = augmenter
		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._is_label_augmented = is_label_augmented
		self._process_pool = process_pool

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._images is not None:
			self._num_examples = self._images.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // batch_size + 1) if self._num_examples > 0 else 0
		#if self._images is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	"""
	def getBatches(self, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._num_steps):
			yield AugmentationBatchManager._getBatches(self._augmenter, self._images, self._labels, self._batch_size, self._is_label_augmented, indices, step)
	"""

	def getBatches(self, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		if self._process_pool is None:
			for step in range(self._num_steps):
				yield AugmentationBatchManager._getBatches(self._augmenter, self._images, self._labels, self._batch_size, self._is_label_augmented, indices, step)
		else:
			# TODO [improve] >> Starts yielding after generating all batches.
			retval = self._process_pool.map(partial(AugmentationBatchManager._getBatches, self._augmenter, self._images, self._labels, self._batch_size, self._is_label_augmented, indices), range(self._num_steps))
			for rv in retval:
				yield rv

	@staticmethod
	def _getBatches(augmenter, images, labels, batch_size, is_label_augmented, indices, step):
		start = step * batch_size
		end = start + batch_size
		batch_indices = indices[start:end]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			batch_images = images[batch_indices]
			batch_labels = labels[batch_indices]
			if batch_images.size > 0 and batch_labels.size > 0:  # If batch_images and batch_labels are non-empty.
				# augmenter.augment() can be run in an individual thread or process.
				return augmenter.augment(batch_images, batch_labels, is_label_augmented)

#%%------------------------------------------------------------------
# AugmentationBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates and augments batches.
#	An augmenter has to support a function, augment(images, labels, is_label_augmented=False). 
class AugmentationBatchManagerWithFileInput(AugmentationBatchManager):
	def __init__(self, augmenter, npy_filepath_pairs, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False, process_pool=None):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major, process_pool)

#%%------------------------------------------------------------------
# AugmentationFileBatchManager.
#	Generates, augments, saves, and loads batches through npy files.
#	An augmenter has to support a function, augment(images, labels, is_label_augmented=False). 
class AugmentationFileBatchManager(FileBatchManager):
	def __init__(self, augmenter, images, labels, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False, process_pool=None, image_file_format=None, label_file_format=None):
		super().__init__()

		self._augmenter = augmenter
		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._is_label_augmented = is_label_augmented
		self._process_pool = process_pool

		self._image_file_format = 'batch_images_{}.npy' if image_file_format is None else image_file_format
		self._label_file_format = 'batch_labels_{}.npy' if label_file_format is None else label_file_format

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._images is not None:
			self._num_examples = self._images.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // self._batch_size + 1) if self._num_examples > 0 else 0
		#if self._images is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def getBatches(self, dir_path, *args, **kwargs):
		for step in range(self._num_steps):
			batch_images = np.load(os.path.join(dir_path, self._image_file_format.format(step)))
			batch_labels = np.load(os.path.join(dir_path, self._label_file_format.format(step)))
			print('=====')
			yield batch_images, batch_labels

	def putBatches(self, dir_path, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		if self._process_pool is None:
			for step in range(self._num_steps):
				AugmentationFileBatchManager._putBatches(self._augmenter, self._images, self._labels, dir_path, self._image_file_format, self._label_file_format, self._batch_size, self._is_label_augmented, indices, step)
		else:
			self._process_pool.map(partial(AugmentationFileBatchManager._putBatches, self._augmenter, self._images, self._labels, dir_path, self._image_file_format, self._label_file_format, self._batch_size, self._is_label_augmented, indices), range(self._num_steps))

	@staticmethod
	def _putBatches(augmenter, images, labels, dir_path, image_file_format, label_file_format, batch_size, is_label_augmented, indices, step):
		start = step * batch_size
		end = start + batch_size
		batch_indices = indices[start:end]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			batch_images = images[batch_indices]
			batch_labels = labels[batch_indices]
			if batch_images.size > 0 and batch_labels.size > 0:  # If batch_images and batch_labels are non-empty.
				# augmenter.augment() can be run in an individual thread or process.
				batch_images, batch_labels = augmenter.augment(batch_images, batch_labels, is_label_augmented)

				np.save(os.path.join(dir_path, image_file_format.format(step)), batch_images)
				np.save(os.path.join(dir_path, label_file_format.format(step)), batch_labels)

#%%------------------------------------------------------------------
# AugmentationFileBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates, augments, saves, and loads batches through npy files.
#	An augmenter has to support a function, augment(images, labels, is_label_augmented=False). 
class AugmentationFileBatchManagerWithFileInput(AugmentationFileBatchManager):
	def __init__(self, augmenter, npy_filepath_pairs, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False, process_pool=None):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major, process_pool)
