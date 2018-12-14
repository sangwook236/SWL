import os
import numpy as np
from swl.machine_learning.batch_manager import BatchManager, FileBatchManager

#%%------------------------------------------------------------------
# AugmentationBatchManager.
#	Generates and augments batches.
#	An augmenter supports a function, augment(images, labels=None). 
class AugmentationBatchManager(BatchManager):
	def __init__(self, augmenter, images, labels, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False):
		super().__init__()

		self._augmenter = augmenter
		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._is_label_augmented = is_label_augmented

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._images is not None:
			self._num_examples = self._images.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // batch_size + 1) if self._num_examples > 0 else 0
		#if self._images is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def getBatches(self, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_images = self._images[batch_indices]
				batch_labels = self._labels[batch_indices]
				if batch_images.size > 0 and batch_labels.size > 0:  # If batch_images and batch_labels are non-empty.
					if self._is_label_augmented:
						batch_images, batch_labels = self._augmenter.augment(batch_images, batch_labels)
					else:
						batch_images = self._augmenter.augment(batch_images)
					yield batch_images, batch_labels

#%%------------------------------------------------------------------
# AugmentationBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates and augments batches.
#	An augmenter supports a function, augment(images, labels=None). 
class AugmentationBatchManagerWithFileInput(AugmentationBatchManager):
	def __init__(self, augmenter, npy_filepath_pairs, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major)

#%%------------------------------------------------------------------
# AugmentationFileBatchManager.
#	Generates, augments, saves, and loads batches through npy files.
#	An augmenter supports a function, augment(images, labels=None). 
class AugmentationFileBatchManager(FileBatchManager):
	def __init__(self, augmenter, images, labels, dir_path, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False):
		super().__init__()

		self._augmenter = augmenter
		self._images = images
		self._labels = labels
		self._dir_path = dir_path
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._is_label_augmented = is_label_augmented

		self._image_file_format = 'batch_images_{}.npy'
		self._label_file_format = 'batch_labels_{}.npy'

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._images is not None:
			self._num_examples = self._images.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // self._batch_size + 1) if self._num_examples > 0 else 0
		#if self._images is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def getBatches(self, *args, **kwargs):
		for step in range(self._num_steps):
			batch_images = np.load(os.path.join(self._dir_path, self._image_file_format.format(step)))
			batch_labels = np.load(os.path.join(self._dir_path, self._label_file_format.format(step)))
			yield batch_images, batch_labels

	def putBatches(self, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_images = self._images[batch_indices]
				batch_labels = self._labels[batch_indices]
				if batch_images.size > 0 and batch_labels.size > 0:  # If batch_images and batch_labels are non-empty.
					if self._is_label_augmented:
						batch_images, batch_labels = self._augmenter.augment(batch_images, batch_labels)
					else:
						batch_images = self._augmenter.augment(batch_images)

					np.save(os.path.join(self._dir_path, self._image_file_format.format(step)), batch_images)
					np.save(os.path.join(self._dir_path, self._label_file_format.format(step)), batch_labels)

#%%------------------------------------------------------------------
# AugmentationFileBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates, augments, saves, and loads batches through npy files.
#	An augmenter supports a function, augment(images, labels=None). 
class AugmentationFileBatchManagerWithFileInput(AugmentationFileBatchManager):
	def __init__(self, augmenter, npy_filepath_pairs, dir_path, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(augmenter, images, labels, dir_path, batch_size, shuffle, is_label_augmented, is_time_major)
