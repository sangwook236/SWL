import os, abc
import numpy as np

#--------------------------------------------------------------------
# BatchManager.
#	Abstract batch manager.
class BatchManager(abc.ABC):
	def __init__(self):
		super().__init__()

	# Return a batch generator.
	@abc.abstractmethod
	def getBatches(self, *args, **kwargs):
		raise NotImplementedError

#--------------------------------------------------------------------
# FileBatchManager.
#	Loads, saves, and augments batches using files.
class FileBatchManager(abc.ABC):
	def __init__(self):
		super().__init__()

	# Return a batch generator.
	@abc.abstractmethod
	def getBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def putBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

#--------------------------------------------------------------------
# SimpleBatchManager.
#	Generates batches without augmentation.
class SimpleBatchManager(BatchManager):
	def __init__(self, images, labels, batch_size, shuffle=True, is_time_major=False):
		super().__init__()

		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle

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
					yield batch_images, batch_labels

#--------------------------------------------------------------------
# SimpleBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates batches without augmentation.
class SimpleBatchManagerWithFileInput(SimpleBatchManager):
	def __init__(self, npy_filepath_pairs, batch_size, shuffle=True, is_time_major=False):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(images, labels, batch_size, shuffle, is_time_major)

#--------------------------------------------------------------------
# SimpleFileBatchManager.
#	Generates, saves, and loads batches through npy files without augmentation.
class SimpleFileBatchManager(FileBatchManager):
	def __init__(self, images, labels, batch_size, shuffle=True, is_time_major=False, image_file_format=None, label_file_format=None):
		super().__init__()

		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle

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
			yield batch_images, batch_labels

	def putBatches(self, dir_path, *args, **kwargs):
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
					np.save(os.path.join(dir_path, self._image_file_format.format(step)), batch_images)
					np.save(os.path.join(dir_path, self._label_file_format.format(step)), batch_labels)

#--------------------------------------------------------------------
# SimpleFileBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates, saves, and loads batches through npy files without augmentation.
class SimpleFileBatchManagerWithFileInput(SimpleFileBatchManager):
	def __init__(self, npy_filepath_pairs, batch_size, shuffle=True, is_time_major=False):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(images, labels, batch_size, shuffle, is_time_major)
