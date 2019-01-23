import os, abc
import numpy as np

#%%------------------------------------------------------------------
# BatchGenerator.
#	Generates batches.
class BatchGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a batch generator.
	@abc.abstractmethod
	def generateBatches(self, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# FileBatchGenerator.
#	Generates and saves batches to files.
class FileBatchGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a list of filepath pairs, (image filepath, label filepath).
	@abc.abstractmethod
	def saveBatches(self, dir_path, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# FileBatchLoader.
#	Loads batches from files.
class FileBatchLoader(abc.ABC):
	def __init__(self):
		super().__init__()

	# Returns a batch generator.
	@abc.abstractmethod
	def loadBatches(self, filepath_pairs, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# SimpleBatchGenerator.
#	Generates batches.
class SimpleBatchGenerator(BatchGenerator):
	# functor: images, labels = functor(images, labels).
	def __init__(self, images, labels, batch_size, shuffle=True, is_time_major=False, functor=None):
		super().__init__()

		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._functor = functor

		batch_axis = 1 if is_time_major else 0
		self._num_examples, self._num_steps = 0, 0
		if self._images is not None:
			self._num_examples = self._images.shape[batch_axis]
			self._num_steps = ((self._num_examples - 1) // batch_size + 1) if self._num_examples > 0 else 0
		#if self._images is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def generateBatches(self, *args, **kwargs):
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
					if self._functor is None:
						yield batch_images, batch_labels
					else:
						yield self._functor(batch_images, batch_labels)

#%%------------------------------------------------------------------
# NpyFileBatchGenerator.
#	Generates and saves batches to npy files.
class NpyFileBatchGenerator(FileBatchGenerator):
	def __init__(self, images, labels, batch_size, shuffle=True, is_time_major=False, functor=None, image_file_format=None, label_file_format=None):
		super().__init__()

		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._functor = functor

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

	def saveBatches(self, dir_path, *args, **kwargs):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		npy_filepath_pairs = list()
		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_images = self._images[batch_indices]
				batch_labels = self._labels[batch_indices]
				if batch_images.size > 0 and batch_labels.size > 0:  # If batch_images and batch_labels are non-empty.
					if self._functor is not None:
						batch_images, batch_labels = self._functor(batch_images, batch_labels)
					image_filepath, label_filepath = os.path.join(dir_path, self._image_file_format.format(step)), os.path.join(dir_path, self._label_file_format.format(step))
					np.save(image_filepath, batch_images)
					np.save(label_filepath, batch_labels)
					npy_filepath_pairs.append((image_filepath, label_filepath))
		return npy_filepath_pairs

#%%------------------------------------------------------------------
# NpyFileBatchLoader.
#	Loads batches from npy files.
class NpyFileBatchLoader(FileBatchLoader):
	def __init__(self):
		super().__init__()

	def loadBatches(self, npy_filepath_pairs, *args, **kwargs):
		for image_filepath, label_filepath in npy_filepath_pairs:
			batch_images = np.load(image_filepath)
			batch_labels = np.load(label_filepath)
			yield batch_images, batch_labels
