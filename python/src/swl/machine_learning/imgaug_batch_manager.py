import os
import numpy as np
import imgaug as ia
from swl.machine_learning.batch_manager import BatchManager, FileBatchManager

#%%------------------------------------------------------------------
# ImgaugBatchManager.
#	Generates and augments batches using imgaug library in background processes.
class ImgaugBatchManager(BatchManager):
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
		if self._is_label_augmented:
			# FIXME [fix] >> Do not check.
			batch_loader = ia.BatchLoader(self._loadBatchPairs)
			augmenter_det = self._augmenter.to_deterministic()
			bg_augmenter = ia.BackgroundAugmenter(batch_loader, augmenter_det)

			while True:
				batch = bg_augmenter.get_batch()
				if batch is None:
					break

				yield batch.images_aug, batch.keypoints_aug

			batch_loader.terminate()
			bg_augmenter.terminate()
		else:
			batch_loader = ia.BatchLoader(self._loadBatches)
			bg_augmenter = ia.BackgroundAugmenter(batch_loader, self._augmenter)

			while True:
				batch = bg_augmenter.get_batch()
				if batch is None:
					break

				#images = batch.images
				#images_aug = batch.images_aug
				#keypoints = batch.keypoints
				#keypoints_aug = batch.keypoints_aug
				#data = batch.data

				yield batch.images_aug, self._labels[batch.data]

			batch_loader.terminate()
			bg_augmenter.terminate()

	# A generator that loads batches from a numpy array.
	def _loadBatches(self):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._num_steps):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_images = self._images[batch_indices]
				if batch_images.size > 0:  # If batch_images is non-empty.
					# Create the batch object to send to the background processes.
					yield ia.Batch(images=batch_images, data=batch_indices)

	# A generator that loads batches from a numpy array.
	def _loadBatchPairs(self):
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
					# Create the batch object to send to the background processes.
					#yield ia.Batch(images=batch_images, data=batch_labels)
					yield ia.Batch(images=batch_images, keypoints=batch_labels)

#%%------------------------------------------------------------------
# ImgaugBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates and augment batches using imgaug library in background processes.
class ImgaugBatchManagerWithFileInput(ImgaugBatchManager):
	def __init__(self, augmenter, npy_filepath_pairs, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(augmenter, images, labels, batch_size, shuffle, is_label_augmented, is_time_major)

#%%------------------------------------------------------------------
# ImgaugFileBatchManager.
#	Generates, augments, saves, and loads batches through npy files using imgaug library.
class ImgaugFileBatchManager(FileBatchManager):
	def __init__(self, augmenter, images, labels, dir_path, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False, image_file_format=None, label_file_format=None):
		super().__init__()

		self._augmenter = augmenter
		self._images = images
		self._labels = labels
		self._dir_path = dir_path
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._is_label_augmented = is_label_augmented

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
						augseq_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
						batch_images = augseq_det.augment_images(batch_images)
						batch_labels = augseq_det.augment_images(batch_labels)
					else:
						batch_images = self._augmenter.augment_images(batch_images)

					np.save(os.path.join(self._dir_path, self._image_file_format.format(step)), batch_images)
					np.save(os.path.join(self._dir_path, self._label_file_format.format(step)), batch_labels)

#%%------------------------------------------------------------------
# ImgaugFileBatchManagerWithFileInput.
#	Loads dataset from multiple npy files.
#	Generates, augments, saves, and loads batches through npy files using imgaug library.
class ImgaugFileBatchManagerWithFileInput(ImgaugFileBatchManager):
	def __init__(self, augmenter, npy_filepath_pairs, dir_path, batch_size, shuffle=True, is_label_augmented=False, is_time_major=False):
		images, labels = None, None
		for image_filepath, label_filepath in npy_filepath_pairs:
			imgs = np.load(image_filepath)
			lbls = np.load(label_filepath)
			images = imgs if images is None else np.concatenate((images, imgs), axis=0)
			labels = lbls if labels is None else np.concatenate((labels, lbls), axis=0)

		super().__init__(augmenter, images, labels, dir_path, batch_size, shuffle, is_label_augmented, is_time_major)
