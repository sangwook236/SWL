import os
import numpy as np
import imgaug as ia
from swl.machine_learning.batch_manager import BatchManager, FileBatchManager

# Load and augment batches using imgaug library in background processes.
class ImgaugBatchManager(BatchManager):
	def __init__(self, images, labels, augseq, batch_size, shuffle=True, is_time_major=False):
		super().__init__()

		self._images = images
		self._labels = labels
		self._augseq = augseq
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

	def getBatches(self, num_epoches, *args, **kwargs):
		for epoch in range(num_epoches):
			batch_loader = ia.BatchLoader(self._loadBatchesForImage)
			bg_augmenter = ia.BackgroundAugmenter(batch_loader, self._augseq)

			while True:
				batch = bg_augmenter.get_batch()
				if batch is None:
					break
				#images = batch.images
				images_aug = batch.images_aug
				#keypoints = batch.keypoints
				#keypoints_aug = batch.keypoints_aug

				yield images_aug, self._labels[batch.data]

			batch_loader.terminate()
			bg_augmenter.terminate()

	def getBatches2(self, num_epoches, *args, **kwargs):
		for epoch in range(num_epoches):
			# FIXME [fix] >> Incorrect.
			batch_loader = ia.BatchLoader(self._loadBatchPairs)
			augseq_det = self._augseq.to_deterministic()
			bg_augmenter = ia.BackgroundAugmenter(batch_loader, augseq_det)
			bg_augmenter = ia.BackgroundAugmenter(batch_loader, augseq_det)

			while True:
				batch = bg_augmenter.get_batch()
				if batch is None:
					break
				#images = batch.images
				images_aug = batch.images_aug
				#keypoints = batch.keypoints
				#keypoints_aug = batch.keypoints_aug

				yield images_aug, self._labels[batch.data]

			batch_loader.terminate()
			bg_augmenter.terminate()

	# A generator that loads batches from a numpy array.
	def _loadBatchesForImage(self):
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
					yield ia.Batch(images=batch_images, data=batch_labels)

#%%------------------------------------------------------------------

# Load, save, and augment batches from files using imgaug library.
class ImgaugFileBatchManager(FileBatchManager):
	def __init__(self, images, labels, augseq, is_label_image=False, is_time_major=False):
		super().__init__()

		self._images = images
		self._labels = labels
		self._augseq = augseq
		self._is_label_image = is_label_image
		self._is_time_major = is_time_major

	def getBatches(self, dir_path, filename_pairs, num_epoches, *args, **kwargs):
		for epoch in range(num_epoches):
			for img_filename, lbl_filename in filename_pairs:
				batch_images = np.load(os.path.join(dir_path, img_filename))
				batch_labels = np.load(os.path.join(dir_path, lbl_filename))

				if self._is_label_image:
					augseq_det = self._augseq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
					batch_images = augseq_det.augment_images(batch_images)
					batch_labels = augseq_det.augment_images(batch_labels)
				else:
					batch_images = self._augseq.augment_images(batch_images)

				yield batch_images, batch_labels

	def putBatches(self, dir_path, filename_pairs, shuffle, *args, **kwargs):
		batch_axis = 1 if self._is_time_major else 0
		num_steps = len(filename_pairs)

		num_examples, batch_size = 0, 0
		if self._images is not None:
			num_examples = self._images.shape[batch_axis]
			batch_size = (num_examples // num_steps + 1) if num_examples > 0 else 0
		#if self._images is None:
		if num_examples <= 0:
			raise ValueError('Invalid argument')

		indices = np.arange(num_examples)
		if shuffle:
			np.random.shuffle(indices)

		for step, (img_filename, lbl_filename) in enumerate(filename_pairs):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_images = self._images[batch_indices]
				batch_labels = self._labels[batch_indices]
				if batch_images.size > 0 and batch_labels.size > 0:  # If batch_images and batch_labels are non-empty.
					np.save(os.path.join(dir_path, img_filename), batch_images)
					np.save(os.path.join(dir_path, lbl_filename), batch_labels)
