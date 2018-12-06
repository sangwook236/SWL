from swl.machine_learning.batch_loader import BatchLoader
import imgaug as ia
import numpy as np

# Load and augment batches using imgaug library.
class ImgaugBatchLoader(BatchLoader):
	def __init__(self, augseq, images, labels, batch_size, shuffle=True, is_time_major=False):
		super().__init__()

		self._augseq = augseq
		self._images = images
		self._labels = labels
		self._batch_size = batch_size
		self._shuffle = shuffle
		batch_dim = 1 if is_time_major else 0

		self._num_examples, self._steps_per_epoch = 0, 0
		if self._images is not None:
			self._num_examples = self._images.shape[batch_dim]
			self._steps_per_epoch = ((self._num_examples - 1) // batch_size + 1) if self._num_examples > 0 else 0
		#if self._images is None:
		if self._num_examples <= 0:
			raise ValueError('Invalid argument')

	def getBatches(self, num_epoches):
		for epoch in range(num_epoches):
			batch_loader = ia.BatchLoader(self._loadBatches)
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

	# A generator that loads batches from a numpy array.
	def _loadBatches(self):
		indices = np.arange(self._num_examples)
		if self._shuffle:
			np.random.shuffle(indices)

		for step in range(self._steps_per_epoch):
			start = step * self._batch_size
			end = start + self._batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				batch_images = self._images[batch_indices]
				if batch_images.size > 0:  # If batch_images is non-empty.
					#batch_data = []
					#for idx in batch_indices:
					#	batch_data.append((step, idx))

					# Create the batch object to send to the background processes.
					batch = ia.Batch(
						images=batch_images,
						#data=batch_data
						data=batch_indices
					)

					yield batch
