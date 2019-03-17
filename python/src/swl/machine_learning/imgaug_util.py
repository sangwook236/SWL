import numpy as np
import imgaug as ia
#from imgaug import augmenters as iaa

#%%------------------------------------------------------------------

def generateBatchesInParallelWithoutOutputAugmentation(imgaug_augmenter, processes, chunksize, inputs, outputs, batch_size, shuffle=True, *args, **kwargs):
	#if not isinstance(imgaug_augmenter, iaa.Sequential):
	#	raise ValueError('The augmenter has to be an instance of imgaug.augmenters.Sequential to augment in parallel')

	def createBatchGeneratorInParallel(inputs, outputs, batch_size, shuffle):
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
					yield ia.Batch(images=batch_inputs, data=batch_outputs)

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	# Start a pool to augment on multiple CPU cores.
	#	processes=-1 means that all CPU cores except one are used for the augmentation, so one is kept free to move data to the GPU.
	#	maxtasksperchild=20 restarts child workers every 20 tasks.
	#		Only use this if you encounter problems such as memory leaks.
	#		Restarting child workers decreases performance.
	#	seed=123 makes the result of the whole augmentation process deterministic between runs of this script, i.e. reproducible results.
	#with imgaug_augmenter.pool(processes=-1, maxtasksperchild=20, seed=123) as pool:
	with imgaug_augmenter.pool(processes=processes) as pool:
		batch_gen = createBatchGeneratorInParallel(inputs, outputs, batch_size, shuffle)

		# Augment on multiple CPU cores.
		#	The result of imap_batches() is also a generator.
		#	Use map_batches() if your input is a list.
		#	chunksize=10 controls how much data to send to each child worker per transfer, set it higher for better performance.
		batch_aug_gen = pool.imap_batches(batch_gen, chunksize=chunksize)

		for batch in batch_aug_gen:
			yield (batch.images_aug, batch.data), len(batch.images_aug)

def generateBatchesInParallelWithOutputAugmentation(imgaug_augmenter, processes, chunksize, inputs, outputs, batch_size, shuffle=True, *args, **kwargs):
	#if not isinstance(imgaug_augmenter, iaa.Sequential):
	#	raise ValueError('The augmenter has to be an instance of imgaug.augmenters.Sequential to augment in parallel')

	def createBatchGeneratorInParallel(inputs, outputs, batch_size, shuffle):
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
					yield ia.Batch(images=batch_inputs, segmentation_maps=batch_outputs)

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	# Start a pool to augment on multiple CPU cores.
	#	processes=-1 means that all CPU cores except one are used for the augmentation, so one is kept free to move data to the GPU.
	#	maxtasksperchild=20 restarts child workers every 20 tasks.
	#		Only use this if you encounter problems such as memory leaks.
	#		Restarting child workers decreases performance.
	#	seed=123 makes the result of the whole augmentation process deterministic between runs of this script, i.e. reproducible results.
	#with imgaug_augmenter.pool(processes=-1, maxtasksperchild=20, seed=123) as pool:
	with imgaug_augmenter.pool(processes=processes) as pool:
		batch_gen = createBatchGeneratorInParallel(inputs, outputs, batch_size, shuffle)

		# Augment on multiple CPU cores.
		#	The result of imap_batches() is also a generator.
		#	Use map_batches() if your input is a list.
		#	chunksize=10 controls how much data to send to each child worker per transfer, set it higher for better performance.
		batch_aug_gen = pool.imap_batches(batch_gen, chunksize=chunksize)

		for batch in batch_aug_gen:
			yield (batch.images_aug, batch.segmentation_maps_aug), len(batch.images_aug)
