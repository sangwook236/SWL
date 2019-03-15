import abc
import numpy as np

#%%------------------------------------------------------------------
# DataGenerator.

class DataGenerator(abc.ABC):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def initialize(self, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def getTrainBatches(self, batch_size, shuffle=True, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def hasValidationData(self):
		raise NotImplementedError

	@abc.abstractmethod
	def getValidationData(self, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def getValidationBatches(self, batch_size=None, shuffle=False, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def hasTestData(self):
		raise NotImplementedError

	@abc.abstractmethod
	def getTestData(self, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def getTestBatches(self, batch_size=None, shuffle=False, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------
# Data2Generator.

# Data generator with 2 data.
class Data2Generator(DataGenerator):
	@staticmethod
	def _generateBatchesWithoutAugmentation(data1, data2, batch_size, shuffle=True, *args, **kwargs):
		num_examples = len(data1)
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
				batch_data1, batch_data2 = data1[batch_indices], data2[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0:  # If batch_data1 and batch_data2 are non-empty.
					yield (batch_data1, batch_data2), batch_indices.size

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	@staticmethod
	def _generateBatchesWithAugmentation(augmenter, data1, data2, batch_size, shuffle=True, *args, **kwargs):
		num_examples = len(data1)
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
				batch_data1, batch_data2 = data1[batch_indices], data2[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0:  # If batch_data1 and batch_data2 are non-empty.
					batch_data1, batch_data2 = augmenter(batch_data1, batch_data2)
					yield (batch_data1, batch_data2), batch_indices.size

			if end_idx >= num_examples:
				break
			start_idx = end_idx

#%%------------------------------------------------------------------
# Data3Generator.

# Data generator with 3 data.
class Data3Generator(DataGenerator):
	@staticmethod
	def _generateBatchesWithoutAugmentation(data1, data2, data3, batch_size, shuffle=True, *args, **kwargs):
		num_examples = len(data1)
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
				batch_data1, batch_data2, batch_data3 = data1[batch_indices], data2[batch_indices], data3[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0 and batch_data3.size > 0:  # If batch_data1, batch_data2, and batch_data3 are non-empty.
					yield (batch_data1, batch_data2, batch_data3), batch_indices.size

			if end_idx >= num_examples:
				break
			start_idx = end_idx

	@staticmethod
	def _generateBatchesWithAugmentation(augmenter, data1, data2, data3, batch_size, shuffle=True, *args, **kwargs):
		num_examples = len(data1)
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
				batch_data1, batch_data2, batch_data3 = data1[batch_indices], data2[batch_indices], data3[batch_indices]
				if batch_data1.size > 0 and batch_data2.size > 0 and batch_data3.size > 0:  # If batch_data1, batch_data2, and batch_data3 are non-empty.
					batch_data1, batch_data2, batch_data3 = augmenter(batch_data1, batch_data2, batch_data3)
					yield (batch_data1, batch_data2, batch_data3), batch_indices.size

			if end_idx >= num_examples:
				break
			start_idx = end_idx
