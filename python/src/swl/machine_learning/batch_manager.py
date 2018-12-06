import abc

# Abstract batch loader.
class BatchManager(abc.ABC):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def getBatches(self, num_epoches, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------

# Load, save, and augment batches using files.
class FileBatchManager(BatchManager):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def getBatches(self, dir_path, filename_pairs, num_epoches, *args, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def putBatches(self, dir_path, filename_pairs, shuffle, *args, **kwargs):
		raise NotImplementedError
