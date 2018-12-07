import abc

# Abstract batch loader.
class BatchManager(abc.ABC):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def getBatches(self, *args, **kwargs):
		raise NotImplementedError

#%%------------------------------------------------------------------

# Load, save, and augment batches using files.
class FileBatchManager(BatchManager):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def putBatches(self, *args, **kwargs):
		raise NotImplementedError
