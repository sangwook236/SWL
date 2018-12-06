import abc

# Abstract batch loader.
class BatchLoader(abc.ABC):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def getBatches(self, num_epoches):
		raise NotImplementedError

#%%------------------------------------------------------------------

# Load batches from directories.
class DirectoryBatchLoader(BatchLoader):
	def __init__(self):
		super().__init__()

	def getBatches(self, num_epoches):
		raise NotImplementedError
