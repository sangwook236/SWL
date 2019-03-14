import abc

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
