import abc

#--------------------------------------------------------------------

class LearningModel(abc.ABC):
	"""Toplevel class for learning model.
	"""

	def __init__(self):
		super().__init__()

		# model_output is used in training, evaluation, and inference steps.
		self._model_output = None

	@property
	def model_output(self):
		if self._model_output is None:
			raise TypeError
		return self._model_output

	@abc.abstractmethod
	def create_training_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_evaluation_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_inference_model(self):
		raise NotImplementedError
