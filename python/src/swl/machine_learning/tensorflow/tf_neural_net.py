import abc
import tensorflow as tf

#%%------------------------------------------------------------------

class TensorFlowNeuralNet(abc.ABC):
	def __init__(self, input_shape, output_shape):
		super().__init__()

		self._input_shape = input_shape
		self._output_shape = output_shape

		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor_ph')
		#self._output_tensor_ph = tf.placeholder(tf.int32, shape=output_shape, name='output_tensor_ph')
		self._output_tensor_ph = tf.placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')
		#self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and inference steps.
		self._model_output = None

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = None
		self._accuracy = None

	@property
	def model_output(self):
		if self._model_output is None:
			raise TypeError
		return self._model_output

	@property
	def loss(self):
		if self._loss is None:
			raise TypeError
		return self._loss

	@property
	def accuracy(self):
		if self._accuracy is None:
			raise TypeError
		return self._accuracy

	def get_feed_dict(self, data, labels=None, **kwargs):
		if labels is None:
			feed_dict = {self._input_tensor_ph: data}
		else:
			feed_dict = {self._input_tensor_ph: data, self._output_tensor_ph: labels}
		return feed_dict

	@abc.abstractmethod
	def create_training_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_evaluation_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_inference_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_loss(self, y, t):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_accuracy(self, y, t):
		raise NotImplementedError

#%%------------------------------------------------------------------

class TensorFlowBasicSeq2SeqNeuralNet(abc.ABC):
	def __init__(self, input_shape, output_shape):
		self._input_shape = input_shape
		self._output_shape = output_shape

		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor_ph')
		#self._output_tensor_ph = tf.placeholder(tf.int32, shape=output_shape, name='output_tensor_ph')
		self._output_tensor_ph = tf.placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')
		#self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and prediction steps.
		self._model_output = None

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = None
		self._accuracy = None

	@property
	def input_shape(self):
		if self._input_shape is None:
			raise TypeError
		return self._input_shape

	@property
	def output_shape(self):
		if self._output_shape is None:
			raise TypeError
		return self._output_shape

	@property
	def model_output(self):
		if self._model_output is None:
			raise TypeError
		return self._model_output

	@property
	def loss(self):
		if self._loss is None:
			raise TypeError
		return self._loss

	@property
	def accuracy(self):
		if self._accuracy is None:
			raise TypeError
		return self._accuracy

	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		if outputs is None:
			feed_dict = {self._input_tensor_ph: inputs}
		else:
			feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs}
		return feed_dict

	@abc.abstractmethod
	def create_training_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_evaluation_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_inference_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_loss(self, y, t):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_accuracy(self, y, t):
		raise NotImplementedError

#%%------------------------------------------------------------------

class TensorFlowSeq2SeqNeuralNet(abc.ABC):
	def __init__(self, encoder_input_shape, decoder_input_shape, decoder_output_shape):
		self._encoder_input_shape = encoder_input_shape
		self._decoder_input_shape = decoder_input_shape
		self._decoder_output_shape = decoder_output_shape

		self._encoder_input_tensor_ph = tf.placeholder(tf.float32, shape=encoder_input_shape, name='encoder_input_tensor_ph')
		self._decoder_input_tensor_ph = tf.placeholder(tf.float32, shape=decoder_input_shape, name='decoder_input_tensor_ph')
		#self._decoder_output_tensor_ph = tf.placeholder(tf.int32, shape=decoder_output_shape, name='decoder_output_tensor_ph')
		self._decoder_output_tensor_ph = tf.placeholder(tf.float32, shape=decoder_output_shape, name='decoder_output_tensor_ph')
		#self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and prediction steps.
		self._model_output = None

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = None
		self._accuracy = None

	@property
	def encoder_input_shape(self):
		if self._encoder_input_shape is None:
			raise TypeError
		return self._encoder_input_shape

	@property
	def decoder_input_shape(self):
		if self._decoder_input_shape is None:
			raise TypeError
		return self._decoder_input_shape

	@property
	def decoder_output_shape(self):
		if self._decoder_output_shape is None:
			raise TypeError
		return self._decoder_output_shape

	@property
	def model_output(self):
		if self._model_output is None:
			raise TypeError
		return self._model_output

	@property
	def loss(self):
		if self._loss is None:
			raise TypeError
		return self._loss

	@property
	def accuracy(self):
		if self._accuracy is None:
			raise TypeError
		return self._accuracy

	def get_feed_dict(self, encoder_inputs, decoder_inputs=None, decoder_outputs=None, **kwargs):
		if decoder_inputs is None or decoder_outputs is None:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs}
		else:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs, self._decoder_input_tensor_ph: decoder_inputs, self._decoder_output_tensor_ph: decoder_outputs}
		return feed_dict

	@abc.abstractmethod
	def create_training_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_evaluation_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def create_inference_model(self):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_loss(self, y, t):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_accuracy(self, y, t):
		raise NotImplementedError
