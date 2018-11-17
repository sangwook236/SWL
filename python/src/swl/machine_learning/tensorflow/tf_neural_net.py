import tensorflow as tf

#%%------------------------------------------------------------------

class TensorFlowNeuralNet(object):
	def __init__(self, input_shape, output_shape, is_sparse_output=False):
		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor_ph')
		if is_sparse_output:
			self._output_tensor_ph = tf.sparse_placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')
			#self._output_tensor_ph = tf.sparse_placeholder(tf.float32, name='output_tensor_ph')
		else:
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

	def create_training_model(self):
		raise NotImplementedError

	def create_evaluation_model(self):
		raise NotImplementedError

	def create_inference_model(self):
		raise NotImplementedError

	def _get_loss(self, y, t):
		raise NotImplementedError

	def _get_accuracy(self, y, t):
		raise NotImplementedError

#%%------------------------------------------------------------------

class TensorFlowBasicSeq2SeqNeuralNet(object):
	def __init__(self, input_shape, output_shape, is_sparse_output=False):
		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor_ph')
		if is_sparse_output:
			self._output_tensor_ph = tf.sparse_placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')
		else:
			self._output_tensor_ph = tf.placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')
		#self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and prediction steps.
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

	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		if inputs is None or outputs is None:
			feed_dict = {self._input_tensor_ph: inputs}
		else:
			feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs}
		return feed_dict

	def create_training_model(self):
		raise NotImplementedError

	def create_evaluation_model(self):
		raise NotImplementedError

	def create_inference_model(self):
		raise NotImplementedError

	def _get_loss(self, y, t):
		raise NotImplementedError

	def _get_accuracy(self, y, t):
		raise NotImplementedError

#%%------------------------------------------------------------------

class TensorFlowSeq2SeqNeuralNet(object):
	def __init__(self, encoder_input_shape, decoder_input_shape, decoder_output_shape, is_sparse_output=False):
		self._encoder_input_tensor_ph = tf.placeholder(tf.float32, shape=encoder_input_shape, name='encoder_input_tensor_ph')
		self._decoder_input_tensor_ph = tf.placeholder(tf.float32, shape=decoder_input_shape, name='decoder_input_tensor_ph')
		if is_sparse_output:
			self._decoder_output_tensor_ph = tf.sparse_placeholder(tf.float32, shape=decoder_output_shape, name='decoder_output_tensor_ph')
		else:
			self._decoder_output_tensor_ph = tf.placeholder(tf.float32, shape=decoder_output_shape, name='decoder_output_tensor_ph')
		#self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and prediction steps.
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

	def get_feed_dict(self, encoder_inputs, decoder_inputs=None, decoder_outputs=None, **kwargs):
		if decoder_inputs is None or decoder_outputs is None:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs}
		else:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs, self._decoder_input_tensor_ph: decoder_inputs, self._decoder_output_tensor_ph: decoder_outputs}
		return feed_dict

	def create_training_model(self):
		raise NotImplementedError

	def create_evaluation_model(self):
		raise NotImplementedError

	def create_inference_model(self):
		raise NotImplementedError

	def _get_loss(self, y, t):
		raise NotImplementedError

	def _get_accuracy(self, y, t):
		raise NotImplementedError
