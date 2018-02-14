import tensorflow as tf

#%%------------------------------------------------------------------

class TensorFlowNeuralNet(object):
	def __init__(self, input_shape, output_shape):
		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor_ph')
		self._output_tensor_ph = tf.placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')
		self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and prediction steps.
		self._model_output = self._create_model(self._input_tensor_ph, self._output_tensor_ph, self._is_training_tensor_ph, input_shape, output_shape)

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = self._loss(self._model_output, self._output_tensor_ph)
		self._accuracy = self._accuracy(self._model_output, self._output_tensor_ph)

	@property
	def model_output(self):
		return self._model_output

	@property
	def loss(self):
		return self._loss

	@property
	def accuracy(self):
		return self._accuracy

	def get_feed_dict(self, data, labels=None, is_training=True, **kwargs):
		if labels is None:
			feed_dict = {self._input_tensor_ph: data, self._is_training_tensor_ph: is_training}
		else:
			feed_dict = {self._input_tensor_ph: data, self._output_tensor_ph: labels, self._is_training_tensor_ph: is_training}
		return feed_dict

	def _create_model(self, input_tensor, output_tensor, is_training_tensor, input_shape, output_shape):
		raise NotImplementedError

	def _loss(self, y, t):
		raise NotImplementedError

	def _accuracy(self, y, t):
		raise NotImplementedError

#%%------------------------------------------------------------------

class TensorFlowSeq2SeqNeuralNet(object):
	def __init__(self, encoder_input_shape, decoder_input_shape, decoder_output_shape):
		self._encoder_input_tensor_ph = tf.placeholder(tf.float32, shape=encoder_input_shape, name='encoder_input_tensor_ph')
		self._decoder_input_tensor_ph = tf.placeholder(tf.float32, shape=decoder_input_shape, name='decoder_input_tensor_ph')
		self._decoder_output_tensor_ph = tf.placeholder(tf.float32, shape=decoder_output_shape, name='decoder_output_tensor_ph')
		self._is_training_tensor_ph = tf.placeholder(tf.bool, name='is_training_tensor_ph')

		# model_output is used in training, evaluation, and prediction steps.
		self._model_output = self._create_model(self._encoder_input_tensor_ph, self._decoder_input_tensor_ph, self._decoder_output_tensor_ph, self._is_training_tensor_ph, encoder_input_shape, decoder_input_shape, decoder_output_shape)

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = self._loss(self._model_output, self._decoder_output_tensor_ph)
		self._accuracy = self._accuracy(self._model_output, self._decoder_output_tensor_ph)

	@property
	def model_output(self):
		return self._model_output

	@property
	def loss(self):
		return self._loss

	@property
	def accuracy(self):
		return self._accuracy

	def get_feed_dict(self, encoder_inputs, decoder_inputs=None, decoder_outputs=None, is_training=True, **kwargs):
		if decoder_inputs is None or decoder_outputs is None:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs, self._is_training_tensor_ph: is_training}
		else:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs, self._decoder_input_tensor_ph: decoder_inputs, self._decoder_output_tensor_ph: decoder_outputs, self._is_training_tensor_ph: is_training}
		return feed_dict

	def _create_model(self, encoder_input_tensor, decoder_input_tensor, decoder_output_tensor, is_training_tensor, encoder_input_shape, decoder_input_shape, decoder_output_shape):
		raise NotImplementedError

	def _loss(self, y, t):
		raise NotImplementedError

	def _accuracy(self, y, t):
		raise NotImplementedError
