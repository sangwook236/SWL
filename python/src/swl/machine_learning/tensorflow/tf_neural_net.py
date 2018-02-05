import tensorflow as tf

#%%------------------------------------------------------------------

class TensorFlowNeuralNet(object):
	def __init__(self, input_shape, output_shape):
		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape)
		self._output_tensor_ph = tf.placeholder(tf.float32, shape=output_shape)
		self._is_training_tensor_ph = tf.placeholder(tf.bool)

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

	def get_feed_dict(self, data, labels=None, is_training=True):
		if labels is None:
			feed_dict = { self._input_tensor_ph: data, self._is_training_tensor_ph: is_training}
		else:
			feed_dict = { self._input_tensor_ph: data, self._output_tensor_ph: labels, self._is_training_tensor_ph: is_training}
		return feed_dict

	def _create_model(self, input_tensor, is_training_tensor, num_classes):
		raise NotImplementedError

	def _loss(self, y, t):
		raise NotImplementedError

	def _accuracy(self, y, t):
		raise NotImplementedError
