import tensorflow as tf

#%%------------------------------------------------------------------

class TensorFlowNeuralNet(object):
	def __init__(self, input_shape, output_shape):
		self._input_tensor_ph = tf.placeholder(tf.float32, shape=(None,) + input_shape)
		self._output_tensor_ph = tf.placeholder(tf.float32, shape=(None,) + output_shape)
		self._is_training_tensor_ph = tf.placeholder(tf.bool)

		# model_output is used in training, evaluation, & prediction steps.
		num_classes = output_shape[-1]
		self._model_output = self._create_model(self._input_tensor_ph, self._is_training_tensor_ph, num_classes)

		# Loss & accuracy are used in training & evaluation steps.
		self._loss, self._accuracy = self._get_metrics(self._model_output, self._output_tensor_ph)

	@property
	def model_output(self):
		return self._model_output

	@property
	def loss(self):
		return self._loss

	@property
	def accuracy(self):
		return self._accuracy

	def fill_feed_dict(self, data, labels=None, is_training=True):
		if labels is None:
			feed_dict = { self._input_tensor_ph: data, self._is_training_tensor_ph: is_training}
		else:
			feed_dict = { self._input_tensor_ph: data, self._output_tensor_ph: labels, self._is_training_tensor_ph: is_training}
		return feed_dict

	def _create_model(self, input_tensor, is_training_tensor, num_classes):
		raise NotImplementedError

	def _loss(self, y, t):
		#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
		#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
		#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
		return cross_entropy

	def _accuracy(self, y, t):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def _get_metrics(self, y_pred, y_true):
		with tf.name_scope('loss'):
			loss = self._loss(y_pred, y_true)
			tf.summary.scalar('loss', loss)

		with tf.name_scope('accuracy'):
			accuracy = self._accuracy(y_pred, y_true)
			tf.summary.scalar('accuracy', accuracy)

		return loss, accuracy
