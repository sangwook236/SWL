import abc
import tensorflow as tf
from swl.machine_learning.learning_model import LearningModel

#%%------------------------------------------------------------------

class TensorFlowModel(LearningModel):
	def __init__(self):
		super().__init__()

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = None
		self._accuracy = None

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

	@abc.abstractmethod
	def get_feed_dict(self, data, *args, **kwargs):
		raise NotImplementedError

	def _get_loss(self, y, t):
		with tf.name_scope('loss'):
			"""
			if 1 == num_classes:
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
			elif num_classes >= 2:
				#loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
				#loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
			else:
				assert num_classes > 0, 'Invalid number of classes.'
			"""
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t):
		with tf.name_scope('accuracy'):
			"""
			if 1 == num_classes:
				correct_prediction = tf.equal(tf.round(y), tf.round(t))
			elif num_classes >= 2:
				correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
			else:
				assert num_classes > 0, 'Invalid number of classes.'
			"""
			correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			tf.summary.scalar('accuracy', accuracy)
			return accuracy

#%%------------------------------------------------------------------

class SimpleTensorFlowModel(TensorFlowModel):
	def __init__(self, input_shape, output_shape):
		super().__init__()

		self._input_shape = input_shape
		self._output_shape = output_shape

		self._input_tensor_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor_ph')
		self._output_tensor_ph = tf.placeholder(tf.int32, shape=output_shape, name='output_tensor_ph')
		#self._output_tensor_ph = tf.placeholder(tf.float32, shape=output_shape, name='output_tensor_ph')

	def create_training_model(self):
		self._model_output = self._create_single_model(self._input_tensor_ph, self._input_shape, self._output_shape, True)

		self._loss = self._get_loss(self._model_output, self._output_tensor_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_tensor_ph)

	def create_evaluation_model(self):
		self._model_output = self._create_single_model(self._input_tensor_ph, self._input_shape, self._output_shape, False)

		self._loss = self._get_loss(self._model_output, self._output_tensor_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_tensor_ph)

	def create_inference_model(self):
		self._model_output = self._create_single_model(self._input_tensor_ph, self._input_shape, self._output_shape, False)

		self._loss = None
		self._accuracy = None

	@abc.abstractmethod
	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		raise NotImplementedError

#%%------------------------------------------------------------------

class SimpleTwoInputTensorFlowModel(TensorFlowModel):
	def __init__(self, encoder_input_shape, decoder_input_shape, decoder_output_shape):
		super().__init__()

		self._encoder_input_shape = encoder_input_shape
		self._decoder_input_shape = decoder_input_shape
		self._decoder_output_shape = decoder_output_shape

		self._encoder_input_tensor_ph = tf.placeholder(tf.float32, shape=encoder_input_shape, name='encoder_input_tensor_ph')
		self._decoder_input_tensor_ph = tf.placeholder(tf.float32, shape=decoder_input_shape, name='decoder_input_tensor_ph')
		self._decoder_output_tensor_ph = tf.placeholder(tf.int32, shape=decoder_output_shape, name='decoder_output_tensor_ph')
		#self._decoder_output_tensor_ph = tf.placeholder(tf.float32, shape=decoder_output_shape, name='decoder_output_tensor_ph')

	def create_training_model(self):
		self._model_output = self._create_single_model(self._encoder_input_tensor_ph, self._decoder_input_tensor_ph, self._encoder_input_shape, self._decoder_input_shape, self._decoder_output_shape, True)

		self._loss = self._get_loss(self._model_output, self._decoder_output_tensor_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._decoder_output_tensor_ph)

	def create_evaluation_model(self):
		self._model_output = self._create_single_model(self._encoder_input_tensor_ph, self._decoder_input_tensor_ph, self._encoder_input_shape, self._decoder_input_shape, self._decoder_output_shape, False)

		self._loss = self._get_loss(self._model_output, self._decoder_output_tensor_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._decoder_output_tensor_ph)

	def create_inference_model(self):
		self._model_output = self._create_single_model(self._encoder_input_tensor_ph, self._decoder_input_tensor_ph, self._encoder_input_shape, self._decoder_input_shape, self._decoder_output_shape, False)

		self._loss = None
		self._accuracy = None

	@abc.abstractmethod
	def _create_single_model(self, encoder_input_tensor, decoder_input_tensor, encoder_input_shape, decoder_input_shape, decoder_output_shape, is_training):
		raise NotImplementedError
