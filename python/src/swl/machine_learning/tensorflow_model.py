import abc
import tensorflow as tf
from swl.machine_learning.learning_model import LearningModel

#%%------------------------------------------------------------------

class TensorFlowModel(LearningModel):
	"""Learning model for TensorFlow library.
	"""

	def __init__(self):
		super().__init__()

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = None
		self._accuracy = None

	@property
	def loss(self):
		if self._loss is None:
			raise ValueError('Loss is None')
		return self._loss

	@property
	def accuracy(self):
		if self._accuracy is None:
			raise ValueError('Accuracy is None')
		return self._accuracy

	@abc.abstractmethod
	def get_feed_dict(self, data, num_data, *args, **kwargs):
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
	"""Single-input single-output learning model for TensorFlow library.

	- Fixed-length inputs and outputs.
	- Dense tensors for input and output.
	"""

	def __init__(self, input_shape, output_shape):
		super().__init__()

		self._input_shape = input_shape
		self._output_shape = output_shape

		self._input_ph = tf.placeholder(tf.float32, shape=self._input_shape, name='input_ph')
		self._output_ph = tf.placeholder(tf.int32, shape=self._output_shape, name='output_ph')
		#self._output_ph = tf.placeholder(tf.float32, shape=self._output_shape, name='output_ph')

	@abc.abstractmethod
	def _create_single_model(self, inputs, input_shape, output_shape, is_training):
		raise NotImplementedError

	def create_training_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._input_shape, self._output_shape, True)

		self._loss = self._get_loss(self._model_output, self._output_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_ph)

	def create_evaluation_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._input_shape, self._output_shape, False)

		self._loss = self._get_loss(self._model_output, self._output_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_ph)

	def create_inference_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._input_shape, self._output_shape, False)

		self._loss = None
		self._accuracy = None

#%%------------------------------------------------------------------

class SimpleAuxiliaryInputTensorFlowModel(TensorFlowModel):
	"""Single-input single-output learning model for TensorFlow library.

	- Auxiliary inputs for training.
	- Fixed- or variable-length inputs and outputs.
	- Dense tensors for input and output.
	"""

	def __init__(self, input_shape, aux_input_shape, output_shape):
		super().__init__()

		self._input_shape = input_shape
		self._aux_input_shape = aux_input_shape
		self._output_shape = output_shape

		self._input_ph = tf.placeholder(tf.float32, shape=self._input_shape, name='input_ph')
		self._aux_input_ph = tf.placeholder(tf.float32, shape=self._aux_input_shape, name='aux_input_ph')
		self._output_ph = tf.placeholder(tf.int32, shape=self._output_shape, name='output_ph')
		#self._output_ph = tf.placeholder(tf.float32, shape=self._output_shape, name='output_ph')

	@abc.abstractmethod
	def _create_single_model(self, inputs, aux_inputs, input_shape, aux_input_shape, output_shape, is_training):
		raise NotImplementedError

	def create_training_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._aux_input_ph, self._input_shape, self._aux_input_shape, self._output_shape, True)

		self._loss = self._get_loss(self._model_output, self._output_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_ph)

	def create_evaluation_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._aux_input_ph, self._input_shape, self._aux_input_shape, self._output_shape, False)

		self._loss = self._get_loss(self._model_output, self._output_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_ph)

	def create_inference_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._aux_input_ph, self._input_shape, self._aux_input_shape, self._output_shape, False)

		self._loss = None
		self._accuracy = None

#%%------------------------------------------------------------------

class SimpleSequentialTensorFlowModel(TensorFlowModel):
	"""Single-input single-output learning model for TensorFlow library.

	- Fixed- or variable-length inputs and outputs.
	- Dense tensors for input.
	- Dense or sparse tensors for output.
	"""

	def __init__(self, input_shape, output_shape, num_classes, is_sparse_output=False, is_time_major=False):
		super().__init__()

		self._input_shape = input_shape
		self._output_shape = output_shape
		self._num_classes = num_classes
		self._is_sparse_output = is_sparse_output
		self._is_time_major = is_time_major

		self._input_ph = tf.placeholder(tf.float32, shape=self._input_shape, name='input_ph')
		if self._is_sparse_output:
			self._output_ph = tf.sparse_placeholder(tf.int32, shape=self._output_shape, name='output_ph')
		else:
			self._output_ph = tf.placeholder(tf.int32, shape=self._output_shape, name='output_ph')
		self._output_len_ph = tf.placeholder(tf.int32, [None], name='output_len_ph')
		self._model_output_len_ph = tf.placeholder(tf.int32, [None], name='model_output_len_ph')

	@abc.abstractmethod
	def _get_loss(self, y, t, y_len, t_len):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_accuracy(self, y, t, y_len):
		raise NotImplementedError

	@abc.abstractmethod
	def _create_single_model(self, inputs, input_shape, num_classes, is_training):
		raise NotImplementedError

	def create_training_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._input_shape, self._num_classes, True)

		self._loss = self._get_loss(self._model_output, self._output_ph, self._model_output_len_ph, self._output_len_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_ph, self._model_output_len_ph)

	def create_evaluation_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._input_shape, self._num_classes, False)

		self._loss = self._get_loss(self._model_output, self._output_ph, self._model_output_len_ph, self._output_len_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_ph, self._model_output_len_ph)

	def create_inference_model(self):
		self._model_output = self._create_single_model(self._input_ph, self._input_shape, self._num_classes, False)

		self._loss = None
		self._accuracy = None
