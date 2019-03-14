import tensorflow as tf
from swl.machine_learning.tensorflow_model import SimpleTensorFlowModel

#%%------------------------------------------------------------------

class MnistCnn(SimpleTensorFlowModel):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def get_feed_dict(self, data, *args, **kwargs):
		len_data = len(data)
		if 1 == len_data:
			feed_dict = {self._input_tensor_ph: data[0]}
		elif 2 == len_data:
			feed_dict = {self._input_tensor_ph: data[0], self._output_tensor_ph: data[1]}
		else:
			raise ValueError('Invalid number of feed data: {}'.format(len_data))
		return feed_dict

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		num_classes = output_shape[-1]
		with tf.variable_scope('mnist_cnn', reuse=tf.AUTO_REUSE):
			return self._create_model(input_tensor, is_training, num_classes)

	def _create_model(self, input_tensor, is_training, num_classes):
		dropout_prob = 0.25

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv')
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv')
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(conv2, name='flatten')

			fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='dense')
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			fc1 = tf.layers.dropout(fc1, rate=dropout_prob, training=is_training, name='dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, name='dense')
				#fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			elif num_classes >= 2:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='dense')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc2
