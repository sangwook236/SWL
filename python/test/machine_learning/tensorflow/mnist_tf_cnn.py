import tensorflow as tf
from mnist_cnn import MnistCNN

#%%------------------------------------------------------------------

class MnistTensorFlowCNN(MnistCNN):
	def __init__(self, input_shape, output_shape, model_type):
		self._model_type = model_type
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, num_classes):
		with tf.variable_scope('mnist_tf_cnn', reuse=tf.AUTO_REUSE):
			if 0 == self._model_type:
				return self._create_model_1(input_tensor, is_training_tensor, num_classes)
			elif 1 == self._model_type:
				return self._create_model_2(input_tensor, is_training_tensor, num_classes)
			else:
				assert False, 'Invalid model type.'
				return None

	def _create_model_1(self, input_tensor, is_training_tensor, num_classes):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv')
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv')
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(conv2, name='flatten')

			fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc')
			# NOTE [info] >> If dropout rate=0.0, droput layer is not created.
			fc1 = tf.layers.dropout(fc1, rate=0.75, training=is_training_tensor, name='dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, name='fc')
				#fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='fc')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc2

	def _create_model_2(self, input_tensor, is_training_tensor, num_classes):
		# FIXME [fix] >> Too slow and too low accuracy.

		#keep_prob = 0.25 if True == is_training_tensor else 1.0  # Error: Not working.
		keep_prob = tf.cond(tf.equal(is_training_tensor, tf.constant(True)), lambda: tf.constant(0.25), lambda: tf.constant(1.0))

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = self._conv_layer(input_tensor, 32, (5, 5, 1), (1, 1, 1, 1), padding='SAME', layer_name='conv', act=tf.nn.relu)
			conv1 = self._max_pool_layer(conv1, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID', layer_name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = self._conv_layer(conv1, 64, (3, 3, 32), (1, 1, 1, 1), padding='SAME', layer_name='conv', act=tf.nn.relu)
			conv2 = self._max_pool_layer(conv2, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID', layer_name='maxpool')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = self._flatten_layer(conv2, 7 * 7 * 64, layer_name='flatten')

			fc1 = self._fc_layer(fc1, 7 * 7 * 64, 1024, layer_name='fc', act=tf.nn.relu)
			# NOTE [info] >> If keep_prob=1.0, droput layer is not created.
			fc1 = self._dropout_layer(fc1, keep_prob, 'dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc2 = self._fc_layer(fc1, 1024, 1, layer_name='fc', act=tf.sigmoid)
			elif num_classes >= 2:
				fc2 = self._fc_layer(fc1, 1024, num_classes, layer_name='fc', act=tf.nn.softmax)
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc2

	# We can't initialize these variables to 0 - the network will get stuck.
	def _weight_variable(self, shape, name):
		"""Create a weight variable with appropriate initialization."""
		#initial = tf.truncated_normal(shape, stddev=0.1)
		#return tf.Variable(initial, name=name)
		return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

	def _bias_variable(self, shape, name):
		"""Create a bias variable with appropriate initialization."""
		#initial = tf.constant(0.1, shape=shape)
		#return tf.Variable(initial, name=name)
		return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

	def _variable_summaries(self, var, is_filter=False):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)
			if True == is_filter:
				tf.summary.image('filter', var)  # To visualize filters.

	def _conv_layer(self, input_tensor, output_dim, kernel_shape, strides, padding, layer_name, act=tf.nn.relu):
		with tf.name_scope(layer_name):
			# This variable will hold the state of the weights for the layer.
			"""
			kernel = tf.Variable(tf.truncated_normal(kernel_shape + (output_dim,), dtype=tf.float32, stddev=1e-1, name='weights'))
			conv = tf.nn.conv2d(input=input_tensor, filter=kernel, strides=strides, padding=padding)
			biases = tf.Variable(tf.constant(0.0, shape=(output_dim,), dtype=tf.float32), name='biases')
			preactivations = tf.nn.bias_add(conv, biases)
			activations = act(preactivations, name='activation')
			"""
			with tf.name_scope('weights'):
				kernel = self._weight_variable(kernel_shape + (output_dim,))
				self._variable_summaries(kernel, True)
			conv = tf.nn.conv2d(input_tensor, filter=kernel, strides=strides, padding=padding)
			#tf.summary.histogram('convolution', conv)
			with tf.name_scope('biases'):
				biases = self._bias_variable((output_dim,))
				self._variable_summaries(biases)
			with tf.name_scope('conv_x_plus_b'):
				preactivation = tf.nn.bias_add(conv, biases)
				tf.summary.histogram('preactivations', preactivation)
			activation = act(preactivation, name='activations')
			tf.summary.histogram('activations', activation)
			return activation

	def _fc_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		"""Add a name scope ensures logical grouping of the layers in the graph."""
		with tf.name_scope(layer_name):
			# This variable will hold the state of the weights for the layer.
			with tf.name_scope('weights'):
				weights = self._weight_variable((input_dim, output_dim))
				self._variable_summaries(weights)
			with tf.name_scope('biases'):
				biases = self._bias_variable((output_dim,))
				self._variable_summaries(biases)
			with tf.name_scope('W_x_plus_b'):
				preactivation = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram('preactivations', preactivation)
			activation = act(preactivation, name='activations')
			tf.summary.histogram('activations', activation)
			return activation

	def _max_pool_layer(self, input_tensor, ksize, strides, padding, layer_name):
		with tf.name_scope(layer_name):
			#tf.summary.scalar('max_pool_ksize', ksize)
			maxpool = tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)
			return maxpool

	def _dropout_layer(self, input_tensor, keep_prob, layer_name):
		with tf.name_scope(layer_name):
			tf.summary.scalar('dropout_keep_probability', keep_prob)
			dropped = tf.nn.dropout(input_tensor, keep_prob)
			return dropped

	def _flatten_layer(self, input_tensor, output_dim, layer_name):
		with tf.name_scope(layer_name):
			flatten = tf.reshape(input_tensor, (-1, output_dim))
			return flatten
