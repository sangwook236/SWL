import math
import tensorflow as tf

#%%------------------------------------------------------------------

class TensorFlowCnnModel:
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.model_output = None

	def __call__(self, input_tensor, is_training=True):
		self.model_output = self._create_model_1(input_tensor, self.num_classes, is_training)
		#self.model_output = self._create_model_2(input_tensor, self.num_classes, is_training)
		return self.model_output

	def train(self, train_data, train_labels, batch_size, num_epochs, shuffle, initial_epoch=0):
		pass
		#return history

	def load(self, model_filepath):
		pass

	def save(self, model_filepath):
		pass

	def _create_model_1(self, input_tensor, num_classes, is_training=True):
		with tf.variable_scope('tf_cnn_model', reuse=True):
			conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv1_1')
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool1_1')

			conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv2_1')
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool2_1')

			fc1 = tf.contrib.layers.flatten(conv2)

			fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc1_1')
			fc1 = tf.layers.dropout(fc1, rate=0.25, training=is_training, name='dropout1_1')

			if 2 == num_classes:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.sigmoid, name='fc2_1')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc2_1')
			else:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='fc2_1')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc2_1')

			return fc2

	def _create_model_2(self, input_tensor, num_classes, is_training=True):
		input_shape = input_tensor.get_shape()

		with tf.variable_scope('tf_cnn_model', reuse=True):
			conv1 = self._conv_layer(input_tensor, 28 * 28, 32, [5, 5], [1, 1, 1, 1], padding='SAME', layer_name='conv1_1', act=tf.nn.relu)
			conv1 = self._max_pool_layer(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', layer_name='maxpool1_1')

			conv2 = self._conv_layer(conv1, 32, 64, [3, 3], [1, 1, 1, 1], padding='SAME', layer_name='conv2_1', act=tf.nn.relu)
			conv2 = self._max_pool_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', layer_name='maxpool2_1')

			fc1 = tf.layers.flatten(conv2)

			fc1 = self._fc_layer(fc1, input_dim, 1024, layer_name, act=tf.nn.relu)
			fc1 = self._droupout_layer(conv1, 0.25, 'droupout1_1')

			if 2 == num_classes:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.sigmoid, name='fc2_1')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc2_1')
			else:
				fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='fc2_1')
				#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc2_1')

			return fc2

	# We can't initialize these variables to 0 - the network will get stuck.
	def _weight_variable(shape):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def _bias_variable(shape):
		"""Create a bias variable with appropriate initialization."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def _variable_summaries(var):
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

	def _conv_layer(input_tensor, input_dim, output_dim, kernel_size, strides, padding, layer_name, act=tf.nn.relu):
		with tf.name_scope(layer_name):
			# This variable will hold the state of the weights for the layer.
			with tf.name_scope('weights'):
				kernel = self._weight_variable(kernel_size + [input_dim, output_dim])
				self._variable_summaries(kernel)
			conv = tf.nn.conv2d(input_tensor, filter=kernel, strides=strides, padding=padding)
			#tf.summary.histogram('convolution', conv)
			with tf.name_scope('biases'):
				biases = self._bias_variable([output_dim])
				self._variable_summaries(biases)
			preactivations = tf.nn.bias_add(conv, biases)
			activations = act(preactivations, name='activation')
			tf.summary.histogram('activations', activations)
			"""
			kernel = tf.Variable(tf.truncated_normal(kernel_size + [input_dim, output_dim], dtype=tf.float32, stddev=1e-1, name='weights'))
			conv = tf.nn.conv2d(input=input_val, filter=kernel, strides=strides, padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), name='biases')
			preactivations = tf.nn.bias_add(conv, biases)
			activations = act(preactivations, name='activation')
			"""
			return activations

	def _fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		"""Add a name scope ensures logical grouping of the layers in the graph."""
		with tf.name_scope(layer_name):
			# This variable will hold the state of the weights for the layer.
			with tf.name_scope('weights'):
				weights = self._weight_variable([input_dim, output_dim])
				self._variable_summaries(weights)
			with tf.name_scope('biases'):
				biases = self._bias_variable([output_dim])
				self._variable_summaries(biases)
			with tf.name_scope('Wx_plus_b'):
				preactivation = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram('pre_activations', preactivation)
			activations = act(preactivation, name='activation')
			tf.summary.histogram('activations', activations)
			return activations

	def _max_pool_layer(input_tensor, ksize, strides, padding, layer_name):
		with tf.name_scope(layer_name):
			#tf.summary.scalar('max_pool_ksize', ksize)
			maxpool = tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)
			return maxpool

	def _droupout_layer(input_tensor, keep_prob, layer_name):
		with tf.name_scope(layer_name):
			#keep_prob = tf.placeholder(tf.float32)
			tf.summary.scalar('dropout_keep_probability', keep_prob)
			dropped = tf.nn.dropout(input_tensor, keep_prob)
			return dropped
