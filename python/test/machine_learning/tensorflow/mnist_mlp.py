import tensorflow as tf
import math
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet

#%%------------------------------------------------------------------

class MnistMLP(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape, model_type, max_neuron_count):
		super().__init__(input_shape, output_shape)

		self._model_type = model_type
		self._max_neuron_count = max_neuron_count

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		num_classes = output_shape[-1]
		with tf.variable_scope('mnist_mlp', reuse=tf.AUTO_REUSE):
			if 1 == self._model_type:  # A wide and shallow MLP with a single hidden layer.
				return self._create_wide_and_shallow_model(input_tensor, is_training, num_classes, self._max_neuron_count)
			elif 2 == self._model_type:  # A narrow and deep MLP with three hidden layers of triangular shape.
				return self._create_narrow_and_deep_model_with_triangular_shape(input_tensor, is_training, num_classes, self._max_neuron_count)
			elif 3 == self._model_type:  # A narrow and deep MLP with three hidden layers of diamond shape.
				return self._create_narrow_and_deep_model_with_diamond_shape(input_tensor, is_training, num_classes, self._max_neuron_count)
			elif 4 == self._model_type:  # A narrow and deep MLP with three hidden layers of hourglass shape.
				return self._create_narrow_and_deep_model_with_hourglass_shape(input_tensor, is_training, num_classes, self._max_neuron_count)
			else:
				assert False, 'Invalid model type.'
				return None

	def _create_wide_and_shallow_model(self, input_tensor, is_training, num_classes, max_neuron_count):
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(input_tensor, name='flatten')

			fc1 = tf.layers.dense(fc1, max_neuron_count, activation=tf.nn.relu, name='fc')
			#fc1 = tf.layers.dropout(fc1, rate=0.25, training=is_training, name='dropout')

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

	def _create_narrow_and_deep_model_with_triangular_shape(self, input_tensor, is_training, num_classes, max_neuron_count):
		num_neurons = math.ceil((max_neuron_count / 8.0) ** (1.0 / 3.0))

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(input_tensor, name='flatten')

			fc1 = tf.layers.dense(fc1, num_neurons, activation=tf.nn.relu, name='fc')
			#fc1 = tf.layers.dropout(fc1, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			fc2 = tf.layers.dense(fc1, num_neurons * 2, activation=tf.nn.relu, name='fc')
			#fc2 = tf.layers.dropout(fc2, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc3', reuse=tf.AUTO_REUSE):
			fc3 = tf.layers.dense(fc2, num_neurons * 4, activation=tf.nn.relu, name='fc')
			#fc3 = tf.layers.dropout(fc3, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc4', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc4 = tf.layers.dense(fc3, 1, activation=tf.sigmoid, name='fc')
				#fc4 = tf.layers.dense(fc3, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc4 = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax, name='fc')
				#fc4 = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc4

	def _create_narrow_and_deep_model_with_diamond_shape(self, input_tensor, is_training, num_classes, max_neuron_count):
		num_neurons = math.ceil((max_neuron_count / 2.0) ** (1.0 / 3.0))

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(input_tensor, name='flatten')

			fc1 = tf.layers.dense(fc1, num_neurons, activation=tf.nn.relu, name='fc')
			#fc1 = tf.layers.dropout(fc1, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			fc2 = tf.layers.dense(fc1, num_neurons * 2, activation=tf.nn.relu, name='fc')
			#fc2 = tf.layers.dropout(fc2, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc3', reuse=tf.AUTO_REUSE):
			fc3 = tf.layers.dense(fc2, num_neurons, activation=tf.nn.relu, name='fc')
			#fc3 = tf.layers.dropout(fc3, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc4', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc4 = tf.layers.dense(fc3, 1, activation=tf.sigmoid, name='fc')
				#fc4 = tf.layers.dense(fc3, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc4 = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax, name='fc')
				#fc4 = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc4

	def _create_narrow_and_deep_model_with_hourglass_shape(self, input_tensor, is_training, num_classes, max_neuron_count):
		num_neurons = math.ceil((max_neuron_count / 4.0) ** (1.0 / 3.0))

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			fc1 = tf.layers.flatten(input_tensor, name='flatten')

			fc1 = tf.layers.dense(fc1, num_neurons * 2, activation=tf.nn.relu, name='fc')
			#fc1 = tf.layers.dropout(fc1, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
			fc2 = tf.layers.dense(fc1, num_neurons, activation=tf.nn.relu, name='fc')
			#fc2 = tf.layers.dropout(fc2, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc3', reuse=tf.AUTO_REUSE):
			fc3 = tf.layers.dense(fc2, num_neurons * 2, activation=tf.nn.relu, name='fc')
			#fc3 = tf.layers.dropout(fc3, rate=0.25, training=is_training, name='dropout')

		with tf.variable_scope('fc4', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc4 = tf.layers.dense(fc3, 1, activation=tf.sigmoid, name='fc')
				#fc4 = tf.layers.dense(fc3, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc4 = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax, name='fc')
				#fc4 = tf.layers.dense(fc3, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc4
