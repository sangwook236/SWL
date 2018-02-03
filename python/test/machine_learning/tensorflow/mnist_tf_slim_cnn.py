import tensorflow.contrib.slim as slim
import tensorflow as tf
from simple_neural_net import SimpleNeuralNet

#%%------------------------------------------------------------------

class MnistTfSlimCNN(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		# REF [site] >> https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

		#keep_prob = 0.25 if True == is_training_tensor else 1.0  # Error: Not working.
		keep_prob = tf.cond(tf.equal(is_training_tensor, tf.constant(True)), lambda: tf.constant(0.25), lambda: tf.constant(1.0))

		num_classes = output_shape[-1]
		with tf.variable_scope('mnist_tf_slim_cnn', reuse=tf.AUTO_REUSE):
			with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
				conv1 = slim.conv2d(input_tensor, num_outputs=32, kernel_size=[5, 5], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv')
				conv1 = slim.max_pool2d(conv1, kernel_size=[2, 2], stride=2, scope='maxpool')

			with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
				conv2 = slim.conv2d(conv1, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv')
				conv2 = slim.max_pool2d(conv2, kernel_size=[2, 2], stride=2, scope='maxpool')

			with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
				fc1 = slim.flatten(conv2, scope='flatten')

				fc1 = slim.fully_connected(fc1, num_outputs=1024, activation_fn=tf.nn.relu, scope='fc')
				# NOTE [info] >> If keep_prob=1.0, dropout layer is not created.
				fc1 = slim.dropout(fc1, keep_prob=keep_prob, scope='dropout')

			with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
				if 1 == num_classes:
					fc2 = slim.fully_connected(fc1, num_outputs=1, activation_fn=tf.sigmoid, scope='fc')
				elif num_classes >= 2:
					fc2 = slim.fully_connected(fc1, num_outputs=num_classes, activation_fn=tf.nn.softmax, scope='fc')
				else:
					assert num_classes > 0, 'Invalid number of classes.'

				return fc2
