import tflearn
import tensorflow as tf
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet

#--------------------------------------------------------------------

class MnistCnnUsingTfLearn(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

		#tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		# REF [site] >> http://tflearn.org/getting_started/

		keep_prob = 0.25 if is_training else 1.0

		num_classes = output_shape[-1]
		with tf.variable_scope('mnist_cnn_using_tflearn', reuse=tf.AUTO_REUSE):
			with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
				#net = tflearn.input_data(shape=input_shape)

				net = tflearn.conv_2d(input_tensor, nb_filter=32, filter_size=5, strides=1, padding='same', activation='relu', name='conv')
				net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, name='maxpool')

			with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
				net = tflearn.conv_2d(net, nb_filter=64, filter_size=3, strides=1, padding='same', activation='relu', name='conv')
				net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, name='maxpool')

			with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
				net = tflearn.flatten(net, name='flatten')

				net = tflearn.fully_connected(net, n_units=1024, activation='relu', name='fc')
				# NOTE [info] >> If keep_prob=1.0, dropout layer is not created.
				net = tflearn.dropout(net, keep_prob=keep_prob, name='dropout')

			with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
				if 1 == num_classes:
					net = tflearn.fully_connected(net, n_units=1, activation='sigmoid', name='fc')
				elif num_classes >= 2:
					net = tflearn.fully_connected(net, n_units=num_classes, activation='softmax', name='fc')
				else:
					assert num_classes > 0, 'Invalid number of classes.'

				return net
