import tflearn
import tensorflow as tf
from mnist_cnn import MnistCNN

#%%------------------------------------------------------------------

class MnistTfLearnCNN(MnistCNN):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

		#tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

	def _create_model(self, input_tensor, is_training_tensor, num_classes):
		# REF [site] >> http://tflearn.org/getting_started/

		#keep_prob = 0.25 if True == is_training_tensor else 1.0  # Error: Not working.
		keep_prob = tf.cond(tf.equal(is_training_tensor, tf.constant(True)), lambda: tf.constant(0.25), lambda: tf.constant(1.0))

		with tf.variable_scope('mnist_tflearn_cnn', reuse=tf.AUTO_REUSE):
			with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
				#net = tflearn.input_data(shape=input_shape)

				net = tflearn.conv_2d(input_tensor, nb_filter=32, filter_size=5, strides=1, padding='same', activation='relu', name='conv')
				net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, name='maxpool')

			with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
				net = tflearn.conv_2d(net, nb_filter=64, filter_size=3, strides=1, padding='same', activation='relu', name='conv')
				net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, name='maxpool')

				net = tflearn.flatten(net, name='flatten')

			with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
				net = tflearn.fully_connected(net, n_units=1024, activation='relu', name='fc')
				# NOTE [info] >> If keep_prob=1.0, droput layer is not created.
				net = tflearn.dropout(net, keep_prob=keep_prob, name='dropout')

			with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
				if 2 == num_classes:
					net = tflearn.fully_connected(net, n_units=1, activation='sigmoid', name='fc')
				else:
					net = tflearn.fully_connected(net, n_units=num_classes, activation='softmax', name='fc')

				return net
