import tensorflow as tf
from swl.machine_learning.tensorflow.tf_neural_net import TensorFlowNeuralNet

#%%------------------------------------------------------------------

class SimpleNeuralNet(TensorFlowNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _loss(self, y, t):
		with tf.name_scope('loss'):
			"""
			if 1 == num_classes:
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
			elif num_classes >= 2:
				#loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
				#loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
			else:
				assert num_classes > 0, 'Invalid number of classes.'
			"""
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
			tf.summary.scalar('loss', loss)
			return loss

	def _accuracy(self, y, t):
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
