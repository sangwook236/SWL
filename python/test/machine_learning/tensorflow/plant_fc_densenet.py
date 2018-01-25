import densenet_fc as dc
import tensorflow as tf
from swl.machine_learning.tensorflow.tf_neural_net import TensorFlowNeuralNet

#%%------------------------------------------------------------------

class PlantFcDenseNet(TensorFlowNeuralNet):
	def __init__(self, input_shape, output_shape):
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, num_classes):
		input_shape = input_tensor.get_shape().as_list()

		with tf.name_scope('plant_fc_densenet'):
			fc_densenet_model = dc.DenseNetFCN(input_shape[1:], nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)

			# Display the model summary.
			#fc_densenet_model.summary()

			return fc_densenet_model(input_tensor)

	def _loss(self, y, t):
		with tf.name_scope('loss'):
			"""
			if num_classes <= 2:
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
			else:
				#loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
				#loss = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
			"""
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
			tf.summary.scalar('loss', loss)
			return loss

	def _accuracy(self, y, t):
		with tf.name_scope('accuracy'):
			"""
			if num_classes <= 2:
				correct_prediction = tf.equal(tf.round(y), tf.round(t))
			else:
				correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
			"""
			correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			tf.summary.scalar('accuracy', accuracy)
			return accuracy
