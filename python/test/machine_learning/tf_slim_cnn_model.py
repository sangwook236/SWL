import tensorflow.contrib.slim as slim
import tensorflow as tf
from dnn_model import DnnBaseModel

#%%------------------------------------------------------------------

class TfSlimCnnModel(DnnBaseModel):
	def __init__(self, num_classes):
		super(TfSlimCnnModel, self).__init__(num_classes)

	def __call__(self, input_tensor, is_training=True):
		self.model_output_ = self._create_model(input_tensor, self.num_classes_, is_training)
		return self.model_output_

	def _create_model(self, input_tensor, num_classes, is_training=True):
		# REF [site] >> https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
		keep_prob = 0.25 if is_training is True else 1.0

		with tf.variable_scope('tf_slim_cnn_model', reuse=None):
			conv1 = slim.conv2d(input_tensor, num_outputs=32, kernel_size=[5, 5], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv1_1')
			conv1 = slim.max_pool2d(conv1, kernel_size=[2, 2], stride=2, scope='maxpool1_1')

			conv2 = slim.conv2d(conv1, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv2_1')
			conv2 = slim.max_pool2d(conv2, kernel_size=[2, 2], stride=2, scope='maxpool2_1')

			fc1 = slim.flatten(conv2, scope='flatten1_1')

			fc1 = slim.fully_connected(fc1, num_outputs=1024, activation_fn=tf.nn.relu, scope='fc1_1')
			fc1 = slim.dropout(fc1, keep_prob=keep_prob, scope='dropout1_1')

			if 2 == num_classes:
				fc2 = slim.fully_connected(fc1, num_outputs=num_classes, activation_fn=tf.sigmoid, scope='fc2_1')
			else:
				fc2 = slim.fully_connected(fc1, num_outputs=num_classes, activation_fn=tf.nn.softmax, scope='fc2_1')

			return fc2
