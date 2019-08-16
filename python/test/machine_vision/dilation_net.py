import abc
import tensorflow as tf
from swl.machine_learning.tensorflow_model import SimpleTensorFlowModel

#--------------------------------------------------------------------

class DilationNet(SimpleTensorFlowModel):
	"""Dilation network.
	
	Refer to "Multi-Scale Context Aggregation by Dilated Convolutions", ICLR 2016.
	"""

	def __init__(self, input_shape, output_shape, is_final_relu_applied, model_name):
		super().__init__(input_shape, output_shape)

		self._is_final_relu_applied = is_final_relu_applied
		self._model_name = model_name

	def get_feed_dict(self, data, num_data, *args, **kwargs):
		len_data = len(data)
		if 1 == len_data:
			feed_dict = {self._input_ph: data[0]}
		elif 2 == len_data:
			feed_dict = {self._input_ph: data[0], self._output_ph: data[1]}
		else:
			raise ValueError('Invalid number of feed data: {}'.format(len_data))
		return feed_dict

	def _create_single_model(self, inputs, input_shape, output_shape, is_training):
		num_classes = output_shape[-1]
		with tf.variable_scope(self._model_name, reuse=tf.AUTO_REUSE):
			# Preprocessing.
			with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
				inputs = tf.nn.local_response_normalization(inputs, depth_radius=5, bias=1, alpha=1, beta=0.5, name='lrn')

			#--------------------
			with tf.variable_scope('front_end_module', reuse=tf.AUTO_REUSE):
				front_end_outputs = self._create_front_end_module(inputs, num_classes, self._is_final_relu_applied, is_training)

			#--------------------
			with tf.variable_scope('context_module', reuse=tf.AUTO_REUSE):
				context_outputs = self._create_context_module(front_end_outputs, num_classes, is_training)
				return tf.nn.softmax(context_outputs, axis=3, name='softmax')

	@abc.abstractmethod
	def _create_context_module(self, inputs, num_classes, is_training):
		raise NotImplementedError

	def _create_front_end_module(self, inputs, num_classes, is_final_relu_applied, is_training):
		"""A network model based on VGG-16.
		"""

		dropout_rate = 0.5

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.layers.conv2d(conv1, filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.layers.conv2d(conv2, filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool')

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv3 = tf.nn.relu(conv3, name='relu1')

			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv3 = tf.nn.relu(conv3, name='relu2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool')

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv4 = tf.nn.relu(conv4, name='relu1')

			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv4 = tf.nn.relu(conv4, name='relu2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool')

		with tf.variable_scope('atrous_conv5', reuse=tf.AUTO_REUSE):
			#conv5 = tf.nn.atrous_conv2d(conv4, filters=(3, 3, 512, 512), rate=2, padding='valid', name='atrous_conv1')
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv5 = tf.nn.relu(conv5, name='relu1')

			#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, 512, 512), rate=2, padding='valid', name='atrous_conv2')
			conv5 = tf.layers.conv2d(conv5, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv2')
			conv5 = tf.nn.relu(conv5, name='relu2')

			#conv5 = tf.nn.atrous_conv2d(conv5, filters=(3, 3, 512, 512), rate=2, padding='valid', name='atrous_conv3')
			conv5 = tf.layers.conv2d(conv5, filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv3')
			conv5 = tf.nn.relu(conv5, name='relu3')

		with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE):
			#dense6 = tf.nn.atrous_conv2d(conv5, filters=(7, 7, 512, 4096), rate=4, padding='valid', name='atrous_conv')
			dense6 = tf.layers.conv2d(conv5, filters=4096, kernel_size=(7, 7), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='dense')
			dense6 = tf.nn.relu(dense6, name='relu')
			# TODO [check] >> Which is better, dropout or batch normalization?
			dense6 = tf.layers.dropout(dense6, rate=dropout_rate, training=is_training, name='dropout')
			#dense6 = tf.layers.batch_normalization(dense6, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')

		with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE):
			dense7 = tf.layers.conv2d(dense6, filters=4096, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense')
			dense7 = tf.nn.relu(dense7, name='relu')
			# TODO [check] >> Which is better, dropout or batch normalization?
			dense7 = tf.layers.dropout(dense7, rate=dropout_rate, training=is_training, name='dropout')
			#dense7 = tf.layers.batch_normalization(dense7, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')

		with tf.variable_scope('fc_final', reuse=tf.AUTO_REUSE):
			dense_final = tf.layers.conv2d(dense7, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense')
			return tf.nn.relu(dense_final, name='relu') if is_final_relu_applied else dense_final

#--------------------------------------------------------------------

class PascalVocDilationNet(DilationNet):
	""" A dilation network for the PASCAL VOC dataset.
	"""

	def __init__(self, input_shape, output_shape):
		# TODO [check] >> is_final_relu_applied = True?
		super().__init__(input_shape, output_shape, is_final_relu_applied=True, model_name='pascal_voc_dilation_net')

	def _create_context_module(self, inputs, num_classes, is_training):
		with tf.variable_scope('ctx_conv', reuse=tf.AUTO_REUSE):
			conv1 = tf.pad(inputs, [[0, 0], [33, 33], [33, 33], [0, 0]], mode='CONSTANT', constant_values=0, name='pad')

			# Layer 1.
			conv1 = tf.layers.conv2d(conv1, filters=2 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			# Layer 2.
			conv1 = tf.layers.conv2d(conv1, filters=2 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

		with tf.variable_scope('ctx_atrous_conv', reuse=tf.AUTO_REUSE):
			# Layer 3.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, 2 * num_classes, 4 * num_classes), rate=2, padding='valid', name='atrous_conv1')
			conv2 = tf.layers.conv2d(conv2, filters=4 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			# Layer 4.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, 4 * num_classes, 8 * num_classes), rate=4, padding='valid', name='atrous_conv2')
			conv2 = tf.layers.conv2d(conv2, filters=8 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			# Layer 5.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, 8 * num_classes, 16 * num_classes), rate=8, padding='valid', name='atrous_conv3')
			conv2 = tf.layers.conv2d(conv2, filters=16 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
			conv2 = tf.nn.relu(conv2, name='relu3')

			# Layer 6.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, 16 * num_classes, 32 * num_classes), rate=16, padding='valid', name='atrous_conv4')
			conv2 = tf.layers.conv2d(conv2, filters=32 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='valid', name='conv4')
			conv2 = tf.nn.relu(conv2, name='relu4')

		with tf.variable_scope('ctx_final', reuse=tf.AUTO_REUSE):
			# Layer 7.
			dense_final = tf.layers.conv2d(conv2, filters=32 * num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense1')
			dense_final = tf.nn.relu(dense_final, name='relu1')

			# Layer 8.
			return tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense2')

#--------------------------------------------------------------------

class CamVidDilationNet(DilationNet):
	""" A dilation network for the CamVid dataset.
	"""

	def __init__(self, input_shape, output_shape):
		# TODO [check] >> is_final_relu_applied = False?
		super().__init__(input_shape, output_shape, is_final_relu_applied=False, model_name='camvid_dilation_net')

	def _create_context_module(self, inputs, num_classes, is_training):
		with tf.variable_scope('ctx_conv', reuse=tf.AUTO_REUSE):
			conv1 = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 1.
			conv1 = tf.layers.conv2d(conv1, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 2.
			conv1 = tf.layers.conv2d(conv1, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

		with tf.variable_scope('ctx_atrous_conv', reuse=tf.AUTO_REUSE):
			conv2 = tf.pad(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 3.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=2, padding='valid', name='atrous_conv1')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.pad(conv2, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 4.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=4, padding='valid', name='atrous_conv2')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			conv2 = tf.pad(conv2, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', constant_values=0, name='pad3')

			# Layer 5.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=8, padding='valid', name='atrous_conv3')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
			conv2 = tf.nn.relu(conv2, name='relu3')

			conv2 = tf.pad(conv2, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', constant_values=0, name='pad4')

			# Layer 6.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=16, padding='valid', name='atrous_conv4')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='valid', name='conv4')
			conv2 = tf.nn.relu(conv2, name='relu4')

		with tf.variable_scope('ctx_final', reuse=tf.AUTO_REUSE):
			dense_final = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad')

			# Layer 7.
			dense_final = tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense1')
			dense_final = tf.nn.relu(dense_final, name='relu1')

			# Layer 8.
			return tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense2')

#--------------------------------------------------------------------

class KittiDilationNet(DilationNet):
	""" A dilation network for the KITTI dataset.
	"""

	def __init__(self, input_shape, output_shape):
		# TODO [check] >> is_final_relu_applied = False?
		super().__init__(input_shape, output_shape, is_final_relu_applied=False, model_name='kitti_dilation_net')

	def _create_context_module(self, inputs, num_classes, is_training):
		with tf.variable_scope('ctx_conv', reuse=tf.AUTO_REUSE):
			conv1 = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 1.
			conv1 = tf.layers.conv2d(conv1, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 2.
			conv1 = tf.layers.conv2d(conv1, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

		with tf.variable_scope('ctx_atrous_conv', reuse=tf.AUTO_REUSE):
			conv2 = tf.pad(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 3.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=2, padding='valid', name='atrous_conv1')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.pad(conv2, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 4.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=4, padding='valid', name='atrous_conv2')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			conv2 = tf.pad(conv2, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', constant_values=0, name='pad3')

			# Layer 5.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=8, padding='valid', name='atrous_conv3')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
			conv2 = tf.nn.relu(conv2, name='relu3')

		with tf.variable_scope('ctx_final', reuse=tf.AUTO_REUSE):
			dense_final = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad')

			# Layer 6.
			dense_final = tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense1')
			dense_final = tf.nn.relu(dense_final, name='relu1')

			# Layer 7.
			return tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense2')

#--------------------------------------------------------------------

class CityscapesDilationNet(DilationNet):
	""" A dilation network for the Cityscapes dataset.
	"""

	def __init__(self, input_shape, output_shape):
		# TODO [check] >> is_final_relu_applied = False?
		super().__init__(input_shape, output_shape, is_final_relu_applied=False, model_name='cityscapes_dilation_net')

	def _create_context_module(self, inputs, num_classes, is_training):
		with tf.variable_scope('ctx_conv', reuse=tf.AUTO_REUSE):
			conv1 = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 1.
			conv1 = tf.layers.conv2d(conv1, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 2.
			conv1 = tf.layers.conv2d(conv1, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

		with tf.variable_scope('ctx_atrous_conv', reuse=tf.AUTO_REUSE):
			conv2 = tf.pad(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 3.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=2, padding='valid', name='atrous_conv1')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.pad(conv2, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 4.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=4, padding='valid', name='atrous_conv2')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			conv2 = tf.pad(conv2, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', constant_values=0, name='pad3')

			# Layer 5.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=8, padding='valid', name='atrous_conv3')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
			conv2 = tf.nn.relu(conv2, name='relu3')

			conv2 = tf.pad(conv2, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', constant_values=0, name='pad4')

			# Layer 6.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=16, padding='valid', name='atrous_conv4')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='valid', name='conv4')
			conv2 = tf.nn.relu(conv2, name='relu4')

			conv2 = tf.pad(conv2, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', constant_values=0, name='pad5')

			# Layer 7.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=32, padding='valid', name='atrous_conv5')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(32, 32), padding='valid', name='conv5')
			conv2 = tf.nn.relu(conv2, name='relu5')

			conv2 = tf.pad(conv2, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', constant_values=0, name='pad6')

			# Layer 8.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_classes, num_classes), rate=64, padding='valid', name='atrous_conv6')
			conv2 = tf.layers.conv2d(conv2, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(64, 64), padding='valid', name='conv6')
			conv2 = tf.nn.relu(conv2, name='relu6')

		with tf.variable_scope('ctx_final', reuse=tf.AUTO_REUSE):
			dense_final = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0, name='pad')

			# Layer 9.
			dense_final = tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense1')
			dense_final = tf.nn.relu(dense_final, name='relu1')

			# Layer 10.
			dense_final = tf.layers.conv2d(dense_final, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense2')

		with tf.variable_scope('ctx_upsample', reuse=tf.AUTO_REUSE):
			conv_upsample = tf.image.resize_bilinear(dense_final, size=(1024, 1024))
			# FIXME [fix] >> Check if filters=num_classes, kernel_size=(1, 1).
			conv_upsample = tf.layers.conv2d(conv_upsample, filters=num_classes, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='same', name='dense')
			return tf.nn.relu(conv_upsample, name='relu')
