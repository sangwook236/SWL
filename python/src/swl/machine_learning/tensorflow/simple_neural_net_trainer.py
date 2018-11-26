import tensorflow as tf
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer

#%%------------------------------------------------------------------

class BasicNeuralNetTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, optimizer, initial_epoch=0):
		self._optimizer = optimizer
		super().__init__(neuralNet, initial_epoch)

	def _get_train_operation(self, loss, global_step=None):
		with tf.name_scope('train'):
			train_op = self._optimizer.minimize(loss, global_step=global_step)
			return train_op

class BasicGradientClippingNeuralNetTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, optimizer, max_gradient_norm, initial_epoch=0):
		self._optimizer = optimizer
		self._max_gradient_norm = max_gradient_norm
		super().__init__(neuralNet, initial_epoch)

	def _get_train_operation(self, loss, global_step=None):
		with tf.name_scope('train'):
			# Method 1.
			gradients = self._optimizer.compute_gradients(loss)
			for i, (g, v) in enumerate(gradients):
				if g is not None:
					gradients[i] = (tf.clip_by_norm(g, self._max_gradient_norm), v)  # Clip gradients.
			train_op = self._optimizer.apply_gradients(gradients, global_step=global_step)
			"""
			# Method 2.
			#	REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
			params = tf.trainable_variables()
			gradients = tf.gradients(loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)  # Clip gradients.
			train_op = self._optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
			"""
			return train_op

#%%------------------------------------------------------------------

class SimpleNeuralNetTrainer(BasicNeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		with tf.name_scope('learning_rate'):
			#learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			learning_rate = 0.001
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)

		super().__init__(neuralNet, optimizer, initial_epoch)

class SimpleGradientClippingNeuralNetTrainer(BasicGradientClippingNeuralNetTrainer):
	def __init__(self, neuralNet, max_gradient_norm, initial_epoch=0):
		with tf.name_scope('learning_rate'):
			#learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			learning_rate = 0.001
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)

		super().__init__(neuralNet, optimizer, max_gradient_norm, initial_epoch)
