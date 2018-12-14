import tensorflow as tf
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer, GradientClippingNeuralNetTrainer

#%%------------------------------------------------------------------

class SimpleNeuralNetTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 0.001
			#learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

		super().__init__(neuralNet, optimizer, global_step)

class SimpleGradientClippingNeuralNetTrainer(GradientClippingNeuralNetTrainer):
	def __init__(self, neuralNet, max_gradient_norm, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 0.001
			#learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

		super().__init__(neuralNet, optimizer, max_gradient_norm, global_step)
