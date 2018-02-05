import tensorflow as tf
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer

#%%------------------------------------------------------------------

class SimpleSeq2SeqTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		super().__init__(neuralNet, initial_epoch)

	def _get_train_step(self, loss, global_step=None):
		with tf.name_scope('learning_rate'):
			#learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
			learning_rate = 0.001
			tf.summary.scalar('learning_rate', learning_rate)

		with tf.name_scope('train'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#train_step = optimizer.minimize(loss, global_step=global_step)
			params = tf.trainable_variables()
			gradients = tf.gradients(loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
			#train_step = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
			train_step = optimizer.apply_gradients(zip(clipped_gradients, params))
			return train_step
