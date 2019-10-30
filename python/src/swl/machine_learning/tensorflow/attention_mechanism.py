import abc
import tensorflow as tf

#--------------------------------------------------------------------

# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
class AttentionMechanism(abc.ABC):
	@abc.abstractmethod
	def __call__(self, source_states, target_state):
		raise NotImplementedError

	# REF [function] >> _weight_variable() in ./mnist_cnn_tf.py.
	# We can't initialize these variables to 0 - the network will get stuck.
	def _weight_variable(self, shape, name):
		"""Create a weight variable with appropriate initialization."""
		#initial = tf.truncated_normal(shape, stddev=0.1)
		#return tf.Variable(initial, name=name)
		return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

	# REF [function] >> _variable_summaries() in ./mnist_cnn_tf.py.
	def _variable_summaries(self, var, is_filter=False):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)
			if is_filter:
				tf.summary.image('filter', var)  # Visualizes filters.

# Additive attention mechanism.
# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
class BahdanauAttentionMechanism(AttentionMechanism):
	def __init__(self, source_dim, target_dim):
		self._W1, self._W2, self._V = self._create_attention_variables(source_dim, target_dim)

	# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
	# REF [site] >> https://www.tensorflow.org/api_guides/python/contrib.seq2seq
	# REF [site] >> https://talbaumel.github.io/attention/
	# FIXME [improve] >> Too slow.
	def __call__(self, source_states, target_state):
		attention_weights = []
		for src_state in source_states:
			# score = v^T * tanh(W1 * h + W2 * )
			attention_weight = tf.matmul(src_state, self._W1) + tf.matmul(target_state, self._W2)
			attention_weight = tf.matmul(tf.tanh(attention_weight), self._V)
			attention_weights.append(attention_weight)

		attention_weights = tf.nn.softmax(attention_weights)  # alpha.
		attention_weights = tf.unstack(attention_weights, len(source_states), axis=0)
		return tf.reduce_sum([src_state * weight for src_state, weight in zip(source_states, attention_weights)], axis=0)  # Context, c.

	def _create_attention_variables(self, source_dim, target_dim):
		with tf.name_scope('attention_W1'):
			W1 = self._weight_variable((source_dim, source_dim), 'W1')
			self._variable_summaries(W1)
		with tf.name_scope('attention_W2'):
			W2 = self._weight_variable((target_dim, source_dim), 'W2')
			self._variable_summaries(W2)
		with tf.name_scope('attention_V'):
			V = self._weight_variable((source_dim, 1), 'V')
			self._variable_summaries(V)
		return W1, W2, V

# Multiplicative attention mechanism.
# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
class LuongAttentionMechanism(AttentionMechanism):
	def __init__(self, source_dim, target_dim):
		self._W = self._create_attention_variables(source_dim, target_dim)

	# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
	# REF [site] >> https://www.tensorflow.org/api_guides/python/contrib.seq2seq
	def __call__(self, source_states, target_state):
		attention_weights = []
		for src_state in source_states:
			# FIXME [fix] >> Not working.
			#attention_weight = tf.matmul(tf.matmul(target_state, self._W), src_state)
			attention_weight = tf.einsum('ij,ij->i', tf.matmul(target_state, self._W), src_state)
			attention_weights.append(attention_weight)

		attention_weights = tf.nn.softmax(attention_weights)  # alpha.
		attention_weights = tf.unstack(attention_weights, len(source_states), axis=0)
		return tf.reduce_sum([src_state * weight for src_state, weight in zip(source_states, attention_weights)], axis=0)  # Context, c.

	def _create_attention_variables(self, source_dim, target_dim):
		with tf.name_scope('attention_W'):
			W = self._weight_variable((target_dim, source_dim), 'W')
			self._variable_summaries(W)
		return W
