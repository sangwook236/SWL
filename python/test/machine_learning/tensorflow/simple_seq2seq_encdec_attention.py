import tensorflow as tf
from simple_neural_net import SimpleNeuralNet

#%%------------------------------------------------------------------

class SimpleSeq2SeqEncoderDecoderWithAttention(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape, is_bidirectional=True, is_time_major=False):
		self._input_seq_len_ph = tf.placeholder(tf.int32, [None])
		self._output_seq_len_ph = tf.placeholder(tf.int32, [None])

		self._is_bidirectional = is_bidirectional
		self._is_time_major = is_time_major
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, output_tensor, is_training_tensor, input_shape, output_shape):
		with tf.variable_scope('simple_seq2seq_encdec_attention', reuse=tf.AUTO_REUSE):
			num_classes = output_shape[-1]
			if self._is_bidirectional:
				return self._create_dynamic_bidirectional_model(input_tensor, output_tensor, is_training_tensor, num_classes, self._is_time_major)
			else:
				return self._create_dynamic_model(input_tensor, output_tensor, is_training_tensor, num_classes, self._is_time_major)

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
			masks = tf.sequence_mask(self._output_seq_len_ph, tf.reduce_max(self._output_seq_len_ph), dtype=tf.float32)
			print('***********', y.get_shape().as_list(), t.get_shape().as_list())
			loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=t, weights=masks)
			tf.summary.scalar('loss', loss)
			return loss

	def _create_dynamic_model(self, input_tensor, output_tensor, is_training_tensor, num_classes, is_time_major):
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 256
		num_dec_hidden_units = 256
		keep_prob = 0.5
		num_attention_units = 128

		# Defines cells.
		enc_cell = self._create_unit_cell(num_enc_hidden_units)
		enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell = tf.contrib.rnn.AttentionCellWrapper(enc_cell, attention_window_len, state_is_tuple=True)
		dec_cell = self._create_unit_cell(num_dec_hidden_units)
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#dec_cell = tf.contrib.rnn.AttentionCellWrapper(dec_cell, attention_window_len, state_is_tuple=True)

		# Encoder.
		enc_cell_outputs, enc_cell_state = tf.nn.dynamic_rnn(enc_cell, input_tensor, sequence_length=self._input_seq_len_ph, time_major=is_time_major, dtype=tf.float32, scope='enc')

		# Attention.
		# Additive attention.
		# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_attention_units, memory=enc_cell_outputs, memory_sequence_length=self._input_seq_len_ph)
		# Multiplicative attention.
		# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
		#attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_attention_units, memory=enc_cell_outputs)
		dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size=num_attention_units)

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

		# Decoder.
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				output_layer = tf.layers.Dense(1)
			elif num_classes >= 2:
				output_layer = tf.layers.Dense(num_classes)
			else:
				assert num_classes > 0, 'Invalid number of classes.'

		training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=output_tensor, sequence_length=self._output_seq_len_ph, time_major=is_time_major)
		decoder = tf.contrib.seq2seq.BasicDecoder(
			dec_cell, helper=training_helper,
			#initial_state=enc_cell_state,
			# FIXME [restore] >>
			#initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_state),
			initial_state=dec_cell.zero_state(4, tf.float32).clone(cell_state=enc_cell_state),
			output_layer=output_layer)
		decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=tf.reduce_max(self._output_seq_len_ph))

		return decoder_output.rnn_output

	def _create_dynamic_bidirectional_model(self, input_tensor, output_tensor, is_training_tensor, num_classes, is_time_major):
		"""
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 256
		keep_prob = 0.5
		num_attention_units = 128

		# Defines cells.
		enc_cell_fw = self._create_unit_cell(num_enc_hidden_units)  # Forward cell.
		enc_cell_fw = tf.contrib.rnn.DropoutWrapper(enc_cell_fw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell_fw = tf.contrib.rnn.AttentionCellWrapper(enc_cell_fw, attention_window_len, state_is_tuple=True)
		enc_cell_bw = self._create_unit_cell(num_enc_hidden_units)  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell_bw = tf.contrib.rnn.AttentionCellWrapper(enc_cell_bw, attention_window_len, state_is_tuple=True)
		dec_cell = self._create_unit_cell(num_dec_hidden_units)
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#dec_cell = tf.contrib.rnn.AttentionCellWrapper(dec_cell, attention_window_len, state_is_tuple=True)

		# Encoder.
		enc_cell_outputs, enc_cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, time_major=is_time_major, dtype=tf.float32)
		enc_cell_outputs = tf.concat(enc_cell_outputs, 2)
		enc_cell_states = tf.concat(enc_cell_states, 2)

		# Attention.
		# Additive attention.
		# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_attention_units, memory=enc_cell_outputs)
		# Multiplicative attention.
		# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
		#attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_attention_units, memory=enc_cell_outputs)
		dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size=num_attention_units)

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

		# Decoder.
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				output_layer = tf.layers.Dense(1)
			elif num_classes >= 2:
				output_layer = tf.layers.Dense(num_classes)
			else:
				assert num_classes > 0, 'Invalid number of classes.'

		training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=output_tensor, sequence_length=self._output_seq_len_ph, time_major=is_time_major)
		decoder = tf.contrib.seq2seq.BasicDecoder(
			dec_cell, helper=training_helper,
			initial_state=enc_cell_states,
			#initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_states),
			output_layer=output_layer)
		decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=tf.reduce_max(self._output_seq_len_ph))

		return decoder_output.rnn_output

	def _create_unit_cell(self, num_units):
		#return tf.contrib.rnn.BasicRNNCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.RNNCell(num_units, forget_bias=1.0)

		return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.LSTMCell(num_units, forget_bias=1.0)

		#return tf.contrib.rnn.GRUCell(num_units, forget_bias=1.0)

	# REF [function] >> _weight_variable() in ./mnist_tf_cnn.py.
	# We can't initialize these variables to 0 - the network will get stuck.
	def _weight_variable(self, shape, name):
		"""Create a weight variable with appropriate initialization."""
		#initial = tf.truncated_normal(shape, stddev=0.1)
		#return tf.Variable(initial, name=name)
		return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

	# REF [function] >> _variable_summaries() in ./mnist_tf_cnn.py.
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

	def _create_variables_for_additive_attention(self, inputs, state):
		#input_shape = tf.shape(inputs[0])
		#state_shape = tf.shape(state)
		# TODO [caution] >> inputs is a list.
		input_shape = inputs[0].get_shape().as_list()
		state_shape = state.get_shape().as_list()

		with tf.name_scope('attention_W1'):
			W1 = self._weight_variable((input_shape[-1], input_shape[-1]), 'W1')
			self._variable_summaries(W1)
		with tf.name_scope('attention_W2'):
			W2 = self._weight_variable((state_shape[-1], input_shape[-1]), 'W2')
			self._variable_summaries(W2)
		with tf.name_scope('attention_V'):
			V = self._weight_variable((input_shape[-1], 1), 'V')
			self._variable_summaries(V)
		return W1, W2, V

	# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
	# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
	# REF [site] >> https://www.tensorflow.org/api_guides/python/contrib.seq2seq
	# REF [site] >> https://talbaumel.github.io/attention/
	# FIXME [improve] >> Too slow.
	def _attend_additively(self, inputs, state, W1, W2, V):
		attention_weights = []
		for inp in inputs:
			attention_weight = tf.matmul(inp, W1) + tf.matmul(state, W2)
			attention_weight = tf.matmul(tf.tanh(attention_weight), V)
			attention_weights.append(attention_weight)

		attention_weights = tf.nn.softmax(attention_weights)  # alpha.
		attention_weights = tf.unstack(attention_weights, len(inputs), axis=0)
		return tf.reduce_sum([inp * weight for inp, weight in zip(inputs, attention_weights)], axis=0)  # Context, c.

	# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
	# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
	# REF [site] >> https://www.tensorflow.org/api_guides/python/contrib.seq2seq
	def _attend_multiplicatively(self, inputs, state):
		raise NotImplementedError
