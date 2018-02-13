import numpy as np
import tensorflow as tf
from simple_neural_net import SimpleNeuralNet

#%%------------------------------------------------------------------

class SimpleTfSeq2SeqEncoderDecoderWithAttention(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape, start_token, end_token, is_bidirectional=True, is_time_major=False):
		self._input_seq_lens_ph = tf.placeholder(tf.int32, [None], name='input_seq_lens_ph')
		self._output_seq_lens_ph = tf.placeholder(tf.int32, [None], name='output_seq_lens_ph')
		self._batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		self._start_token = start_token
		self._end_token = end_token

		self._is_bidirectional = is_bidirectional
		self._is_time_major = is_time_major
		super().__init__(input_shape, output_shape)

	def get_feed_dict(self, data, labels=None, is_training=True, **kwargs):
		#input_seq_lens = tf.constant(max_time_steps, tf.int32, shape=[batch_size])
		#output_seq_lens = tf.constant(max_time_steps, tf.int32, shape=[batch_size])
		if self._is_time_major:
			input_seq_lens = np.full(data.shape[1], data.shape[0], np.int32)
			if labels is None:
				output_seq_lens = np.full(data.shape[1], data.shape[0], np.int32)
			else:
				output_seq_lens = np.full(labels.shape[1], labels.shape[0], np.int32)
			batch_size = [data.shape[1]]
		else:
			input_seq_lens = np.full(data.shape[0], data.shape[1], np.int32)
			if labels is None:
				output_seq_lens = np.full(data.shape[0], data.shape[1], np.int32)
			else:
				output_seq_lens = np.full(labels.shape[0], labels.shape[1], np.int32)
			batch_size = [data.shape[0]]

		if labels is None:
			feed_dict = {self._input_tensor_ph: data, self._is_training_tensor_ph: is_training, self._input_seq_lens_ph: input_seq_lens, self._output_seq_lens_ph: output_seq_lens, self._batch_size_ph: batch_size}
		else:
			feed_dict = {self._input_tensor_ph: data, self._output_tensor_ph: labels, self._is_training_tensor_ph: is_training, self._input_seq_lens_ph: input_seq_lens, self._output_seq_lens_ph: output_seq_lens, self._batch_size_ph: batch_size}
		return feed_dict

	def _create_model(self, input_tensor, output_tensor, is_training_tensor, input_shape, output_shape):
		with tf.variable_scope('simple_tf_seq2seq_encdec_attention', reuse=tf.AUTO_REUSE):
			num_classes = output_shape[-1]
			if self._is_bidirectional:
				return self._create_dynamic_bidirectional_model(input_tensor, output_tensor, is_training_tensor, self._input_seq_lens_ph, self._output_seq_lens_ph, self._batch_size_ph, num_classes, self._is_time_major)
			else:
				return self._create_dynamic_model(input_tensor, output_tensor, is_training_tensor, self._input_seq_lens_ph, self._output_seq_lens_ph, self._batch_size_ph, num_classes, self._is_time_major)

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
			masks = tf.sequence_mask(self._output_seq_lens_ph, tf.reduce_max(self._output_seq_lens_ph), dtype=tf.float32)
			#loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=t, weights=masks)
			loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=tf.argmax(t, axis=-1), weights=masks)
			tf.summary.scalar('loss', loss)
			return loss

	def _create_dynamic_model(self, input_tensor, output_tensor, is_training_tensor, input_seq_lens, output_seq_lens, batch_size, num_classes, is_time_major):
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 256
		num_dec_hidden_units = 256
		keep_prob = 0.5
		"""
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
		enc_cell_outputs, enc_cell_state = tf.nn.dynamic_rnn(enc_cell, input_tensor, sequence_length=input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')

		# Attention.
		if True:
			# Additive attention.
			# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_attention_units, memory=enc_cell_outputs, memory_sequence_length=input_seq_lens)
		else:
			# Multiplicative attention.
			# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
			attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_attention_units, memory=enc_cell_outputs, memory_sequence_length=input_seq_lens)
		dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size=num_attention_units)

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('enc-dec-attn', reuse=tf.AUTO_REUSE):
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

		def get_training_helper():
			return tf.contrib.seq2seq.TrainingHelper(inputs=input_tensor, sequence_length=input_seq_lens, time_major=is_time_major)
		def get_testing_helper():
			return tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=input_tensor, start_tokens=tf.tile([self._start_token], batch_size), end_token=self._end_token)
		def get_decoder_outputs(helper):
			decoder = tf.contrib.seq2seq.BasicDecoder(
				dec_cell, helper=helper,
				initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_state),  # tf.contrib.seq2seq.AttentionWrapperState.
				output_layer=output_layer)
			#decoder_outputs, decoder_state, decoder_seq_lens = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
			decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
			return decoder_outputs
		decoder_outputs = tf.cond(is_training_tensor, lambda: get_decoder_outputs(get_training_helper()), lambda: get_decoder_outputs(get_testing_helper()))
		"""
		helper = tf.contrib.seq2seq.TrainingHelper(inputs=input_tensor, sequence_length=input_seq_lens, time_major=is_time_major)
		decoder = tf.contrib.seq2seq.BasicDecoder(
			dec_cell, helper=helper,
			initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_state),  # tf.contrib.seq2seq.AttentionWrapperState.
			output_layer=output_layer)
		#decoder_outputs, decoder_state, decoder_seq_lens = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		"""

		return decoder_outputs.rnn_output

	def _create_dynamic_bidirectional_model(self, input_tensor, output_tensor, is_training_tensor, input_seq_lens, output_seq_lens, batch_size, num_classes, is_time_major):
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 256
		keep_prob = 0.5
		"""
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
		enc_cell_outputs, enc_cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, sequence_length=input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs = tf.concat(enc_cell_outputs, axis=-1)
		enc_cell_states = tf.contrib.rnn.LSTMStateTuple(tf.concat((enc_cell_states[0].c, enc_cell_states[1].c), axis=-1), tf.concat((enc_cell_states[0].h, enc_cell_states[1].h), axis=-1))

		# Attention.
		if True:
			# Additive attention.
			# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_attention_units, memory=enc_cell_outputs)
		else:
			# Multiplicative attention.
			# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
			attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_attention_units, memory=enc_cell_outputs)
		dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size=num_attention_units)

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('enc-dec-attn', reuse=tf.AUTO_REUSE):
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

		def get_training_helper():
			return tf.contrib.seq2seq.TrainingHelper(inputs=input_tensor, sequence_length=input_seq_lens, time_major=is_time_major)
		def get_testing_helper():
			return tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=input_tensor, start_tokens=tf.tile([self._start_token], batch_size), end_token=self._end_token)
		def get_decoder_outputs(helper):
			decoder = tf.contrib.seq2seq.BasicDecoder(
				dec_cell, helper=helper,
				initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_states),  # tf.contrib.seq2seq.AttentionWrapperState.
				output_layer=output_layer)
			#decoder_outputs, decoder_state, decoder_seq_lens = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
			decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
			return decoder_outputs
		decoder_outputs = tf.cond(is_training_tensor, lambda: get_decoder_outputs(get_training_helper()), lambda: get_decoder_outputs(get_testing_helper()))
		"""
		training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=input_tensor, sequence_length=input_seq_lens, time_major=is_time_major)
		testing_helper = tf.contrib.seq2seq.TrainingHelper(inputs=input_tensor, sequence_length=input_seq_lens, time_major=is_time_major)
		#testing_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=input_tensor, start_tokens=tf.tile([self._start_token], batch_size), end_token=self._end_token)
		training_decoder = tf.contrib.seq2seq.BasicDecoder(
			dec_cell, helper=training_helper,
			initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_states),  # tf.contrib.seq2seq.AttentionWrapperState.
			output_layer=output_layer)
		testing_decoder = tf.contrib.seq2seq.BasicDecoder(
			dec_cell, helper=testing_helper,
			initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_states),  # tf.contrib.seq2seq.AttentionWrapperState.
			output_layer=output_layer)
		#decoder_outputs, decoder_state, decoder_seq_lens = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		#decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		testing_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(testing_decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		decoder_outputs = tf.cond(is_training_tensor, lambda: training_decoder_outputs, lambda: testing_decoder_outputs)
		"""
		"""
		helper = tf.contrib.seq2seq.TrainingHelper(inputs=input_tensor, sequence_length=input_seq_lens, time_major=is_time_major)
		decoder = tf.contrib.seq2seq.BasicDecoder(
			dec_cell, helper=helper,
			initial_state=dec_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_cell_states),  # tf.contrib.seq2seq.AttentionWrapperState.
			output_layer=output_layer)
		#decoder_outputs, decoder_state, decoder_seq_lens = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=is_time_major, impute_finished=True, maximum_iterations=None if input_seq_lens is None else tf.reduce_max(input_seq_lens), scope='dec')
		"""

		return decoder_outputs.rnn_output

	def _create_unit_cell(self, num_units):
		#return tf.contrib.rnn.BasicRNNCell(num_units)
		#return tf.contrib.rnn.RNNCell(num_units)

		return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.LSTMCell(num_units, forget_bias=1.0)

		#return tf.contrib.rnn.GRUCell(num_units)
