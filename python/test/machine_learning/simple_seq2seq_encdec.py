import numpy as np
import tensorflow as tf
from swl.machine_learning.tensorflow_model import SimpleAuxiliaryInputTensorFlowModel

#--------------------------------------------------------------------

class SimpleSeq2SeqEncoderDecoder(SimpleAuxiliaryInputTensorFlowModel):
	def __init__(self, encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_bidirectional=True, is_time_major=False):
		self._input_seq_lens_ph = tf.placeholder(tf.int32, [None], name='encoder_input_seq_lens_ph')
		self._output_seq_lens_ph = tf.placeholder(tf.int32, [None], name='decoder_output_seq_lens_ph')
		self._batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		self._start_token = start_token
		self._end_token = end_token

		self._is_bidirectional = is_bidirectional
		self._is_time_major = is_time_major

		super().__init__(encoder_input_shape, decoder_input_shape, decoder_output_shape)

	def get_feed_dict(self, data, num_data, *args, **kwargs):
		len_data = len(data)
		if 1 == len_data:
			encoder_inputs = data[0]

			if self._is_time_major:
				encoder_input_seq_lens = np.full(encoder_inputs.shape[1], encoder_inputs.shape[0], np.int32)
				decoder_output_seq_lens = np.full(encoder_inputs.shape[1], encoder_inputs.shape[0], np.int32)
			else:
				encoder_input_seq_lens = np.full(encoder_inputs.shape[0], encoder_inputs.shape[1], np.int32)
				decoder_output_seq_lens = np.full(encoder_inputs.shape[0], encoder_inputs.shape[1], np.int32)

			feed_dict = {self._input_ph: data[0], self._input_seq_lens_ph: encoder_input_seq_lens, self._output_seq_lens_ph: decoder_output_seq_lens, self._batch_size_ph: [num_data]}
		elif 3 == len_data:
			encoder_inputs, decoder_inputs, decoder_outputs = data

			if self._is_time_major:
				encoder_input_seq_lens = np.full(encoder_inputs.shape[1], encoder_inputs.shape[0], np.int32)
				if decoder_inputs is None or decoder_outputs is None:
					decoder_output_seq_lens = np.full(encoder_inputs.shape[1], encoder_inputs.shape[0], np.int32)
				else:
					decoder_output_seq_lens = np.full(decoder_outputs.shape[1], decoder_outputs.shape[0], np.int32)
			else:
				encoder_input_seq_lens = np.full(encoder_inputs.shape[0], encoder_inputs.shape[1], np.int32)
				if decoder_inputs is None or decoder_outputs is None:
					decoder_output_seq_lens = np.full(encoder_inputs.shape[0], encoder_inputs.shape[1], np.int32)
				else:
					decoder_output_seq_lens = np.full(decoder_outputs.shape[0], decoder_outputs.shape[1], np.int32)

			if decoder_inputs is None or decoder_outputs is None:
				feed_dict = {self._input_ph: encoder_inputs, self._input_seq_lens_ph: encoder_input_seq_lens, self._output_seq_lens_ph: decoder_output_seq_lens, self._batch_size_ph: [num_data]}
			else:
				feed_dict = {self._input_ph: encoder_inputs, self._aux_input_ph: decoder_inputs, self._output_ph: decoder_outputs, self._input_seq_lens_ph: encoder_input_seq_lens, self._output_seq_lens_ph: decoder_output_seq_lens, self._batch_size_ph: [num_data]}
		else:
			raise ValueError('Invalid number of feed data: {}'.format(len_data))
		return feed_dict

	def _get_loss(self, y, t):
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
			masks = tf.sequence_mask(self._output_seq_lens_ph, maxlen=tf.reduce_max(self._output_seq_lens_ph), dtype=tf.float32)
			# Weighted cross-entropy loss for a sequence of logits.
			#loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=t, weights=masks)
			loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=tf.argmax(t, axis=-1), weights=masks)
			tf.summary.scalar('loss', loss)
			return loss

	def _create_single_model(self, encoder_inputs, decoder_inputs, encoder_input_shape, decoder_input_shape, decoder_output_shape, is_training):
		with tf.variable_scope('simple_seq2seq_encdec', reuse=tf.AUTO_REUSE):
			# TODO [improve] >> It is not good to use num_time_steps.
			#num_classes = decoder_output_shape[-1]
			if self._is_time_major:
				num_time_steps, num_classes = decoder_output_shape[0], decoder_output_shape[-1]
			else:
				num_time_steps, num_classes = decoder_output_shape[1], decoder_output_shape[-1]
			if self._is_bidirectional:
				return self._create_dynamic_bidirectional_model(encoder_inputs, decoder_inputs, is_training, self._input_seq_lens_ph, self._batch_size_ph, num_time_steps, num_classes, self._is_time_major)
			else:
				return self._create_dynamic_model(encoder_inputs, decoder_inputs, is_training, self._input_seq_lens_ph, self._batch_size_ph, num_time_steps, num_classes, self._is_time_major)

	def _create_dynamic_model(self, encoder_inputs, decoder_inputs, is_training, encoder_input_seq_lens, batch_size, num_time_steps, num_classes, is_time_major):
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 256
		num_dec_hidden_units = 256
		keep_prob = 0.5
		"""

		# Defines cells.
		enc_cell = self._create_unit_cell(num_enc_hidden_units, 'enc_unit_cell')
		enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell = tf.contrib.rnn.AttentionCellWrapper(enc_cell, attention_window_len, state_is_tuple=True)
		dec_cell = self._create_unit_cell(num_dec_hidden_units, 'dec_unit_cell')
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#dec_cell = tf.contrib.rnn.AttentionCellWrapper(dec_cell, attention_window_len, state_is_tuple=True)

		# Encoder.
		enc_cell_outputs, enc_cell_state = tf.nn.dynamic_rnn(enc_cell, encoder_inputs, sequence_length=encoder_input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')

		# Attention.
		# REF [function] >> SimpleSeq2SeqEncoderDecoderWithTfAttention._create_dynamic_model() in ./simple_seq2seq_encdec_tf_attention.py.

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('simple_seq2seq_encdec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		"""
		# REF [site] >> https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn

		# Method #1: Uses only inputs of the cell.
		cell_state = cell.zero_state(batch_size, tf.float32)  # Initial state.
		cell_outputs = []
		for inp in cell_inputs:
			cell_output, cell_state = cell(inp, cell_state, scope='cell')
			cell_outputs.append(cell_output)

		# Method #2: Uses only the previous output of the cell.
		cell_state = cell.zero_state(batch_size, tf.float32)  # Initial state.
		cell_input = tf.fill(tf.concat((batch_size, tf.constant([num_classes])), axis=-1), float(start_token))  # Initial input.
		cell_outputs = []
		for _ in range(num_time_steps):
			cell_output, cell_state = cell(cell_input, cell_state, scope='cell')
			cell_input = f(cell_output)  # TODO [implement] >> e.g.) num_dec_hidden_units -> num_classes.
			cell_outputs.append(cell_input)
			#cell_outputs.append(cell_output)

		# Method #3: Uses both inputs and the previous output of the cell.
		cell_state = cell.zero_state(batch_size, tf.float32)  # Initial state.
		cell_input = tf.fill(tf.concat((batch_size, tf.constant([num_classes])), axis=-1), float(start_token))  # Initial input.
		cell_outputs = []
		for inp in cell_inputs:
			cell_output, cell_state = cell(tf.concat([inp, cell_input], axis=-1), cell_state, scope='cell')
			#cell_output, cell_state = cell(cell_input, tf.concat([inp, cell_state], axis=-1), scope='cell')
			cell_input = f(cell_output)  # TODO [implement] >> e.g.) num_dec_hidden_units -> num_classes.
			cell_outputs.append(cell_input)
			#cell_outputs.append(cell_output)
		"""

		# Decoder.
		# NOTICE [info] {important} >> The same model has to be used in training and inference steps.
		if is_training:
			return self._get_decoder_output_for_training(dec_cell, enc_cell_state, decoder_inputs, num_time_steps, num_classes, is_time_major)
		else:
			return self._get_decoder_output_for_inference(dec_cell, enc_cell_state, batch_size, num_time_steps, num_classes, is_time_major)

	def _create_dynamic_bidirectional_model(self, encoder_inputs, decoder_inputs, is_training, encoder_input_seq_lens, batch_size, num_time_steps, num_classes, is_time_major):
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 256
		keep_prob = 0.5
		"""

		# Defines cells.
		enc_cell_fw = self._create_unit_cell(num_enc_hidden_units, 'enc_fw_unit_cell')  # Forward cell.
		enc_cell_fw = tf.contrib.rnn.DropoutWrapper(enc_cell_fw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell_fw = tf.contrib.rnn.AttentionCellWrapper(enc_cell_fw, attention_window_len, state_is_tuple=True)
		enc_cell_bw = self._create_unit_cell(num_enc_hidden_units, 'enc_bw_unit_cell')  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell_bw = tf.contrib.rnn.AttentionCellWrapper(enc_cell_bw, attention_window_len, state_is_tuple=True)
		dec_cell = self._create_unit_cell(num_dec_hidden_units, 'dec_unit_cell')
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#dec_cell = tf.contrib.rnn.AttentionCellWrapper(dec_cell, attention_window_len, state_is_tuple=True)

		# Encoder.
		enc_cell_outputs, enc_cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, encoder_inputs, sequence_length=encoder_input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs = tf.concat(enc_cell_outputs, axis=-1)
		enc_cell_states = tf.contrib.rnn.LSTMStateTuple(tf.concat((enc_cell_states[0].c, enc_cell_states[1].c), axis=-1), tf.concat((enc_cell_states[0].h, enc_cell_states[1].h), axis=-1))

		# Attention.
		# REF [function] >> SimpleSeq2SeqEncoderDecoderWithTfAttention._create_dynamic_bidirectional_model() in ./simple_seq2seq_encdec_tf_attention.py.

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('simple_seq2seq_encdec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		# Decoder.
		# NOTICE [info] {important} >> The same model has to be used in training and inference steps.
		if is_training:
			return self._get_decoder_output_for_training(dec_cell, enc_cell_states, decoder_inputs, num_time_steps, num_classes, is_time_major)
		else:
			return self._get_decoder_output_for_inference(dec_cell, enc_cell_states, batch_size, num_time_steps, num_classes, is_time_major)

	def _create_projection_layer(self, dec_cell_outputs, num_classes):
		with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				return tf.layers.dense(dec_cell_outputs, 1, activation=tf.sigmoid, name='dense')
				#return tf.layers.dense(dec_cell_outputs, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			elif num_classes >= 2:
				return tf.layers.dense(dec_cell_outputs, num_classes, activation=tf.nn.softmax, name='dense')
				#return tf.layers.dense(dec_cell_outputs, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			else:
				assert num_classes > 0, 'Invalid number of classes.'
				return None

	def _get_decoder_output_for_training(self, dec_cell, initial_cell_state, decoder_inputs, num_time_steps, num_classes, is_time_major):
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#dec_cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, decoder_inputs, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')
		#dec_cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, decoder_inputs, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		decoder_inputs = tf.unstack(decoder_inputs, num_time_steps, axis=0 if is_time_major else 1)

		dec_cell_state = initial_cell_state
		dec_cell_outputs = []
		for inp in decoder_inputs:
			dec_cell_output, dec_cell_state = dec_cell(inp, dec_cell_state, scope='dec')
			dec_cell_outputs.append(dec_cell_output)

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		dec_cell_outputs = tf.stack(dec_cell_outputs, axis=0 if is_time_major else 1)

		return self._create_projection_layer(dec_cell_outputs, num_classes)

	def _get_decoder_output_for_inference(self, dec_cell, initial_cell_state, batch_size, num_time_steps, num_classes, is_time_major):
		dec_cell_state = initial_cell_state
		dec_cell_input = tf.fill(tf.concat((batch_size, tf.constant([num_classes])), axis=-1), float(self._start_token))  # Initial input.
		projection_outputs = []
		for _ in range(num_time_steps):
			dec_cell_output, dec_cell_state = dec_cell(dec_cell_input, dec_cell_state, scope='dec')

			#dec_cell_output = tf.reshape(dec_cell_output, [None, 1, num_dec_hidden_units])
			dec_cell_input = self._create_projection_layer(dec_cell_output, num_classes)
			projection_outputs.append(dec_cell_input)

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		return tf.stack(projection_outputs, axis=0 if is_time_major else 1)

	def _create_unit_cell(self, num_units, name):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, name=name)
