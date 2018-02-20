import numpy as np
import tensorflow as tf
from simple_neural_net import SimpleSeq2SeqNeuralNet

#%%------------------------------------------------------------------

class SimpleSeq2SeqEncoderDecoder(SimpleSeq2SeqNeuralNet):
	def __init__(self, encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_bidirectional=True, is_time_major=False):
		self._encoder_input_seq_lens_ph = tf.placeholder(tf.int32, [None], name='encoder_input_seq_lens_ph')
		self._decoder_output_seq_lens_ph = tf.placeholder(tf.int32, [None], name='decoder_output_seq_lens_ph')
		self._batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		self._start_token = start_token
		self._end_token = end_token

		self._is_bidirectional = is_bidirectional
		self._is_time_major = is_time_major
		super().__init__(encoder_input_shape, decoder_input_shape, decoder_output_shape)

	def get_feed_dict(self, encoder_inputs, decoder_inputs=None, decoder_outputs=None, is_training=True, **kwargs):
		#encoder_input_seq_lens = tf.constant(max_time_steps, tf.int32, shape=[batch_size])
		#decoder_output_seq_lens = tf.constant(max_time_steps, tf.int32, shape=[batch_size])
		if self._is_time_major:
			encoder_input_seq_lens = np.full(encoder_inputs.shape[1], encoder_inputs.shape[0], np.int32)
			if decoder_inputs is None or decoder_outputs is None:
				decoder_output_seq_lens = np.full(encoder_inputs.shape[1], encoder_inputs.shape[0], np.int32)
			else:
				decoder_output_seq_lens = np.full(decoder_outputs.shape[1], decoder_outputs.shape[0], np.int32)
			batch_size = [encoder_inputs.shape[1]]
		else:
			encoder_input_seq_lens = np.full(encoder_inputs.shape[0], encoder_inputs.shape[1], np.int32)
			if decoder_inputs is None or decoder_outputs is None:
				decoder_output_seq_lens = np.full(encoder_inputs.shape[0], encoder_inputs.shape[1], np.int32)
			else:
				decoder_output_seq_lens = np.full(decoder_outputs.shape[0], decoder_outputs.shape[1], np.int32)
			batch_size = [encoder_inputs.shape[0]]

		if decoder_inputs is None or decoder_outputs is None:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs, self._is_training_tensor_ph: is_training, self._encoder_input_seq_lens_ph: encoder_input_seq_lens, self._decoder_output_seq_lens_ph: decoder_output_seq_lens, self._batch_size_ph: batch_size}
		else:
			feed_dict = {self._encoder_input_tensor_ph: encoder_inputs, self._decoder_input_tensor_ph: decoder_inputs, self._decoder_output_tensor_ph: decoder_outputs, self._is_training_tensor_ph: is_training, self._encoder_input_seq_lens_ph: encoder_input_seq_lens, self._decoder_output_seq_lens_ph: decoder_output_seq_lens, self._batch_size_ph: batch_size}
		return feed_dict

	def _create_single_model(self, encoder_input_tensor, decoder_input_tensor, decoder_output_tensor, is_training, encoder_input_shape, decoder_input_shape, decoder_output_shape):
		with tf.variable_scope('simple_seq2seq_encdec', reuse=tf.AUTO_REUSE):
			# TODO [improve] >> It is not good to use num_time_steps.
			#num_classes = decoder_output_shape[-1]
			if self._is_time_major:
				num_time_steps, num_classes = decoder_output_shape[0], decoder_output_shape[-1]
			else:
				num_time_steps, num_classes = decoder_output_shape[1], decoder_output_shape[-1]
			if self._is_bidirectional:
				return self._create_dynamic_bidirectional_model(encoder_input_tensor, decoder_input_tensor, is_training, self._encoder_input_seq_lens_ph, self._batch_size_ph, num_time_steps, num_classes, self._is_time_major)
			else:
				return self._create_dynamic_model(encoder_input_tensor, decoder_input_tensor, is_training, self._encoder_input_seq_lens_ph, self._batch_size_ph, num_time_steps, num_classes, self._is_time_major)

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
			masks = tf.sequence_mask(self._decoder_output_seq_lens_ph, tf.reduce_max(self._decoder_output_seq_lens_ph), dtype=tf.float32)
			#loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=t, weights=masks)
			loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=tf.argmax(t, axis=-1), weights=masks)
			tf.summary.scalar('loss', loss)
			return loss

	def _create_dynamic_model(self, encoder_input_tensor, decoder_input_tensor, is_training, encoder_input_seq_lens, batch_size, num_time_steps, num_classes, is_time_major):
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 256
		num_dec_hidden_units = 256
		keep_prob = 0.5
		"""

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
		enc_cell_outputs, enc_cell_state = tf.nn.dynamic_rnn(enc_cell, encoder_input_tensor, sequence_length=encoder_input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')

		# Attention.
		# REF [function] >> SimpleTfSeq2SeqEncoderDecoderWithAttention._create_dynamic_model() in ./simple_tf_seq2seq_encdec_attention.py.

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('seq2seq_encdec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		# Decoder.
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, decoder_input_tensor, initial_state=enc_cell_state, time_major=is_time_major, dtype=tf.float32, scope='dec')
		cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, decoder_input_tensor, initial_state=enc_cell_state, time_major=is_time_major, dtype=tf.float32, scope='dec')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc1 = tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, name='fc')
				#fc1 = tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc1 = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, name='fc')
				#fc1 = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

		return fc1

	def _create_dynamic_bidirectional_model(self, encoder_input_tensor, decoder_input_tensor, is_training, encoder_input_seq_lens, batch_size, num_time_steps, num_classes, is_time_major):
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 256
		keep_prob = 0.5
		"""

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
		enc_cell_outputs, enc_cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, encoder_input_tensor, sequence_length=encoder_input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs = tf.concat(enc_cell_outputs, axis=-1)
		enc_cell_states = tf.contrib.rnn.LSTMStateTuple(tf.concat((enc_cell_states[0].c, enc_cell_states[1].c), axis=-1), tf.concat((enc_cell_states[0].h, enc_cell_states[1].h), axis=-1))

		# Attention.
		# REF [function] >> SimpleTfSeq2SeqEncoderDecoderWithAttention._create_dynamic_bidirectional_model() in ./simple_tf_seq2seq_encdec_attention.py.

		# FIXME [implement] >> How to add dropout?
		#with tf.variable_scope('seq2seq_encdec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		# Decoder.
		def get_training_decoder_outputs():
			# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
			#dec_cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, decoder_input_tensor, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')
			dec_cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, decoder_input_tensor, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')
			return dec_cell_outputs
		def get_testing_decoder_outputs():
			dec_cell_state = enc_cell_states  # Initial state.
			dec_cell_output = tf.fill(tf.concat((batch_size, tf.constant([num_dec_hidden_units])), axis=-1), float(self._start_token))  # Initial input.
			print('^^^^^^', dec_cell_output.shape.as_list(), dec_cell_state.c.shape.as_list(), dec_cell_state.h.shape.as_list())
			dec_cell_outputs = []
			for _ in range(num_time_steps):
				print('*******')
				dec_cell_output, dec_cell_state = dec_cell(dec_cell_output, dec_cell_state, scope='dec')
				print('*******11', dec_cell_output.shape.as_list(), dec_cell_state.c.shape.as_list(), dec_cell_state.h.shape.as_list())
				dec_cell_outputs.append(dec_cell_output)
			if is_time_major:
				dec_cell_outputs = tf.stack(dec_cell_outputs, axis=0)
			else:
				dec_cell_outputs = tf.stack(dec_cell_outputs, axis=1)
			return dec_cell_outputs
			"""
			print('***********2')
			dec_cell_state = enc_cell_states  # Initial state.
			dec_cell_output = tf.fill(tf.concat((batch_size, tf.constant([num_dec_hidden_units])), axis=0), float(self._start_token))  # Initial input.
			dec_cell_outputs = None
			max_time_steps = tf.reduce_max(encoder_input_seq_lens)
			def body(i, dec_cell_output, dec_cell_state, dec_cell_outputs):
				tf.add(i, 1)
				dec_cell_output, dec_cell_state = dec_cell(dec_cell_output, dec_cell_state, scope='dec')
				#dec_cell_outputs.append(dec_cell_output)
				if dec_cell_outputs is None:
					dec_cell_outputs = dec_cell_output
				else:
					dec_cell_outputs = tf.concat([dec_cell_outputs, dec_cell_output], axis=1)
				return [i+1, dec_cell_output, dec_cell_state, dec_cell_outputs]
			i = tf.constant(0)
			cond = lambda i, dec_cell_output, dec_cell_state, dec_cell_outputs: tf.less(i, max_time_steps)
			print('***********3')
			print('^^^^^^^^^^^^^^1', dec_cell_output.get_shape().as_list(), dec_cell_outputs.get_shape().as_list())
			r = tf.while_loop(cond, body, loop_vars=[i, dec_cell_output, dec_cell_state, dec_cell_outputs])
			print('***********4', len(dec_cell_outputs))
			return dec_cell_outputs
			"""
		#cell_outputs = get_training_decoder_outputs if is_training else get_testing_decoder_outputs)
		#dec_cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, decoder_input_tensor, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')
		cell_outputs = get_testing_decoder_outputs()

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc1 = tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, name='fc')
				#fc1 = tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc1 = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, name='fc')
				#fc1 = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

		return fc1

	def _create_unit_cell(self, num_units):
		#return tf.contrib.rnn.BasicRNNCell(num_units)
		#return tf.contrib.rnn.RNNCell(num_units)

		return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.LSTMCell(num_units, forget_bias=1.0)

		#return tf.contrib.rnn.GRUCell(num_units)
