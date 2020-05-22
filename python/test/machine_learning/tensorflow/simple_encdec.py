import tensorflow as tf
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet

#--------------------------------------------------------------------

class SimpleEncoderDecoder(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape, is_dynamic=True, is_bidirectional=True, is_time_major=False):
		super().__init__(input_shape, output_shape)

		self._is_dynamic = is_dynamic
		self._is_bidirectional = is_bidirectional
		self._is_time_major = is_time_major

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		with tf.variable_scope('simple_encdec', reuse=tf.AUTO_REUSE):
			if self._is_dynamic:
				num_classes = output_shape[-1]
				if self._is_bidirectional:
					return self._create_dynamic_bidirectional_model(input_tensor, is_training, num_classes, self._is_time_major)
				else:
					return self._create_dynamic_model(input_tensor, is_training, num_classes, self._is_time_major)
			else:
				if self._is_time_major:
					num_time_steps, num_classes = input_shape[0], output_shape[-1]
				else:
					num_time_steps, num_classes = input_shape[1], output_shape[-1]
				if self._is_bidirectional:
					return self._create_static_bidirectional_model(input_tensor, is_training, num_time_steps, num_classes, self._is_time_major)
				else:
					return self._create_static_model(input_tensor, is_training, num_time_steps, num_classes, self._is_time_major)

	def _create_dynamic_model(self, input_tensor, is_training, num_classes, is_time_major):
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 256
		num_dec_hidden_units = 256
		keep_prob = 0.5

		# Defines cells.
		enc_cell = self._create_unit_cell(num_enc_hidden_units, 'enc_unit_cell')
		enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_unit_cell(num_dec_hidden_units, 'dec_unit_cell')
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Encoder.
		#enc_cell_outputs, enc_cell_state = tf.nn.dynamic_rnn(enc_cell, input_tensor, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _ = tf.nn.dynamic_rnn(enc_cell, input_tensor, time_major=is_time_major, dtype=tf.float32, scope='enc')

		# Uses the last output of the encoder only.
		# TODO [enhance] >> The dimension of tensors is fixed as 3.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[:,-1,:] is used?
		#enc_cell_outputs = tf.tile(enc_cell_outputs[:,-1,:], [1, num_time_steps, 1])
		enc_cell_output_shape = tf.shape(enc_cell_outputs)
		enc_cell_outputs = tf.tile(tf.reshape(enc_cell_outputs[:,-1,:], [-1, 1, num_enc_hidden_units]), [1, enc_cell_output_shape[1], 1])

		# Decoder.
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, time_major=is_time_major, dtype=tf.float32, scope='dec')
		cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, time_major=is_time_major, dtype=tf.float32, scope='dec')

		#with tf.variable_scope('enc_dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		return self._create_projection_layer(cell_outputs, num_classes)

	def _create_dynamic_bidirectional_model(self, input_tensor, is_training, num_classes, is_time_major):
		"""
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 1.0
		"""
		num_enc_hidden_units = 128
		num_dec_hidden_units = 256
		keep_prob = 0.5

		# Defines cells.
		enc_cell_fw = self._create_unit_cell(num_enc_hidden_units, 'enc_fw_unit_cell')  # Forward cell.
		enc_cell_fw = tf.contrib.rnn.DropoutWrapper(enc_cell_fw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		enc_cell_bw = self._create_unit_cell(num_enc_hidden_units, 'enc_bw_unit_cell')  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_unit_cell(num_dec_hidden_units, 'dec_unit_cell')
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Encoder.
		#enc_cell_outputs, enc_cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _ = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs = tf.concat(enc_cell_outputs, axis=-1)
		#enc_cell_states = tf.contrib.rnn.LSTMStateTuple(tf.concat((enc_cell_states[0].c, enc_cell_states[1].c), axis=-1), tf.concat((enc_cell_states[0].h, enc_cell_states[1].h), axis=-1))

		# Uses the last output of the encoder only.
		# TODO [enhance] >> The dimension of tensors is fixed as 3.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[:,-1,:] is used?
		#enc_cell_outputs = tf.tile(enc_cell_outputs[:,-1,:], [1, num_time_steps, 1])
		enc_cell_output_shape = tf.shape(enc_cell_outputs)
		enc_cell_outputs = tf.tile(tf.reshape(enc_cell_outputs[:,-1,:], [-1, 1, num_enc_hidden_units * 2]), [1, enc_cell_output_shape[1], 1])

		# Decoder.
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, time_major=is_time_major, dtype=tf.float32, scope='dec')
		cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, time_major=is_time_major, dtype=tf.float32, scope='dec')

		#with tf.variable_scope('enc_dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		return self._create_projection_layer(cell_outputs, num_classes)

	def _create_static_model(self, input_tensor, is_training, num_time_steps, num_classes, is_time_major):
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
		dec_cell = self._create_unit_cell(num_dec_hidden_units, 'dec_unit_cell')
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=0 if is_time_major else 1)

		# Encoder.
		#enc_cell_outputs, enc_cell_state = tf.nn.static_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _ = tf.nn.static_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')

		# Uses the last output of the encoder only.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[-1] is used?
		enc_cell_outputs = [enc_cell_outputs[-1]] * num_time_steps

		# Decoder.
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#dec_cell_outputs, dec_cell_state = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
		dec_cell_outputs, _ = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		cell_outputs = tf.stack(dec_cell_outputs, axis=0 if is_time_major else 1)

		#with tf.variable_scope('enc_dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		return self._create_projection_layer(cell_outputs, num_classes)

	def _create_static_bidirectional_model(self, input_tensor, is_training, num_time_steps, num_classes, is_time_major):
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
		enc_cell_bw = self._create_unit_cell(num_enc_hidden_units, 'end_bw_unit_cell')  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_unit_cell(num_dec_hidden_units, 'dec_unit_cell')
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=0 if is_time_major else 1)

		# Encoder.
		#enc_cell_outputs, enc_cell_state_fw, enc_cell_state_bw = tf.nn.static_bidirectional_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32, scope='enc')
		#enc_cell_outputs = tf.concat(enc_cell_outputs, axis=-1)  # Don't need.
		#enc_cell_states = tf.contrib.rnn.LSTMStateTuple(tf.concat((enc_cell_state_fw.c, enc_cell_state_bw.c), axis=-1), tf.concat((enc_cell_state_fw.h, enc_cell_state_bw.h), axis=-1))

		# Uses the last output of the encoder only.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[-1] is used?
		enc_cell_outputs = [enc_cell_outputs[-1]] * num_time_steps

		# Decoder.
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#dec_cell_outputs, dec_cell_state = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
		dec_cell_outputs, _ = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		cell_outputs = tf.stack(dec_cell_outputs, axis=0 if is_time_major else 1)

		#with tf.variable_scope('enc_dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

		return self._create_projection_layer(cell_outputs, num_classes)

	def _create_projection_layer(self, cell_outputs, num_classes):
		with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				return tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, name='dense')
				#return tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			elif num_classes >= 2:
				return tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, name='dense')
				#return tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			else:
				assert num_classes > 0, 'Invalid number of classes.'
				return None

	def _create_unit_cell(self, num_units, name):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, name=name)
