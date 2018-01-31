import tensorflow as tf
from reverse_function_rnn import ReverseFunctionRNN

#%%------------------------------------------------------------------

class ReverseFunctionTensorFlowEncoderDecoder(ReverseFunctionRNN):
	def __init__(self, input_shape, output_shape, is_dynamic=True, is_bidirectional=True):
		self._is_dynamic = is_dynamic
		self._is_bidirectional = is_bidirectional
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		with tf.variable_scope('reverse_function_tf_encdec', reuse=tf.AUTO_REUSE):
			if self._is_dynamic:
				num_classes = output_shape[-1]
				if self._is_bidirectional:
					return self._create_dynamic_bidirectional_model(input_tensor, is_training_tensor, num_classes)
				else:
					return self._create_dynamic_model(input_tensor, is_training_tensor, num_classes)
			else:
				num_time_steps, num_classes = input_shape[0], output_shape[-1]
				if self._is_bidirectional:
					return self._create_static_bidirectional_model(input_tensor, is_training_tensor, num_time_steps, num_classes)
				else:
					return self._create_static_model(input_tensor, is_training_tensor, num_time_steps, num_classes)

	def _create_dynamic_model(self, input_tensor, is_training_tensor, num_classes):
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 0.75

		# Defines cells.
		enc_cell = self._create_cell(num_enc_hidden_units)
		enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_cell(num_dec_hidden_units)
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Encoder.
		#enc_cell_outputs, enc_cell_state = tf.nn.dynamic_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _ = tf.nn.dynamic_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')

		# Uses the last output of the encoder only.
		# TODO [enhance] >> The dimension of tensors is fixed as 3.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[:,-1,:] is used?
		#enc_cell_outputs = tf.tile(enc_cell_outputs[:,-1,:], [1, num_time_steps, 1])
		enc_cell_output_shape = tf.shape(enc_cell_outputs)
		enc_cell_outputs = tf.tile(tf.reshape(enc_cell_outputs[:,-1,:], [-1, 1, num_enc_hidden_units]), [1, enc_cell_output_shape[1], 1])

		# Decoder.
		#cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
		cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')

		#with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_dynamic_bidirectional_model(self, input_tensor, is_training_tensor, num_classes):
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 0.75

		# Defines cells.
		enc_cell_fw = self._create_cell(num_enc_hidden_units)  # Forward cell.
		enc_cell_fw = tf.contrib.rnn.DropoutWrapper(enc_cell_fw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		enc_cell_bw = self._create_cell(num_enc_hidden_units)  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_cell(num_dec_hidden_units)
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Encoder.
		#enc_cell_outputs, cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32)
		enc_cell_outputs, _ = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32)
		enc_cell_outputs = tf.concat(enc_cell_outputs, 2)
		#enc_cell_states = tf.concat(enc_cell_states, 2)

		# Uses the last output of the encoder only.
		# TODO [enhance] >> The dimension of tensors is fixed as 3.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[:,-1,:] is used?
		#enc_cell_outputs = tf.tile(enc_cell_outputs[:,-1,:], [1, num_time_steps, 1])
		enc_cell_output_shape = tf.shape(enc_cell_outputs)
		enc_cell_outputs = tf.tile(tf.reshape(enc_cell_outputs[:,-1,:], [-1, 1, num_enc_hidden_units * 2]), [1, enc_cell_output_shape[1], 1])

		# Decoder.
		#cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
		cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')

		#with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_static_model(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		keep_prob = 0.75

		# Defines cells.
		enc_cell = self._create_cell(num_enc_hidden_units)
		enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_cell(num_dec_hidden_units)
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Encoder.
		#enc_cell_outputs, enc_cell_state = tf.nn.static_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _ = tf.nn.static_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')

		# Decoder.
		if True:
			# Uses the last output of the encoder only.
			# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[-1] is used?
			enc_cell_outputs = [enc_cell_outputs[-1]] * num_time_steps
	
			# Decoder.
			#dec_cell_outputs, dec_cell_state = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
			dec_cell_outputs, _ = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
		else:
			# When using the last output of the encoder (context) and the previous output of the decoder together.
			input_shape = tf.shape(input_tensor)
			batch_size = input_shape[0]

			context = enc_cell_outputs[-1]
			dec_cell_state = dec_cell.zero_state(batch_size, tf.float32)
			# FIXME [fix] >>
			dec_cell_output = None  # The start word.
			dec_cell_outputs = []
			for _ in range(num_time_steps):
				dec_cell_output, dec_cell_state = tf.nn.static_rnn(dec_cell, [context, dec_cell_output], initial_state=dec_cell_state, dtype=tf.float32, scope='dec')
				dec_cell_outputs.append(dec_cell_output)

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		cell_outputs = tf.stack(dec_cell_outputs, axis=1)

		#with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_static_bidirectional_model(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_enc_hidden_units = 64
		num_dec_hidden_units = 128
		keep_prob = 0.75

		# Defines cells.
		enc_cell_fw = self._create_cell(num_enc_hidden_units)  # Forward cell.
		enc_cell_fw = tf.contrib.rnn.DropoutWrapper(enc_cell_fw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		enc_cell_bw = self._create_cell(num_enc_hidden_units)  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		dec_cell = self._create_cell(num_dec_hidden_units)
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Encoder.
		#enc_cell_outputs, enc_cell_state_fw, enc_cell_state_bw = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32, scope='enc')
		#enc_cell_states = tf.concat((enc_cell_state_fw, enc_cell_state_bw), 2)  # ?
		enc_cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32, scope='enc')

		# Uses the last output of the encoder only.
		# TODO [check] >> Is it correct that the last output of the encoder enc_cell_outputs[-1] is used?
		enc_cell_outputs = [enc_cell_outputs[-1]] * num_time_steps

		# Decoder.
		#dec_cell_outputs, dec_cell_state = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')
		dec_cell_outputs, _ = tf.nn.static_rnn(dec_cell, enc_cell_outputs, dtype=tf.float32, scope='dec')

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		cell_outputs = tf.stack(dec_cell_outputs, axis=1)

		#with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
		#	dropout_rate = 1 - keep_prob
		#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
		#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_cell(self, num_units):
		#return tf.contrib.rnn.BasicRNNCell(num_units, forget_bias=1.0)
		return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.RNNCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.LSTMCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.GRUCell(num_units, forget_bias=1.0)
