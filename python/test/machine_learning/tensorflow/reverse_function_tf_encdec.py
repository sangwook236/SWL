import tensorflow as tf
from tensorflow.contrib import rnn
from reverse_function_rnn import ReverseFunctionRNN

#%%------------------------------------------------------------------

class ReverseFunctionTensorFlowEncoderDecoder(ReverseFunctionRNN):
	def __init__(self, input_shape, output_shape, is_bidirectional=True, is_dynamic=True):
		if is_dynamic:
			raise Exception('Not supported')
		self._is_bidirectional = is_bidirectional
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		num_time_steps, num_classes = input_shape[0], output_shape[-1]
		with tf.variable_scope('reverse_function_tf_encdec', reuse=tf.AUTO_REUSE):
			if self._is_bidirectional:
				return self._create_static_bidirectional_model(input_tensor, is_training_tensor, num_time_steps, num_classes)
			else:
				return self._create_static_model(input_tensor, is_training_tensor, num_time_steps, num_classes)

	def _create_static_model(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_enc_hidden_units = 128
		num_dec_hidden_units = 128
		dropout_rate = 0.5

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		enc_cell = self._create_cell(num_enc_hidden_units)
		dec_cell = self._create_cell(num_dec_hidden_units)

		# Gets cell outputs.
		#enc_cell_outputs, enc_cell_state = tf.nn.static_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')
		enc_cell_outputs, _ = tf.nn.static_rnn(enc_cell, input_tensor, dtype=tf.float32, scope='enc')

		dec_inputs = [enc_cell_outputs[-1]] * num_time_steps
		#dec_cell_outputs, dec_cell_state = tf.nn.static_rnn(dec_cell, dec_inputs, dtype=tf.float32, scope='dec')
		dec_cell_outputs, _ = tf.nn.static_rnn(dec_cell, dec_inputs, dtype=tf.float32, scope='dec')

		with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(dec_cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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
		dropout_rate = 0.5

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		enc_cell_fw = self._create_cell(num_enc_hidden_units)  # Forward cell.
		enc_cell_bw = self._create_cell(num_enc_hidden_units)  # Backward cell.
		dec_cell = self._create_cell(num_dec_hidden_units)

		# Gets cell outputs.
		#enc_cell_outputs, enc_cell_state_fw, enc_cell_state_bw = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32, scope='enc')
		#enc_cell_states = tf.concat((enc_cell_state_fw, enc_cell_state_bw), 2)  # ?
		enc_cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(enc_cell_fw, enc_cell_bw, input_tensor, dtype=tf.float32, scope='enc')

		# Gets cell outputs.
		dec_inputs = [enc_cell_outputs[-1]] * num_time_steps
		#dec_cell_outputs, dec_cell_state = tf.nn.static_rnn(dec_cell, dec_inputs, dtype=tf.float32, scope='dec')
		dec_cell_outputs, _ = tf.nn.static_rnn(dec_cell, dec_inputs, dtype=tf.float32, scope='dec')

		with tf.variable_scope('enc-dec', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(dec_cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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
		#return rnn.BasicRNNCell(num_units, forget_bias=1.0)
		return rnn.BasicLSTMCell(num_units, forget_bias=1.0)
		#return rnn.RNNCell(num_units, forget_bias=1.0)
		#return rnn.LSTMCell(num_units, forget_bias=1.0)
		#return rnn.GRUCell(num_units, forget_bias=1.0)
