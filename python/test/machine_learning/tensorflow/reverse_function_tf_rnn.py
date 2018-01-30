import tensorflow as tf
from tensorflow.contrib import rnn
from reverse_function_rnn import ReverseFunctionRNN

#%%------------------------------------------------------------------

class ReverseFunctionTensorFlowRNN(ReverseFunctionRNN):
	def __init__(self, input_shape, output_shape, model_type=0, is_dynamic=True):
		self._model_type = model_type
		self._is_dynamic = is_dynamic
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		num_time_steps, num_classes = input_shape[0], output_shape[-1]
		with tf.variable_scope('reverse_function_tf_rnn', reuse=tf.AUTO_REUSE):
			if self._is_dynamic:
				if 0 == self._model_type:
					return self._create_dynamic_rnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				elif 1 == self._model_type:
					return self._create_dynamic_stacked_rnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				elif 2 == self._model_type:
					return self._create_dynamic_birnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				elif 3 == self._model_type:
					return self._create_dynamic_stacked_birnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				else:
					assert False, 'Invalid model type.'
					return None
			else:
				if 0 == self._model_type:
					return self._create_static_rnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				elif 1 == self._model_type:
					return self._create_static_stacked_rnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				elif 2 == self._model_type:
					return self._create_static_birnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				elif 3 == self._model_type:
					return self._create_static_stacked_birnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
				else:
					assert False, 'Invalid model type.'
					return None

	def _create_dynamic_rnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 256
		dropout_rate = 0.5

		# Defines a cell.
		cell = self._create_cell(num_hidden_units)

		# Gets cell outputs.
		#cell_outputs, cell_state = tf.nn.dynamic_rnn(cell, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.dynamic_rnn(cell, input_tensor, dtype=tf.float32)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_dynamic_stacked_rnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 128
		num_layers = 2
		dropout_rate = 0.5

		# Defines a cell.
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		stacked_cell = rnn.MultiRNNCell([self._create_cell(num_hidden_units) for _ in range(num_layers)])

		# Gets cell outputs.
		#cell_outputs, cell_state = tf.nn.dynamic_rnn(stacked_cell, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.dynamic_rnn(stacked_cell, input_tensor, dtype=tf.float32)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_dynamic_birnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 128
		dropout_rate = 0.5

		# Defines a cell.
		cell_fw = self._create_cell(num_hidden_units)  # Forward cell.
		cell_bw = self._create_cell(num_hidden_units)  # Backward cell.

		# Gets cell outputs.
		#cell_outputs, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)
		cell_outputs = tf.concat(cell_outputs, 2)
		#cell_states = tf.concat(cell_states, 2)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_dynamic_stacked_birnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 64
		num_layers = 2
		dropout_rate = 0.5

		# Defines a cell.
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		stacked_cell_fw = rnn.MultiRNNCell([self._create_cell(num_hidden_units) for _ in range(num_layers)])  # Forward cell.
		stacked_cell_bw = rnn.MultiRNNCell([self._create_cell(num_hidden_units) for _ in range(num_layers)])  # Backward cell.

		# Gets cell outputs.
		#cell_outputs, cell_states = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)
		cell_outputs = tf.concat(cell_outputs, 2)
		#cell_states = tf.concat(cell_states, 2)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_static_rnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 256
		dropout_rate = 0.5

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		cell = self._create_cell(num_hidden_units)

		# Gets cell outputs.
		"""
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		#cell_state = cell.zero_state(batch_size, tf.float32)
		cell_state = tf.zeros([batch_size, cell.state_size])
		cell_output_list = []
		probabilities = []
		loss = 0.0
		for i in range(num_time_steps):
			#cell_output, cell_state = cell(input_tensor[:, i], cell_state)
			cell_outputs, _ = cell(input_tensor[:, i], cell_state)
			cell_output_list.append(cell_outputs)

			#logits = tf.matmul(cell_output, weights) + biases
			# TODO [check] >>
			logits = tf.layers.dense(cell_output, 1024, activation=tf.nn.softmax, name='fc')
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			logits = tf.layers.dropout(logits, rate=dropout_rate, training=is_training_tensor, name='dropout')

			probabilities.append(tf.nn.softmax(logits))
			loss += loss_function(probabilities, output_tensor[:, i])
		"""
		#cell_outputs, cell_state = tf.nn.static_rnn(cell, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.static_rnn(cell, input_tensor, dtype=tf.float32)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_static_stacked_rnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 128
		num_layers = 2
		dropout_rate = 0.5

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		"""
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		# Defines a cell.
		stacked_cells = [self._create_cell(num_hidden_units) for _ in range(num_layers)]

		# Gets cell outputs.
		def run_stacked_cells(cells, cell_inputs, cell_state_list):
			cell_outputs = cell_inputs
			new_cell_state_list = []
			for (cell, cell_state) in zip(cells, cell_state_list):
				cell_outputs, cell_state = cell(cell_outputs, cell_state)
				new_cell_state_list.append(cell_state)
			return cell_outputs, new_cell_state_list

		cell_state_list = tf.zeros([[batch_size, cell.state_size] for cell in cells])
		cell_output_list = []
		probabilities = []
		loss = 0.0
		for i in range(num_time_steps):
			cell_output, cell_state_list = run_stacked_cells(stacked_cells, input_tensor[:, i], cell_state_list)
			cell_output_list.append(cell_output)

			#logits = tf.matmul(cell_output, weights) + biases
			# TODO [check] >>
			logits = tf.layers.dense(cell_output, 1024, activation=tf.nn.softmax, name='fc')
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			logits = tf.layers.dropout(logits, rate=dropout_rate, training=is_training_tensor, name='dropout')

			probabilities.append(tf.nn.softmax(logits))
			loss += loss_function(probabilities, output_tensor[:, i])
		"""
		# Defines a cell.
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		stacked_cell = rnn.MultiRNNCell([self._create_cell(num_hidden_units) for _ in range(num_layers)])

		# Gets cell outputs.
		#cell_outputs, cell_state = tf.nn.static_rnn(stacked_cell, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.static_rnn(stacked_cell, input_tensor, dtype=tf.float32)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_static_birnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 128
		dropout_rate = 0.5

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		cell_fw = self._create_cell(num_hidden_units)  # Forward cell.
		cell_bw = self._create_cell(num_hidden_units)  # Backward cell.

		# Gets cell outputs.
		#cell_outputs, cell_state_fw, cell_state_bw = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)
		#cell_states = tf.concat((cell_state_fw, cell_state_bw), 2)  # ?
		cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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

	def _create_static_stacked_birnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 64
		num_layers = 2
		dropout_rate = 0.5

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		stacked_cell_fw = rnn.MultiRNNCell([self._create_cell(num_hidden_units) for _ in range(num_layers)])  # Forward cell.
		stacked_cell_bw = rnn.MultiRNNCell([self._create_cell(num_hidden_units) for _ in range(num_layers)])  # Backward cell.

		# Gets cell outputs.
		#cell_outputs, cell_state_fw, cell_state_bw = tf.nn.static_bidirectional_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)
		#cell_states = tf.concat((cell_state_fw, cell_state_bw), 2)  # ?
		cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)

		with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
			# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
			cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training_tensor, name='dropout')

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
