import tensorflow as tf
from tensorflow.contrib import rnn
from reverse_function_rnn import ReverseFunctionRNN

#%%------------------------------------------------------------------

class ReverseFunctionTensorFlowRNN(ReverseFunctionRNN):
	def __init__(self, input_shape, output_shape, model_type=0):
		self._model_type = model_type
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		num_time_steps, num_classes = input_shape[0], output_shape[-1]
		with tf.variable_scope('reverse_function_tf_rnn', reuse=tf.AUTO_REUSE):
			if 0 == self._model_type:
				return self._create_rnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
			elif 1 == self._model_type:
				return self._create_stacked_rnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
			elif 2 == self._model_type:
				return self._create_birnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
			elif 3 == self._model_type:
				return self._create_stacked_birnn(input_tensor, is_training_tensor, num_time_steps, num_classes)
			else:
				assert False, 'Invalid model type.'
				return None

	def _create_rnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 256
		# FIXME [implement] >> Add dropout layers.
		dropout_rate = 0.5

		# For tf.nn.static_rnn().
		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		#input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		cell = rnn.BasicLSTMCell(num_hidden_units, forget_bias=1.0)

		# Gets cell outputs.
		#cell_outputs, cell_state = tf.nn.static_rnn(cell, input_tensor, dtype=tf.float32)
		#cell_outputs, _ = tf.nn.static_rnn(cell, input_tensor, dtype=tf.float32)
		#cell_outputs, cell_state = tf.nn.dynamic_rnn(cell, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.dynamic_rnn(cell, input_tensor, dtype=tf.float32)

		lstm = cell_outputs[-1]  # Uses the final output only.
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, name='fc')
				#fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, name='fc')
				#fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', fc1.get_shape().as_list())
			return fc1

	def _create_stacked_rnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		def create_cell(num_units):
			#return rnn.BasicRNNCell(num_units, forget_bias=1.0)
			return rnn.BasicLSTMCell(num_units, forget_bias=1.0)
			#return rnn.RNNCell(num_units, forget_bias=1.0)
			#return rnn.LSTMCell(num_units, forget_bias=1.0)
			#return rnn.GRUCell(num_units, forget_bias=1.0)

		num_hidden_units = 128
		num_layers = 2
		# FIXME [implement] >> Add dropout layers.
		dropout_rate = 0.5

		# For tf.nn.static_rnn().
		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		#input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		stacked_cell = rnn.MultipleRNNCell([create_cell(num_hidden_units) for _ in range(num_layers)])

		# Gets cell outputs.
		#cell_outputs, cell_state = tf.nn.static_rnn(stacked_cell, input_tensor, dtype=tf.float32)
		#cell_outputs, _ = tf.nn.static_rnn(stacked_cell, input_tensor, dtype=tf.float32)
		#cell_outputs, cell_state = tf.nn.dynamic_rnn(stacked_cell, input_tensor, dtype=tf.float32)
		cell_outputs, _ = tf.nn.dynamic_rnn(stacked_cell, input_tensor, dtype=tf.float32)

		lstm = cell_outputs[-1]  # Uses the final output only.
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, name='fc')
				#fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, name='fc')
				#fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc1

	def _create_birnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		num_hidden_units = 256
		# FIXME [implement] >> Add dropout layers.
		dropout_rate = 0.5

		# For tf.nn.static_rnn().
		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		#input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		cell_fw = rnn.BasicLSTMCell(num_hidden_units, forget_bias=1.0)  # Forward cell.
		cell_bw = rnn.BasicLSTMCell(num_hidden_units, forget_bias=1.0)  # Backward cell.

		# Gets cell outputs.
		#cell_outputs, cell_state_fw, cell_state_bw = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)
		#cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)
		#cell_outputs, cell_state_fw, cell_state_bw = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)
		cell_outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_tensor, dtype=tf.float32)

		lstm = cell_outputs[-1]  # Uses the final output only.
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, name='fc')
				#fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, name='fc')
				#fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc1

	def _create_stacked_birnn(self, input_tensor, is_training_tensor, num_time_steps, num_classes):
		def create_cell(num_units):
			#return rnn.BasicRNNCell(num_units, forget_bias=1.0)
			return rnn.BasicLSTMCell(num_units, forget_bias=1.0)
			#return rnn.RNNCell(num_units, forget_bias=1.0)
			#return rnn.LSTMCell(num_units, forget_bias=1.0)
			#return rnn.GRUCell(num_units, forget_bias=1.0)

		num_hidden_units = 128
		num_layers = 2
		# FIXME [implement] >> Add dropout layers.
		dropout_rate = 0.5

		# For tf.nn.static_rnn().
		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		#input_tensor = tf.unstack(input_tensor, num_time_steps, axis=1)

		# Defines a cell.
		# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
		stacked_cell_fw = rnn.MultipleRNNCell([create_cell(num_hidden_units) for _ in range(num_layers)])  # Forward cell.
		stacked_cell_bw = rnn.MultipleRNNCell([create_cell(num_hidden_units) for _ in range(num_layers)])  # Backward cell.

		# Gets cell outputs.
		#cell_outputs, cell_state_fw, cell_state_bw = tf.nn.static_bidirectional_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)
		#cell_outputs, _, _ = tf.nn.static_bidirectional_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)
		#cell_outputs, cell_state_fw, cell_state_bw = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)
		cell_outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw, stacked_cell_bw, input_tensor, dtype=tf.float32)

		lstm = cell_outputs[-1]  # Uses the final output only.
		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, name='fc')
				#fc1 = tf.layers.dense(lstm, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			elif num_classes >= 2:
				fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, name='fc')
				#fc1 = tf.layers.dense(lstm, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
			else:
				assert num_classes > 0, 'Invalid number of classes.'

			return fc1
