import tensorflow as tf
from keras.layers import LSTM, Dense, Bidirectional
from reverse_function_rnn import ReverseFunctionRNN

#%%------------------------------------------------------------------

class ReverseFunctionKerasRNN(ReverseFunctionRNN):
	def __init__(self, input_shape, output_shape, model_type=0):
		self._model_type = model_type
		super().__init__(input_shape, output_shape)

	def _create_model(self, input_tensor, is_training_tensor, input_shape, output_shape):
		# Note [info] >> Because is_training_tensor is a TensorFlow tensor, it can not be used as an argument in Keras.
		#	In Keras, K.set_learning_phase(1) or K.set_learning_phase(0) has to be used to set the learning phase, 'train' or 'test' before defining a model.
		#		K.set_learning_phase(1)  # Set the learning phase to 'train'.
		#		K.set_learning_phase(0)  # Set the learning phase to 'test'.
		#dropout_rate = 0.75 if True == is_training_tensor else 0.0  # Error: Not working.
		#dropout_rate = tf.cond(tf.equal(is_training_tensor, tf.constant(True)), lambda: tf.constant(0.75), lambda: tf.constant(0.0))  # Error: Not working.
		dropout_rate = 0.25

		num_classes = output_shape[-1]
		with tf.variable_scope('reverse_function_keras_rnn', reuse=tf.AUTO_REUSE):
			if 0 == self._model_type:
				return self._create_rnn(input_tensor, num_classes, dropout_rate)
			elif 1 == self._model_type:
				return self._create_stacked_rnn(input_tensor, num_classes, dropout_rate)
			elif 2 == self._model_type:
				return self._create_birnn(input_tensor, num_classes, dropout_rate)
			elif 3 == self._model_type:
				return self._create_stacked_birnn(input_tensor, num_classes, dropout_rate)
			else:
				assert False, 'Invalid model type.'
				return None

	def _create_rnn(self, input_tensor, num_classes, dropout_rate):
		num_hidden_units = 256
		dropout_rate = 0.5

		# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
		x = LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(input_tensor)
		# Output shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
		if 1 == num_classes:
			x = Dense(1, activation='sigmoid')(x)
			#x = Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		elif num_classes >= 2:
			x = Dense(num_classes, activation='softmax')(x)
			#x = Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		else:
			assert num_classes > 0, 'Invalid number of classes.'

		return x

	def _create_stacked_rnn(self, input_tensor, num_classes, dropout_rate):
		num_hidden_units = 128
		dropout_rate = 0.5

		# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
		x = LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(input_tensor)
		x = LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(x)
		# Output shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
		if 1 == num_classes:
			x = Dense(1, activation='sigmoid')(x)
			#x = Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		elif num_classes >= 2:
			x = Dense(num_classes, activation='softmax')(x)
			#x = Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		else:
			assert num_classes > 0, 'Invalid number of classes.'

		return x

	def _create_birnn(self, input_tensor, num_classes, dropout_rate):
		num_hidden_units = 256
		dropout_rate = 0.5

		# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
		# FIXME [check] >>.
		# Output shape = (None, num_hidden_units * 2).
		#x = Bidirectional(LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))(input_tensor)
		# Output shape = (None, None, num_hidden_units * 2).
		x = Bidirectional(LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(input_tensor)
		if 1 == num_classes:
			x = Dense(1, activation='sigmoid')(x)
			#x = Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		elif num_classes >= 2:
			x = Dense(num_classes, activation='softmax')(x)
			#x = Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		else:
			assert num_classes > 0, 'Invalid number of classes.'

		return x

	def _create_stacked_birnn(self, input_tensor, num_classes, dropout_rate):
		num_hidden_units = 128
		dropout_rate = 0.5

		# Input shape = (samples, time-steps, features) = (None, None, VOCAB_SIZE).
		x = Bidirectional(LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(input_tensor)
		# FIXME [check] >> I don't know why.
		# Output shape = (None, num_hidden_units * 2).
		#x = Bidirectional(LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
		# Output shape = (None, None, num_hidden_units * 2).
		x = Bidirectional(LSTM(num_hidden_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(x)
		if 1 == num_classes:
			x = Dense(1, activation='sigmoid')(x)
			#x = Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		elif num_classes >= 2:
			x = Dense(num_classes, activation='softmax')(x)
			#x = Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001))(x)
		else:
			assert num_classes > 0, 'Invalid number of classes.'

		return x
