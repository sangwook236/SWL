import tensorflow as tf
from tf.keras.layers import LSTM, Dense, Bidirectional
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet

#--------------------------------------------------------------------

class SimpleRnnUsingKeras(SimpleNeuralNet):
	def __init__(self, input_shape, output_shape, is_bidirectional=True, is_stacked=True):
		super().__init__(input_shape, output_shape)

		self._is_bidirectional = is_bidirectional
		self._is_stacked = is_stacked

	def _create_single_model(self, input_tensor, input_shape, output_shape, is_training):
		# In Keras, K.set_learning_phase(1) or K.set_learning_phase(0) has to be used to set the learning phase, 'train' or 'test' before defining a model.
		#	K.set_learning_phase(1)  # Set the learning phase to 'train'.
		#	K.set_learning_phase(0)  # Set the learning phase to 'test'.
		dropout_rate = 0.5 if is_training else 0.0

		num_classes = output_shape[-1]
		with tf.variable_scope('simple_rnn_using_keras', reuse=tf.AUTO_REUSE):
			if self._is_bidirectional:
				if self._is_stacked:
					return self._create_stacked_birnn(input_tensor, num_classes, dropout_rate)
				else:
					return self._create_birnn(input_tensor, num_classes, dropout_rate)
			else:
				if self._is_stacked:
					return self._create_stacked_rnn(input_tensor, num_classes, dropout_rate)
				else:
					return self._create_rnn(input_tensor, num_classes, dropout_rate)

	def _create_rnn(self, input_tensor, num_classes, dropout_rate):
		num_hidden_units = 512

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
		num_hidden_units = 256

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
