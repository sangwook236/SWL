import abc
import numpy as np
import tensorflow as tf
from swl.machine_learning.tensorflow_model import SimpleSequentialTensorFlowModel
import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class HangeulCrnn(SimpleSequentialTensorFlowModel):
	def __init__(self, input_shape, output_shape, num_classes, is_sparse_output):
		super().__init__(input_shape, output_shape, num_classes, is_sparse_output, is_time_major=False)

		self._model_output_len = 0

	def get_feed_dict(self, data, num_data, *args, **kwargs):
		len_data = len(data)
		model_output_len = [self._model_output_len] * num_data
		if 1 == len_data:
			feed_dict = {self._input_ph: data[0], self._model_output_len_ph: model_output_len}
		elif 2 == len_data:
			"""
			feed_dict = {self._input_ph: data[0], self._output_ph: data[1], self._model_output_len_ph: model_output_len}
			"""
			# Use output lengths.
			output_len = list(map(lambda lbl: len(lbl), data[1]))
			feed_dict = {self._input_ph: data[0], self._output_ph: data[1], self._output_len_ph: output_len, self._model_output_len_ph: model_output_len}
		else:
			raise ValueError('Invalid number of feed data: {}'.format(len_data))
		return feed_dict

	def _create_single_model(self, inputs, input_shape, num_classes, is_training):
		with tf.variable_scope('hangeul_crnn', reuse=tf.AUTO_REUSE):
			return self._create_crnn(inputs, num_classes, is_training)

	def _create_crnn(self, inputs, num_classes, is_training):
		#kernel_initializer = None
		#kernel_initializer = tf.initializers.he_normal()
		#kernel_initializer = tf.initializers.glorot_normal()  # Xavier normal initialization.
		kernel_initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal')

		# Preprocessing.
		with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
			inputs = tf.nn.local_response_normalization(inputs, depth_radius=5, bias=1, alpha=1, beta=0.5, name='lrn')
			# (samples, height, width, channels) -> (samples, width, height, channels).
			inputs = tf.transpose(inputs, perm=(0, 2, 1, 3), name='transpose')

		#--------------------
		# Convolutional layer.

		# TODO [check] >> The magic number (64).
		num_cnn_features = 64

		with tf.variable_scope('convolutional_layer', reuse=tf.AUTO_REUSE):
			cnn_outputs = self._create_convolutional_layer(inputs, num_cnn_features, kernel_initializer, is_training)

		#--------------------
		# Recurrent layer.
		with tf.variable_scope('recurrent_layer', reuse=tf.AUTO_REUSE):
			rnn_outputs = self._create_recurrent_layer(cnn_outputs, self._model_output_len_ph, kernel_initializer, is_training)

		#--------------------
		# Transcription layer.
		with tf.variable_scope('transcription_layer', reuse=tf.AUTO_REUSE):
			logits = self._create_transcription_layer(rnn_outputs, num_classes, kernel_initializer, is_training)

		#--------------------
		# Decoding layer.
		with tf.variable_scope('decoding_layer', reuse=tf.AUTO_REUSE):
			decoded = self._create_decoding_layer(logits)

		return {'logit': logits, 'decoded_label': decoded}

	def _create_convolutional_layer(self, inputs, num_features, kernel_initializer, is_training):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv1 = tf.layers.batch_normalization(conv1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			conv1 = tf.nn.relu(conv1, name='relu')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv2 = tf.layers.batch_normalization(conv2, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			conv2 = tf.nn.relu(conv2, name='relu')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv1')
			conv3 = tf.layers.batch_normalization(conv3, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm1')
			conv3 = tf.nn.relu(conv3, name='relu1')

			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv2')
			conv3 = tf.layers.batch_normalization(conv3, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm2')
			conv3 = tf.nn.relu(conv3, name='relu2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=(1, 2), strides=(1, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv1')
			conv4 = tf.layers.batch_normalization(conv4, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm1')
			conv4 = tf.nn.relu(conv4, name='relu1')

			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv2')
			conv4 = tf.layers.batch_normalization(conv4, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm2')
			conv4 = tf.nn.relu(conv4, name='relu2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=(1, 2), strides=(1, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv')
			#conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=kernel_initializer, name='conv')
			conv5 = tf.layers.batch_normalization(conv5, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			conv5 = tf.nn.relu(conv5, name='relu')

		# Dilation.
		with tf.variable_scope('dilation', reuse=tf.AUTO_REUSE):
			conv5_shape = conv5.shape
			conv5 = self._create_dilation_layer(conv5, conv5_shape[-1], is_training)

		with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
			self._model_output_len = conv5_shape[1]

			#dense = tf.reshape(conv5, shape=conv5_shape[:2] + (-1,), name='reshape')
			#dense = tf.reshape(conv5, shape=conv5_shape[:2] + (conv5_shape[2] * conv5_shape[3]), name='reshape')
			outputs = tf.reshape(conv5, shape=(-1, conv5_shape[1], conv5_shape[2] * conv5_shape[3]), name='reshape')
			return tf.layers.dense(outputs, num_features, activation=tf.nn.relu, kernel_initializer=kernel_initializer, name='dense')  # (None, ???, 64).

	def _create_dilation_layer(self, inputs, num_features, is_training):
		return inputs

	def _create_recurrent_layer(self, inputs, input_len, kernel_initializer, is_training):
		num_hidden_units = 256
		keep_prob = 1.0
		#keep_prob = 0.5

		with tf.variable_scope('rnn1', reuse=tf.AUTO_REUSE):
			cell_fw1 = self._create_unit_cell(num_hidden_units, kernel_initializer, 'fw_unit_cell')  # Forward cell.
			#cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
			cell_bw1 = self._create_unit_cell(num_hidden_units, kernel_initializer, 'bw_unit_cell')  # Backward cell.
			#cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

			#rnn_outputs1, rnn_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, inputs, sequence_length=None, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs1, rnn_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, inputs, sequence_length=input_len, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs1 = tf.concat(rnn_outputs1, axis=-1)
			rnn_outputs1 = tf.layers.batch_normalization(rnn_outputs1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			#rnn_states1 = tf.contrib.rnn.LSTMStateTuple(tf.concat((rnn_states1[0].c, rnn_states1[1].c), axis=-1), tf.concat((rnn_states1[0].h, rnn_states1[1].h), axis=-1))
			#rnn_states1 = tf.layers.batch_normalization(rnn_states1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')

		with tf.variable_scope('rnn2', reuse=tf.AUTO_REUSE):
			cell_fw2 = self._create_unit_cell(num_hidden_units, kernel_initializer, 'fw_unit_cell')  # Forward cell.
			#cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
			cell_bw2 = self._create_unit_cell(num_hidden_units, kernel_initializer, 'bw_unit_cell')  # Backward cell.
			#cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

			#rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, rnn_outputs1, sequence_length=None, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, rnn_outputs1, sequence_length=input_len, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs2 = tf.concat(rnn_outputs2, axis=-1)
			rnn_outputs2 = tf.layers.batch_normalization(rnn_outputs2, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')
			#rnn_states2 = tf.contrib.rnn.LSTMStateTuple(tf.concat((rnn_states2[0].c, rnn_states2[1].c), axis=-1), tf.concat((rnn_states2[0].h, rnn_states2[1].h), axis=-1))
			#rnn_states2 = tf.layers.batch_normalization(rnn_states2, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_training, name='batchnorm')

			return rnn_outputs2

	@abc.abstractmethod
	def _create_transcription_layer(self, inputs, num_classes, kernel_initializer, is_training):
		raise NotImplementedError

	@abc.abstractmethod
	def _create_decoding_layer(self, logits):
		raise NotImplementedError

	def _create_unit_cell(self, num_units, kernel_initializer, name):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, initializer=kernel_initializer, forget_bias=1.0, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=kernel_initializer, name=name)

#--------------------------------------------------------------------

class HangeulCrnnWithCrossEntropyLoss(HangeulCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		super().__init__((None, image_height, image_width, image_channel), (None, None, num_classes), num_classes, is_sparse_output=False)

	def _create_transcription_layer(self, inputs, num_classes, kernel_initializer, is_training):
		outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=kernel_initializer, name='dense')
		#outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=kernel_initializer, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')

		return outputs  # (None, ???, num_classes).

	def _create_decoding_layer(self, logits):
		return None

	def _get_loss(self, y, t, y_len, t_len):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			masks = tf.sequence_mask(lengths=y_len, maxlen=tf.reduce_max(y_len), dtype=tf.float32)
			# Weighted cross-entropy loss for a sequence of logits.
			#loss = tf.contrib.seq2seq.sequence_loss(logits=y['logit'], targets=t, weights=masks)
			loss = tf.contrib.seq2seq.sequence_loss(logits=y['logit'], targets=tf.argmax(t, axis=-1), weights=masks)

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t, y_len):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			correct_prediction = tf.equal(tf.argmax(y['logit'], axis=-1), tf.argmax(t, axis=-1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

#--------------------------------------------------------------------

class HangeulCrnnWithCtcLoss(HangeulCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		super().__init__((None, image_height, image_width, image_channel), (None, None), num_classes, is_sparse_output=True)

	def _create_transcription_layer(self, inputs, num_classes, kernel_initializer, is_training):
		outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.relu, kernel_initializer=kernel_initializer, name='dense')
		#outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.relu, kernel_initializer=kernel_initializer, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')

		return outputs  # (None, ???, num_classes).

	def _create_decoding_layer(self, logits):
		# CTC beam search decoding.
		logits = tf.transpose(logits, (1, 0, 2))  # Time-major.
		# NOTE [info] >> CTC beam search decoding is too slow. It seems to run on CPU, not GPU.
		#	If the number of classes increases, its computation time becomes much slower.
		beam_width = 10 #100
		#decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=self._model_output_len_ph, beam_width=beam_width, top_paths=1, merge_repeated=True)
		decoded, log_prob = tf.nn.ctc_beam_search_decoder_v2(inputs=logits, sequence_length=self._model_output_len_ph, beam_width=beam_width, top_paths=1)
		#decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=self._model_output_len_ph, merge_repeated=True)
		return decoded[0]  # Sparse tensor.
		#return tf.sparse.to_dense(decoded[0], default_value=self._default_value)  # Dense tensor.

	def _get_loss(self, y, t_sparse, y_len, t_len):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			# Connectionist temporal classification (CTC) loss.
			# TODO [check] >> The case of preprocess_collapse_repeated=True & ctc_merge_repeated=True is untested.
			loss = tf.reduce_mean(tf.nn.ctc_loss(labels=t_sparse, inputs=y['logit'], sequence_length=y_len, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=False))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t_sparse, y_len):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			# Inaccuracy: label error rate.
			# NOTE [info] >> tf.edit_distance() is too slow. It seems to run on CPU, not GPU.
			#	Accuracy may not be calculated to speed up the training.
			ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(y['decoded_label'], tf.int32), truth=t_sparse, normalize=True))  # int64 -> int32.
			accuracy = 1.0 - ler
			#accuracy = tf.constant(-1, tf.float32)

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

#--------------------------------------------------------------------

class HangeulCrnnWithKerasCtcLoss(HangeulCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		super().__init__((None, image_height, image_width, image_channel), (None, None), num_classes, is_sparse_output=True)

		# FIXME [fix] >>
		self._eos_token = 2350

	def _create_transcription_layer(self, inputs, num_classes, kernel_initializer, is_training):
		outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=kernel_initializer, name='dense')
		#outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=kernel_initializer, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')

		return outputs  # (None, ???, num_classes).

	def _create_decoding_layer(self, logits):
		"""
		# Decoding.
		decoded = tf.argmax(logits, axis=-1)
		# FIXME [fix] >> This does not work correctly.
		#	Refer to MyModel._decode_label() in ${SWL_PYTHON_HOME}/test/language_processing/run_simple_hangeul_crrn.py.
		decoded = tf.numpy_function(lambda x: list(map(lambda lbl: list(k for k, g in itertools.groupby(lbl) if k < self._blank_label), x)), [decoded], [tf.int32])  # Removes repetitive labels.
		return decoded
		"""
		return None

	def _get_loss(self, y, t, y_len, t_len):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			# Connectionist temporal classification (CTC) loss.
			y_len, t_len = tf.reshape(y_len, (-1, 1)), tf.reshape(t_len, (-1, 1))
			loss = tf.reduce_mean(tf.keras.backend.ctc_batch_cost(y_true=t, y_pred=y['logit'], input_length=y_len, label_length=t_len))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t, y_len):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			"""
			correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.cast(t, tf.int64))  # Error: The time-steps of y and t are different.
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			return accuracy
			"""
			"""
			# FIXME [implement] >> The below logic has to be implemented in TensorFlow.
			correct_prediction = len(list(filter(lambda xx: len(list(filter(lambda x: x[0] == x[1], zip(xx[0], xx[1])))) == max(len(xx[0]), len(xx[1])), zip(tf.argmax(y, axis=-1), t))))
			accuracy = correct_prediction / max(len(y), len(t))

			return accuracy
			"""
			accuracy = tf.constant(-1.0, tf.float32)

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

#--------------------------------------------------------------------

class HangeulDilatedCrnnWithCtcLoss(HangeulCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		super().__init__((None, image_height, image_width, image_channel), (None, None), num_classes, is_sparse_output=True)

	def _create_transcription_layer(self, inputs, num_classes, kernel_initializer, is_training):
		outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.relu, kernel_initializer=kernel_initializer, name='dense')
		#outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.relu, kernel_initializer=kernel_initializer, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')

		return outputs  # (None, ???, num_classes).

	def _create_decoding_layer(self, logits):
		# CTC beam search decoding.
		logits = tf.transpose(logits, (1, 0, 2))  # Time-major.
		# NOTE [info] >> CTC beam search decoding is too slow. It seems to run on CPU, not GPU.
		#	If the number of classes increases, its computation time becomes much slower.
		beam_width = 10 #100
		#decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=self._model_output_len_ph, beam_width=beam_width, top_paths=1, merge_repeated=True)
		decoded, log_prob = tf.nn.ctc_beam_search_decoder_v2(inputs=logits, sequence_length=self._model_output_len_ph, beam_width=beam_width, top_paths=1)
		#decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=self._model_output_len_ph, merge_repeated=True)
		return decoded[0]  # Sparse tensor.
		#return tf.sparse.to_dense(decoded[0], default_value=self._default_value)  # Dense tensor.

	def _get_loss(self, y, t_sparse, y_len, t_len):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			# Connectionist temporal classification (CTC) loss.
			# TODO [check] >> The case of preprocess_collapse_repeated=True & ctc_merge_repeated=True is untested.
			loss = tf.reduce_mean(tf.nn.ctc_loss(labels=t_sparse, inputs=y, sequence_length=y_len, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=False))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t_sparse, y_len):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			# Inaccuracy: label error rate.
			# NOTE [info] >> tf.edit_distance() is too slow. It seems to run on CPU, not GPU.
			#	Accuracy may not be calculated to speed up the training.
			ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(y['decoded_label'], tf.int32), truth=t_sparse, normalize=True))  # int64 -> int32.
			accuracy = 1.0 - ler
			#accuracy = tf.constant(-1, tf.float32)

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

	def _create_dilation_layer(self, inputs, num_features, is_training):
		with tf.variable_scope('ctx_conv', reuse=tf.AUTO_REUSE):
			conv1 = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 1.
			conv1 = tf.layers.conv2d(conv1, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.pad(conv1, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 2.
			conv1 = tf.layers.conv2d(conv1, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

		with tf.variable_scope('ctx_atrous_conv', reuse=tf.AUTO_REUSE):
			conv2 = tf.pad(conv1, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 3.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=2, padding='valid', name='atrous_conv1')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.pad(conv2, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 4.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=4, padding='valid', name='atrous_conv2')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			conv2 = tf.pad(conv2, ((0, 0), (8, 8), (8, 8), (0, 0)), mode='CONSTANT', constant_values=0, name='pad3')

			# Layer 5.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=8, padding='valid', name='atrous_conv3')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
			conv2 = tf.nn.relu(conv2, name='relu3')

			conv2 = tf.pad(conv2, ((0, 0), (16, 16), (16, 16), (0, 0)), mode='CONSTANT', constant_values=0, name='pad4')

			# Layer 6.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=16, padding='valid', name='atrous_conv4')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='valid', name='conv4')
			conv2 = tf.nn.relu(conv2, name='relu4')

		with tf.variable_scope('ctx_final', reuse=tf.AUTO_REUSE):
			dense_final = tf.pad(conv2, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0, name='pad')

			# Layer 7.
			dense_final = tf.layers.conv2d(dense_final, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense1')
			dense_final = tf.nn.relu(dense_final, name='relu1')

			# Layer 8.
			return tf.layers.conv2d(dense_final, filters=num_features, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense2')

#--------------------------------------------------------------------

class HangeulDilatedCrnnWithKerasCtcLoss(HangeulCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		super().__init__((None, image_height, image_width, image_channel), (None, None), num_classes, is_sparse_output=True)

		# FIXME [fix] >>
		self._eos_token = 2350

	def _create_transcription_layer(self, inputs, num_classes, kernel_initializer, is_training):
		outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=kernel_initializer, name='dense')
		#outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=kernel_initializer, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')

		return outputs  # (None, ???, num_classes).

	def _create_decoding_layer(self, logits):
		"""
		# Decoding.
		decoded = tf.argmax(logits, axis=-1)
		# FIXME [fix] >> This does not work correctly.
		#	Refer to MyModel._decode_label() in ${SWL_PYTHON_HOME}/test/language_processing/run_simple_hangeul_crrn.py.
		decoded = tf.numpy_function(lambda x: list(map(lambda lbl: list(k for k, g in itertools.groupby(lbl) if k < self._blank_label), x)), [decoded], [tf.int32])  # Removes repetitive labels.
		return decoded
		"""
		return None

	def _get_loss(self, y, t, y_len, t_len):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			# Connectionist temporal classification (CTC) loss.
			y_len, t_len = tf.reshape(y_len, (-1, 1)), tf.reshape(t_len, (-1, 1))
			loss = tf.reduce_mean(tf.keras.backend.ctc_batch_cost(y_true=t, y_pred=y['logit'], input_length=y_len, label_length=t_len))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t, y_len):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			"""
			correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.cast(t, tf.int64))  # Error: The time-steps of y and t are different.
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			return accuracy
			"""
			"""
			# FIXME [implement] >> The below logic has to be implemented in TensorFlow.
			correct_prediction = len(list(filter(lambda xx: len(list(filter(lambda x: x[0] == x[1], zip(xx[0], xx[1])))) == max(len(xx[0]), len(xx[1])), zip(tf.argmax(y, axis=-1), t))))
			accuracy = correct_prediction / max(len(y), len(t))

			return accuracy
			"""
			accuracy = tf.constant(-1.0, tf.float32)

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

	def _create_dilation_layer(self, inputs, num_features, is_training):
		with tf.variable_scope('ctx_conv', reuse=tf.AUTO_REUSE):
			conv1 = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 1.
			conv1 = tf.layers.conv2d(conv1, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv1')
			conv1 = tf.nn.relu(conv1, name='relu1')

			conv1 = tf.pad(conv1, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 2.
			conv1 = tf.layers.conv2d(conv1, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='conv2')
			conv1 = tf.nn.relu(conv1, name='relu2')

		with tf.variable_scope('ctx_atrous_conv', reuse=tf.AUTO_REUSE):
			conv2 = tf.pad(conv1, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='CONSTANT', constant_values=0, name='pad1')

			# Layer 3.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=2, padding='valid', name='atrous_conv1')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='valid', name='conv1')
			conv2 = tf.nn.relu(conv2, name='relu1')

			conv2 = tf.pad(conv2, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='CONSTANT', constant_values=0, name='pad2')

			# Layer 4.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=4, padding='valid', name='atrous_conv2')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='valid', name='conv2')
			conv2 = tf.nn.relu(conv2, name='relu2')

			conv2 = tf.pad(conv2, ((0, 0), (8, 8), (8, 8), (0, 0)), mode='CONSTANT', constant_values=0, name='pad3')

			# Layer 5.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=8, padding='valid', name='atrous_conv3')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='valid', name='conv3')
			conv2 = tf.nn.relu(conv2, name='relu3')

			conv2 = tf.pad(conv2, ((0, 0), (16, 16), (16, 16), (0, 0)), mode='CONSTANT', constant_values=0, name='pad4')

			# Layer 6.
			#conv2 = tf.nn.atrous_conv2d(conv2, filters=(3, 3, num_features, num_features), rate=16, padding='valid', name='atrous_conv4')
			conv2 = tf.layers.conv2d(conv2, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='valid', name='conv4')
			conv2 = tf.nn.relu(conv2, name='relu4')

		with tf.variable_scope('ctx_final', reuse=tf.AUTO_REUSE):
			dense_final = tf.pad(conv2, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0, name='pad')

			# Layer 7.
			dense_final = tf.layers.conv2d(dense_final, filters=num_features, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense1')
			dense_final = tf.nn.relu(dense_final, name='relu1')

			# Layer 8.
			return tf.layers.conv2d(dense_final, filters=num_features, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', name='dense2')
