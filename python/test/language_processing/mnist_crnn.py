import numpy as np
import tensorflow as tf
from swl.machine_learning.tensorflow.simple_neural_net import BasicSeq2SeqNeuralNet

#%%------------------------------------------------------------------

class MnistCRNN(BasicSeq2SeqNeuralNet):
	def __init__(self, input_shape, output_shape, is_time_major=False, has_decoder=True):
		self._input_seq_lens_ph = tf.placeholder(tf.int32, [None], name='input_seq_lens_ph')
		self._output_seq_lens_ph = tf.placeholder(tf.int32, [None], name='output_seq_lens_ph')
		self._batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		self._is_time_major = is_time_major
		self._has_decoder = has_decoder
		super().__init__(input_shape, output_shape)

	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		#input_seq_lens = tf.constant(max_time_steps, tf.int32, shape=[batch_size])
		#output_seq_lens = tf.constant(max_time_steps, tf.int32, shape=[batch_size])
		if self._is_time_major:
			input_seq_lens = np.full(inputs.shape[1], inputs.shape[0], np.int32)
			if outputs is None:
				output_seq_lens = np.full(inputs.shape[1], inputs.shape[0], np.int32)
			else:
				output_seq_lens = np.full(outputs.shape[1], outputs.shape[0], np.int32)
			batch_size = [inputs.shape[1]]
		else:
			input_seq_lens = np.full(inputs.shape[0], inputs.shape[1], np.int32)
			if outputs is None:
				output_seq_lens = np.full(inputs.shape[0], inputs.shape[1], np.int32)
			else:
				output_seq_lens = np.full(outputs.shape[0], outputs.shape[1], np.int32)
			batch_size = [inputs.shape[0]]

		if outputs is None:
			feed_dict = {self._input_tensor_ph: inputs, self._input_seq_lens_ph: input_seq_lens, self._output_seq_lens_ph: output_seq_lens, self._batch_size_ph: batch_size}
		else:
			feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs, self._input_seq_lens_ph: input_seq_lens, self._output_seq_lens_ph: output_seq_lens, self._batch_size_ph: batch_size}
		return feed_dict

	def _create_single_model(self, input_tensor, output_tensor, is_training, input_shape, output_shape):
		with tf.variable_scope('mnist_crnn', reuse=tf.AUTO_REUSE):
			# TODO [improve] >> It is not good to use num_time_steps.
			#num_classes = output_shape[-1]
			if self._is_time_major:
				num_time_steps, num_classes = output_shape[0], output_shape[-1]
			else:
				num_time_steps, num_classes = output_shape[1], output_shape[-1]
			return self._create_dynamic_bidirectional_model(input_tensor, is_training, self._input_seq_lens_ph, self._batch_size_ph, num_time_steps, num_classes, self._is_time_major)

	def _create_dynamic_bidirectional_model(self, input_tensor, is_training, input_seq_lens, batch_size, num_time_steps, num_classes, is_time_major):
		keep_prob = 1.0
		#keep_prob = 0.5

		#--------------------
		# CNN.
		cnn_output = self._get_cnn_output(input_tensor, num_time_steps, num_classes, is_time_major)

		#--------------------
		# Encoder.
		num_enc_hidden_units = 256
		enc_cell_fw = self._create_unit_cell(num_enc_hidden_units)  # Forward cell.
		enc_cell_fw = tf.contrib.rnn.DropoutWrapper(enc_cell_fw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell_fw = tf.contrib.rnn.AttentionCellWrapper(enc_cell_fw, attention_window_len, state_is_tuple=True)
		enc_cell_bw = self._create_unit_cell(num_enc_hidden_units)  # Backward cell.
		enc_cell_bw = tf.contrib.rnn.DropoutWrapper(enc_cell_bw, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
		# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
		#enc_cell_bw = tf.contrib.rnn.AttentionCellWrapper(enc_cell_bw, attention_window_len, state_is_tuple=True)

		enc_cell_outputs, enc_cell_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, cnn_output, sequence_length=input_seq_lens, time_major=is_time_major, dtype=tf.float32, scope='enc')
		enc_cell_outputs = tf.concat(enc_cell_outputs, axis=-1)
		enc_cell_states = tf.contrib.rnn.LSTMStateTuple(tf.concat((enc_cell_states[0].c, enc_cell_states[1].c), axis=-1), tf.concat((enc_cell_states[0].h, enc_cell_states[1].h), axis=-1))

		if self._has_decoder:
			#--------------------
			# Attention.
			# REF [function] >> SimpleSeq2SeqEncoderDecoderWithTfAttention._create_dynamic_bidirectional_model() in ./simple_seq2seq_encdec_tf_attention.py.

			#--------------------
			# Decoder.
			num_dec_hidden_units = 512
			dec_cell = self._create_unit_cell(num_dec_hidden_units)
			dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
			# REF [paper] >> "Long Short-Term Memory-Networks for Machine Reading", arXiv 2016.
			#dec_cell = tf.contrib.rnn.AttentionCellWrapper(dec_cell, attention_window_len, state_is_tuple=True)

			return self._get_decoder_output(dec_cell, enc_cell_states, enc_cell_outputs, num_time_steps, num_classes, is_time_major)
		else:
			return enc_cell_outputs

	def _create_unit_cell(self, num_units):
		#return tf.contrib.rnn.BasicRNNCell(num_units)
		#return tf.contrib.rnn.RNNCell(num_units)

		return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
		#return tf.contrib.rnn.LSTMCell(num_units, forget_bias=1.0)

		#return tf.contrib.rnn.GRUCell(num_units)

	def _create_cnn_model(self, sliced_input_tensor, num_classes):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(sliced_input_tensor, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv1')
			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=(1, 2), strides=(1, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv1')
			conv4 = tf.layers.batch_normalization(conv4, axis=-1, name='batchnorm1')
			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv2')
			conv4 = tf.layers.batch_normalization(conv4, axis=-1, name='batchnorm2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=(1, 2), strides=(1, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
			#conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv')
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv')
			conv5 = tf.layers.flatten(conv5, name='flatten')

			return conv5

	def _create_fc_layer(self, dec_cell_outputs, num_classes):
		with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
			if 1 == num_classes:
				return tf.layers.dense(dec_cell_outputs, 1, activation=tf.sigmoid, name='dense')
				#return tf.layers.dense(dec_cell_outputs, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			elif num_classes >= 2:
				return tf.layers.dense(dec_cell_outputs, num_classes, activation=tf.nn.softmax, name='dense')
				#return tf.layers.dense(dec_cell_outputs, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
			else:
				assert num_classes > 0, 'Invalid number of classes.'
				return None

	def _get_cnn_output(self, input_tensor, num_time_steps, num_classes, is_time_major):
		# (samples, time-steps, features) -> (time-steps, samples, features).
		if is_time_major:
			input_tensor_time_major = input_tensor
		else:
			dims = list(range(len(input_tensor.shape.as_list())))
			input_tensor_time_major = tf.transpose(input_tensor, [1, 0] + dims[2:], name='transpose1')  # Time-major.

		cnn_model_fn = lambda sliced: self._create_cnn_model(sliced, num_classes)
		cnn_output = tf.map_fn(cnn_model_fn, input_tensor_time_major, swap_memory=True, name='map_fn')

		if not is_time_major:
			# (time-steps, samples, features) -> (samples, time-steps, features).
			dims = list(range(len(cnn_output.shape.as_list())))
			cnn_output = tf.transpose(cnn_output, [1, 0] + dims[2:], name='transpose2')

		return cnn_output

	def _get_decoder_output(self, dec_cell, initial_cell_state, enc_cell_outputs, num_time_steps, num_classes, is_time_major):
		# dec_cell_state is an instance of LSTMStateTuple, which stores (c, h), where c is the hidden state and h is the output.
		#dec_cell_outputs, dec_cell_state = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')
		#dec_cell_outputs, _ = tf.nn.dynamic_rnn(dec_cell, enc_cell_outputs, initial_state=enc_cell_states, time_major=is_time_major, dtype=tf.float32, scope='dec')

		# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
		enc_cell_outputs = tf.unstack(enc_cell_outputs, num_time_steps, axis=0 if is_time_major else 1)

		dec_cell_state = initial_cell_state
		dec_cell_outputs = []
		for inp in enc_cell_outputs:
			dec_cell_output, dec_cell_state = dec_cell(inp, dec_cell_state, scope='dec')
			dec_cell_outputs.append(dec_cell_output)

		# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
		dec_cell_outputs = tf.stack(dec_cell_outputs, axis=0 if is_time_major else 1)

		fc_outputs = self._create_fc_layer(dec_cell_outputs, num_classes)

		return fc_outputs

#%%------------------------------------------------------------------

class MnistCrnnWithCrossEntropyLoss(MnistCRNN):
	def __init__(self, input_shape, output_shape, is_time_major=False):
		super().__init__(input_shape, output_shape, is_time_major, has_decoder=True)

	def _get_loss(self, y, t):
		with tf.name_scope('loss'):
			# Fixed-length outputs.
			# Decoder is required, 

			masks = tf.sequence_mask(self._output_seq_lens_ph, tf.reduce_max(self._output_seq_lens_ph), dtype=tf.float32)
			# Weighted cross-entropy loss for a sequence of logits.
			#loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=t, weights=masks)
			loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=tf.argmax(t, axis=-1), weights=masks)

			tf.summary.scalar('loss', loss)
			return loss

#%%------------------------------------------------------------------

class MnistCrnnWithCtcLoss(MnistCRNN):
	def __init__(self, input_shape, output_shape, is_time_major=False):
		super().__init__(input_shape, output_shape, is_time_major, has_decoder=True)

	def _get_loss(self, y, t):
		with tf.name_scope('loss'):
			# Variable-length outputs.
			# Decoder is required.

			# Connectionist temporal classification (CTC) loss.
			loss = tf.reduce_mean(tf.nn.ctc_loss(t, y, sequence_length=self._output_seq_lens_ph, ctc_merge_repeated=True, time_major=self._is_time_major))

			tf.summary.scalar('loss', loss)
			return loss

#%%------------------------------------------------------------------

class MnistCrnnWithCtcBeamSearchDecoding(MnistCRNN):
	def __init__(self, input_shape, output_shape, is_time_major=False):
		super().__init__(input_shape, output_shape, is_time_major, has_decoder=False)

	def _get_loss(self, y, t):
		with tf.name_scope('loss'):
			# Variable-length outputs.
			# No decoder is required.

			dims = list(range(len(y.shape.as_list())))
			y_time_major = y if self._is_time_major else tf.transpose(y, [1, 0] + dims[2:])  # Time-major.

			# Beam search decoding on the logits given in input.
			#	Slower but get better results.
			#decoded, log_prob = tf.nn.ctc_beam_search_decoder(y_time_major, sequence_length=self._output_seq_lens_ph, beam_width=100, top_paths=1, merge_repeated=True)
			decoded, log_prob = tf.nn.ctc_greedy_decoder(y_time_major, sequence_length=self._output_seq_lens_ph, merge_repeated=True)

			# Inaccuracy: label error rate.
			loss = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), t))

			tf.summary.scalar('loss', loss)
			return loss
