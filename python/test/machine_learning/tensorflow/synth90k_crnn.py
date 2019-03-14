import abc
import numpy as np
import tensorflow as tf

#%%------------------------------------------------------------------

class Synth90kCrnn(abc.ABC):
	def __init__(self, input_tensor_ph, output_tensor_ph, batch_size_ph, image_height, image_width, image_channel, num_classes):
		super().__init__()

		self._input_tensor_ph = input_tensor_ph
		self._output_tensor_ph = output_tensor_ph
		#self._output_seq_lens_ph = output_seq_lens_ph
		self._batch_size_ph = batch_size_ph

		self._image_height, self._image_width, self._image_channel = image_height, image_width, image_channel
		self._num_classes = num_classes
		#self._is_time_major = False

		# model_output is used in training, evaluation, and inference steps.
		self._model_output = None

		# Loss and accuracy are used in training and evaluation steps.
		self._loss = None
		self._accuracy = None

	@property
	def model_output(self):
		if self._model_output is None:
			raise TypeError
		return self._model_output

	@property
	def loss(self):
		if self._loss is None:
			raise TypeError
		return self._loss

	@property
	def accuracy(self):
		if self._loss is None:
			raise TypeError
		return self._accuracy

	@abc.abstractmethod
	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_loss(self, y, t, seq_lens):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_accuracy(self, y, t):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_final_output(self, logits, seq_lens):
		raise NotImplementedError

	def create_training_model(self):
		self._model_output, model_output_for_loss, model_output_lens = self._create_single_model(self._input_tensor_ph, self._num_classes, True)

		self._loss = self._get_loss(model_output_for_loss, self._output_tensor_ph, model_output_lens)
		self._accuracy = self._get_accuracy(self._model_output, self._output_tensor_ph)

	def create_evaluation_model(self):
		self._model_output, model_output_for_loss, model_output_lens = self._create_single_model(self._input_tensor_ph, self._num_classes, False)

		self._loss = self._get_loss(model_output_for_loss, self._output_tensor_ph, model_output_lens)
		self._accuracy = self._get_accuracy(self._model_output, self._output_tensor_ph)

	def create_inference_model(self):
		self._model_output, _, _ = self._create_single_model(self._input_tensor_ph, self._num_classes, False)

		self._loss = None
		self._accuracy = None

	def _create_single_model(self, input_tensor, num_classes, is_training):
		with tf.variable_scope('synth90k_crnn', reuse=tf.AUTO_REUSE):
			crnn_outputs = self._create_crnn(input_tensor, num_classes, is_training)

			crnn_output_lens = tf.fill(self._batch_size_ph, crnn_outputs.shape[1])
			return self._get_final_output(crnn_outputs, crnn_output_lens)

	def _create_crnn(self, input_tensor, num_classes, is_training):
		input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1, 3])  # (batches, height, width, channels) -> (batches, width, height, channels).

		#--------------------
		# Convolutional layer.
		with tf.variable_scope('convolutional_layer', reuse=tf.AUTO_REUSE):
			cnn_outputs = self._create_convolutional_layer(input_tensor)

		#--------------------
		# Recurrent layer.
		with tf.variable_scope('recurrent_layer', reuse=tf.AUTO_REUSE):
			rnn_outputs = self._create_recurrent_layer(cnn_outputs)

		#--------------------
		# Transcription layer.
		with tf.variable_scope('transcription_layer', reuse=tf.AUTO_REUSE):
			return self._create_transcription_layer(rnn_outputs, num_classes)

	def _create_convolutional_layer(self, inputs):
		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv')
			conv1 = tf.layers.batch_normalization(conv1, axis=-1, name='batchnorm')
			conv1 = tf.nn.relu(conv1, name='relu')
			conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			conv2 = tf.layers.conv2d(conv1, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv')
			conv2 = tf.layers.batch_normalization(conv2, axis=-1, name='batchnorm')
			conv2 = tf.nn.relu(conv2, name='relu')
			conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			conv3 = tf.layers.conv2d(conv2, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv1')
			conv3 = tf.layers.batch_normalization(conv3, axis=-1, name='batchnorm1')
			conv3 = tf.nn.relu(conv3, name='relu1')
			conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv2')
			conv3 = tf.layers.batch_normalization(conv3, axis=-1, name='batchnorm2')
			conv3 = tf.nn.relu(conv3, name='relu2')
			conv3 = tf.layers.max_pooling2d(conv3, pool_size=(1, 2), strides=(1, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
			conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv1')
			conv4 = tf.layers.batch_normalization(conv4, axis=-1, name='batchnorm1')
			conv4 = tf.nn.relu(conv4, name='relu1')
			conv4 = tf.layers.conv2d(conv4, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv2')
			conv4 = tf.layers.batch_normalization(conv4, axis=-1, name='batchnorm2')
			conv4 = tf.nn.relu(conv4, name='relu2')
			conv4 = tf.layers.max_pooling2d(conv4, pool_size=(1, 2), strides=(1, 2), padding='same', name='maxpool')

		with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
			conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(2, 2), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv')
			#conv5 = tf.layers.conv2d(conv4, filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='conv')
			conv5 = tf.layers.batch_normalization(conv5, axis=-1, name='batchnorm')
			conv5 = tf.nn.relu(conv5, name='relu')

		with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
			conv5_shape = conv5.shape.as_list()
			#dense = tf.reshape(conv5, shape=conv5_shape[:2] + [-1], name='reshape')
			#dense = tf.reshape(conv5, shape=conv5_shape[:2] + [conv5_shape[2] * conv5_shape[3]], name='reshape')
			outputs = tf.reshape(conv5, shape=[-1, conv5_shape[1], conv5_shape[2] * conv5_shape[3]], name='reshape')
			outputs = tf.layers.dense(outputs, 64, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='dense')

			return outputs

	def _create_recurrent_layer(self, inputs):
		num_hidden_units = 256
		keep_prob = 1.0
		#keep_prob = 0.5

		with tf.variable_scope('rnn1', reuse=tf.AUTO_REUSE):
			cell_fw1 = self._create_unit_cell(num_hidden_units, 'fw_unit_cell')  # Forward cell.
			#cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
			cell_bw1 = self._create_unit_cell(num_hidden_units, 'bw_unit_cell')  # Backward cell.
			#cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

			#rnn_outputs1, rnn_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, inputs, sequence_length=input_seq_lens, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs1, rnn_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, inputs, sequence_length=None, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs1 = tf.concat(rnn_outputs1, axis=-1)
			rnn_outputs1 = tf.layers.batch_normalization(rnn_outputs1, axis=-1, name='batchnorm')
			#rnn_states1 = tf.contrib.rnn.LSTMStateTuple(tf.concat((rnn_states1[0].c, rnn_states1[1].c), axis=-1), tf.concat((rnn_states1[0].h, rnn_states1[1].h), axis=-1))
			#rnn_states1 = tf.layers.batch_normalization(rnn_states1, axis=-1, name='batchnorm')

		with tf.variable_scope('rnn2', reuse=tf.AUTO_REUSE):
			cell_fw2 = self._create_unit_cell(num_hidden_units, 'fw_unit_cell')  # Forward cell.
			#cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)
			cell_bw2 = self._create_unit_cell(num_hidden_units, 'bw_unit_cell')  # Backward cell.
			#cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

			#rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, rnn_outputs1, sequence_length=input_seq_lens, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, rnn_outputs1, sequence_length=None, time_major=False, dtype=tf.float32, scope='rnn')
			rnn_outputs2 = tf.concat(rnn_outputs2, axis=-1)
			rnn_outputs2 = tf.layers.batch_normalization(rnn_outputs2, axis=-1, name='batchnorm')
			#rnn_states2 = tf.contrib.rnn.LSTMStateTuple(tf.concat((rnn_states2[0].c, rnn_states2[1].c), axis=-1), tf.concat((rnn_states2[0].h, rnn_states2[1].h), axis=-1))
			#rnn_states2 = tf.layers.batch_normalization(rnn_states2, axis=-1, name='batchnorm')

			return rnn_outputs2

	def _create_transcription_layer(self, inputs, num_classes):
		outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name='dense')
		#outputs = tf.layers.dense(inputs, num_classes, activation=tf.nn.softmax, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')

		return outputs

	def _create_unit_cell(self, num_units, name):
		#return tf.nn.rnn_cell.RNNCell(num_units, name=name)
		return tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), forget_bias=1.0, name=name)
		#return tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal'), name=name)

#%%------------------------------------------------------------------

class Synth90kCrnnWithCrossEntropyLoss(Synth90kCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes):
		input_tensor_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_tensor_ph')
		output_tensor_ph = tf.placeholder(tf.int32, shape=[None, None, num_classes], name='output_tensor_ph')
		batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		super().__init__(input_tensor_ph, output_tensor_ph, batch_size_ph, image_height, image_width, image_channel, num_classes)

	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		batch_size = [inputs.shape[0]]

		if outputs is None:
			feed_dict = {self._input_tensor_ph: inputs, self._batch_size_ph: batch_size}
		else:
			feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs, self._batch_size_ph: batch_size}
		return feed_dict

	def _get_loss(self, y, t, seq_lens):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			masks = tf.sequence_mask(seq_lens, tf.reduce_max(seq_lens), dtype=tf.float32)
			# Weighted cross-entropy loss for a sequence of logits.
			#loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=t, weights=masks)
			loss = tf.contrib.seq2seq.sequence_loss(logits=y, targets=tf.argmax(t, axis=-1), weights=masks)

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

	def _get_final_output(self, logits, seq_lens):
		return logits, logits, seq_lens

#%%------------------------------------------------------------------

class Synth90kCrnnWithCtcLoss(Synth90kCrnn):
	def __init__(self, image_height, image_width, image_channel, num_classes, eos_token=-1):
		input_tensor_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_tensor_ph')
		output_tensor_ph = tf.sparse_placeholder(tf.int32, shape=[None, None], name='output_tensor_ph')
		batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')
		self._eos_token = eos_token

		super().__init__(input_tensor_ph, output_tensor_ph, batch_size_ph, image_height, image_width, image_channel, num_classes)

	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		batch_size = [inputs.shape[0]]

		# For checking dataset.
		#Synth90kCrnnWithCtcLoss._visualize_data(inputs, outputs)

		if outputs is None:
			feed_dict = {self._input_tensor_ph: inputs, self._batch_size_ph: batch_size}
		else:
			feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs, self._batch_size_ph: batch_size}
		return feed_dict

	def _get_loss(self, y, t, seq_lens):
		with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
			# Connectionist temporal classification (CTC) loss.
			# TODO [check] >> The case of preprocess_collapse_repeated=True & ctc_merge_repeated=True is untested.
			loss = tf.reduce_mean(tf.nn.ctc_loss(labels=t, inputs=y, sequence_length=seq_lens, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=False))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t):
		with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
			# Inaccuracy: label error rate.
			ler = tf.reduce_mean(tf.edit_distance(tf.cast(y, tf.int32), t, normalize=True))
			accuracy = 1.0 - ler

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

	def _get_final_output(self, logits, seq_lens):
		#decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(logits, (1, 0, 2)), sequence_length=seq_lens, beam_width=100, top_paths=1, merge_repeated=True)
		decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(logits, (1, 0, 2)), sequence_length=seq_lens, merge_repeated=True)
		decoded_best = decoded[0]  # tf.SparseTensor.

		return decoded_best, logits, seq_lens

	@staticmethod
	def _visualize_data(inputs, outputs):
		import cv2
		import swl.machine_learning.util as swl_ml_util

		if outputs is not None:
			max_label_len = 23  # Max length of words in lexicon.

			# Label: 0~9 + a~z + A~Z.
			#label_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
			# Label: 0~9 + a~z.
			label_characters = '0123456789abcdefghijklmnopqrstuvwxyz'

			SOS = '<SOS>'  # All strings will start with the Start-Of-String token.
			EOS = '<EOS>'  # All strings will end with the End-Of-String token.
			#extended_label_list = [SOS] + list(label_characters) + [EOS]
			extended_label_list = list(label_characters) + [EOS]
			#extended_label_list = list(label_characters)

			label_int2char = extended_label_list
			label_char2int = {c:i for i, c in enumerate(extended_label_list)}

			num_labels = len(extended_label_list)
			num_classes = num_labels + 1  # extended labels + blank label.
			# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
			blank_label = num_classes - 1
			label_eos_token = label_char2int[EOS]
			#label_eos_token = blank_label

			dense_outputs = swl_ml_util.sparse_to_dense(outputs[0], outputs[1], outputs[2], default_value=label_eos_token, dtype=np.int32)
			#dense_outputs = np.argmax(dense_outputs, -1)
		else:
			dense_outputs = None

		if dense_outputs is not None:
			print('Image shape: {}, dtype: {}.'.format(inputs.shape, inputs.dtype))
			print('Image min = {}, max = {}.'.format(np.min(inputs), np.max(inputs)))
			print('Label shape: {}, dtype: {}.'.format(dense_outputs.shape, dense_outputs.dtype))

			for inp, outp in zip(inputs, dense_outputs):
				label = [label_int2char[lbl] for lbl in outp]
				print('Label =', label)
				
				cv2.imshow('Image', inp)
				ch = cv2.waitKey(0)
				if 27 == ch:  # ESC.
					break
			cv2.destroyAllWindows()
