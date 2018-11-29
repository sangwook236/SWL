#!/usr/bin/env python

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example/blob/master/ctc_tensorflow_example.py

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	#lib_home_dir_path = 'D:/lib_repo/python'
	lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append('../../../src')

#--------------------
import abc, time, datetime, math, random
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from PIL import Image
from python_speech_features import mfcc
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.machine_learning.util as swl_ml_util
import traceback

#%%------------------------------------------------------------------

class SimpleRnnBase(abc.ABC):
	def __init__(self, input_tensor_ph, output_tensor_ph, seq_lens_ph, num_classes, is_time_major=False):
		super().__init__()

		self._input_tensor_ph = input_tensor_ph
		self._output_tensor_ph = output_tensor_ph
		self._seq_lens_ph = seq_lens_ph
		#self._batch_size_ph = batch_size_ph

		self._num_classes = num_classes
		self._is_time_major = is_time_major

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

	def get_feed_dict(self, inputs, outputs=None, **kwargs):
		if self._is_time_major:
			seq_lens = np.full(inputs.shape[1], inputs.shape[0], np.int32)
			#batch_size = [inputs.shape[1]]
		else:
			seq_lens = np.full(inputs.shape[0], inputs.shape[1], np.int32)
			#batch_size = [inputs.shape[0]]

		if outputs is None:
			#feed_dict = {self._input_tensor_ph: inputs, self._seq_lens_ph: seq_lens, self._batch_size_ph: batch_size}
			feed_dict = {self._input_tensor_ph: inputs, self._seq_lens_ph: seq_lens}
		else:
			#feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs, self._seq_lens_ph: seq_lens, self._batch_size_ph: batch_size}
			feed_dict = {self._input_tensor_ph: inputs, self._output_tensor_ph: outputs, self._seq_lens_ph: seq_lens}
		return feed_dict

	def create_training_model(self):
		self._model_output, model_output_for_loss = self._create_single_model(self._input_tensor_ph, self._seq_lens_ph, self._num_classes, True)

		self._loss = self._get_loss(model_output_for_loss, self._output_tensor_ph, self._seq_lens_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_tensor_ph)

	def create_evaluation_model(self):
		self._model_output, model_output_for_loss = self._create_single_model(self._input_tensor_ph, self._seq_lens_ph, self._num_classes, False)

		self._loss = self._get_loss(model_output_for_loss, self._output_tensor_ph, self._seq_lens_ph)
		self._accuracy = self._get_accuracy(self._model_output, self._output_tensor_ph)

	def create_inference_model(self):
		self._model_output, _ = self._create_single_model(self._input_tensor_ph, self._seq_lens_ph, self._num_classes, False)

		self._loss = None
		self._accuracy = None

	def _create_single_model(self, input_tensor, seq_lens, num_classes, is_training):
		num_hidden = 50
		num_layers = 1
		num_units=50  # Number of units in the LSTM cell.

		if self._is_time_major:
			input_tensor = tf.transpose(input_tensor, (1, 0, 2))

		shape = tf.shape(input_tensor)
		batch_size, max_time_steps = shape[0], shape[1]
		#seq_lens = tf.fill(batch_size, max_time_steps)  # Error.

		cells = []
		for _ in range(num_layers):
			cell = tf.contrib.rnn.LSTMCell(num_units)   # Or LSTMCell(num_units).
			cells.append(cell)
		stack = tf.contrib.rnn.MultiRNNCell(cells)

		# The second output is the last state and we will no use that.
		outputs, _ = tf.nn.dynamic_rnn(stack, input_tensor, seq_lens, dtype=tf.float32, time_major=False)

		# Reshape to apply the same weights over the timesteps.
		outputs = tf.reshape(outputs, [-1, num_hidden])

		# Truncated normal with mean 0 and stdev=0.1.
		# Tip: Try another initialization.
		# 	REF [site] >> https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
		W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
		# Zero initialization.
		# Tip: Is tf.zeros_initializer the same?
		b = tf.Variable(tf.constant(0., shape=[num_classes]))

		# Do the affine projection.
		logits = tf.matmul(outputs, W) + b

		# Reshape back to the original shape.
		logits = tf.reshape(logits, [-1, batch_size, num_classes])  # Time-major.

		#decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_lens, beam_width=100, top_paths=1, merge_repeated=True)
		decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_lens, merge_repeated=True)
		decoded_best = decoded[0]  # tf.SparseTensor.

		if not self._is_time_major:
			logits = tf.transpose(logits, (1, 0, 2))

		return decoded_best, logits

	@abc.abstractmethod
	def _get_loss(self, y_for_loss, t, seq_lens):
		raise NotImplementedError

	@abc.abstractmethod
	def _get_accuracy(self, y, t):
		raise NotImplementedError

class SimpleRnnWithDenseLabel(SimpleRnnBase):
	def __init__(self, num_features, num_classes, label_eos_token, is_time_major=False):
		# e.g: log filter bank or MFCC features.
		# Has size [batch_size, time_steps, num_features], but the batch_size and max_stepsize can vary along each step.
		input_tensor_ph = tf.placeholder(tf.float32, [None, None, num_features], name='input_tensor_ph')
		output_tensor_ph = tf.placeholder(tf.int32, [None, None], name='output_tensor_ph')
		# 1D array of size [batch_size].
		seq_lens_ph = tf.placeholder(tf.int32, [None], name='seq_lens_ph')
		#batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		#super().__init__(input_tensor_ph, output_tensor_ph, seq_lens_ph, batch_size_ph, num_classes, is_time_major)
		super().__init__(input_tensor_ph, output_tensor_ph, seq_lens_ph, num_classes, is_time_major)

		self._label_eos_token = label_eos_token

	def _get_loss(self, y_for_loss, t, seq_lens):
		with tf.name_scope('loss'):
			if not self._is_time_major:
				y_for_loss = tf.transpose(y_for_loss, (1, 0, 2))

			# Dense tensor -> sparse tensor.
			t = tf.contrib.layers.dense_to_sparse(t, eos_token=self._label_eos_token)

			loss = tf.reduce_mean(tf.nn.ctc_loss(t, y_for_loss, seq_lens, time_major=True))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t):
		with tf.name_scope('accuracy'):
			# Dense tensor -> sparse tensor.
			t = tf.contrib.layers.dense_to_sparse(t, eos_token=self._label_eos_token)

			# Inaccuracy: label error rate.
			ler = tf.reduce_mean(tf.edit_distance(tf.cast(y, tf.int32), t, normalize=True))
			accuracy = 1.0 - ler

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

class SimpleRnnWithSparseLabel(SimpleRnnBase):
	def __init__(self, num_features, num_classes, is_time_major=False):
		# e.g: log filter bank or MFCC features.
		# Has size [batch_size, time_steps, num_features], but the batch_size and max_stepsize can vary along each step.
		input_tensor_ph = tf.placeholder(tf.float32, [None, None, num_features], name='input_tensor_ph')
		# Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
		output_tensor_ph = tf.sparse_placeholder(tf.int32, name='output_tensor_ph')
		# 1D array of size [batch_size].
		seq_lens_ph = tf.placeholder(tf.int32, [None], name='seq_lens_ph')
		#batch_size_ph = tf.placeholder(tf.int32, [1], name='batch_size_ph')

		#super().__init__(input_tensor_ph, output_tensor_ph, seq_lens_ph, batch_size_ph, num_classes, is_time_major)
		super().__init__(input_tensor_ph, output_tensor_ph, seq_lens_ph, num_classes, is_time_major)

	def _get_loss(self, y_for_loss, t, seq_lens):
		with tf.name_scope('loss'):
			# Connectionist temporal classification (CTC) loss.
			loss = tf.reduce_mean(tf.nn.ctc_loss(labels=t, inputs=y_for_loss, sequence_length=seq_lens, ctc_merge_repeated=True, time_major=self._is_time_major))

			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t):
		with tf.name_scope('accuracy'):
			# Inaccuracy: label error rate.
			ler = tf.reduce_mean(tf.edit_distance(tf.cast(y, tf.int32), t, normalize=True))
			accuracy = 1.0 - ler

			tf.summary.scalar('accuracy', accuracy)
			return accuracy

#%%------------------------------------------------------------------

class SimpleRnnTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		with tf.name_scope('learning_rate'):
			learning_rate = 1e-2
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

		super().__init__(neuralNet, optimizer, initial_epoch)

#%%------------------------------------------------------------------

# Supports lists of dense and sparse labels.
def train_neural_net_by_batch_lists(session, nnTrainer, train_inputs_list, train_outputs_list, val_inputs_list, val_outputs_list, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_label):
	num_train_batches, num_val_batches = len(train_inputs_list), len(val_inputs_list)
	if len(train_outputs_list) != num_train_batches or len(val_outputs_list) != num_val_batches:
		raise ValueError('Invalid parameter length')

	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	# Create writers to write all the summaries out to a directory.
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, session.graph) if train_summary_dir_path is not None else None
	val_summary_writer = tf.summary.FileWriter(val_summary_dir_path) if val_summary_dir_path is not None else None

	history = {
		'acc': [],
		'loss': [],
		'val_acc': [],
		'val_loss': []
	}

	batch_dim = 1 if is_time_major else 0

	best_val_acc = 0.0
	for epoch in range(1, num_epochs + 1):
		print('Epoch {}/{}'.format(epoch, num_epochs))

		start_time = time.time()

		indices = np.arange(num_train_batches)
		if shuffle:
			np.random.shuffle(indices)

		print('>-', sep='', end='')
		processing_ratio = 0.05
		train_loss, train_acc, num_train_examples = 0.0, 0.0, 0
		for step in indices:
			train_inputs, train_outputs = train_inputs_list[step], train_outputs_list[step]
			batch_acc, batch_loss = nnTrainer.train_by_batch(session, train_inputs, train_outputs, train_summary_writer, is_time_major, is_sparse_label)

			# TODO [check] >> Are these calculation correct?
			batch_size = train_inputs.shape[batch_dim]
			train_acc += batch_acc * batch_size
			train_loss += batch_loss * batch_size
			num_train_examples += batch_size

			if step / num_train_batches >= processing_ratio:
				print('-', sep='', end='')
				processing_ratio = round(step / num_train_batches, 2) + 0.05
		print('<')

		train_acc /= num_train_examples
		train_loss /= num_train_examples

		#--------------------
		indices = np.arange(num_val_batches)
		np.random.shuffle(indices)

		val_loss, val_acc, num_val_examples = 0.0, 0.0, 0
		for step in indices:
			val_inputs, val_outputs = val_inputs_list[step], val_outputs_list[step]
			batch_acc, batch_loss = nnTrainer.evaluate_training_by_batch(session, val_inputs, val_outputs, val_summary_writer, is_time_major, is_sparse_label)

			# TODO [check] >> Are these calculation correct?
			batch_size = val_inputs.shape[batch_dim]
			val_acc += batch_acc * batch_size
			val_loss += batch_loss * batch_size
			num_val_examples += batch_size

		val_acc /= num_val_examples
		val_loss /= num_val_examples

		history['acc'].append(train_acc)
		history['loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		# Save a model.
		if saver is not None and checkpoint_dir_path is not None and val_acc >= best_val_acc:
			saved_model_path = saver.save(session, checkpoint_dir_path + '/model.ckpt', global_step=nnTrainer.global_step)
			best_val_acc = val_acc
			print('[SWL] Info: Accurary is improved and the model is saved at {}.'.format(saved_model_path))

		print('\tTraining time = {}'.format(time.time() - start_time))
		print('\tLoss = {}, accuracy = {}, validation loss = {}, validation accurary = {}'.format(train_loss, train_acc, val_loss, val_acc))

	# Close writers.
	if train_summary_writer is not None:
		train_summary_writer.close()
	if val_summary_writer is not None:
		val_summary_writer.close()

	#--------------------
	# Save a graph.
	#tf.train.write_graph(session.graph_def, output_dir_path, 'crnn_graph.pb', as_text=False)
	##tf.train.write_graph(session.graph_def, output_dir_path, 'crnn_graph.pbtxt', as_text=True)

	# Save a serving model.
	#builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	#builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	#builder.save(as_text=False)

	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

# Supports a dense label only.
def train_neural_net(session, nnTrainer, train_inputs, train_outputs, val_inputs, val_outputs, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train(session, train_inputs, train_outputs, val_inputs, val_outputs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	#--------------------
	# Save a graph.
	#tf.train.write_graph(session.graph_def, output_dir_path, 'crnn_graph.pb', as_text=False)
	##tf.train.write_graph(session.graph_def, output_dir_path, 'crnn_graph.pbtxt', as_text=True)

	# Save a serving model.
	#builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	#builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	#builder.save(as_text=False)

	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, val_inputs, val_outputs, batch_size, saver=None, checkpoint_dir_path=None, is_time_major=False, is_sparse_label=False):
	batch_dim = 1 if is_time_major else 0

	num_val_examples = 0
	if val_inputs is not None and val_outputs is not None:
		if is_sparse_label:
			num_val_examples = val_inputs.shape[batch_dim]
		else:
			if val_inputs.shape[batch_dim] == val_outputs.shape[batch_dim]:
				num_val_examples = val_inputs.shape[batch_dim]

	if num_val_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start evaluation...')
		start_time = time.time()
		val_loss, val_acc = nnEvaluator.evaluate(session, val_inputs, val_outputs, batch_size)
		print('\tEvaluation time = {}'.format(time.time() - start_time))
		print('\tValidation loss = {}, validation accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The number of validation images is not equal to that of validation labels.')

def infer_by_neural_net(session, nnInferrer, test_inputs, batch_size, saver=None, checkpoint_dir_path=None, is_time_major=False):
	batch_dim = 1 if is_time_major else 0

	num_inf_examples = 0
	if test_inputs is not None:
		num_inf_examples = test_inputs.shape[batch_dim]

	if num_inf_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		inferences = nnInferrer.infer(session, test_inputs, batch_size)
		print('\tInference time = {}'.format(time.time() - start_time))
		print('[SWL] Info: End inferring...')

		return inferences
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')
		return None

#%%------------------------------------------------------------------

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as ex:
			if os.errno.EEXIST != ex.errno:
				raise

def create_rnn(num_features, num_classes, label_eos_token, is_time_major=False, is_sparse_label=True):
	if is_sparse_label:
		return SimpleRnnWithSparseLabel(num_features, num_classes, is_time_major=is_time_major)
	else:
		return SimpleRnnWithDenseLabel(num_features, num_classes, label_eos_token, is_time_major=is_time_major)

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'ctc_example'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181129T122700'

	is_sparse_label = True
	is_time_major = False
	label_eos_token = -1

	num_features = 13
	# Account the 0th indice + space + blank label = 28 characters.
	num_classes = ord('z') - ord('a') + 1 + 1 + 1
	num_examples = 1

	batch_size = 1  # Number of samples per gradient update.
	num_epochs = 200  # Number of times to iterate over training data.
	#num_batches_per_epoch = int(num_examples / batch_size)
	shuffle = True

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	make_dir(checkpoint_dir_path)
	make_dir(inference_dir_path)
	make_dir(train_summary_dir_path)
	make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mnist/0_download'

	# Constants.
	SPACE_TOKEN = '<space>'
	SPACE_INDEX = 0
	FIRST_INDEX = ord('a') - 1  # 0 is reserved to space.

	# Load the data.
	audio_filepath = '../../../data/machine_learning/LDC93S1.wav'
	target_filepath = '../../../data/machine_learning/LDC93S1.txt'

	fs, audio = wav.read(audio_filepath)

	inputs = mfcc(audio, samplerate=fs)
	# Tranform in 3D array.
	train_inputs = np.asarray(inputs[np.newaxis, :])
	train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
	train_seq_len = [train_inputs.shape[1]]

	# Read targets.
	with open(target_filepath, 'r') as f:
		# Only the last line is necessary.
		line = f.readlines()[-1]

		# Get only the words between [a-z] and replace period for none.
		original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
		targets = original.replace(' ', '  ')
		targets = targets.split(' ')

	# Add blank label.
	targets = np.hstack([SPACE_TOKEN if '' == x else list(x) for x in targets])

	# Transform char into index.
	targets = np.asarray([SPACE_INDEX if SPACE_TOKEN == x else ord(x) - FIRST_INDEX for x in targets])
	print('*******************************1', targets)

	if is_sparse_label:
		# Create sparse representation to feed the placeholder.
		# NOTE [info] {important} >> A tuple (indices, values, shape) for a sparse tensor, not tf.SparseTensor.
		train_outputs = swl_ml_util.generate_sparse_tuple([targets], eos_token=-1)
		#train_outputs = swl_ml_util.generate_sparse_tuple([targets, targets], eos_token=-1)
		#train_outputs = swl_ml_util.generate_sparse_tuple(np.vstack([targets, targets]), eos_token=-1)
	else:
		train_outputs = targets

	# We don't have a validation dataset.
	val_inputs, val_outputs, val_seq_len = train_inputs, train_outputs, train_seq_len
	print('*******************************1', train_outputs)

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Create a model.
			cnnModelForTraining = create_rnn(num_features, num_classes, label_eos_token, is_time_major, is_sparse_label)
			cnnModelForTraining.create_training_model()

			# Create a trainer.
			initial_epoch = 0
			nnTrainer = SimpleRnnTrainer(cnnModelForTraining, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			# Create a model.
			cnnModelForEvaluation = create_rnn(num_features, num_classes, label_eos_token, is_time_major, is_sparse_label)
			cnnModelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(cnnModelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		# Create a model.
		cnnModelForInference = create_rnn(num_features, num_classes, label_eos_token, is_time_major, is_sparse_label)
		cnnModelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(cnnModelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	config = tf.ConfigProto()
	#config.device_count = {'GPU': 2}
	#config.allow_soft_placement = True
	config.log_device_placement = True
	config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=config)
		eval_session = tf.Session(graph=eval_graph, config=config)
	infer_session = tf.Session(graph=infer_graph, config=config)

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#%%------------------------------------------------------------------
	# Train and evaluate.

	if does_need_training:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				# Supports lists of dense and sparse labels.
				train_neural_net_by_batch_lists(sess, nnTrainer, [train_inputs], [train_outputs], [val_inputs], [val_outputs], num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_label)
				# Supports a dense label only.
				#train_neural_net(sess, nnTrainer, train_inputs, train_outputs, val_inputs, val_outputs, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - start_time))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				evaluate_neural_net(sess, nnEvaluator, val_inputs, val_outputs, batch_size, eval_saver, checkpoint_dir_path, is_time_major, is_sparse_label)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infer.

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			# type(inferences) = tf.SparseTensorValue.
			inferences = infer_by_neural_net(sess, nnInferrer, val_inputs, batch_size, infer_saver, checkpoint_dir_path, is_time_major)

			str_decoded = ''.join([chr(x) for x in np.asarray(inferences.values) + FIRST_INDEX])
			# Replaces blank label to none.
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replaces space label to space.
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

			print('Original:\n%s' % original)
			print('Decoded:\n%s' % str_decoded)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	#--------------------
	# Close sessions.

	if does_need_training:
		train_session.close()
		del train_session
		eval_session.close()
		del eval_session
	infer_session.close()
	del infer_session

#%%------------------------------------------------------------------

if '__main__' == __name__:
	try:
		main()
	except:
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		traceback.print_exc(limit=None, file=sys.stdout)
