# REF [site] >> https://www.tensorflow.org/tutorials/recurrent

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	lib_home_dir_path = 'D:/lib_repo/python'
	#lib_home_dir_path = 'D:/lib_repo/python/rnd'
#sys.path.append('../../../src')
sys.path.append(swl_python_home_dir_path + '/src')

#os.chdir(swl_python_home_dir_path + '/test/machine_learning/tensorflow')

#--------------------
#import numpy as np
import tensorflow as tf
from simple_rnn_tf import SimpleRnnUsingTF
from simple_rnn_keras import SimpleRnnUsingKeras
from simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
from swl.machine_learning.tensorflow.neural_net_trainer import TrainingMode
from reverse_function_util import ReverseFunctionDataset
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'reverse_function_rnn'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180116T212902'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
inference_dir_path = './result/{}_inference_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as exception:
			if os.errno.EEXIST != exception.errno:
				raise
make_dir(model_dir_path)
make_dir(inference_dir_path)
make_dir(train_summary_dir_path)
make_dir(val_summary_dir_path)

#%%------------------------------------------------------------------
# Prepare data.

characters = list('abcd')
dataset = ReverseFunctionDataset(characters)

# FIXME [modify] >> In order to use a time-major dataset, trainer, evaluator, and inferrer have to be modified.
is_time_major = False
train_rnn_input_seqs, train_rnn_output_seqs, _, val_rnn_input_seqs, val_rnn_output_seqs, _ = dataset.generate_dataset(is_time_major)
#train_rnn_input_seqs, _, train_rnn_output_seqs, val_rnn_input_seqs, _, val_rnn_output_seqs = dataset.generate_dataset(is_time_major)

#%%------------------------------------------------------------------

def train_neural_net(session, nnTrainer, saver, train_input_seqs, train_output_seqs, val_input_seqs, val_output_seqs, batch_size, num_epochs, shuffle, trainingMode, model_dir_path, train_summary_dir_path, val_summary_dir_path):
	if TrainingMode.START_TRAINING == trainingMode:
		print('[SWL] Info: Start training...')
	elif TrainingMode.RESUME_TRAINING == trainingMode:
		print('[SWL] Info: Resume training...')
	elif TrainingMode.USE_SAVED_MODEL == trainingMode:
		print('[SWL] Info: Use a saved model.')
	else:
		assert False, '[SWL] Error: Invalid training mode.'

	if TrainingMode.RESUME_TRAINING == trainingMode or TrainingMode.USE_SAVED_MODEL == trainingMode:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		start_time = time.time()
		history = nnTrainer.train(session, train_input_seqs, train_output_seqs, val_input_seqs, val_output_seqs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		end_time = time.time()

		print('\tTraining time = {}'.format(end_time - start_time))

		# Display results.
		nnTrainer.display_history(history)

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, saver, val_input_seqs, val_output_seqs, batch_size, model_dir_path):
	# Load a model.
	ckpt = tf.train.get_checkpoint_state(model_dir_path)
	saver.restore(session, ckpt.model_checkpoint_path)
	#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

	print('[SWL] Info: Loaded a model.')
	print('[SWL] Info: Start evaluation...')

	start_time = time.time()
	val_loss, val_acc = nnEvaluator.evaluate(session, val_input_seqs, val_output_seqs, batch_size)
	end_time = time.time()

	print('\tEvaluation time = {}'.format(end_time - start_time))
	print('\tTest loss = {}, test accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

def infer_by_neural_net(session, nnInferrer, saver, test_strs, batch_size, model_dir_path):
	# Character strings -> numeric data.
	test_data = dataset.to_numeric_data(test_strs)

	# Load a model.
	ckpt = tf.train.get_checkpoint_state(model_dir_path)
	saver.restore(session, ckpt.model_checkpoint_path)
	#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

	print('[SWL] Info: Loaded a model.')
	print('[SWL] Info: Start inferring...')

	start_time = time.time()
	inferences = nnInferrer.infer(session, test_data, batch_size)
	end_time = time.time()

	# Numeric data -> character strings.
	inferred_strs = dataset.to_char_strings(inferences, has_start_token=True)

	print('\tInference time = {}'.format(end_time - start_time))
	print('\tTest strings = {}, inferred strings = {}'.format(test_strs, inferred_strs))
	print('[SWL] Info: End inferring...')

is_dynamic = False
if is_dynamic:
	# Dynamic RNNs use variable-length dataset.
	# TODO [improve] >> Training & validation datasets are still fixed-length (static).
	input_shape = (None, None, dataset.vocab_size)
	output_shape = (None, None, dataset.vocab_size)
else:
	# Static RNNs use fixed-length dataset.
	if is_time_major:
		# (time-steps, samples, features).
		input_shape = (dataset.max_token_len, None, dataset.vocab_size)
		output_shape = (dataset.max_token_len, None, dataset.vocab_size)
	else:
		# (samples, time-steps, features).
		input_shape = (None, dataset.max_token_len, dataset.vocab_size)
		output_shape = (None, dataset.max_token_len, dataset.vocab_size)

#%%------------------------------------------------------------------
# RNN models, sessions, and graphs.

#from keras import backend as K

# REF [site] >> https://talbaumel.github.io/attention/
def create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major):
	return SimpleRnnUsingTF(input_shape, output_shape, is_dynamic=is_dynamic, is_bidirectional=is_bidirectional, is_stacked=is_stacked, is_time_major=is_time_major)
	#return SimpleRnnUsingKeras(input_shape, output_shape, is_bidirectional=False, is_stacked=is_stacked)

is_bidirectional = True  # Uses a bidirectional model.
is_stacked = True  # Uses multiple layers.
if is_bidirectional:
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 50  # Number of times to iterate over training data.
else:
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.

#--------------------
# Create graphs.
train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():
#with train_session:
	# Create a model.
	rnnModel = create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major)
	rnnModel.create_training_model()

	# Create a trainer.
	initial_epoch = 0
	nnTrainer = SimpleNeuralNetTrainer(rnnModel, initial_epoch)

	# Create a saver.
	#	Save a model every 2 hours and maximum 5 latest models are saved.
	train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	initializer = tf.global_variables_initializer()

with eval_graph.as_default():
#with eval_session:
	# Create a model.
	rnnModel = create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major)
	rnnModel.create_evaluation_model()

	# Create an evaluator.
	nnEvaluator = NeuralNetEvaluator(rnnModel)

	# Create a saver.
	eval_saver = tf.train.Saver()

with infer_graph.as_default():
#with infer_session:
	# Create a model.
	rnnModel = create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major)
	rnnModel.create_inference_model()

	# Create an inferrer.
	nnInferrer = NeuralNetInferrer(rnnModel)

	# Create a saver.
	infer_saver = tf.train.Saver()

#--------------------
# Configuration.
config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

# Create sessions.
train_session = tf.Session(graph=train_graph, config=config)
eval_session = tf.Session(graph=eval_graph, config=config)
infer_session = tf.Session(graph=infer_graph, config=config)

# Initialize.
train_session.run(initializer)

#%%------------------------------------------------------------------
# Train.

total_elapsed_time = time.time()
with train_session.as_default() as sess:
	#K.set_learning_phase(1)  # Set the learning phase to 'train'.
	shuffle = True
	trainingMode = TrainingMode.START_TRAINING
	train_neural_net(sess, nnTrainer, train_saver, train_rnn_input_seqs, train_rnn_output_seqs, val_rnn_input_seqs, val_rnn_output_seqs, batch_size, num_epochs, shuffle, trainingMode, model_dir_path, train_summary_dir_path, val_summary_dir_path)
print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

#%%------------------------------------------------------------------
# Evaluate and infer.

total_elapsed_time = time.time()
with eval_session.as_default() as sess:
	#K.set_learning_phase(0)  # Set the learning phase to 'test'.
	evaluate_neural_net(sess, nnEvaluator, eval_saver, val_rnn_input_seqs, val_rnn_output_seqs, batch_size, model_dir_path)
print('\tTotal evaluation time = {}'.format(time.time() - total_elapsed_time))

total_elapsed_time = time.time()
with infer_session.as_default() as sess:
	#K.set_learning_phase(0)  # Set the learning phase to 'test'.
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	infer_by_neural_net(sess, nnInferrer, infer_saver, test_strs, batch_size, model_dir_path)
print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

#%%------------------------------------------------------------------
# Close sessions.

train_session.close()
eval_session.close()
infer_session.close()
