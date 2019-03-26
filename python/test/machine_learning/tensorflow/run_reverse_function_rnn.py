#!/usr/bin/env python

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
sys.path.append(os.path.join(swl_python_home_dir_path, 'src'))

#os.chdir(os.path.join(swl_python_home_dir_path, 'test/machine_learning/tensorflow'))

#--------------------
import time, datetime
#import numpy as np
import tensorflow as tf
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from reverse_function_util import ReverseFunctionDataset
from simple_rnn_tf import SimpleRnnUsingTF
#from simple_rnn_keras import SimpleRnnUsingKeras
#from keras import backend as K
import traceback

#%%------------------------------------------------------------------

# REF [site] >> https://talbaumel.github.io/attention/
def create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major):
	return SimpleRnnUsingTF(input_shape, output_shape, is_dynamic=is_dynamic, is_bidirectional=is_bidirectional, is_stacked=is_stacked, is_time_major=is_time_major)
	#return SimpleRnnUsingKeras(input_shape, output_shape, is_bidirectional=False, is_stacked=is_stacked)

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'reverse_function_rnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180116T212902'

	initial_epoch = 0

	characters = list('abcd')

	# FIXME [modify] >> In order to use a time-major dataset, trainer, evaluator, and inferrer have to be modified.
	is_time_major = False
	is_dynamic = False
	is_bidirectional = True  # Uses a bidirectional model.
	is_stacked = True  # Uses multiple layers.
	if is_bidirectional:
		batch_size = 4  # Number of samples per gradient update.
		num_epochs = 50  # Number of times to iterate over training data.
	else:
		batch_size = 4  # Number of samples per gradient update.
		num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True

	augmenter = None
	is_output_augmented = False

	sess_config = tf.ConfigProto()
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	dataset = ReverseFunctionDataset(characters)

	train_rnn_input_seqs, train_rnn_output_seqs, _, val_rnn_input_seqs, val_rnn_output_seqs, _ = dataset.generate_dataset(is_time_major)
	#train_rnn_input_seqs, _, train_rnn_output_seqs, val_rnn_input_seqs, _, val_rnn_output_seqs = dataset.generate_dataset(is_time_major)

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

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			#K.set_learning_phase(1)  # Set the learning phase to 'train'. (Required)

			# Create a model.
			modelForTraining = create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major)
			modelForTraining.create_training_model()

			# Create a trainer.
			nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch, augmenter, is_output_augmented)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			#K.set_learning_phase(0)  # Set the learning phase to 'test'. (Required)

			# Create a model.
			modelForEvaluation = create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major)
			modelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		#K.set_learning_phase(0)  # Set the learning phase to 'test'. (Required)

		# Create a model.
		modelForInference = create_rnn(input_shape, output_shape, is_dynamic, is_bidirectional, is_stacked, is_time_major)
		modelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(modelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#%%------------------------------------------------------------------
	# Train and evaluate.

	if does_need_training:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(1)  # Set the learning phase to 'train'.
				swl_tf_util.train_neural_net(sess, nnTrainer, train_rnn_input_seqs, train_rnn_output_seqs, val_rnn_input_seqs, val_rnn_output_seqs, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - start_time))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(0)  # Set the learning phase to 'test'.
				swl_tf_util.evaluate_neural_net(sess, nnEvaluator, val_rnn_input_seqs, val_rnn_output_seqs, batch_size, eval_saver, checkpoint_dir_path, is_time_major)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infer.

	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	# String data -> numeric data.
	test_data = dataset.to_numeric(test_strs)

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			#K.set_session(sess)
			#K.set_learning_phase(0)  # Set the learning phase to 'test'.
			inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_data, batch_size, infer_saver, checkpoint_dir_path, is_time_major)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	if inferences is not None:
		# Numeric data -> string data.
		inferred_strs = dataset.to_string(inferences, has_start_token=True)
		print('\tTest strings = {}, inferred strings = {}'.format(test_strs, inferred_strs))
	else:
		print('[SWL] Warning: Invalid inference results.')

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
