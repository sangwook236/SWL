# REF [site] >> https://talbaumel.github.io/attention/ ==> Neural Attention Mechanism - Sequence To Sequence Attention Models In DyNet.pdf
# REF [site] >> https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py
# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
# REF [site] >> https://www.tensorflow.org/tutorials/recurrent

# REF [site] >> https://blog.heuritech.com/2016/01/20/attention-mechanism/
# REF [site] >> https://github.com/philipperemy/keras-attention-mechanism

# REF [paper] >> "Describing Multimedia Content Using Attention-Based Encoder-Decoder Networks", ToM 2015.
# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
# REF [paper] >> "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML 2015.

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
sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_home_dir_path + '/tflearn_github')
#sys.path.append('../../../src')

#os.chdir(swl_python_home_dir_path + '/test/machine_learning/tensorflow')

#--------------------
#import numpy as np
import tensorflow as tf
from simple_seq2seq_encdec_attention import SimpleSeq2SeqEncoderDecoderWithAttention
from simple_seq2seq_trainer import SimpleSeq2SeqTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_predictor import NeuralNetPredictor
#from swl.machine_learning.tensorflow.neural_net_trainer import TrainingMode
from reverse_function_util import ReverseFunctionDataset
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'reverse_function_seq2seq'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180116T212902'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
prediction_dir_path = './result/{}_prediction_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

#%%------------------------------------------------------------------
# Prepare data.
	
characters = list('abcd')
dataset = ReverseFunctionDataset(characters)

# FIXME [modify] >> In order to use a time-major dataset, trainer, evaluator, and predictor have to be modified.
is_time_major = False
train_input_seqs, train_output_seqs, train_output_seqs_of_one_timestep, val_input_seqs, val_output_seqs, val_output_seqs_ahead_of_one_timestep = dataset.generate_dataset(is_time_major)

#%%------------------------------------------------------------------
# Configure tensorflow.

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

# REF [site] >> https://stackoverflow.com/questions/45093688/how-to-understand-sess-as-default-and-sess-graph-as-default
#graph = tf.Graph()
#session = tf.Session(graph=graph, config=config)
session = tf.Session(config=config)

#%%------------------------------------------------------------------

def train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch):
	nnTrainer = SimpleSeq2SeqTrainer(rnnModel, initial_epoch)
	session.run(tf.global_variables_initializer())
	with session.as_default() as sess:
		# Save a model every 2 hours and maximum 5 latest models are saved.
		saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		start_time = time.time()
		history = nnTrainer.train(sess, train_input_seqs, train_output_seqs, val_input_seqs, val_output_seqs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		end_time = time.time()

		print('\tTraining time = {}'.format(end_time - start_time))

		# Display results.
		nnTrainer.display_history(history)

def evaluate_model(session, rnnModel, batch_size):
	nnEvaluator = NeuralNetEvaluator()
	with session.as_default() as sess:
		start_time = time.time()
		test_loss, test_acc = nnEvaluator.evaluate(sess, rnnModel, val_input_seqs, val_output_seqs, batch_size)
		end_time = time.time()

		print('\tEvaluation time = {}'.format(end_time - start_time))
		print('\tTest loss = {}, test accurary = {}'.format(test_loss, test_acc))

def predict_model(session, rnnModel, batch_size, test_strs):
	nnPredictor = NeuralNetPredictor()
	with session.as_default() as sess:
		# Character strings -> numeric data.
		test_data = dataset.to_numeric_data(test_strs)

		start_time = time.time()
		predictions = nnPredictor.predict(sess, rnnModel, test_data, batch_size)
		end_time = time.time()

		# Numeric data -> character strings.
		predicted_strs = dataset.to_char_strings(predictions)

		print('\tPrediction time = {}'.format(end_time - start_time))
		print('\tTest strings = {}, predicted strings = {}'.format(test_strs, predicted_strs))

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
# Sequence-to-sequence model w/ attention.

if True:
	is_bidirectional = False
	rnnModel = SimpleSeq2SeqEncoderDecoderWithAttention(input_shape, output_shape, is_bidirectional=is_bidirectional, is_time_major=is_time_major)

	#--------------------
	batch_size = 4  # Number of samples per gradient update.
	num_epochs = 50  # Number of times to iterate over training data.

	shuffle = True
	initial_epoch = 0

	train_model(session, rnnModel, batch_size, num_epochs, shuffle, initial_epoch)
	evaluate_model(session, rnnModel, batch_size)
	test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
	predict_model(session, rnnModel, batch_size, test_strs)
