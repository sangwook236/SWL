#!/usr/bin/env python

# REF [paper] >> "Describing Multimedia Content Using Attention-Based Encoder-Decoder Networks", ToM 2015.
# REF [paper] >> "Neural Machine Translation by Jointly Learning to Align and Translate", arXiv 2016.
# REF [paper] >> "Effective Approaches to Attention-based Neural Machine Translation", arXiv 2015.
# REF [paper] >> "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML 2015.

# REF [site] >> https://talbaumel.github.io/attention/ ==> Neural Attention Mechanism - Sequence To Sequence Attention Models In DyNet.pdf
# REF [site] >> https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py
# REF [site] >> https://www.tensorflow.org/tutorials/seq2seq
# REF [site] >> https://www.tensorflow.org/tutorials/recurrent

# REF [site] >> https://blog.heuritech.com/2016/01/20/attention-mechanism/
# REF [site] >> https://github.com/philipperemy/keras-attention-mechanism

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
#import numpy as np
import tensorflow as tf
from simple_seq2seq_encdec import SimpleSeq2SeqEncoderDecoder
from simple_seq2seq_encdec_tf_attention import SimpleSeq2SeqEncoderDecoderWithTfAttention
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetGradientTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.machine_learning.util as swl_ml_util
from reverse_function_util import ReverseFunctionDataset
import time
import traceback

#%%------------------------------------------------------------------

def train_neural_net(session, nnTrainer, train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train_seq2seq(session, train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, saver=None, checkpoint_dir_path=None):
	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start evaluation...')
	start_time = time.time()
	val_loss, val_acc = nnEvaluator.evaluate_seq2seq(session, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size)
	print('\tEvaluation time = {}'.format(time.time() - start_time))
	print('\tTest loss = {}, test accurary = {}'.format(val_loss, val_acc))
	print('[SWL] Info: End evaluation...')

def infer_by_neural_net(session, nnInferrer, dataset, test_strs, batch_size, saver=None, checkpoint_dir_path=None):
	# Character strings -> numeric data.
	test_data = dataset.to_numeric_data(test_strs)

	if saver is not None and checkpoint_dir_path is not None:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Loaded a model.')

	print('[SWL] Info: Start inferring...')
	start_time = time.time()
	inferences = nnInferrer.infer(session, test_data, batch_size)
	print('\tInference time = {}'.format(time.time() - start_time))

	# Numeric data -> character strings.
	inferred_strs = dataset.to_char_strings(inferences, has_start_token=False)

	print('\tTest strings = {}, inferred strings = {}'.format(test_strs, inferred_strs))
	print('[SWL] Info: End inferring...')

#%%------------------------------------------------------------------

import datetime

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as ex:
			if os.errno.EEXIST != ex.errno:
				raise

def create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_attentive, is_bidirectional, is_time_major):
	if is_attentive:
		# Sequence-to-sequence encoder-decoder model w/ TF attention.
		return SimpleSeq2SeqEncoderDecoderWithTfAttention(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset.start_token, dataset.end_token, is_bidirectional=is_bidirectional, is_time_major=is_time_major)
	else:
		# Sequence-to-sequence encoder-decoder model w/o attention.
		return SimpleSeq2SeqEncoderDecoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset.start_token, dataset.end_token, is_bidirectional=is_bidirectional, is_time_major=is_time_major)

def main():
	#np.random.seed(7)

	does_need_training = True
	does_resume_training = False

	#--------------------
	# Prepare directories.

	output_dir_prefix = 'reverse_function_seq2seq'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180222T144236'

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	make_dir(checkpoint_dir_path)
	make_dir(inference_dir_path)
	make_dir(train_summary_dir_path)
	make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	characters = list('abcd')
	dataset = ReverseFunctionDataset(characters)

	# FIXME [modify] >> In order to use a time-major dataset, trainer, evaluator, and inferrer have to be modified.
	is_time_major = False
	# NOTICE [info] >> How to use the hidden state c of an encoder in a decoder?
	#	1) The hidden state c of the encoder is used as the initial state of the decoder and the previous output of the decoder may be used as its only input.
	#	2) The previous output of the decoder is used as its input along with the hidden state c of the encoder.
	train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs = dataset.generate_dataset(is_time_major)

	is_dynamic = False
	if is_dynamic:
		# Dynamic RNNs use variable-length dataset.
		# TODO [improve] >> Training & validation datasets are still fixed-length (static).
		encoder_input_shape = (None, None, dataset.vocab_size)
		decoder_input_shape = (None, None, dataset.vocab_size)
		decoder_output_shape = (None, None, dataset.vocab_size)
	else:
		# Static RNNs use fixed-length dataset.
		if is_time_major:
			# (time-steps, samples, features).
			encoder_input_shape = (dataset.max_token_len, None, dataset.vocab_size)
			decoder_input_shape = (dataset.max_token_len, None, dataset.vocab_size)
			decoder_output_shape = (dataset.max_token_len, None, dataset.vocab_size)
		else:
			# (samples, time-steps, features).
			encoder_input_shape = (None, dataset.max_token_len, dataset.vocab_size)
			decoder_input_shape = (None, dataset.max_token_len, dataset.vocab_size)
			decoder_output_shape = (None, dataset.max_token_len, dataset.vocab_size)

	#--------------------
	# Create models, sessions, and graphs.

	is_attentive = True  # Uses attention mechanism.
	is_bidirectional = True  # Uses a bidirectional model.
	if is_attentive:
		batch_size = 4  # Number of samples per gradient update.
		num_epochs = 10  # Number of times to iterate over training data.
	else:
		batch_size = 4  # Number of samples per gradient update.
		num_epochs = 70  # Number of times to iterate over training data.
	shuffle = True

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Create a model.
			rnnModelForTraining = create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_attentive, is_bidirectional, is_time_major)
			rnnModelForTraining.create_training_model()

			# Create a trainer.
			initial_epoch = 0
			#nnTrainer = SimpleNeuralNetTrainer(rnnModelForTraining, initial_epoch)
			nnTrainer = SimpleNeuralNetGradientTrainer(rnnModelForTraining, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			# Create a model.
			rnnModelForEvaluation = create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_attentive, is_bidirectional, is_time_major)
			rnnModelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(rnnModelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		# Create a model.
		rnnModelForInference = create_seq2seq_encoder_decoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, dataset, is_attentive, is_bidirectional, is_time_major)
		rnnModelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(rnnModelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	config = tf.ConfigProto()
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
		total_elapsed_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				train_neural_net(sess, nnTrainer, train_encoder_input_seqs, train_decoder_input_seqs, train_decoder_output_seqs, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

		total_elapsed_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				evaluate_neural_net(sess, nnEvaluator, val_encoder_input_seqs, val_decoder_input_seqs, val_decoder_output_seqs, batch_size, eval_saver, checkpoint_dir_path)
		print('\tTotal evaluation time = {}'.format(time.time() - total_elapsed_time))

	#%%------------------------------------------------------------------
	# Infer.

	total_elapsed_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
			infer_by_neural_net(sess, nnInferrer, dataset, test_strs, batch_size, infer_saver, checkpoint_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

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
