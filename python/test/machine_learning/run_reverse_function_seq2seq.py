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

#--------------------
import sys
sys.path.append('../../src')

import os, time, datetime
#import numpy as np
import tensorflow as tf
from swl.machine_learning.model_trainer import SimpleModelTrainer, SimpleGradientClippingModelTrainer
from swl.machine_learning.model_evaluator import ModelEvaluator
from swl.machine_learning.model_inferrer import ModelInferrer
import swl.util.util as swl_util
from simple_seq2seq_encdec import SimpleSeq2SeqEncoderDecoder
from simple_seq2seq_encdec_tf_attention import SimpleSeq2SeqEncoderDecoderWithTfAttention
from reverse_function_data import ReverseFunctionDataGenerator

#--------------------------------------------------------------------

def create_learning_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_attentive, is_bidirectional, is_time_major):
	if is_attentive:
		# Sequence-to-sequence encoder-decoder model w/ TF attention.
		return SimpleSeq2SeqEncoderDecoderWithTfAttention(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_bidirectional, is_time_major)
	else:
		# Sequence-to-sequence encoder-decoder model w/o attention.
		return SimpleSeq2SeqEncoderDecoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_bidirectional, is_time_major)

#--------------------------------------------------------------------

def main():
	#random.seed(a=None, version=2)
	#np.random.seed(None)
	#tf.set_random_seed(1234)  # Sets a graph-level seed.

	#--------------------
	# Parameters.

	is_training_required, is_evaluation_required = True, True
	is_training_resumed = False

	output_dir_prefix = 'reverse_function_seq2seq'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181210T003513'

	max_gradient_norm = 5
	initial_epoch = 0

	# FIXME [modify] >> In order to use a time-major dataset, trainer, evaluator, and inferrer have to be modified.
	is_time_major = False
	is_dynamic = False  # Uses variable-length time steps.
	is_attentive = True  # Uses attention mechanism.
	is_bidirectional = True  # Uses a bidirectional model.
	if is_attentive:
		batch_size = 4  # Number of samples per gradient update.
		num_epochs = 10  # Number of times to iterate over training data.
	else:
		batch_size = 4  # Number of samples per gradient update.
		num_epochs = 70  # Number of times to iterate over training data.
	shuffle = True

	sess_config = tf.ConfigProto()
	#sess_config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})  # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'.
	sess_config.allow_soft_placement = True
	#sess_config.log_device_placement = True
	#sess_config.operation_timeout_in_ms = 50000
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	# REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Graph#device
	# Can use os.environ['CUDA_VISIBLE_DEVICES'] to specify devices.
	train_device_name = None #'/device:GPU:0'
	eval_device_name = None #'/device:GPU:0'
	infer_device_name = None #'/device:GPU:0'

	#--------------------
	# Prepares directories.

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
	# Prepares data.

	dataGenerator = ReverseFunctionDataGenerator(is_time_major, is_dynamic)
	encoder_input_shape, decoder_input_shape, decoder_output_shape = dataGenerator.shapes
	start_token, end_token = dataGenerator.dataset.start_token, dataGenerator.dataset.end_token

	dataGenerator.initialize()

	#%%------------------------------------------------------------------
	# Trains.

	if is_training_required:
		# Creates a graph.
		train_graph = tf.Graph()
		with train_graph.as_default():
			with tf.device(train_device_name):
				# Creates a model.
				modelForTraining = create_learning_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_attentive, is_bidirectional, is_time_major)
				modelForTraining.create_training_model()

				# Creates a trainer.
				#modelTrainer = SimpleModelTrainer(modelForTraining, dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch, var_list=None)
				modelTrainer = SimpleGradientClippingModelTrainer(modelForTraining, dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, max_gradient_norm, initial_epoch, var_list=None)

				initializer = tf.global_variables_initializer()

		# Creates a session.
		train_session = tf.Session(graph=train_graph, config=sess_config)

		# Initializes.
		train_session.run(initializer)

		#--------------------
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				dataGenerator.initializeTraining(batch_size, shuffle)
				modelTrainer.train(sess, batch_size, num_epochs, shuffle, is_training_resumed)
				dataGenerator.finalizeTraining()
		print('\tTotal training time = {}.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		train_session.close()
		del train_session
		#train_graph.reset_default_graph()
		del train_graph

	#%%------------------------------------------------------------------
	# Evaluates.

	if is_evaluation_required:
		# Creates a graph.
		eval_graph = tf.Graph()
		with eval_graph.as_default():
			with tf.device(eval_device_name):
				# Creates a model.
				modelForEvaluation = create_learning_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_attentive, is_bidirectional, is_time_major)
				modelForEvaluation.create_evaluation_model()

				# Creates an evaluator.
				modelEvaluator = ModelEvaluator(modelForEvaluation, dataGenerator, checkpoint_dir_path)

		# Creates a session.
		eval_session = tf.Session(graph=eval_graph, config=sess_config)

		#--------------------
		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				modelEvaluator.evaluate(sess, batch_size=None, shuffle=False)
		print('\tTotal evaluation time = {}.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		eval_session.close()
		del eval_session
		#eval_graph.reset_default_graph()
		del eval_graph

	#%%------------------------------------------------------------------
	# Infers.

	if True:
		# Creates a graph.
		infer_graph = tf.Graph()
		with infer_graph.as_default():
			with tf.device(infer_device_name):
				# Creates a model.
				modelForInference = create_learning_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_attentive, is_bidirectional, is_time_major)
				modelForInference.create_inference_model()

				# Creates an inferrer.
				modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

		# Creates a session.
		infer_session = tf.Session(graph=infer_graph, config=sess_config)

		#--------------------
		test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
		# String data -> numeric data.
		test_inputs = dataGenerator.dataset.to_numeric(test_strs)

		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences = modelInferrer.infer(sess, test_inputs)
		print('\tTotal inference time = {}.'.format(time.time() - start_time))

		if inferences is not None:
			# Numeric data -> string data.
			inferred_strs = dataGenerator.dataset.to_string(inferences, has_start_token=False)
			print('\tTest strings = {}, inferred strings = {}.'.format(test_strs, inferred_strs))
		else:
			print('[SWL] Warning: Invalid inference results.')

		#--------------------
		# Closes the session and the graph.
		infer_session.close()
		del infer_session
		#infer_graph.reset_default_graph()
		del infer_graph

#%%------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
