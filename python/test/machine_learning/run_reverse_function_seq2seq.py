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
import numpy as np
import tensorflow as tf
from swl.machine_learning.model_trainer import SimpleModelTrainer, SimpleGradientClippingModelTrainer
from swl.machine_learning.model_evaluator import ModelEvaluator
from swl.machine_learning.model_inferrer import ModelInferrer
from simple_seq2seq_encdec import SimpleSeq2SeqEncoderDecoder
from simple_seq2seq_encdec_tf_attention import SimpleSeq2SeqEncoderDecoderWithTfAttention
from reverse_function_data import ReverseFunctionDataGenerator

#--------------------------------------------------------------------

def create_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_attentive, is_bidirectional, is_time_major):
	if is_attentive:
		# Sequence-to-sequence encoder-decoder model w/ TF attention.
		return SimpleSeq2SeqEncoderDecoderWithTfAttention(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_bidirectional, is_time_major)
	else:
		# Sequence-to-sequence encoder-decoder model w/o attention.
		return SimpleSeq2SeqEncoderDecoder(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, is_bidirectional, is_time_major)

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, is_time_major, is_dynamic, is_attentive, is_bidirectional):
		# Sets parameters.
		self._is_time_major, self._is_attentive, self._is_bidirectional = is_time_major, is_attentive, is_bidirectional

		self._sess_config = tf.ConfigProto()
		#self._sess_config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})  # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'.
		self._sess_config.allow_soft_placement = True
		#self._sess_config.log_device_placement = True
		#self._sess_config.operation_timeout_in_ms = 50000
		self._sess_config.gpu_options.allow_growth = True
		#self._sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

		#--------------------
		# Prepares data.

		self._dataGenerator = ReverseFunctionDataGenerator(is_time_major, is_dynamic)

		self._dataGenerator.initialize()

	def train(self, checkpoint_dir_path, output_dir_path, num_epochs, batch_size, shuffle=True, max_gradient_norm=5, initial_epoch=0, is_training_resumed=False, device_name=None):
		# Prepares directories.
		inference_dir_path = os.path.join(output_dir_path, 'inference')
		train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
		val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

		os.makedirs(inference_dir_path, exist_ok=True)
		os.makedirs(train_summary_dir_path, exist_ok=True)
		os.makedirs(val_summary_dir_path, exist_ok=True)

		#--------------------
		# Creates a graph.
		train_graph = tf.Graph()
		with train_graph.as_default():
			with tf.device(device_name):
				encoder_input_shape, decoder_input_shape, decoder_output_shape = self._dataGenerator.shapes
				start_token, end_token = self._dataGenerator.dataset.start_token, self._dataGenerator.dataset.end_token

				# Creates a model.
				modelForTraining = create_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, self._is_attentive, self._is_bidirectional, self._is_time_major)
				modelForTraining.create_training_model()

				# Creates a trainer.
				#modelTrainer = SimpleModelTrainer(modelForTraining, self._dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch, var_list=None)
				modelTrainer = SimpleGradientClippingModelTrainer(modelForTraining, self._dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, max_gradient_norm, initial_epoch, var_list=None)

				initializer = tf.global_variables_initializer()

		# Creates a session.
		train_session = tf.Session(graph=train_graph, config=self._sess_config)

		# Initializes.
		train_session.run(initializer)

		#--------------------
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				self._dataGenerator.initializeTraining(batch_size, shuffle)
				modelTrainer.train(sess, batch_size, num_epochs, shuffle, is_training_resumed)
				self._dataGenerator.finalizeTraining()
		print('\tTotal training time = {}.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		train_session.close()
		del train_session
		#train_graph.reset_default_graph()
		del train_graph

	def evaluate(self, checkpoint_dir_path, batch_size=None, shuffle=False, device_name=None):
		# Creates a graph.
		eval_graph = tf.Graph()
		with eval_graph.as_default():
			with tf.device(device_name):
				encoder_input_shape, decoder_input_shape, decoder_output_shape = self._dataGenerator.shapes
				start_token, end_token = self._dataGenerator.dataset.start_token, self._dataGenerator.dataset.end_token

				# Creates a model.
				modelForEvaluation = create_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, self._is_attentive, self._is_bidirectional, self._is_time_major)
				modelForEvaluation.create_evaluation_model()

				# Creates an evaluator.
				modelEvaluator = ModelEvaluator(modelForEvaluation, self._dataGenerator, checkpoint_dir_path)

		# Creates a session.
		eval_session = tf.Session(graph=eval_graph, config=self._sess_config)

		#--------------------
		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				modelEvaluator.evaluate(sess, batch_size=batch_size, shuffle=shuffle)
		print('\tTotal evaluation time = {}.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		eval_session.close()
		del eval_session
		#eval_graph.reset_default_graph()
		del eval_graph

	def infer(self, checkpoint_dir_path, device_name=None):
		# Creates a graph.
		infer_graph = tf.Graph()
		with infer_graph.as_default():
			with tf.device(device_name):
				encoder_input_shape, decoder_input_shape, decoder_output_shape = self._dataGenerator.shapes
				start_token, end_token = self._dataGenerator.dataset.start_token, self._dataGenerator.dataset.end_token

				# Creates a model.
				modelForInference = create_model(encoder_input_shape, decoder_input_shape, decoder_output_shape, start_token, end_token, self._is_attentive, self._is_bidirectional, self._is_time_major)
				modelForInference.create_inference_model()

				# Creates an inferrer.
				modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

		# Creates a session.
		infer_session = tf.Session(graph=infer_graph, config=self._sess_config)

		#--------------------
		test_strs = ['abc', 'cba', 'dcb', 'abcd', 'dcba', 'cdacbd', 'bcdaabccdb']
		# String data -> one-hot data.
		test_inputs = self._dataGenerator.dataset.encode_data(test_strs)

		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences = modelInferrer.infer(sess, test_inputs)
		print('\tTotal inference time = {}.'.format(time.time() - start_time))

		if inferences:
			print('\tInference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			# One-hot data -> string data.
			inferred_strs = self._dataGenerator.dataset.decode_data(inferences, has_start_token=False)
			print('\tTest strings = {}, inferred strings = {}.'.format(test_strs, inferred_strs))
		else:
			print('[SWL] Warning: Invalid inference results.')

		#--------------------
		# Closes the session and the graph.
		infer_session.close()
		del infer_session
		#infer_graph.reset_default_graph()
		del infer_graph

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	#random.seed(a=None, version=2)
	#np.random.seed(None)
	#tf.set_random_seed(1234)  # Sets a graph-level seed.

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
	max_gradient_norm = 5
	initial_epoch = 0
	is_training_resumed = False

	# REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Graph#device
	# Can use os.environ['CUDA_VISIBLE_DEVICES'] to specify devices.
	train_device_name = None #'/device:GPU:0'
	eval_device_name = None #'/device:GPU:0'
	infer_device_name = None #'/device:GPU:0'

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		output_dir_prefix = 'reverse_function_seq2seq'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		#output_dir_suffix = '20181210T003513'
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')

	#--------------------
	runner = MyRunner(is_time_major, is_dynamic, is_attentive, is_bidirectional)

	if True:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		runner.train(checkpoint_dir_path, output_dir_path, num_epochs, batch_size, shuffle=True, max_gradient_norm=max_gradient_norm, initial_epoch=initial_epoch, is_training_resumed=is_training_resumed, device_name=train_device_name)

	if True:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.evaluate(checkpoint_dir_path, device_name=eval_device_name)

	if True:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.infer(checkpoint_dir_path, device_name=infer_device_name)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
