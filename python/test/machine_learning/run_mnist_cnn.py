#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, time, datetime
import numpy as np
import tensorflow as tf
from swl.machine_learning.model_trainer import SimpleModelTrainer
from swl.machine_learning.model_evaluator import ModelEvaluator
from swl.machine_learning.model_inferrer import ModelInferrer
from mnist_cnn import MnistCnn
from mnist_data import MnistDataGenerator

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self):
		# Sets parameters.
		is_output_augmented = False  # Fixed.
		is_augmented_in_parallel = True

		self._sess_config = tf.ConfigProto()
		#self._sess_config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})  # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'.
		self._sess_config.allow_soft_placement = True
		#self._sess_config.log_device_placement = True
		#self._sess_config.operation_timeout_in_ms = 50000
		self._sess_config.gpu_options.allow_growth = True
		#self._sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

		#--------------------
		# Prepares data.

		self._dataGenerator = MnistDataGenerator(is_output_augmented, is_augmented_in_parallel)
		self._input_shape, self._output_shape, self._num_classes = self._dataGenerator.shapes

		self._dataGenerator.initialize()

	def train(self, checkpoint_dir_path, output_dir_path, num_epochs, batch_size, shuffle=True, initial_epoch=0, is_training_resumed=False, device_name=None):
		# Prepares directories.
		train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
		val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

		os.makedirs(train_summary_dir_path, exist_ok=True)
		os.makedirs(val_summary_dir_path, exist_ok=True)

		#--------------------
		# Creates a graph.
		train_graph = tf.Graph()
		with train_graph.as_default():
			with tf.device(device_name):
				# Creates a model.
				modelForTraining = MnistCnn(self._input_shape, self._output_shape)
				modelForTraining.create_training_model()

				# Creates a trainer.
				modelTrainer = SimpleModelTrainer(modelForTraining, self._dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch, var_list=None)

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
		print('\tTotal training time = {} secs.'.format(time.time() - start_time))

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
				# Creates a model.
				modelForEvaluation = MnistCnn(self._input_shape, self._output_shape)
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
		print('\tTotal evaluation time = {} secs.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		eval_session.close()
		del eval_session
		#eval_graph.reset_default_graph()
		del eval_graph

	def infer(self, checkpoint_dir_path, batch_size=None, shuffle=False, device_name=None):
		# Creates a graph.
		infer_graph = tf.Graph()
		with infer_graph.as_default():
			with tf.device(device_name):
				# Creates a model.
				modelForInference = MnistCnn(self._input_shape, self._output_shape)
				modelForInference.create_inference_model()

				# Creates an inferrer.
				modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

		# Creates a session.
		infer_session = tf.Session(graph=infer_graph, config=self._sess_config)

		#--------------------
		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences, ground_truths = list(), list()
				num_test_examples = 0
				for batch_data, num_batch_examples in self._dataGenerator.getTestBatches(batch_size=batch_size, shuffle=shuffle):  # Gets the whole test data at a time if batch_size = None.
					batch_inputs, batch_outputs = batch_data
					inferences.append(modelInferrer.infer(sess, batch_inputs))
					ground_truths.append(batch_outputs)
				inferences = np.array(inferences)
				ground_truths = np.array(ground_truths)
		print('\tTotal inference time = {} secs.'.format(time.time() - start_time))

		if inferences is not None and ground_truths is not None:
			print('\tInference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			if self._num_classes >= 2:
				inferences = np.argmax(inferences, -1)
				ground_truths = np.argmax(ground_truths, -1)
			else:
				inferences = np.around(inferences)
				#ground_truths = ground_truths
			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			print('\tInference: accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
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

	num_epochs = 30  # Number of times to iterate over training data.
	batch_size = 128  # Number of samples per gradient update.
	initial_epoch = 0
	is_training_resumed = False

	# REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Graph#device
	# Can use os.environ['CUDA_VISIBLE_DEVICES'] to specify devices.
	train_device_name = None #'/device:GPU:0'
	eval_device_name = None #'/device:GPU:0'
	infer_device_name = None #'/device:GPU:0'

	#--------------------
	output_dir_path = None
	if not output_dir_path:
		output_dir_prefix = 'mnist_cnn'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		#output_dir_suffix = '20180302T155710'
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')

	#--------------------
	runner = MyRunner()

	if True:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		runner.train(checkpoint_dir_path, output_dir_path, num_epochs, batch_size, shuffle=True, initial_epoch=initial_epoch, is_training_resumed=is_training_resumed, device_name=train_device_name)

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
