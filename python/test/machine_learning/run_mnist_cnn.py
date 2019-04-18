#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, time, datetime
import numpy as np
import tensorflow as tf
from swl.machine_learning.model_trainer import SimpleModelTrainer
from swl.machine_learning.model_evaluator import ModelEvaluator
from swl.machine_learning.model_inferrer import ModelInferrer
import swl.util.util as swl_util
from mnist_cnn import MnistCnn
from mnist_data import MnistDataGenerator

#--------------------------------------------------------------------

def create_learning_model(input_shape, output_shape):
	return MnistCnn(input_shape, output_shape)

#--------------------------------------------------------------------

def main():
	#random.seed(a=None, version=2)
	#np.random.seed(None)
	#tf.set_random_seed(1234)  # Sets a graph-level seed.

	#--------------------
	# Sets parameters.

	is_training_required, is_evaluation_required = True, False
	is_training_resumed = False

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	initial_epoch = 0

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 30  # Number of times to iterate over training data.
	shuffle = True

	is_output_augmented = False  # Fixed.
	is_augmented_in_parallel = True

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

	dataGenerator = MnistDataGenerator(is_output_augmented, is_augmented_in_parallel)
	input_shape, output_shape, num_classes = dataGenerator.shapes

	dataGenerator.initialize()

	#%%------------------------------------------------------------------
	# Trains.

	if is_training_required:
		# Creates a graph.
		train_graph = tf.Graph()
		with train_graph.as_default():
			with tf.device(train_device_name):
				# Creates a model.
				modelForTraining = create_learning_model(input_shape, output_shape)
				modelForTraining.create_training_model()

				# Creates a trainer.
				modelTrainer = SimpleModelTrainer(modelForTraining, dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch)

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
				modelForEvaluation = create_learning_model(input_shape, output_shape)
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
				modelForInference = create_learning_model(input_shape, output_shape)
				modelForInference.create_inference_model()

				# Creates an inferrer.
				modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

		# Creates a session.
		infer_session = tf.Session(graph=infer_graph, config=sess_config)

		#--------------------
		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				inferences, ground_truths = list(), list()
				num_test_examples = 0
				#for batch_data, num_batch_examples in dataGenerator.getTestBatches(batch_size, shuffle=False):
				for batch_data, num_batch_examples in dataGenerator.getTestBatches(batch_size=None, shuffle=False):  # Gets the whole test data at a time.
					batch_inputs, batch_outputs = batch_data
					inferences.append(modelInferrer.infer(sess, batch_inputs))
					ground_truths.append(batch_outputs)
				inferences = np.array(inferences)
				ground_truths = np.array(ground_truths)
		print('\tTotal inference time = {}.'.format(time.time() - start_time))

		if inferences is not None:
			if num_classes >= 2:
				inferences = np.argmax(inferences, -1)
				ground_truths = np.argmax(ground_truths, -1)
			else:
				inferences = np.around(inferences)
				#ground_truths = ground_truths
			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			print('\tAccurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
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
