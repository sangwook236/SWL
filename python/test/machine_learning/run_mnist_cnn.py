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
from mnist_data import MnistDataGenerator, MnistDataPreprocessor, ImgaugDataAugmenter

#%%------------------------------------------------------------------

def create_learning_model(input_shape, output_shape):
	return MnistCnn(input_shape, output_shape)

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Sets parameters.

	is_training_required = True
	is_training_resumed = False

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	initial_epoch = 0

	num_classes = 10
	input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 30  # Number of times to iterate over training data.
	shuffle = True
	is_output_augmented = False  # Fixed.

	sess_config = tf.ConfigProto()
	#sess_config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})  # os.environ['CUDA_VISIBLE_DEVICES'] = 0,1.
	sess_config.allow_soft_placement = True
	#sess_config.log_device_placement = True
	#sess_config.operation_timeout_in_ms = 50000
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	train_device_name = '/device:GPU:1'
	eval_device_name = '/device:GPU:1'
	infer_device_name = '/device:GPU:1'

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

	augmenter = ImgaugDataAugmenter(is_output_augmented, is_augmented_in_parallel=True)
	#augmenter = None
	preprocessor = MnistDataPreprocessor(input_shape[1:], num_classes)
	dataGenerator = MnistDataGenerator(preprocessor, augmenter)

	#--------------------
	# Creates models, sessions, and graphs.

	# Creates graphs.
	if is_training_required:
		train_graph = tf.Graph()
	eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if is_training_required:
		with train_graph.as_default():
			with tf.device(train_device_name):
				# Creates a model.
				modelForTraining = create_learning_model(input_shape, output_shape)
				modelForTraining.create_training_model()

				# Creates a trainer.
				modelTrainer = SimpleModelTrainer(modelForTraining, dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch)

				initializer = tf.global_variables_initializer()

	with eval_graph.as_default():
		with tf.device(eval_device_name):
			# Creates a model.
			modelForEvaluation = create_learning_model(input_shape, output_shape)
			modelForEvaluation.create_evaluation_model()

			# Creates an evaluator.
			modelEvaluator = ModelEvaluator(modelForEvaluation, dataGenerator, checkpoint_dir_path)

	with infer_graph.as_default():
		with tf.device(infer_device_name):
			# Creates a model.
			modelForInference = create_learning_model(input_shape, output_shape)
			modelForInference.create_inference_model()

			# Creates an inferrer.
			modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

	# Creates sessions.
	if is_training_required:
		train_session = tf.Session(graph=train_graph, config=sess_config)
	eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initializes.
	if is_training_required:
		train_session.run(initializer)

	dataGenerator.initialize()

	#%%------------------------------------------------------------------
	# Trains.

	if is_training_required:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				modelTrainer.train(sess, batch_size, num_epochs, shuffle, is_training_resumed)
		print('\tTotal training time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Evaluates.

	if True:
		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				modelEvaluator.evaluate(sess, batch_size=None, shuffle=False)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infers.

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			if True:
				(test_inputs, test_outputs), num_test_examples = dataGenerator.getTestData()
				inferences = modelInferrer.infer(sess, test_inputs)
			else:
				inferences, test_outputs = list(), list()
				num_test_examples = 0
				for batch_data, num_batch_examples in dataGenerator.getTestBatches(batch_size, shuffle=False):
					batch_inputs, batch_outputs = batch_data
					inferences.append(modelInferrer.infer(sess, batch_inputs))
					test_outputs.append(batch_outputs)
				inferences = np.array(inferences)
				test_outputs = np.array(test_outputs)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	if inferences is not None:
		if num_classes >= 2:
			inferences = np.argmax(inferences, -1)
			ground_truths = np.argmax(test_outputs, -1)
		else:
			inferences = np.around(inferences)
			ground_truths = test_outputs
		correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
	else:
		print('[SWL] Warning: Invalid inference results.')

	#--------------------
	# Closes sessions.

	if is_training_required:
		train_session.close()
		del train_session
	eval_session.close()
	del eval_session
	infer_session.close()
	del infer_session

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
