#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, time, datetime
from functools import partial
import threading
import numpy as np
import tensorflow as tf
from swl.machine_learning.model_trainer import ModelTrainer
from swl.machine_learning.model_evaluator import ModelEvaluator
from swl.machine_learning.model_inferrer import ModelInferrer
import swl.util.util as swl_util
from synth90k_crnn import Synth90kCrnnWithCrossEntropyLoss, Synth90kCrnnWithCtcLoss
from synth90k_data import Synth90kDataGenerator

#%%------------------------------------------------------------------

def create_learning_model(image_height, image_width, image_channel, num_classes, is_sparse_output):
	if is_sparse_output:
		return Synth90kCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes)
	else:
		return Synth90kCrnnWithCrossEntropyLoss(image_height, image_width, image_channel, num_classes)

#%%------------------------------------------------------------------

class SimpleCrnnTrainer(ModelTrainer):
	def __init__(self, model, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 1.0
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08)

		super().__init__(model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, global_step)

#%%------------------------------------------------------------------

# REF [function] >> training_worker_proc() in ${SWL_PYTHON_HOME}/python/test/machine_learning/batch_generator_and_loader_test.py.
def training_worker_thread_proc(session, modelTrainer, batch_size, num_epochs, shuffle, is_training_resumed):
	print('\t{}({}): Start training worker thread.'.format(os.getpid(), threading.get_ident()))

	modelTrainer.train(session, batch_size, num_epochs, shuffle, is_training_resumed)

	print('\t{}({}): End training worker thread.'.format(os.getpid(), threading.get_ident()))

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Sets parameters.

	is_training_required = True
	is_training_resumed = False

	output_dir_prefix = 'synth90k_crnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	initial_epoch = 0

	# When outputs are not sparse, CRNN model's output shape = (samples, 32, num_classes) and dataset's output shape = (samples, 23, num_classes).
	is_sparse_output = True  # Fixed.
	#is_time_major = False  # Fixed.

	batch_size = 256  # Number of samples per gradient update.
	num_epochs = 100  # Number of times to iterate over training data.
	shuffle = True

	is_output_augmented = False  # Fixed.
	is_augmented_in_parallel = True

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

	dataGenerator = Synth90kDataGenerator(num_epochs, is_sparse_output, is_output_augmented, is_augmented_in_parallel)
	image_height, image_width, image_channel, num_classes = dataGenerator.shapes
	#label_sos_token, label_eos_token = dataGenerator.dataset.start_token, dataGenerator.dataset.end_token
	#label_eos_token = dataGenerator.dataset.end_token

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
				modelForTraining = create_learning_model(image_height, image_width, image_channel, num_classes, is_sparse_output)
				modelForTraining.create_training_model()

				# Creates a trainer.
				modelTrainer = SimpleCrnnTrainer(modelForTraining, dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch)

				initializer = tf.global_variables_initializer()

	with eval_graph.as_default():
		with tf.device(eval_device_name):
			# Creates a model.
			modelForEvaluation = create_learning_model(image_height, image_width, image_channel, num_classes, is_sparse_output)
			modelForEvaluation.create_evaluation_model()

			# Creates an evaluator.
			modelEvaluator = ModelEvaluator(modelForEvaluation, dataGenerator, checkpoint_dir_path)

	with infer_graph.as_default():
		with tf.device(infer_device_name):
			# Creates a model.
			modelForInference = create_learning_model(image_height, image_width, image_channel, num_classes, is_sparse_output)
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
				dataGenerator.initializeTraining(batch_size, shuffle)
				if True:
					modelTrainer.train(sess, batch_size, num_epochs, shuffle, is_training_resumed)
				else:
					# Uses a training worker thread.
					training_worker_thread = threading.Thread(target=training_worker_thread_proc, args=(sess, modelTrainer, batch_size, num_epochs, shuffle, is_training_resumed))
					training_worker_thread.start()

					training_worker_thread.join()
				dataGenerator.finalizeTraining()
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
			inferences, test_outputs = list(), list()
			num_test_examples = 0
			for batch_data, num_batch_examples in dataGenerator.getTestBatches(batch_size, shuffle=False):
				batch_inputs, batch_outputs = batch_data
				inferences.append(modelInferrer.infer(sess, batch_inputs))
				test_outputs.append(batch_outputs)
			inferences = np.array(inferences)
			test_outputs = np.array(test_outputs)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	#--------------------
	if inferences is not None:
		if num_classes >= 2:
			inferences = np.argmax(inferences, -1)
			groundtruths = np.argmax(test_labels, -1)
		else:
			inferences = np.around(inferences)
			groundtruths = test_labels
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
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
