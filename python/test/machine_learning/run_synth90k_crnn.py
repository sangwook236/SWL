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
import swl.machine_learning.util as swl_ml_util
from synth90k_crnn import Synth90kCrnnWithCrossEntropyLoss, Synth90kCrnnWithCtcLoss
from synth90k_data import Synth90kDataGenerator

#%%------------------------------------------------------------------

def create_learning_model(image_height, image_width, image_channel, num_classes, is_sparse_output):
	if is_sparse_output:
		return Synth90kCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes)
	else:
		return Synth90kCrnnWithCrossEntropyLoss(image_height, image_width, image_channel, num_classes)

#%%------------------------------------------------------------------

def user_defined_learning_rate(learning_rate, global_step, gamma, power, name=None):
	if global_step is None:
		raise ValueError('global_step should not be None')

    with tf.name_scope(name, 'user_defined_learning_rate', [learning_rate, global_step, gamma, power]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate, name='learning_rate')
        dtype = learning_rate.dtype
        global_step = tf.cast(global_step, dtype)
        gamma = tf.cast(gamma, dtype)
        power = tf.cast(power, dtype)
        base = tf.multiply(gamma, global_step)
        return tf.multiply(learning_rate, tf.pow(1 + base, -power), name=name)

class SimpleCrnnTrainer(ModelTrainer):
	def __init__(self, model, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			init_learning_rate = 0.001
			decay_steps = 10000
			decay_rate = 0.96
			learning_rate = init_learning_rate
			#learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.inverse_time_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.natural_exp_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.cosine_decay(init_learning_rate, global_step, decay_steps, alpha=0.0)
			#learning_rate = tf.train.linear_cosine_decay(init_learning_rate, global_step, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.noisy_linear_cosine_decay(init_learning_rate, global_step, decay_steps, initial_variance=1.0, variance_decay=0.55, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.polynomial_decay(init_learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False)
			#learning_rate = user_defined_learning_rate(init_learning_rate, global_step, gamma=0.0001, power=0.75)
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

	is_training_required, is_evaluation_required = True, True
	is_training_resumed = False

	output_dir_prefix = 'synth90k_crnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20190320T134245'

	initial_epoch = 0

	# When outputs are not sparse, CRNN model's output shape = (samples, 32, num_classes) and dataset's output shape = (samples, 23, num_classes).
	is_sparse_output = True  # Fixed.
	#is_time_major = False  # Fixed.

	batch_size = 256  # Number of samples per gradient update.
	num_epochs = 100  # Number of times to iterate over training data.
	shuffle = True

	is_output_augmented = False  # Fixed.
	is_augmented_in_parallel = True
	is_npy_files_used_as_input = False  # Specifies whether npy files or image files are used as input. Using npy files is faster.

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

	dataGenerator = Synth90kDataGenerator(num_epochs, is_sparse_output, is_output_augmented, is_augmented_in_parallel, is_npy_files_used_as_input)
	image_height, image_width, image_channel, num_classes = dataGenerator.shapes
	#label_sos_token, label_eos_token = dataGenerator.dataset.start_token, dataGenerator.dataset.end_token
	label_eos_token = dataGenerator.dataset.end_token

	#--------------------
	# Creates models, sessions, and graphs.

	# Creates graphs.
	if is_training_required:
		train_graph = tf.Graph()
	if is_evaluation_required:
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

	if is_evaluation_required:
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
	if is_evaluation_required:
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initializes.
	if is_training_required:
		train_session.run(initializer)

	dataGenerator.initialize(batch_size)

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
		print('\tTotal training time = {}.'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Evaluates.

	if is_evaluation_required:
		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				#modelEvaluator.evaluate(sess, batch_size=None, shuffle=False)  # Exception: NotImplementedError is raised in dataGenerator.getValidationData().
				modelEvaluator.evaluate(sess, batch_size=batch_size, shuffle=False)
		print('\tTotal evaluation time = {}.'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infers.

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			inferences, ground_truths = list(), list()
			num_test_examples = 0
			for batch_data, num_batch_examples in dataGenerator.getTestBatches(batch_size, shuffle=False):
				# A sparse tensor expressed by a tuple with (indices, values, dense_shape) -> a dense tensor of dense_shape.
				stv = modelInferrer.infer(sess, batch_data[0])  # tf.SparseTensorValue.
				print('*******************', stv.dense_shape)
				dense_batch_inferences = swl_ml_util.sparse_to_dense(stv.indices, stv.values, stv.dense_shape, default_value=label_eos_token, dtype=np.int8)
				dense_batch_outputs = swl_ml_util.sparse_to_dense(*batch_data[1], default_value=label_eos_token, dtype=np.int8)
				inferences.append(dense_batch_inferences)
				ground_truths.append(dense_batch_outputs)
			# Variable-length numpy.arrays are not merged into a single numpy.array.
			#inferences, ground_truths = np.array(inferences), np.array(ground_truths)
	print('\tTotal inference time = {}.'.format(time.time() - start_time))

	#--------------------
	if inferences is not None:
		if len(inferences) == len(ground_truths):
			#correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			#print('\tAccurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))

			#for inf, gt in zip(inferences, ground_truths):
			#	#print('Result =\n', np.hstack((inf, gt)))
			#	print('Result =\n', inf, gt)
			for idx in range(3):
				#print('Result =\n', np.hstack((inferences[idx], ground_truths[idx])))
				print('Result =\n', inferences[idx], ground_truths[idx])
	else:
		print('[SWL] Warning: Invalid inference results.')

	#--------------------
	# Closes sessions.

	if is_training_required:
		train_session.close()
		del train_session
	if is_evaluation_required:
		eval_session.close()
		del eval_session
	infer_session.close()
	del infer_session

#%%------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
