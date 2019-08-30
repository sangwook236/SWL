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
from hangeul_crnn import HangeulCrnnWithCrossEntropyLoss, HangeulCrnnWithCtcLoss, HangeulCrnnWithKerasCtcLoss, HangeulDilatedCrnnWithCtcLoss, HangeulDilatedCrnnWithKerasCtcLoss
from hangeul_data import HangeulDataGenerator

#--------------------------------------------------------------------

def create_model(image_height, image_width, image_channel, num_classes, is_sparse_output):
	if is_sparse_output:
		return HangeulCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes)
		#return HangeulDilatedCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes)
	else:
		# NOTE [info] >> The time-steps of model outputs and ground truths are different.
		#return HangeulCrnnWithCrossEntropyLoss(image_height, image_width, image_channel, num_classes)
		return HangeulCrnnWithKerasCtcLoss(image_height, image_width, image_channel, num_classes)
		#return HangeulDilatedCrnnWithKerasCtcLoss(image_height, image_width, image_channel, num_classes)

#--------------------------------------------------------------------

class SimpleCrnnTrainer(ModelTrainer):
	def __init__(self, model, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch=0, var_list=None):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			initial_learning_rate = 1.0
			decay_steps = 10000
			decay_rate = 0.96
			learning_rate = initial_learning_rate
			#learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.inverse_time_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.natural_exp_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.cosine_decay(initial_learning_rate, global_step, decay_steps, alpha=0.0)
			#learning_rate = tf.train.linear_cosine_decay(initial_learning_rate, global_step, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.noisy_linear_cosine_decay(initial_learning_rate, global_step, decay_steps, initial_variance=1.0, variance_decay=0.55, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.polynomial_decay(initial_learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-07)

		super().__init__(model, optimizer, dataGenerator, output_dir_path, model_save_dir_path, train_summary_dir_path, val_summary_dir_path, global_step, var_list)

#--------------------------------------------------------------------

# REF [function] >> training_worker_proc() in ${SWL_PYTHON_HOME}/python/test/machine_learning/batch_generator_and_loader_test.py.
def training_worker_thread_proc(session, modelTrainer, batch_size, num_epochs, shuffle, is_training_resumed):
	print('\t{}({}): Start training worker thread.'.format(os.getpid(), threading.get_ident()))

	modelTrainer.train(session, batch_size, num_epochs, shuffle, is_training_resumed)

	print('\t{}({}): End training worker thread.'.format(os.getpid(), threading.get_ident()))

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, num_epochs, batch_size, is_sparse_output):
		# Sets parameters.
		self._num_epochs, self._batch_size = num_epochs, batch_size
		self._is_sparse_output = is_sparse_output

		is_output_augmented = False  # Fixed.
		is_augmented_in_parallel = True
		is_npy_files_used_as_input = True  # Specifies whether npy files or image files are used as input. Using npy files is faster.

		self._sess_config = tf.ConfigProto()
		#self._sess_config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})  # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'.
		self._sess_config.allow_soft_placement = True
		#self._sess_config.log_device_placement = True
		#self._sess_config.operation_timeout_in_ms = 50000
		self._sess_config.gpu_options.allow_growth = True
		#self._sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

		#--------------------
		# Prepares data.

		self._dataGenerator = HangeulDataGenerator(self._num_epochs, self._is_sparse_output, is_output_augmented, is_augmented_in_parallel, is_npy_files_used_as_input)
		#self._label_sos_token, self._label_eos_token = self._dataGenerator.dataset.start_token, self._dataGenerator.dataset.end_token
		self._label_eos_token = self._dataGenerator.dataset.end_token

		self._dataGenerator.initialize(self._batch_size)

	def train(self, checkpoint_dir_path, output_dir_path, shuffle=True, initial_epoch=0, is_training_resumed=False, device_name=None):
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
				image_height, image_width, image_channel, num_classes = self._dataGenerator.shapes

				# Creates a model.
				modelForTraining = create_model(image_height, image_width, image_channel, num_classes, self._is_sparse_output)
				modelForTraining.create_training_model()

				# Creates a trainer.
				modelTrainer = SimpleCrnnTrainer(modelForTraining, self._dataGenerator, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, initial_epoch)

				initializer = tf.global_variables_initializer()

		# Creates a session.
		train_session = tf.Session(graph=train_graph, config=self._sess_config)

		# Initializes.
		train_session.run(initializer)

		#--------------------
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				self._dataGenerator.initializeTraining(self._batch_size, shuffle)
				if True:
					modelTrainer.train(sess, self._batch_size, self._num_epochs, shuffle, is_training_resumed)
				else:
					# Uses a training worker thread.
					training_worker_thread = threading.Thread(target=training_worker_thread_proc, args=(sess, modelTrainer, self._batch_size, self._num_epochs, shuffle, is_training_resumed))
					training_worker_thread.start()

					training_worker_thread.join()
				self._dataGenerator.finalizeTraining()
		print('\tTotal training time = {}.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		train_session.close()
		del train_session
		#train_graph.reset_default_graph()
		del train_graph

	def evaluate(self, checkpoint_dir_path, shuffle=False, device_name=None):
		# Creates a graph.
		eval_graph = tf.Graph()
		with eval_graph.as_default():
			with tf.device(device_name):
				image_height, image_width, image_channel, num_classes = self._dataGenerator.shapes

				# Creates a model.
				modelForEvaluation = create_model(image_height, image_width, image_channel, num_classes, self._is_sparse_output)
				modelForEvaluation.create_evaluation_model()

				# Creates an evaluator.
				modelEvaluator = ModelEvaluator(modelForEvaluation, self._dataGenerator, checkpoint_dir_path)

		# Creates a session.
		eval_session = tf.Session(graph=eval_graph, config=self._sess_config)

		#--------------------
		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				#modelEvaluator.evaluate(sess, batch_size=None, shuffle=False)  # Exception: NotImplementedError is raised in dataGenerator.getValidationData().
				modelEvaluator.evaluate(sess, batch_size=self._batch_size, shuffle=shuffle)
		print('\tTotal evaluation time = {}.'.format(time.time() - start_time))

		#--------------------
		# Closes the session and the graph.
		eval_session.close()
		del eval_session
		#eval_graph.reset_default_graph()
		del eval_graph

	def test(self, checkpoint_dir_path, shuffle=False, device_name=None):
		# Creates a graph.
		test_graph = tf.Graph()
		with test_graph.as_default():
			with tf.device(device_name):
				image_height, image_width, image_channel, num_classes = self._dataGenerator.shapes

				# Creates a model.
				modelForInference = create_model(image_height, image_width, image_channel, num_classes, self._is_sparse_output)
				modelForInference.create_inference_model()

				# Creates a tester.
				modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

		# Creates a session.
		test_session = tf.Session(graph=test_graph, config=self._sess_config)

		#--------------------
		start_time = time.time()
		with test_session.as_default() as sess:
			with sess.graph.as_default():
				label_eos_token = self._dataGenerator.dataset.end_token

				inferences, ground_truths = list(), list()
				num_test_examples = 0
				for batch_data, num_batch_examples in self._dataGenerator.getTestBatches(self._batch_size, shuffle=shuffle):
					inferred = modelInferrer.infer(sess, batch_data[0])
					if isinstance(inferred, tf.SparseTensorValue):
						print('*******************', inferred.dense_shape)
						# A sparse tensor expressed by a tuple with (indices, values, dense_shape) -> a dense tensor of dense_shape.
						dense_batch_inferences = swl_ml_util.sparse_to_dense(inferred.indices, inferred.values, inferred.dense_shape, default_value=label_eos_token, dtype=np.int8)
						inferences.append(dense_batch_inferences)
					else:
						inferences.append(inferred)
					# A sparse tensor expressed by a tuple with (indices, values, dense_shape) -> a dense tensor of dense_shape.
					dense_batch_outputs = swl_ml_util.sparse_to_dense(*batch_data[1], default_value=label_eos_token, dtype=np.int8)
					ground_truths.append(dense_batch_outputs)
				# Variable-length numpy.arrays are not merged into a single numpy.array.
				#inferences, ground_truths = np.array(inferences), np.array(ground_truths)
		print('\tTotal test time = {}.'.format(time.time() - start_time))

		#--------------------
		if inferences is not None and ground_truths is not None:
			#print('\tTest: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			if len(inferences) == len(ground_truths):
				num_correct_letters, num_letters = 0, 0
				num_correct_texts, num_texts = 0, 0
				for inference, ground_truth in zip(inferences, ground_truths):
					inference, ground_truth = self._dataGenerator.dataset.decode_label(np.argmax(inference, axis=-1)), self._dataGenerator.dataset.decode_label(ground_truth)
					for inf, gt in zip(inference, ground_truth):
						for ich, gch in zip(inf, gt):
							if ich == gch:
								num_correct_letters += 1
						num_letters += max(len(inf), len(gt))

						if inf == gt:
							num_correct_texts += 1
						num_texts += 1
						print('\tInferred: {}, G/T: {}.'.format(inf, gt))
				print('\tLetter accuracy = {}, Text accuracy = {}.'.format(num_correct_letters / num_letters, num_correct_texts / num_texts))
			else:
				print('[SWL] Error: The lengths of test results and ground truth are different.')
		else:
			print('[SWL] Warning: Invalid test results.')

		#--------------------
		# Closes the session and the graph.
		test_session.close()
		del test_session
		#test_graph.reset_default_graph()
		del test_graph

	def infer(self, checkpoint_dir_path, image_filepaths, inference_dir_path, batch_size=None, shuffle=False, device_name=None):
		# Creates a graph.
		infer_graph = tf.Graph()
		with infer_graph.as_default():
			with tf.device(device_name):
				image_height, image_width, image_channel, num_classes = self._dataGenerator.shapes

				# Creates a model.
				modelForInference = create_model(image_height, image_width, image_channel, num_classes, self._is_sparse_output)
				modelForInference.create_inference_model()

				# Creates an inferrer.
				modelInferrer = ModelInferrer(modelForInference, checkpoint_dir_path)

		# Creates a session.
		infer_session = tf.Session(graph=infer_graph, config=self._sess_config)

		#--------------------
		print('[SWL] Info: Start loading images...')
		inf_images = list()
		for fpath in image_filepaths:
			img = cv2.imread(fpath)
			if image_channel != img.shape[2]:
				print('[SWL] Warning: Failed to load an image from {}.'.format(fpath))
				continue
			if image_height != img.shape[0] or image_width != img.shape[1]:
				img = cv2.resize(img, (image_width, image_height))
			img, _ = self._dataGenerator.preprocess(img, None)
			inf_images.append(img)
		inf_images = np.array(inf_images)
		print('[SWL] Info: End loading images: {} secs.'.format(time.time() - start_time))

		num_examples = len(inf_images)
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		indices = np.arange(num_examples)

		#--------------------
		start_time = time.time()
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				label_eos_token = self._dataGenerator.dataset.end_token

				inferences = list()
				start_idx = 0
				while True:
					end_idx = start_idx + batch_size
					batch_indices = indices[start_idx:end_idx]
					if batch_indices.size > 0:  # If batch_indices is non-empty.
						# FIXME [fix] >> Does not work correctly in time-major data.
						batch_images = inf_images[batch_indices]
						if batch_images.size > 0:  # If batch_images is non-empty.
							inferred = modelInferrer.infer(sess, batch_images)
							if isinstance(inferred, tf.SparseTensorValue):
								print('*******************', inferred.dense_shape)
								# A sparse tensor expressed by a tuple with (indices, values, dense_shape) -> a dense tensor of dense_shape.
								dense_batch_inferences = swl_ml_util.sparse_to_dense(inferred.indices, inferred.values, inferred.dense_shape, default_value=label_eos_token, dtype=np.int8)
								inferences.append(dense_batch_inferences)
							else:
								inferences.append(inferred)

					if end_idx >= num_examples:
						break
					start_idx = end_idx
				# Variable-length numpy.arrays are not merged into a single numpy.array.
				#inferences = np.array(inferences)
		print('\tTotal inference time = {}.'.format(time.time() - start_time))

		#--------------------
		if inferences is not None:
			#print('\tInference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			inferences_str = list()
			for inference in inferences:
				inference = self._dataGenerator.dataset.decode_label(np.argmax(inference, axis=-1))
				inferences_str.extend(inference)

			# Output to a file.
			csv_filepath = os.path.join(inference_dir_path, 'inference_results.csv')
			with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')

				for fpath, inf in zip(image_filepaths, inferences_str):
					writer.writerow([fpath, inf])
		else:
			print('[SWL] Warning: Invalid inference results.')

		#--------------------
		# Closes the session and the graph.
		infer_session.close()
		del infer_session
		#infer_graph.reset_default_graph()
		del infer_graph

#--------------------------------------------------------------------

def check_data(is_sparse_output, num_epochs, batch_size, shuffle):
	is_output_augmented = False  # Fixed.
	is_augmented_in_parallel = True
	is_npy_files_used_as_input = True  # Specifies whether npy files or image files are used as input. Using npy files is faster.

	dataGenerator = HangeulDataGenerator(num_epochs, is_sparse_output, is_output_augmented, is_augmented_in_parallel, is_npy_files_used_as_input)
	label_eos_token = dataGenerator.dataset.end_token
	dataGenerator.initialize(batch_size)

	dataGenerator.initializeTraining(batch_size, shuffle=shuffle)

	if is_sparse_output:
		for batch_step, (batch_data, num_batch_examples) in enumerate(dataGenerator.getTrainBatches(batch_size, shuffle=shuffle)):
			#batch_images (np.array), batch_labels (a sparse tensor, a tuple of (indices, values, dense_shape)) = batch_data

			if 0 == batch_step:
				print('type(batch_data) = {}, len(batch_data) = {}.'.format(type(batch_data), len(batch_data)))
				print('type(batch_data[0]) = {}.'.format(type(batch_data[0])))
				print('\tbatch_data[0].shape = {}, batch_data[0].dtype = {}, (min, max) = ({}, {}).'.format(batch_data[0].shape, batch_data[0].dtype, np.min(batch_data[0]), np.max(batch_data[0])))
				print('type(batch_data[1]) = {}, len(batch_data[1]) = {}.'.format(type(batch_data[1]), len(batch_data[1])))
				print('\tbatch_data[1][0] = {}, batch_data[1][1] = {}, batch_data[1][2] = {}.'.format(batch_data[1][0], batch_data[1][1], batch_data[1][2]))

			if batch_size != batch_data[0].shape[0]:
				print('Invalid image size: {} != {}.'.format(batch_size, batch_data[0].shape[0]))
			if batch_size != batch_data[1][2][0]:
				print('Invalid label size: {} != {}.'.format(batch_size, batch_data[1][2][0]))

			#break
	else:
		for batch_step, (batch_data, num_batch_examples) in enumerate(dataGenerator.getTrainBatches(batch_size, shuffle=shuffle)):
			# TODO [check] >> Not yet tested.
			#batch_images (np.array), batch_labels (np.array) = batch_data

			if 0 == batch_step:
				print('type(batch_data) = {}, len(batch_data) = {}.'.format(type(batch_data), len(batch_data)))
				print('type(batch_data[0]) = {}.'.format(type(batch_data[0])))
				print('\tbatch_data[0].shape = {}, batch_data[0].dtype = {}, (min, max) = ({}, {}).'.format(batch_data[0].shape, batch_data[0].dtype, np.min(batch_data[0]), np.max(batch_data[0])))
				print('type(batch_data[1]) = {}.'.format(type(batch_data[1])))
				print('\tbatch_data[1].shape = {}, batch_data[1].dtype = {}, (min, max) = ({}, {}).'.format(batch_data[1].shape, batch_data[1].dtype, np.min(batch_data[1]), np.max(batch_data[1])))

			if batch_size != batch_data[0].shape[0]:
				print('Invalid image size: {} != {}.'.format(batch_size, batch_data[0].shape[0]))
			if batch_size != batch_data[1].shape[0]:
				print('Invalid label size: {} != {}.'.format(batch_size, batch_data[1].shape[0]))

			#break

	dataGenerator.finalizeTraining()

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#random.seed(a=None, version=2)
	#np.random.seed(None)
	#tf.set_random_seed(1234)  # Sets a graph-level seed.

	#--------------------
	# When outputs are not sparse, CRNN model's output shape = (samples, 32, num_classes) and dataset's output shape = (samples, 23, num_classes).
	is_sparse_output = True
	#is_time_major = False  # Fixed.

	num_epochs = 1000  # Number of times to iterate over training data.
	batch_size = 256  # Number of samples per gradient update.
	initial_epoch = 0
	is_trained, is_evaluated, is_tested, is_inferred = True, True, True, False
	is_training_resumed = False

	# REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Graph#device
	# Can use os.environ['CUDA_VISIBLE_DEVICES'] to specify devices.
	train_device_name = None #'/device:GPU:0'
	eval_device_name = None #'/device:GPU:0'
	test_device_name = None #'/device:GPU:0'
	infer_device_name = None #'/device:GPU:0'

	#--------------------
	if False:
		print('[SWL] Info: Start checking data...')
		start_time = time.time(is_sparse_output, num_epochs, batch_size, shuffle=False)
		check_data()
		print('[SWL] Info: End checking data: {} secs.'.format(time.time() - start_time))
		return

	#--------------------
	output_dir_path = None
	if not output_dir_path:
		output_dir_prefix = 'hangeul_crnn'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = None
	if not inference_dir_path:
		inference_dir_path = os.path.join(output_dir_path, 'inference')

	#--------------------
	runner = MyRunner(num_epochs, batch_size, is_sparse_output)

	if is_trained:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		runner.train(checkpoint_dir_path, output_dir_path, shuffle=True, initial_epoch=initial_epoch, is_training_resumed=is_training_resumed, device_name=train_device_name)

	if is_evaluated:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.evaluate(checkpoint_dir_path, device_name=eval_device_name)

	if is_tested:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return

		runner.test(checkpoint_dir_path, device_name=test_device_name)

	if is_inferred:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return
		if inference_dir_path and inference_dir_path.strip() and not os.path.exists(inference_dir_path):
			os.makedirs(inference_dir_path, exist_ok=True)

		image_filepaths = glob.glob('./images/*.jpg')  # TODO [modify] >>
		# TODO [check] >> Not yet tested.
		runner.infer(checkpoint_dir_path, image_filepaths, inference_dir_path, device_name=infer_device_name)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
