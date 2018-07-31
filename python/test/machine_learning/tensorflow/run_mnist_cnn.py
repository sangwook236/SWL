#!/usr/bin/env python

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
sys.path.append(os.path.join(lib_home_dir_path, 'tflearn_github'))
sys.path.append(os.path.join(lib_home_dir_path, 'tf_cnnvis_github'))

#os.chdir(os.path.join(swl_python_home_dir_path, 'test/machine_learning/tensorflow'))

#--------------------
import numpy as np
import tensorflow as tf
from mnist_cnn_tf import MnistCnnUsingTF
#from mnist_cnn_tf_slim import MnistCnnUsingTfSlim
#from mnist_cnn_keras import MnistCnnUsingKeras
#from mnist_cnn_tflearn import MnistCnnUsingTfLearn
from simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.machine_learning.util as swl_ml_util
import time
import traceback

#%%------------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data

def load_data(data_dir_path, shape):
	mnist = input_data.read_data_sets(data_dir_path, one_hot=True)

	train_images = np.reshape(mnist.train.images, (-1,) + shape)
	train_labels = np.round(mnist.train.labels).astype(np.int)
	test_images = np.reshape(mnist.test.images, (-1,) + shape)
	test_labels = np.round(mnist.test.labels).astype(np.int)

	return train_images, train_labels, test_images, test_labels

def preprocess_data(data, labels, num_classes, axis=0):
	if data is not None:
		# Preprocessing (normalization, standardization, etc.).
		#data = data.astype(np.float32)
		#data /= 255.0
		#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
		#data = np.reshape(data, data.shape + (1,))
		pass

	if labels is not None:
		# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
		#labels = to_one_hot_encoding(labels, num_classes).astype(np.uint8)
		pass

	return data, labels

def train_neural_net(session, nnTrainer, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, model_dir_path, train_summary_dir_path, val_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train(session, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, val_images, val_labels, batch_size, saver=None, model_dir_path=None):
	num_val_examples = 0
	if val_images is not None and val_labels is not None:
		if val_images.shape[0] == val_labels.shape[0]:
			num_val_examples = val_images.shape[0]

	if num_val_examples > 0:
		if saver is not None and model_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(model_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start evaluation...')
		start_time = time.time()
		val_loss, val_acc = nnEvaluator.evaluate(session, val_images, val_labels, batch_size)
		print('\tEvaluation time = {}'.format(time.time() - start_time))
		print('\tValidation loss = {}, validation accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The number of validation images is not equal to that of validation labels.')

def infer_by_neural_net(session, nnInferrer, test_images, test_labels, num_classes, batch_size, saver=None, model_dir_path=None):
	num_inf_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_inf_examples = test_images.shape[0]

	if num_inf_examples > 0:
		if saver is not None and model_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(model_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		inferences = nnInferrer.infer(session, test_images, batch_size)
		print('\tInference time = {}'.format(time.time() - start_time))

		if num_classes >= 2:
			inferences = np.argmax(inferences, -1)
			groundtruths = np.argmax(test_labels, -1)
		else:
			inferences = np.around(inferences)
			groundtruths = test_labels
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))

		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
		print('[SWL] Info: End inferring...')
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

#%%------------------------------------------------------------------

import datetime
#from keras import backend as K

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as ex:
			if os.errno.EEXIST != ex.errno:
				raise

def create_mnist_cnn(input_shape, output_shape):
	model_type = 0  # {0, 1}.
	return MnistCnnUsingTF(input_shape, output_shape, model_type)
	#return MnistCnnUsingTfSlim(input_shape, output_shape)
	#return MnistCnnUsingTfLearn(input_shape, output_shape)
	#return MnistCnnUsingKeras(input_shape, output_shape, model_type)

def main():
	#np.random.seed(7)

	does_need_training = True
	does_resume_training = False

	#--------------------
	# Prepare directories.

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	output_dir_path = './{}_{}'.format(output_dir_prefix, output_dir_suffix)
	model_dir_path = '{}/model'.format(output_dir_path)
	inference_dir_path = '{}/inference'.format(output_dir_path)
	train_summary_dir_path = '{}/train_log'.format(output_dir_path)
	val_summary_dir_path = '{}/val_log'.format(output_dir_path)

	make_dir(model_dir_path)
	make_dir(inference_dir_path)
	make_dir(train_summary_dir_path)
	make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/machine_vision/mnist/0_download'

	num_classes = 10
	input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
	output_shape = (None, num_classes)

	train_images, train_labels, test_images, test_labels = load_data(data_dir_path, input_shape[1:])

	# Pre-process.
	#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
	#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			#K.set_learning_phase(1)  # Set the learning phase to 'train'. (Required)

			# Create a model.
			cnnModelForTraining = create_mnist_cnn(input_shape, output_shape)
			cnnModelForTraining.create_training_model()

			# Create a trainer.
			initial_epoch = 0
			nnTrainer = SimpleNeuralNetTrainer(cnnModelForTraining, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			#K.set_learning_phase(0)  # Set the learning phase to 'test'. (Required)

			# Create a model.
			cnnModelForEvaluation = create_mnist_cnn(input_shape, output_shape)
			cnnModelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(cnnModelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		#K.set_learning_phase(0)  # Set the learning phase to 'test'. (Required)

		# Create a model.
		cnnModelForInference = create_mnist_cnn(input_shape, output_shape)
		cnnModelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(cnnModelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	config = tf.ConfigProto()
	#config.device_count = {'GPU': 2}
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

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.	
	shuffle = True

	if does_need_training:
		total_elapsed_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(1)  # Set the learning phase to 'train'.
				train_neural_net(sess, nnTrainer, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, model_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

		total_elapsed_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(0)  # Set the learning phase to 'test'.
				evaluate_neural_net(sess, nnEvaluator, test_images, test_labels, batch_size, eval_saver, model_dir_path)
		print('\tTotal evaluation time = {}'.format(time.time() - total_elapsed_time))

	#%%------------------------------------------------------------------
	# Infer.

	total_elapsed_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			#K.set_session(sess)
			#K.set_learning_phase(0)  # Set the learning phase to 'test'.
			infer_by_neural_net(sess, nnInferrer, test_images, test_labels, num_classes, batch_size, infer_saver, model_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

	#%%------------------------------------------------------------------
	# Visualize.

	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			#K.set_session(sess)
			#K.set_learning_phase(0)  # Set the learning phase to 'test'.

			#--------------------
			idx = 0
			#vis_images = train_images[idx:(idx+1)]  # Recommend using a single image.
			vis_images = test_images[idx:(idx+1)]  # Recommend using a single image.
			feed_dict = cnnModelForInference.get_feed_dict(vis_images, is_training=False)
			input_tensor = None
			#input_tensor = cnnModelForInference.input_tensor

			print('[SWL] Info: Start visualizing activation...')
			start = time.time()
			is_succeeded = swl_ml_util.visualize_activation(sess, input_tensor, feed_dict, output_dir_path)
			print('\tVisualization time = {}, succeeded? = {}'.format(time.time() - start, 'yes' if is_succeeded else 'no'))
			print('[SWL] Info: End visualizing activation...')

			print('[SWL] Info: Start visualizing by deconvolution...')
			start = time.time()
			is_succeeded = swl_ml_util.visualize_by_deconvolution(sess, input_tensor, feed_dict, output_dir_path)
			print('\tVisualization time = {}, succeeded? = {}'.format(time.time() - start, 'yes' if is_succeeded else 'no'))
			print('[SWL] Info: End visualizing by deconvolution...')

			#import matplotlib.pyplot as plt
			#plt.imsave(output_dir_path + '/vis.png', np.around(vis_images[0].reshape(vis_images[0].shape[:2]) * 255), cmap='gray')

			#--------------------
			#vis_images = train_images[0:10]
			#vis_labels = train_labels[0:10]
			vis_images = test_images[0:100]
			vis_labels = test_labels[0:100]

			print('[SWL] Info: Start visualizing by partial occlusion...')
			start_time = time.time()
			grid_counts = (28, 28)  # (grid count in height, grid count in width).
			grid_size = (4, 4)  # (grid height, grid width).
			occlusion_color = 0  # Black.
			occluded_probilities = swl_ml_util.visualize_by_partial_occlusion(sess, nnInferrer, vis_images, vis_labels, grid_counts, grid_size, occlusion_color, num_classes, batch_size, infer_saver, model_dir_path)
			print('\tVisualization time = {}'.format(time.time() - start_time))
			print('[SWL] Info: End visualizing by partial occlusion...')

			if occluded_probilities is not None:
				import matplotlib.pyplot as plt
				for (idx, prob) in enumerate(occluded_probilities):
					#plt.figure()
					#plt.imshow(1 - prob.reshape(prob.shape[:2]), cmap='gray')
					#plt.figure()
					#plt.imshow(vis_images[idx].reshape(vis_images[idx].shape[:2]), cmap='gray')
					plt.imsave((output_dir_path + '/occluded_prob_{}.png').format(idx), np.around((1 - prob.reshape(prob.shape[:2])) * 255), cmap='gray')
					plt.imsave((output_dir_path + '/vis_{}.png').format(idx), np.around(vis_images[idx].reshape(vis_images[idx].shape[:2]) * 255), cmap='gray')

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
