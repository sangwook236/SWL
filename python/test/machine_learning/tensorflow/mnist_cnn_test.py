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
sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_home_dir_path + '/tflearn_github')
#sys.path.append('../../../src')

#os.chdir(swl_python_home_dir_path + '/test/machine_learning/tensorflow')

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
from swl.machine_learning.tensorflow.neural_net_trainer import TrainingMode
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'mnist'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180116T212902'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
inference_dir_path = './result/{}_inference_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as exception:
			if os.errno.EEXIST != exception.errno:
				raise
make_dir(model_dir_path)
make_dir(inference_dir_path)
make_dir(train_summary_dir_path)
make_dir(val_summary_dir_path)

#%%------------------------------------------------------------------
# Load data.

from tensorflow.examples.tutorials.mnist import input_data

if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'
data_dir_path = data_home_dir_path + '/pattern_recognition/mnist/0_original'

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

num_classes = 10
input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
output_shape = (None, num_classes)

train_images, train_labels, test_images, test_labels = load_data(data_dir_path, input_shape[1:])

# Pre-process.
#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

#%%------------------------------------------------------------------

def train_neural_net(session, nnTrainer, saver, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, trainingMode, model_dir_path, train_summary_dir_path, val_summary_dir_path):
	if TrainingMode.START_TRAINING == trainingMode:
		print('[SWL] Info: Start training...')
	elif TrainingMode.RESUME_TRAINING == trainingMode:
		print('[SWL] Info: Resume training...')
	elif TrainingMode.USE_SAVED_MODEL == trainingMode:
		print('[SWL] Info: Use a saved model.')
	else:
		assert False, '[SWL] Error: Invalid training mode.'

	if TrainingMode.RESUME_TRAINING == trainingMode or TrainingMode.USE_SAVED_MODEL == trainingMode:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		start_time = time.time()
		history = nnTrainer.train(session, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		end_time = time.time()

		print('\tTraining time = {}'.format(end_time - start_time))

		# Display results.
		nnTrainer.display_history(history)

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, saver, val_images, val_labels, batch_size, model_dir_path):
	num_val_examples = 0
	if val_images is not None and val_labels is not None:
		if val_images.shape[0] == val_labels.shape[0]:
			num_val_examples = val_images.shape[0]

	if num_val_examples > 0:
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
		end_time = time.time()

		print('\tEvaluation time = {}'.format(end_time - start_time))
		print('\tValidation loss = {}, validation accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The number of validation images is not equal to that of validation labels.')

def infer_by_neural_net(session, nnInferrer, saver, test_images, test_labels, batch_size, model_dir_path):
	num_inf_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_inf_examples = test_images.shape[0]

	if num_inf_examples > 0:
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
		end_time = time.time()

		if num_classes <= 2:
			inferences = np.around(inferences)
			groundtruths = test_labels
		else:
			inferences = np.argmax(inferences, -1)
			groundtruths = np.argmax(test_labels, -1)
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))

		print('\tInference time = {}'.format(end_time - start_time))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
		print('[SWL] Info: End inferring...')
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

#%%------------------------------------------------------------------
# Configure tensorflow.

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

#%%------------------------------------------------------------------
# CNN models, sessions, and graphs.

#from keras import backend as K

def create_mnist_cnn(input_shape, output_shape):
	model_type = 0
	return MnistCnnUsingTF(input_shape, output_shape, model_type)
	#return MnistCnnUsingTfSlim(input_shape, output_shape)
	#return MnistCnnUsingTfLearn(input_shape, output_shape)
	#return MnistCnnUsingKeras(input_shape, output_shape, model_type)

#--------------------
train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

train_session = tf.Session(graph=train_graph, config=config)
eval_session = tf.Session(graph=eval_graph, config=config)
infer_session = tf.Session(graph=infer_graph, config=config)

with train_session:
#with train_graph.as_default():
	#K.set_learning_phase(1)  # Set the learning phase to 'train'.

	# Create a model.
	cnnModel = create_mnist_cnn(input_shape, output_shape)
	cnnModel.create_training_model()

	# Create a trainer.
	initial_epoch = 0
	nnTrainer = SimpleNeuralNetTrainer(cnnModel, initial_epoch)

	# Save a model every 2 hours and maximum 5 latest models are saved.
	train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	initializer = tf.global_variables_initializer()

with eval_session:
#with eval_graph.as_default():
	#K.set_learning_phase(0)  # Set the learning phase to 'test'.

	# Create a model.
	cnnModel = create_mnist_cnn(input_shape, output_shape)
	cnnModel.create_evaluation_model()

	# Create an evaluator.
	nnEvaluator = NeuralNetEvaluator(cnnModel)

	# Create a saver.
	eval_saver = tf.train.Saver()

with infer_session:
#with infer_graph.as_default():
	#K.set_learning_phase(0)  # Set the learning phase to 'test'.

	# Create a model.
	cnnModel = create_mnist_cnn(input_shape, output_shape)
	cnnModel.create_inference_model()

	# Create an inferrer.
	nnInferrer = NeuralNetInferrer(cnnModel)

	# Create a saver.
	infer_saver = tf.train.Saver()

train_session.run(initializer)

#--------------------
batch_size = 128  # Number of samples per gradient update.
num_epochs = 20  # Number of times to iterate over training data.	

with train_session as sess:
	#K.set_learning_phase(1)  # Set the learning phase to 'train'.
	shuffle = True
	trainingMode = TrainingMode.START_TRAINING
	train_neural_net(sess, nnTrainer, train_saver, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, trainingMode, model_dir_path, train_summary_dir_path, val_summary_dir_path)

with eval_session as sess:
	#K.set_learning_phase(0)  # Set the learning phase to 'test'.
	evaluate_neural_net(sess, nnEvaluator, eval_saver, test_images, test_labels, batch_size, model_dir_path)

with infer_session as sess:
	#K.set_learning_phase(0)  # Set the learning phase to 'test'.
	infer_by_neural_net(sess, nnInferrer, infer_saver, test_images, test_labels, batch_size, model_dir_path)
