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
from mnist_tf_cnn import MnistTensorFlowCNN
#from mnist_tf_slim_cnn import MnistTfSlimCNN
#from mnist_keras_cnn import MnistKerasCNN
#from mnist_tflearn_cnn import MnistTfLearnCNN
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer, TrainingMode
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_predictor import NeuralNetPredictor
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'mnist'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180116T212902'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
prediction_dir_path = './result/{}_prediction_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

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
input_shape = (28, 28, 1)  # 784 = 28 * 28.
output_shape = (num_classes,)

train_images, train_labels, test_images, test_labels = load_data(data_dir_path, input_shape)

# Pre-process.
#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

#%%------------------------------------------------------------------
# Create a model.

cnnModel = MnistTensorFlowCNN(input_shape, output_shape)
#cnnModel = MnistTfSlimCNN(input_shape, output_shape)
#cnnModel = MnistTfLearnCNN(input_shape, output_shape)
#from keras import backend as K
#K.set_learning_phase(1)  # Set the learning phase to 'train'.
##K.set_learning_phase(0)  # Set the learning phase to 'test'.
#cnnModel = MnistKerasCNN(input_shape, output_shape)

print('[SWL] Info: Created a model.')

#%%------------------------------------------------------------------
# Configure tensorflow.

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

# REF [site] >> https://stackoverflow.com/questions/45093688/how-to-understand-sess-as-default-and-sess-graph-as-default
#graph = tf.Graph()
#session = tf.Session(graph=graph, config=config)
session = tf.Session(config=config)

#%%------------------------------------------------------------------
# Train the model.

batch_size = 128  # Number of samples per gradient update.
num_epochs = 20  # Number of times to iterate over training data.

shuffle = True

trainingMode = TrainingMode.START_TRAINING

if TrainingMode.START_TRAINING == trainingMode:
	initial_epoch = 0
	print('[SWL] Info: Start training...')
elif TrainingMode.RESUME_TRAINING == trainingMode:
	initial_epoch = 10
	print('[SWL] Info: Resume training...')
elif TrainingMode.USE_SAVED_MODEL == trainingMode:
	initial_epoch = 0
	print('[SWL] Info: Use a saved model.')
else:
	assert False, '[SWL] Error: Invalid training mode.'

if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
	nnTrainer = NeuralNetTrainer(cnnModel, initial_epoch)
	print('[SWL] Info: Created a trainer.')
else:
	nnTrainer = None

session.run(tf.global_variables_initializer())

with session.as_default() as sess:
	# Save a model every 2 hours and maximum 5 latest models are saved.
	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	if TrainingMode.RESUME_TRAINING == trainingMode or TrainingMode.USE_SAVED_MODEL == trainingMode:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		start_time = time.time()
		history = nnTrainer.train(session, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		print('\tTraining time = {}'.format(time.time() - start_time))

		# Display results.
		nnTrainer.display_history(history)

if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
	print('[SWL] Info: End training...')

#%%------------------------------------------------------------------
# Evaluate the model.

print('[SWL] Info: Start evaluation...')

nnEvaluator = NeuralNetEvaluator()
print('[SWL] Info: Created an evaluator.')

with session.as_default() as sess:
	num_test_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_test_examples = test_images.shape[0]

	if num_test_examples > 0:
		start_time = time.time()
		test_loss, test_acc = nnEvaluator.evaluate(session, cnnModel, test_images, test_labels, batch_size)
		end_time = time.time()

		print('\tTest loss = {}, test accurary = {}, evaluation time = {}'.format(test_loss, test_acc, end_time - start_time))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

print('[SWL] Info: End evaluation...')

#%%------------------------------------------------------------------
# Predict.

print('[SWL] Info: Start prediction...')

nnPredictor = NeuralNetPredictor()
print('[SWL] Info: Created a predictor.')

with session.as_default() as sess:
	num_pred_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_pred_examples = test_images.shape[0]

	if num_pred_examples > 0:
		start_time = time.time()
		predictions = nnPredictor.predict(session, cnnModel, test_images, batch_size)
		end_time = time.time()

		if num_classes <= 2:
			predictions = np.around(predictions)
			groundtruths = test_labels
		else:
			predictions = np.argmax(predictions, -1)
			groundtruths = np.argmax(test_labels, -1)
		correct_estimation_count = np.count_nonzero(np.equal(predictions, groundtruths))

		print('\tAccurary = {} / {} = {}, prediction time = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size, end_time - start_time))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

print('[SWL] Info: End prediction...')
