# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import numpy as np
import tensorflow as tf
from mnist_tensorflow_cnn import MnistTensorFlowCNN
from mnist_tf_slim_cnn import MnistTfSlimCNN
from mnist_keras_cnn import MnistKerasCNN
#from mnist_tflearn_cnn import MnistTfLearnCNN
from neural_net_trainer import NeuralNetTrainer
from neural_net_evaluator import NeuralNetEvaluator
from neural_net_predictor import NeuralNetPredictor
import time

#np.random.seed(7)

#%%------------------------------------------------------------------

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
# Load datasets.

from tensorflow.examples.tutorials.mnist import input_data

def load_data(shape):
	mnist = input_data.read_data_sets("D:/dataset/pattern_recognition/mnist/0_original/", one_hot=True)

	train_images = np.reshape(mnist.train.images, (-1,) + shape)
	train_labels = np.round(mnist.train.labels).astype(np.int)
	test_images = np.reshape(mnist.test.images, (-1,) + shape)
	test_labels = np.round(mnist.test.labels).astype(np.int)

	return train_images, train_labels, test_images, test_labels

def preprocess_data(data, labels, num_classes, axis=0):
	if data is not None:
		# Preprocessing (normalization, standardization, etc.).
		#data = data.astype(np.float)
		#data /= 255.0
		#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
		#data = np.reshape(data, data.shape + (1,))
		pass

	if labels is not None:
		# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
		#if 2 != num_classes:
		#	#labels = np.uint8(keras.utils.to_categorical(labels, num_classes).reshape(labels.shape + (-1,)))
		#	labels = np.uint8(keras.utils.to_categorical(labels, num_classes))
		pass

num_classes = 10
input_shape = (28, 28, 1)  # 784 = 28 * 28.
output_shape = (num_classes,)

train_images, train_labels, test_images, test_labels = load_data(input_shape)

# Pre-process.
#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

model_dir_path = './result/model_' + timestamp
prediction_dir_path = './result/prediction_' + timestamp
train_summary_dir_path = './log/train_' + timestamp
val_summary_dir_path = './log/val_' + timestamp

#%%------------------------------------------------------------------
# Create a model.

cnnForMnist = MnistTensorFlowCNN(input_shape, output_shape)
#cnnForMnist = MnistTfSlimCNN(input_shape, output_shape)
#cnnForMnist = MnistTfLearnCNN(input_shape, output_shape)
#from keras import backend as K
#K.set_learning_phase(1)  # Set the learning phase to 'train'.
##K.set_learning_phase(0)  # Set the learning phase to 'train'.
#cnnForMnist = MnistKerasCNN(input_shape, output_shape)

print('[SWL] Info: Created a model.')

#%%------------------------------------------------------------------
# Train the model.

batch_size = 128  # Number of samples per gradient update.
num_epochs = 10  # Number of times to iterate over training data.

shuffle = True

TRAINING_MODE = 0  # Start training a model.
#TRAINING_MODE = 1  # Resume training a model.
#TRAINING_MODE = 2  # Use a saved model.

if 0 == TRAINING_MODE:
	initial_epoch = 0
	print('[SWL] Info: Start training...')
elif 1 == TRAINING_MODE:
	initial_epoch = 10
	print('[SWL] Info: Resume training...')
elif 2 == TRAINING_MODE:
	initial_epoch = 0
	print('[SWL] Info: Use a saved model.')
else:
	assert False, '[SWL] Error: Invalid TRAINING_MODE.'

nnTrainer = NeuralNetTrainer(cnnForMnist, initial_epoch)
print('[SWL] Info: Created a trainer.')

session.run(tf.global_variables_initializer())

with session.as_default() as sess:
	# Save a model every 2 hours and maximum 5 latest models are saved.
	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	if 1 == TRAINING_MODE or 2 == TRAINING_MODE:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
		start_time = time.time()
		history = nnTrainer.train(session, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		print('\tTraining time = {}'.format(time.time() - start_time))

		# Display results.
		nnTrainer.display_history(history)

if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
	print('[SWL] Info: End training...')

#%%------------------------------------------------------------------
# Evaluate the model.

print('[SWL] Info: Start evaluating...')

nnEvaluator = NeuralNetEvaluator()
print('[SWL] Info: Created an evaluator.')

with session.as_default() as sess:
	num_test_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_test_examples = test_images.shape[0]

	if num_test_examples > 0:
		start_time = time.time()
		test_loss, test_acc = nnEvaluator.evaluate(session, cnnForMnist, test_images, test_labels, batch_size)
		end_time = time.time()

		print('\tTest loss = {}, test accurary = {}, evaluation time = {}'.format(test_loss, test_acc, end_time - start_time))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

print('[SWL] Info: End evaluating...')

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
		predictions = nnPredictor.predict(session, cnnForMnist, test_images, batch_size)
		end_time = time.time()

		groundtruths = np.argmax(test_labels, 1)
		correct_estimation_count = np.count_nonzero(np.equal(predictions, groundtruths))

		print('\tAccurary = {} / {}, prediction time = {}'.format(correct_estimation_count, num_pred_examples, end_time - start_time))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

print('[SWL] Info: End prediction...')
