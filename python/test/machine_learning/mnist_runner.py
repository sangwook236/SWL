# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import numpy as np
import tensorflow as tf
from tensorflow_cnn_model import TensorFlowCnnModel
from tf_slim_cnn_model import TfSlimCnnModel
from keras_cnn_model import KerasCnnModel
#from tflearn_cnn_model import TfLearnCnnModel
from dnn_trainer import DnnTrainer

#np.random.seed(7)

#%%------------------------------------------------------------------

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # only allocate 40% of the total memory of each GPU.

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

def preprocess_dataset(data, labels, num_classes, axis=0):
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

input_shape = (28, 28, 1)  # 784 = 28 * 28.
num_classes = 10

train_images, train_labels, test_images, test_labels = load_data(input_shape)

# Pre-process.
#train_images, train_labels = preprocess_dataset(train_images, train_labels, num_classes)
#test_images, test_labels = preprocess_dataset(test_images, test_labels, num_classes)

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

cnnModel = TensorFlowCnnModel(num_classes)
#cnnModel = TfSlimCnnModel(num_classes)
#cnnModel = KerasCnnModel(num_classes)
#cnnModel = TfLearnCnnModel(num_classes)

dnnTrainer = DnnTrainer(cnnModel, input_shape, num_classes, train_summary_dir_path, val_summary_dir_path)

print('[SWL] Info: Created a model.')

#%%------------------------------------------------------------------

batch_size = 128  # Number of samples per gradient update.
num_epochs = 50  # Number of times to iterate over training data.

shuffle = True

TRAINING_MODE = 0  # Start training a model.
#TRAINING_MODE = 1  # Resume training a model.
#TRAINING_MODE = 2  # Use a trained model.

if 0 == TRAINING_MODE:
	initial_epoch = 0
	print('[SWL] Info: Start training...')
elif 1 == TRAINING_MODE:
	initial_epoch = 200
	print('[SWL] Info: Resume training...')
elif 2 == TRAINING_MODE:
	initial_epoch = 0
	print('[SWL] Info: Use a trained model.')
else:
	raise Exception('[SWL] Error: Invalid TRAINING_MODE')

session.run(tf.global_variables_initializer())

with session.as_default() as sess:
	# Saves a model every 2 hours and maximum 5 latest models are saved.
	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	if 1 == TRAINING_MODE or 2 == TRAINING_MODE:
		# Load a model.
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
		# Train the model.
		history = dnnTrainer.train(session, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path)

		# Display results.
		dnnTrainer.display_history(history)

if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
	print('[SWL] Info: End training...')

#%%------------------------------------------------------------------
# Evaluate the model.

print('[SWL] Info: Start evaluating...')

with session.as_default() as sess:
	num_test_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_test_examples = test_images.shape[0]

	if num_test_examples > 0:
		"""
		#test_loss = loss.eval(session=sess, feed_dict={x_ph: test_data, t_ph: test_labels, is_training_ph: False})
		#test_acc = accuracy.eval(session=sess, feed_dict={x_ph: test_data, t_ph: test_labels, is_training_ph: False})
		test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x_ph: test_data, t_ph: test_labels, is_training_ph: False})
		"""
		test_steps_per_epoch = (num_test_examples - 1) // batch_size + 1

		indices = np.arange(num_test_examples)
		#if True == shuffle:
		#	np.random.shuffle(indices)

		test_loss, test_acc = 0, 0
		for step in range(test_steps_per_epoch):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			data_batch, label_batch = test_images[batch_indices,], test_labels[batch_indices,]
			if data_batch.size > 0 and label_batch.size > 0:  # If data_batch and label_batch are non-empty.
				#batch_loss = loss.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				#batch_acc = accuracy.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				batch_loss, batch_acc = sess.run([loss, accuracy], feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})

				# TODO [check] >> Is test_loss or test_acc correct?
				test_loss += batch_loss * batch_indices.size
				test_acc += batch_acc * batch_indices.size
		test_loss /= num_test_examples
		test_acc /= num_test_examples

		print('Test loss = {}, test accurary = {}'.format(test_loss, test_acc))
	else:
		print('[SWL] Error: The number of test images is not equal to one of test labels.')

print('[SWL] Info: End evaluating...')

#%%------------------------------------------------------------------
# Predict.

print('[SWL] Info: Start prediction...')

with session.as_default() as sess:
	num_pred_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_pred_examples = test_images.shape[0]

	if num_pred_examples > 0:
		"""
		predictions = model_output.eval(session=sess, feed_dict={x_ph: test_images, is_training_ph: False})
		"""
		pred_steps_per_epoch = (num_pred_examples - 1) // batch_size + 1

		indices = np.arange(num_pred_examples)

		predictions = np.array([])
		for step in range(pred_steps_per_epoch):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			data_batch = test_images[batch_indices,]
			if data_batch.size > 0:  # If data_batch is non-empty.
				batch_prediction = model_output.eval(session=sess, feed_dict={x_ph: data_batch, is_training_ph: False})
	
				if predictions.size > 0:  # If predictions is non-empty.
					predictions = np.concatenate((predictions, batch_prediction), axis=0)
				else:
					predictions = batch_prediction

		predictions = np.argmax(predictions, 1)
		groundtruths = np.argmax(test_labels, 1)
		correct_estimation_count = np.count_nonzero(np.equal(predictions, groundtruths))

		print('Accurary = {} / {}'.format(correct_estimation_count, num_pred_examples))
	else:
		print('[SWL] Error: The number of test images is not equal to one of test labels.')

print('[SWL] Info: End prediction...')
