# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_cnn_model import TensorFlowCnnModel
from tf_slim_cnn_model import TfSlimCnnModel
from keras_cnn_model import KerasCnnModel
from tflearn_cnn_model import TfLearnCnnModel
import keras

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

def load_data():
	mnist = input_data.read_data_sets("D:/dataset/pattern_recognition/mnist/0_original/", one_hot=True)

	train_images = np.reshape(mnist.train.images, (-1, 28, 28, 1))  # 784 = 28 * 28.
	train_labels = int(np.round(mnist.train.labels))
	test_images = np.reshape(mnist.test.images, (-1, 28, 28, 1))  # 784 = 28 * 28.
	test_labels = int(np.round(mnist.test.labels))

	return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data()

num_classes = 10
num_examples = train_images.shape[0]
input_shape = train_images.shape[1:]

#--------------------
# Pre-process.

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
		

#train_images, train_labels = preprocess_dataset(train_images, train_labels, num_classes)
#test_images, test_labels = preprocess_dataset(test_images, test_labels, num_classes)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

model_dir_path = './result/model_' + timestamp
prediction_dir_path = './result/prediction_' + timestamp
train_summary_dir_path = './log/train_' + timestamp
test_summary_dir_path = './log/test_' + timestamp

#%%------------------------------------------------------------------
# Create a model.

print('[SWL] Info: Create a model.')

#cnnModel = TensorFlowCnnModel(num_classes)
#cnnModel = TfSlimCnnModel(num_classes)
cnnModel = KerasCnnModel(num_classes)
#cnnModel = TfLearnCnnModel(num_classes)

#%%------------------------------------------------------------------
# Prepare training.

print('[SWL] Info: Prepare training.')

def loss(y, t):
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
	#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=y))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
	return cross_entropy

def accuracy(y, t):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy

def train(loss, learning_rate, global_step=None):
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
	train_step = optimizer.minimize(loss, global_step=global_step)
	return train_step

x_ph = tf.placeholder(tf.float32, shape=(None,) + input_shape)
t_ph = tf.placeholder(tf.float32, shape=(None, num_classes))
is_training_ph = tf.placeholder(tf.bool)

global_step = tf.Variable(0, name='global_step', trainable=False)

model_output = cnnModel(x_ph, is_training_ph)

with tf.name_scope('loss'):
	loss = loss(model_output, t_ph)
	tf.summary.scalar('loss', loss)
with tf.name_scope('accuracy'):
	accuracy = accuracy(model_output, t_ph)
	tf.summary.scalar('accuracy', accuracy)
with tf.name_scope('learning_rate'):
	learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=10, decay_rate=0.995, staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)
with tf.name_scope('train'):
	train_step = train(loss, learning_rate=0.001, global_step=global_step)
	#train_step = train(loss, learning_rate=learning_rate, global_step=global_step)

#%%------------------------------------------------------------------

history = {
	'acc': [],
	'val_acc': [],
	'loss': [],
	'val_loss': []
}

def display_history(history):
	# List all data in history.
	print(history.keys())

	# Summarize history for accuracy.
	fig = plt.figure()
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('model accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

	# Summarize history for loss.
	plt.figure()
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

#%%------------------------------------------------------------------
# Train the model.

batch_size = 128  # Number of samples per gradient update.
num_epochs = 50  # Number of times to iterate over training data.
steps_per_epoch = num_examples // batch_size if num_examples > 0 else 50
if steps_per_epoch < 1:
	steps_per_epoch = 1

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
	# Merge all the summaries and write them out to a directory.
	merged_summary = tf.summary.merge_all()
	train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, sess.graph)
	test_summary_writer = tf.summary.FileWriter(test_summary_dir_path)

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
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}/{}'.format(epoch, num_epochs))

			# Train.
			for step in range(steps_per_epoch):
				batch = mnist.train.next_batch(batch_size=batch_size, shuffle=shuffle)
				data_batch, label_batch = np.reshape(batch[0], (-1,) + input_shape), batch[1]
				summary, _ = sess.run([merged_summary, train_step], feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: True})
				train_summary_writer.add_summary(summary, epoch)

			# Evaluate.
			#if 0 == epoch % 10:
			if True:
				batch = mnist.train.next_batch(batch_size=batch_size, shuffle=shuffle)
				data_batch, label_batch = np.reshape(batch[0], (-1,) + input_shape), batch[1]
				#train_loss = loss.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				#train_acc = accuracy.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})

				data_batch, label_batch = test_images, test_labels
				#summary = merged_summary.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				#val_loss = loss.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				#val_acc = accuracy.eval(session=sess, feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				summary, val_loss, val_acc = sess.run([merged_summary, loss, accuracy], feed_dict={x_ph: data_batch, t_ph: label_batch, is_training_ph: False})
				test_summary_writer.add_summary(summary, epoch)

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)
				history['val_loss'].append(val_loss)
				history['val_acc'].append(val_acc)

				print('epoch {}: loss = {}, accuracy = {}, validation loss = {}, validation accurary = {}'.format(epoch, train_loss, train_acc, val_loss, val_acc))

			# Save a model.
			if 0 == epoch % 10:
				model_saved_path = saver.save(sess, model_dir_path + '/model.ckpt', global_step=global_step)
				print('[SWL] Info: Saved a model.')

		# Display results.
		display_history(history)

		# Close writers.
		train_summary_writer.close()
		test_summary_writer.close()

if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
	print('[SWL] Info: End training...')

#%%------------------------------------------------------------------
# Evaluate the model.

print('[SWL] Info: Start evaluating...')

with session.as_default() as sess:
	test_data, test_label = test_images, test_labels
	#test_loss = loss.eval(session=sess, feed_dict={x_ph: test_data, t_ph: test_label, is_training_ph: False})
	#test_acc = accuracy.eval(session=sess, feed_dict={x_ph: test_data, t_ph: test_label, is_training_ph: False})
	test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x_ph: test_data, t_ph: test_label, is_training_ph: False})

	print('test loss = {}, test accurary = {}'.format(test_loss, test_acc))

print('[SWL] Info: End evaluating...')

#%%------------------------------------------------------------------
# Predict.

print('[SWL] Info: Start prediction...')

with session.as_default() as sess:
	test_data = test_images
	predictions = model_output.eval(session=sess, feed_dict={x_ph: test_data, is_training_ph: False})

	predictions = np.argmax(predictions, 1)
	groundtruths = np.argmax(test_labels, 1)
	count = np.count_nonzero(np.equal(predictions, groundtruths))

	print('accurary = {} / {}'.format(count, predictions.shape[0]))

print('[SWL] Info: End prediction...')
