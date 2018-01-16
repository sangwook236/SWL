# REF [book] >> "Á¤¼®À¸·Î ¹è¿ì´Â µö·¯´×", p.264.
# REF [site] >> http://wikibook.co.kr/deep-learning-with-tensorflow/
# REF [site] >> https://github.com/wikibook/deep-learning-with-tensorflow

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#--------------------
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(1234)

#%%------------------------------------------------------------------
# Load datasets.

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

num_all_examples = len(mnist.data)
#num_examples = num_all_examples
num_examples = 30000
num_train_examples = 20000
num_val_examples = 4000
indices = np.random.permutation(range(num_all_examples))[:num_examples]

X = mnist.data[indices]
X = X / 255.0
X = X - X.mean(axis=1).reshape(len(X), 1)
X = X.reshape(len(X), 28, 28)  # Transform to timeseries data.
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]  # One-hot encoding.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=num_train_examples)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=num_val_examples)

#%%------------------------------------------------------------------

def infer(x, num_in, num_time, num_hidden, num_out):
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.zeros(shape, dtype=tf.float32)
		return tf.Variable(initial)

	# Make a list of length (time steps), each tensor of shape [samples, features].
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, num_in])
	x = tf.split(x, num_time, 0)

	cell_forward = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	cell_backward = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

	outputs, _, _ = rnn.static_bidirectional_rnn(cell_forward, cell_backward, x, dtype=tf.float32)

	W = weight_variable([num_hidden * 2, num_out])
	b = bias_variable([num_out])
	y = tf.nn.softmax(tf.matmul(outputs[-1], W) + b)

	return y

def loss(y, t):
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
	#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
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

class EarlyStopping():
	def __init__(self, patience=0, verbose=0):
		self._step = 0
		self._loss = float('inf')
		self.patience = patience
		self.verbose = verbose

	def validate(self, loss):
		if self._loss < loss:
			self._step += 1
			if self._step > self.patience:
				if self.verbose:
					print('Early stopping.')
				return True
		else:
			self._step = 0
			self._loss = loss

		return False

#%%------------------------------------------------------------------

num_in = 28
num_time = 28
num_hidden = 128
num_out = 10

# [samples, time steps, features].
x_ph = tf.placeholder(tf.float32, shape=[None, num_time, num_in])
t_ph = tf.placeholder(tf.float32, shape=[None, num_out])

y = infer(x_ph, num_in=num_in, num_time=num_time, num_hidden=num_hidden, num_out=num_out)
with tf.name_scope('loss'):
	loss = loss(y, t_ph)
	tf.summary.scalar('loss', loss)
with tf.name_scope('accuracy'):
	accuracy = accuracy(y, t_ph)
	tf.summary.scalar('accuracy', accuracy)
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
	learning_rate = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=100000, decay_rate=0.995, staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)
with tf.name_scope('train'):
	train_step = train(loss, learning_rate=0.001)
	#train_step = train(loss, learning_rate=learning_rate, global_step=global_step)

"""
# Merge all the summaries and write them out to a directory.
merged_summary = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, sess.graph)
test_summary_writer = tf.summary.FileWriter(test_summary_dir_path)

# Saves a model every 2 hours and maximum 5 latest models are saved.
saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
"""

early_stopping = EarlyStopping(patience=10, verbose=1)

#%%------------------------------------------------------------------

num_epoches = 300
batch_size = 250
steps_per_epoch = num_train_examples // batch_size if num_train_examples > 0 else 50
if steps_per_epoch < 1:
	steps_per_epoch = 1

history = {
	'val_loss': [],
	'val_acc': []
}

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	# Train.
	for epoch in range(num_epoches):
		X_, Y_ = shuffle(X_train, Y_train)

		for i in range(steps_per_epoch):
			start = i * batch_size
			end = start + batch_size

			sess.run(train_step, feed_dict={x_ph: X_[start:end], t_ph: Y_[start:end]})

		val_loss = loss.eval(session=sess, feed_dict={x_ph: X_validation, t_ph: Y_validation})
		val_acc = accuracy.eval(session=sess, feed_dict={x_ph: X_validation, t_ph: Y_validation})

		history['val_loss'].append(val_loss)
		history['val_acc'].append(val_acc)

		print('epoch:', epoch, ' validation loss:', val_loss, ' validation accurary:', val_acc)

		if early_stopping.validate(val_loss):
			break

	# Evaluate.
	accuracy_rate = accuracy.eval(session=sess, feed_dict={x_ph: X_test, t_ph: Y_test})
	print('accuracy: ', accuracy_rate)

#%%------------------------------------------------------------------
# Visualize.

loss = history['val_loss']
acc = history['val_acc']

plt.rc('font', family='serif')
plt.figure()
if False:
	plt.plot(range(len(loss)), loss, label='loss', color='red')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.show()
	plt.figure()
	plt.plot(range(len(acc)), acc, label='accuracy', color='blue')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.show()
else:
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ax1.plot(range(len(loss)), loss, label='loss', color='red')
	ax1.set_xlabel('epochs')
	ax1.set_ylabel('loss')

	ax2.plot(range(len(acc)), acc, label='accuracy', color='blue')
	ax2.set_ylabel('accuracy')
