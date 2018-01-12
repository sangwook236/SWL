# REF [book] >> "Á¤¼®À¸·Î ¹è¿ì´Â µö·¯´×", p.264.
# REF [site] >> http://wikibook.co.kr/deep-learning-with-tensorflow/
# REF [site] >> https://github.com/wikibook/deep-learning-with-tensorflow

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#%%------------------------------------------------------------------

import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'

sys.path.append(swl_python_home_dir_path + '/src')

#----------
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(1234)

#%%------------------------------------------------------------------

def infer(x, y, batch_size, is_training, num_input_digits=None, num_output_digits=None, num_hidden=None, num_out=None):
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.zeros(shape, dtype=tf.float32)
		return tf.Variable(initial)

	# Encoder.
	encoder = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	encoder = rnn.AttentionCellWrapper(encoder, num_input_digits, state_is_tuple=True)
	state = encoder.zero_state(batch_size, tf.float32)
	encoder_outputs = []
	encoder_states = []

	with tf.variable_scope('Encoder'):
		for t in range(num_input_digits):
			if t > 0:
				tf.get_variable_scope().reuse_variables()
			# x = (samples, time-steps, features).
			(output, state) = encoder(x[:, t, :], state)
			encoder_outputs.append(output)
			encoder_states.append(state)

	# Decoder.
	decoder = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
	decoder = rnn.AttentionCellWrapper(decoder, num_input_digits, state_is_tuple=True)
	state = encoder_states[-1]
	decoder_outputs = [encoder_outputs[-1]]

	# Pre-define weight and bias of output layer.
	V = weight_variable([num_hidden, num_out])
	c = bias_variable([num_out])
	outputs = []

	with tf.variable_scope('Decoder'):
		for t in range(1, num_output_digits):
			if t > 1:
				tf.get_variable_scope().reuse_variables()

			if is_training is True:
				# y = (samples, time-steps, features).
				(output, state) = decoder(y[:, t-1, :], state)
			else:
				# Use the previous output as an input.
				linear = tf.matmul(decoder_outputs[-1], V) + c
				out = tf.nn.softmax(linear)
				outputs.append(out)
				out = tf.one_hot(tf.argmax(out, -1), depth=num_output_digits)
				(output, state) = decoder(out, state)

			decoder_outputs.append(output)

	if is_training is True:
		output = tf.reshape(tf.concat(decoder_outputs, axis=1), [-1, num_output_digits, num_hidden])

		linear = tf.einsum('ijk,kl->ijl', output, V) + c
		#linear = tf.matmul(output, V) + c
		return tf.nn.softmax(linear)
	else:
		# Compute the final output.
		linear = tf.matmul(decoder_outputs[-1], V) + c
		out = tf.nn.softmax(linear)
		outputs.append(out)

		output = tf.reshape(tf.concat(outputs, axis=1), [-1, num_output_digits, num_out])
		return output

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

#%%------------------------------------------------------------------
# Prepare dataset.

def generate_number(max_digits=3):
    number = ''
    for i in range(np.random.randint(1, max_digits + 1)):
        number += np.random.choice(list('0123456789'))
    return int(number)

def padding(chars, maxlen):
    return chars + ' ' * (maxlen - len(chars))

# Generate data.
num_examples = 20000
num_train_examples = int(num_examples * 0.9)
num_val_examples = num_examples - num_train_examples

max_digits = 3
num_input_digits = max_digits * 2 + 1  # e.g. 123+456.
num_output_digits = max_digits + 1  # 500+500 = 1000.

added = set()
questions = []
answers = []

while len(questions) < num_examples:
	a, b = generate_number(max_digits), generate_number(max_digits)

	pair = tuple(sorted((a, b)))
	if pair in added:
		continue

	question = '{}+{}'.format(a, b)
	question = padding(question, num_input_digits)
	answer = str(a + b)
	answer = padding(answer, num_output_digits)

	added.add(pair)
	questions.append(question)
	answers.append(answer)

chars = '0123456789+ '
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((len(questions), num_input_digits, len(chars)), dtype=np.integer)
Y = np.zeros((len(questions), max_digits + 1, len(chars)), dtype=np.integer)

for i in range(num_examples):
	for t, char in enumerate(questions[i]):
		X[i, t, char_indices[char]] = 1
	for t, char in enumerate(answers[i]):
		Y[i, t, char_indices[char]] = 1

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, train_size=num_train_examples)

#%%------------------------------------------------------------------
# Build a model.

num_in = len(chars)  # 12.
num_hidden = 128
num_out = len(chars)  # 12.

x_ph = tf.placeholder(tf.float32, shape=[None, num_input_digits, num_in])
t_ph = tf.placeholder(tf.float32, shape=[None, num_output_digits, num_out])
#batch_size_ph = tf.placeholder(tf.int32)
batch_size_ph = tf.placeholder(tf.int32, shape=[])
is_training_ph = tf.placeholder(tf.bool)

y = infer(x_ph, t_ph, batch_size_ph, is_training_ph,
		num_input_digits=num_input_digits,
		num_output_digits=num_output_digits,
		num_hidden=num_hidden, num_out=num_out)
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

#%%------------------------------------------------------------------
# Build a model.

num_epochs = 200
batch_size = 200
steps_per_epoch = num_train_examples // batch_size

history = {
	'val_loss': [],
	'val_acc': []
}

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(num_epochs):
		print('=' * 10)
		print('Epoch:', epoch)
		print('=' * 10)

		X_, Y_ = shuffle(X_train, Y_train)

		for i in range(steps_per_epoch):
			start = i * batch_size
			end = start + batch_size

			sess.run(train_step, feed_dict={
				x_ph: X_[start:end],
				t_ph: Y_[start:end],
				batch_size_ph: batch_size,
				is_training_ph: True
			})

		# Evaluate the model.
		val_loss = loss.eval(session=sess, feed_dict={
			x_ph: X_validation,
			t_ph: Y_validation,
			batch_size_ph: num_val_examples,
			is_training_ph: False
		})
		val_acc = acc.eval(session=sess, feed_dict={
			x_ph: X_validation,
			t_ph: Y_validation,
			batch_size_ph: num_val_examples,
			is_training_ph: False
		})

		history['val_loss'].append(val_loss)
		history['val_acc'].append(val_acc)
		print('validation loss:', val_loss)
		print('validation acc: ', val_acc)

		# Check answers.
		for i in range(10):
			index = np.random.randint(0, num_val_examples)
			question = X_validation[np.array([index])]
			answer = Y_validation[np.array([index])]
			prediction = y.eval(session=sess, feed_dict={
				x_ph: question,
				#t_ph: answer,
				batch_size_ph: 1,
				is_training_ph: False
			})
			question = question.argmax(axis=-1)
			answer = answer.argmax(axis=-1)
			prediction = np.argmax(prediction, -1)

			q = ''.join(indices_char[i] for i in question[0])
			a = ''.join(indices_char[i] for i in answer[0])
			p = ''.join(indices_char[i] for i in prediction[0])

			print('-' * 10)
			print('Q:  ', q)
			print('A:  ', p)
			print('T/F:', end=' ')
			if a == p:
				print('T')
			else:
				print('F')
		print('-' * 10)

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
