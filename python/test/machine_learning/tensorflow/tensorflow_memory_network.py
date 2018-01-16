# REF [book] >> "정석으로 배우는 딥러닝", p.264.
# REF [paper] >> "End-To-End Memory Networks", NIPS 2016.
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
from sklearn.utils import shuffle
import re
import tarfile
from functools import reduce
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(1234)

#%%------------------------------------------------------------------

def infer(x, q, batch_size, vocab_size=None, embedding_dim=None, story_maxlen=None, question_maxlen=None):
	def weight_variable(shape, stddev=0.08):
		initial = tf.truncated_normal(shape, stddev=stddev)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.zeros(shape, dtype=tf.float32)
		return tf.Variable(initial)

	A = weight_variable([vocab_size, embedding_dim])
	B = weight_variable([vocab_size, embedding_dim])
	C = weight_variable([vocab_size, question_maxlen])
	m = tf.nn.embedding_lookup(A, x)
	u = tf.nn.embedding_lookup(B, q)
	c = tf.nn.embedding_lookup(C, x)
	p = tf.nn.softmax(tf.einsum('ijk,ilk->ijl', m, u))
	o = tf.add(p, c)
	o = tf.transpose(o, perm=[0, 2, 1])
	ou = tf.concat([o, u], axis=-1)

	cell = tf.contrib.rnn.BasicLSTMCell(embedding_dim//2, forget_bias=1.0)
	initial_state = cell.zero_state(batch_size, tf.float32)
	state = initial_state
	outputs = []
	with tf.variable_scope('LSTM'):
		for t in range(question_maxlen):
			if t > 0:
				tf.get_variable_scope().reuse_variables()
			(cell_output, state) = cell(ou[:, t, :], state)
			outputs.append(cell_output)
	output = outputs[-1]
	W = weight_variable([embedding_dim//2, vocab_size], stddev=0.01)
	a = tf.nn.softmax(tf.matmul(output, W))

	return a

def loss(y, t):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
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

def tokenize(sent):
	return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]

def parse_stories(lines):
	data = []
	story = []
	for line in lines:
		line = line.decode('utf-8').strip()
		nid, line = line.split(' ', 1)
		nid = int(nid)
		if nid == 1:
			story = []
		if '\t' in line:
			q, a, supporting = line.split('\t')
			q = tokenize(q)
			substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append('')
		else:
			sent = tokenize(line)
			story.append(sent)
	return data

def get_stories(f, max_length=None):
	def flatten(data):
		return reduce(lambda x, y: x + y, data)

	data = parse_stories(f.readlines())
	data = [(flatten(story), q, answer)
			for story, q, answer in data
			if not max_length or len(flatten(story)) < max_length]
	return data

def vectorize_stories(data, word_indices, story_maxlen, question_maxlen):
	X = []
	Q = []
	A = []
	for story, question, answer in data:
		x = [word_indices[w] for w in story]
		q = [word_indices[w] for w in question]
		a = np.zeros(len(word_indices) + 1)
		a[word_indices[answer]] = 1
		X.append(x)
		Q.append(q)
		A.append(a)

	return (padding(X, maxlen=story_maxlen), padding(Q, maxlen=question_maxlen), np.array(A))

def padding(words, maxlen):
	for i, word in enumerate(words):
		words[i] = [0] * (maxlen - len(word)) + word
	return np.array(words)

import os
import urllib
from urllib.request import urlretrieve

def get_file(filename, url=None, datadir=None):
	if url is None:
		raise
	if datadir is None:
		datadir = '.'
	if not os.path.exists(datadir):
		os.makedirs(datadir)

	fpath = os.path.join(datadir, filename)

	download = False
	if os.path.exists(fpath):
		pass
	else:
		download = True

	if download:
		print('Downloading data from', url)
		try:
			try:
				urlretrieve(url, fpath)
			except urllib.error.URLError as ex:
				raise
			except urllib.error.HTTPError as ex:
				raise
		except (Exception, KeyboardInterrupt) as ex:
			if os.path.exists(fpath):
				os.remove(fpath)
			raise

	return fpath

#%%------------------------------------------------------------------
# Prepare dataset.

print('Fetching data...')

try:
    path = get_file('babi-tasks-v1-2.tar.gz', url='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except Exception as ex:
    raise
tar = tarfile.open(path)

challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
	vocab |= set(story + q + [answer])
vocab = sorted(vocab)
vocab_size = len(vocab) + 1  # 패딩용으로 +1

story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
question_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('Vectorizing data...')
word_indices = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, questions_train, answers_train = vectorize_stories(train_stories, word_indices, story_maxlen, question_maxlen)

inputs_test, questions_test, answers_test = vectorize_stories(test_stories, word_indices, story_maxlen, question_maxlen)

#%%------------------------------------------------------------------
# Build a model.

print('Building model...')

x_ph = tf.placeholder(tf.int32, shape=[None, story_maxlen])
q_ph = tf.placeholder(tf.int32, shape=[None, question_maxlen])
a_ph = tf.placeholder(tf.float32, shape=[None, vocab_size])
batch_size_ph = tf.placeholder(tf.int32, shape=[])

y = infer(x_ph, q_ph, batch_size_ph, vocab_size=vocab_size, embedding_dim=64, story_maxlen=story_maxlen, question_maxlen=question_maxlen)
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

print('Training model...')

num_epochs = 120
batch_size = 100
steps_per_epoch = len(inputs_train) // batch_size

history = {
	'val_loss': [],
	'val_acc': []
}

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(num_epochs):
		inputs_train_, questions_train_, answers_train_ = shuffle(inputs_train, questions_train, answers_train)

		for i in range(steps_per_epoch):
			start = i * batch_size
			end = start + batch_size

			sess.run(train_step, feed_dict={
				x_ph: inputs_train_[start:end],
				q_ph: questions_train_[start:end],
				a_ph: answers_train_[start:end],
				batch_size_ph: batch_size
			})

		# Evaluate.
		val_loss = loss.eval(session=sess, feed_dict={
			x_ph: inputs_test,
			q_ph: questions_test,
			a_ph: answers_test,
			batch_size_ph: len(inputs_test)
			})
		val_acc = acc.eval(session=sess, feed_dict={
			x_ph: inputs_test,
			q_ph: questions_test,
			a_ph: answers_test,
			batch_size_ph: len(inputs_test)
		})

		history['val_loss'].append(val_loss)
		history['val_acc'].append(val_acc)
		print('epoch:', epoch, ' validation loss:', val_loss, ' validation accuracy:', val_acc)

#%%------------------------------------------------------------------
# Visualize.

loss = history['val_loss']
acc = history['val_acc']

plt.rc('font', family='serif')
plt.figure()
if False:
	plt.plot(range(len(loss)), loss, label='loss', color='red')
	plt.xlabel('num_epochs')
	plt.ylabel('loss')
	plt.show()
	plt.figure()
	plt.plot(range(len(acc)), acc, label='accuracy', color='blue')
	plt.xlabel('num_epochs')
	plt.ylabel('accuracy')
	plt.show()
else:
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ax1.plot(range(len(loss)), loss, label='loss', color='red')
	ax1.set_xlabel('num_epochs')
	ax1.set_ylabel('loss')

	ax2.plot(range(len(acc)), acc, label='accuracy', color='blue')
	ax2.set_ylabel('accuracy')
