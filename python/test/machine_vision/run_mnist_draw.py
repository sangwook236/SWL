#!/usr/bin/env python

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	#lib_home_dir_path = 'D:/lib_repo/python'
	lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append('../../src')

#--------------------
import time, datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from swl.machine_learning.tensorflow.simple_neural_net import SimpleNeuralNet
from swl.machine_learning.tensorflow.neural_net_trainer import GradientClippingNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
from swl.machine_vision.draw_model import DRAW
import swl.util.util as swl_util
import swl.machine_learning.tensorflow.util as swl_tf_util

#%%------------------------------------------------------------------

def create_mnist_draw(image_height, image_width, batch_size, num_time_steps, eps=1e-8):
	use_read_attention, use_write_attention = True, True
	return DRAW(image_height, image_width, batch_size, num_time_steps, use_read_attention, use_write_attention, eps=eps)

#%%------------------------------------------------------------------

class SimpleDrawTrainer(GradientClippingNeuralNetTrainer):
	def __init__(self, neuralNet, max_gradient_norm, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			learning_rate = 1e-3
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.999)

		super().__init__(neuralNet, optimizer, max_gradient_norm, global_step)

#%%------------------------------------------------------------------

def preprocess_data(data, axis=0):
	if data is not None:
		# Preprocessing (normalization, standardization, etc.).
		#data = data.astype(np.float32)
		#data /= 255.0
		#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
		#data = np.reshape(data, data.shape + (1,))
		pass

	return data

def load_data():
	# Pixel value: [0, 255].
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images / 255.0
	train_images = np.reshape(train_images, (train_images.shape[0], -1))
	#train_labels = tf.keras.utils.to_categorical(train_labels).astype(np.uint8)
	test_images = test_images / 255.0
	test_images = np.reshape(test_images, (test_images.shape[0], -1))
	#test_labels = tf.keras.utils.to_categorical(test_labels).astype(np.uint8)

	# Pre-process.
	#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
	#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

	return train_images, test_images

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'mnist_draw'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181203T135011'

	max_gradient_norm = 5
	initial_epoch = 0

	image_height, image_width = 28, 28
	num_time_steps = 10  # MNIST generation sequence length.
	eps = 1e-8  # Epsilon for numerical stability.

	batch_size = 100  # Number of samples per gradient update.
	num_epochs = 50  # Number of times to iterate over training data.
	shuffle = True

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepare data.

	train_images, test_images = load_data()

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	#val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	#swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Create a model.
			drawModelForTraining = create_mnist_draw(image_height, image_width, batch_size, num_time_steps, eps)
			drawModelForTraining.create_training_model()

			# Create a trainer.
			nnTrainer = SimpleDrawTrainer(drawModelForTraining, max_gradient_norm, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

	with infer_graph.as_default():
		# Create a model.
		drawModelForInference = create_mnist_draw(image_height, image_width, batch_size, num_time_steps, eps)
		drawModelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(drawModelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#%%------------------------------------------------------------------
	# Train.

	if does_need_training:
		total_elapsed_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				swl_tf_util.train_neural_net_unsupervisedly(sess, nnTrainer, train_images, test_images, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

	#%%------------------------------------------------------------------
	# Infer.

	# REF [site] >> https://github.com/ericjang/draw/blob/master/plot_data.py
	def xrecons_grid(X, B, A):
		"""
		plots canvas for single time step
		X is x_recons, (batch_size * img_size)
		assumes features = BxA images
		batch is assumed to be a square number
		"""
		padsize, padval = 1, 0.5
		ph, pw = B + 2 * padsize, A + 2 * padsize
		batch_size = X.shape[0]
		N = int(np.sqrt(batch_size))
		X = X.reshape((N, N, B, A))
		img = np.ones((N * ph, N * pw)) * padval
		for i in range(N):
			for j in range(N):
				startr = i * ph + padsize
				endr = startr + B
				startc = j * pw + padsize
				endc = startc + A
				img[startr:endr,startc:endc] = X[i,j,:,:]
		return img

	total_elapsed_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_images[:batch_size], batch_size, infer_saver, checkpoint_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

	if inferences is not None:
		# Reconstruct.
		canvases = np.array(inferences)  # time_steps * batch_size * image_size.
		T, batch_size, img_size = canvases.shape  # T = num_time_steps * num_test_images / 100.
		X = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas).
		#image_height = image_width = int(np.sqrt(img_size))

		for t in range(T):
			img = xrecons_grid(X[t,:,:], image_height, image_width)
			plt.matshow(img, cmap=plt.cm.gray)
			img_filepath = os.path.join(inference_dir_path, '{}_{}.png'.format('mnist_draw', t))  # You can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif.
			plt.savefig(img_filepath)
			print(img_filepath)
	else:
		print('[SWL] Warning: Invalid inference results.')

	#--------------------
	# Close sessions.

	if does_need_training:
		train_session.close()
		del train_session
	infer_session.close()
	del infer_session

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
