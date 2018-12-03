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
import swl.machine_learning.util as swl_ml_util
from swl.machine_vision.draw_model import DRAW

#%%------------------------------------------------------------------

class SimpleDrawTrainer(GradientClippingNeuralNetTrainer):
	def __init__(self, neuralNet, max_gradient_norm, initial_epoch=0):
		with tf.name_scope('learning_rate'):
			learning_rate = 1e-3
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.999)

		super().__init__(neuralNet, optimizer, max_gradient_norm, initial_epoch)

#%%------------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data

def load_data(data_dir_path):
	mnist = input_data.read_data_sets(data_dir_path, one_hot=True)
	return mnist.train.images, mnist.test.images

def preprocess_data(data, axis=0):
	if data is not None:
		# Preprocessing (normalization, standardization, etc.).
		#data = data.astype(np.float32)
		#data /= 255.0
		#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
		#data = np.reshape(data, data.shape + (1,))
		pass

	return data

#%%------------------------------------------------------------------

def train_neural_net(session, nnTrainer, train_images, val_images, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path):
	if does_resume_training:
		print('[SWL] Info: Resume training...')

		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('[SWL] Info: Restored a model.')
	else:
		print('[SWL] Info: Start training...')

	start_time = time.time()
	history = nnTrainer.train_unsupervisedly(session, train_images, val_images, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	#--------------------
	# Save a graph.
	#tf.train.write_graph(session.graph_def, output_dir_path, 'mnist_draw_graph.pb', as_text=False)
	##tf.train.write_graph(session.graph_def, output_dir_path, 'mnist_draw_graph.pbtxt', as_text=True)

	# Save a serving model.
	#builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	#builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	#builder.save(as_text=False)

	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

def infer_by_neural_net(session, nnInferrer, test_images, batch_size, saver=None, checkpoint_dir_path=None):
	num_inf_examples = 0
	if test_images is not None:
		num_inf_examples = test_images.shape[0]

	if num_inf_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		inferences = nnInferrer.infer(session, test_images, batch_size)
		print('\tInference time = {}'.format(time.time() - start_time))
		print('[SWL] Info: End inferring...')

		return inferences
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')
		return None

#%%------------------------------------------------------------------

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as ex:
			if os.errno.EEXIST != ex.errno:
				raise

def create_mnist_draw(image_height, image_width, num_time_steps, eps=1e-8):
	use_read_attention, use_write_attention = True, True
	return DRAW(image_height, image_width, num_time_steps, use_read_attention, use_write_attention, eps=eps)

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'mnist_draw'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181203T135011'

	image_height, image_width = 28, 28
	num_time_steps = 10  # MNIST generation sequence length.
	eps = 1e-8  # Epsilon for numerical stability.

	batch_size = 100  # Number of samples per gradient update.
	num_epochs = 50  # Number of times to iterate over training data.
	shuffle = True
	max_gradient_norm = 5
	initial_epoch = 0

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	#--------------------
	# Prepare directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	#val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	make_dir(checkpoint_dir_path)
	make_dir(inference_dir_path)
	make_dir(train_summary_dir_path)
	#make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mnist/0_download'

	train_images, test_images = load_data(data_dir_path)

	# Pre-process.
	#train_images = preprocess_data(train_images)
	#test_images = preprocess_data(test_images)

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Create a model.
			drawModelForTraining = create_mnist_draw(image_height, image_width, num_time_steps, eps)
			drawModelForTraining.create_training_model()

			# Create a trainer.
			nnTrainer = SimpleDrawTrainer(drawModelForTraining, max_gradient_norm, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

	with infer_graph.as_default():
		# Create a model.
		drawModelForInference = create_mnist_draw(image_height, image_width, num_time_steps, eps)
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
				train_neural_net(sess, nnTrainer, train_images, test_images, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

	#%%------------------------------------------------------------------
	# Infer.

	# REF [site] >> https://github.com/ericjang/draw/blob/master/plot_data.py
	def xrecons_grid(X, B, A):
		"""
		plots canvas for single time step
		X is x_recons, (batch_size x img_size)
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
				img[startr:endr,startc:endc]=X[i,j,:,:]
		return img

	total_elapsed_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			inferences = infer_by_neural_net(sess, nnInferrer, test_images[:100], batch_size, infer_saver, checkpoint_dir_path)

			# Reconstruct.
			canvases = np.array(inferences)  # time_steps x batch_size x image_size.
			T, batch_size, img_size = canvases.shape
			X = 1.0 / (1.0 + np.exp(-canvases))  # x_recons = sigmoid(canvas).
			#image_height = image_width = int(np.sqrt(img_size))

			for t in range(T):
				img = xrecons_grid(X[t,:,:], image_height, image_width)
				plt.matshow(img, cmap=plt.cm.gray)
				img_filepath = os.path.join(inference_dir_path, '{}_{}.png'.format('mnist_draw', t))  # You can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif.
				plt.savefig(img_filepath)
				print(img_filepath)
	print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

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
