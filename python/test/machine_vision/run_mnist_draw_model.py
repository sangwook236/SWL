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
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

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
	#output_dir_suffix = '20180302T155710'

	image_height, image_width = 28, 28
	num_time_steps = 10  # MNIST generation sequence length.
	eps = 1e-8  # Epsilon for numerical stability.

	batch_size = 100  # Number of samples per gradient update.
	num_epochs = 1000  # Number of times to iterate over training data.
	shuffle = True

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
			initial_epoch = 0
			nnTrainer = SimpleDrawTrainer(drawModelForTraining, initial_epoch)

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
	config = tf.ConfigProto()
	#config.device_count = {'GPU': 2}
	#config.allow_soft_placement = True
	config.log_device_placement = True
	config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=config)
	infer_session = tf.Session(graph=infer_graph, config=config)

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

	total_elapsed_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			infer_by_neural_net(sess, nnInferrer, test_images, batch_size, infer_saver, checkpoint_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

	#--------------------
	# Close sessions.

	if does_need_training:
		train_session.close()
		del train_session
	infer_session.close()
	del infer_session

	"""
	#--------------------
	fetches = []
	fetches.extend([Lx, Lz, train_op])
	Lxs = [0] * train_iters
	Lzs = [0] * train_iters

	#saver.restore(sess, '/tmp/draw/drawmodel.ckpt')  # To restore from model, uncomment this line.

	for i in range(train_iters):
		xtrain, _ = train_images.next_batch(batch_size)  # xtrain is (batch_size x image_size).
		feed_dict = {x: xtrain}
		results = sess.run(fetches, feed_dict)
		Lxs[i], Lzs[i], _ = results
		if 0 == i % 100:
			print('iter=%d : Lx: %f Lz: %f' % (i, Lxs[i], Lzs[i]))

	#--------------------
	# Training finished.
	canvases = sess.run(cs, feed_dict)  # Generate some examples.
	canvases = np.array(canvases)  # T x batch x image_size.

	out_file = os.path.join(FLAGS.data_dir, 'draw_data.npy')
	np.save(out_file, [canvases, Lxs, Lzs])
	print('Outputs saved in file: %s' % out_file)

	ckpt_file = os.path.join(FLAGS.data_dir, 'drawmodel.ckpt')
	print('Model saved in file: %s' % saver.save(sess, ckpt_file))

	sess.close()
	"""

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
