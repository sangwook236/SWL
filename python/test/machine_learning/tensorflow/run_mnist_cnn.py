#!/usr/bin/env python

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
#sys.path.append('../../../src')
sys.path.append(os.path.join(swl_python_home_dir_path, 'src'))
sys.path.append(os.path.join(lib_home_dir_path, 'tflearn_github'))
sys.path.append(os.path.join(lib_home_dir_path, 'tf_cnnvis_github'))

#os.chdir(os.path.join(swl_python_home_dir_path, 'test/machine_learning/tensorflow'))

#--------------------
import time, datetime
import numpy as np
import tensorflow as tf
#import imgaug as ia
from imgaug import augmenters as iaa
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from mnist_cnn_tf import MnistCnnUsingTF
#from mnist_cnn_tf_slim import MnistCnnUsingTfSlim
#from mnist_cnn_keras import MnistCnnUsingKeras
#from mnist_cnn_tflearn import MnistCnnUsingTfLearn
#from keras import backend as K
import traceback

#%%------------------------------------------------------------------

def create_mnist_cnn(input_shape, output_shape):
	model_type = 0  # {0, 1}.
	return MnistCnnUsingTF(input_shape, output_shape, model_type)
	#return MnistCnnUsingTfSlim(input_shape, output_shape)
	#return MnistCnnUsingTfLearn(input_shape, output_shape)
	#return MnistCnnUsingKeras(input_shape, output_shape, model_type)

#%%------------------------------------------------------------------

def create_imgaug_augmenter():
	return iaa.Sequential([
		iaa.SomeOf(1, [
			#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
			iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))), # Crop images by 0-10% of their height/width.
			iaa.Fliplr(0.1),  # Horizontally flip 10% of the images.
			iaa.Flipud(0.1),  # Vertically flip 10% of the images.
			iaa.Sometimes(0.5, iaa.Affine(
				scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
				rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
				shear=(-16, 16),  # Shear by -16 to +16 degrees.
				#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			)),
			iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
		]),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

class ImgaugAugmenter(object):
	def __init__(self):
		self._augmenter = create_imgaug_augmenter()

	def __call__(self, inputs, outputs, is_output_augmented=False):
		# Augments here.
		if is_output_augmented:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)
		else:
			return self._augmenter.augment_images(inputs), outputs

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
		#labels = swl_ml_util.to_one_hot_encoding(labels, num_classes).astype(np.uint8)
		pass

	return data, labels

def load_data(image_shape):
	# Pixel value: [0, 255].
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images / 255.0
	train_images = np.reshape(train_images, (-1,) + image_shape)
	train_labels = tf.keras.utils.to_categorical(train_labels).astype(np.uint8)
	test_images = test_images / 255.0
	test_images = np.reshape(test_images, (-1,) + image_shape)
	test_labels = tf.keras.utils.to_categorical(test_labels).astype(np.uint8)

	# Pre-process.
	#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
	#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

	return train_images, train_labels, test_images, test_labels

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Sets parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'mnist_cnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	initial_epoch = 0

	num_classes = 10
	input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
	output_shape = (None, num_classes)

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 30  # Number of times to iterate over training data.
	shuffle = True

	augmenter = ImgaugAugmenter()
	#augmenter = create_imgaug_augmenter()  # If imgaug augmenter is used, data are augmented in background augmentation processes. (faster)
	is_output_augmented = False

	sess_config = tf.ConfigProto()
	#sess_config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 1})  # os.environ['CUDA_VISIBLE_DEVICES'] = 0,1.
	sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	#sess_config.operation_timeout_in_ms = 50000
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

	train_device_name = '/device:GPU:1'
	eval_device_name = '/device:GPU:1'
	# NOTE [info] >> Cannot assign a device for operation save/SaveV2: Could not satisfy explicit device specification '/device:GPU:1' because no supported kernel for GPU devices is available.
	#	Errors occur in tf_cnnvis library when a GPU is assigned.
	#infer_device_name = '/device:GPU:1'
	infer_device_name = '/device:CPU:0'

	#--------------------
	# Prepares directories.

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Prepares data.

	train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])

	#--------------------
	# Creates models, sessions, and graphs.

	# Creates graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			with tf.device(train_device_name):
				#K.set_learning_phase(1)  # Sets the learning phase to 'train'. (Required)

				# Creates a model.
				modelForTraining = create_mnist_cnn(input_shape, output_shape)
				modelForTraining.create_training_model()

				# Creates a trainer.
				nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch, augmenter, is_output_augmented)

				# Creates a saver.
				#	Saves a model every 2 hours and maximum 5 latest models are saved.
				train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

				initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			with tf.device(eval_device_name):
				#K.set_learning_phase(0)  # Sets the learning phase to 'test'. (Required)

				# Creates a model.
				modelForEvaluation = create_mnist_cnn(input_shape, output_shape)
				modelForEvaluation.create_evaluation_model()

				# Creates an evaluator.
				nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

				# Creates a saver.
				eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		with tf.device(infer_device_name):
			#K.set_learning_phase(0)  # Sets the learning phase to 'test'. (Required)

			# Creates a model.
			modelForInference = create_mnist_cnn(input_shape, output_shape)
			modelForInference.create_inference_model()

			# Creates an inferrer.
			nnInferrer = NeuralNetInferrer(modelForInference)

			# Creates a saver.
			infer_saver = tf.train.Saver()

	# Creates sessions.
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initializes.
	if does_need_training:
		train_session.run(initializer)

	#%%------------------------------------------------------------------
	# Trains and evaluates.

	if does_need_training:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(1)  # Sets the learning phase to 'train'.
				swl_tf_util.train_neural_net(sess, nnTrainer, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - start_time))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(0)  # Sets the learning phase to 'test'.
				swl_tf_util.evaluate_neural_net(sess, nnEvaluator, test_images, test_labels, batch_size, eval_saver, checkpoint_dir_path)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infers.

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			#K.set_session(sess)
			#K.set_learning_phase(0)  # Sets the learning phase to 'test'.
			inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_images, batch_size, infer_saver, checkpoint_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	if inferences is not None:
		if num_classes >= 2:
			inferences = np.argmax(inferences, -1)
			groundtruths = np.argmax(test_labels, -1)
		else:
			inferences = np.around(inferences)
			groundtruths = test_labels
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
	else:
		print('[SWL] Warning: Invalid inference results.')

	#%%------------------------------------------------------------------
	# Visualizes.

	if True:
		with infer_session.as_default() as sess:
			with sess.graph.as_default():
				#K.set_session(sess)
				#K.set_learning_phase(0)  # Sets the learning phase to 'test'.

				#--------------------
				idx = 0
				#vis_images = train_images[idx:(idx+1)]  # Recommends using a single image.
				vis_images = test_images[idx:(idx+1)]  # Recommends using a single image.
				feed_dict = modelForInference.get_feed_dict(vis_images, is_training=False)
				input_tensor = None
				#input_tensor = modelForInference.input_tensor

				print('[SWL] Info: Start visualizing activation...')
				start = time.time()
				is_succeeded = swl_ml_util.visualize_activation(sess, input_tensor, feed_dict, output_dir_path)
				print('\tVisualization time = {}, succeeded? = {}'.format(time.time() - start, 'yes' if is_succeeded else 'no'))
				print('[SWL] Info: End visualizing activation...')

				print('[SWL] Info: Start visualizing by deconvolution...')
				start = time.time()
				is_succeeded = swl_ml_util.visualize_by_deconvolution(sess, input_tensor, feed_dict, output_dir_path)
				print('\tVisualization time = {}, succeeded? = {}'.format(time.time() - start, 'yes' if is_succeeded else 'no'))
				print('[SWL] Info: End visualizing by deconvolution...')

				#import matplotlib.pyplot as plt
				#plt.imsave(output_dir_path + '/vis.png', np.around(vis_images[0].reshape(vis_images[0].shape[:2]) * 255), cmap='gray')

				#--------------------
				#vis_images = train_images[0:10]
				#vis_labels = train_labels[0:10]
				vis_images = test_images[0:100]
				vis_labels = test_labels[0:100]

				print('[SWL] Info: Start visualizing by partial occlusion...')
				start_time = time.time()
				grid_counts = (28, 28)  # (grid count in height, grid count in width).
				grid_size = (4, 4)  # (grid height, grid width).
				occlusion_color = 0  # Black.
				occluded_probilities = swl_ml_util.visualize_by_partial_occlusion(sess, nnInferrer, vis_images, vis_labels, grid_counts, grid_size, occlusion_color, num_classes, batch_size, infer_saver, checkpoint_dir_path)
				print('\tVisualization time = {}'.format(time.time() - start_time))
				print('[SWL] Info: End visualizing by partial occlusion...')

				if occluded_probilities is not None:
					import matplotlib.pyplot as plt
					for (idx, prob) in enumerate(occluded_probilities):
						#plt.figure()
						#plt.imshow(1 - prob.reshape(prob.shape[:2]), cmap='gray')
						#plt.figure()
						#plt.imshow(vis_images[idx].reshape(vis_images[idx].shape[:2]), cmap='gray')
						plt.imsave((output_dir_path + '/occluded_prob_{}.png').format(idx), np.around((1 - prob.reshape(prob.shape[:2])) * 255), cmap='gray')
						plt.imsave((output_dir_path + '/vis_{}.png').format(idx), np.around(vis_images[idx].reshape(vis_images[idx].shape[:2]) * 255), cmap='gray')

	#--------------------
	# Closes sessions.

	if does_need_training:
		train_session.close()
		del train_session
		eval_session.close()
		del eval_session
	infer_session.close()
	del infer_session

#%%------------------------------------------------------------------

if '__main__' == __name__:
	try:
		main()
	except:
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		traceback.print_exc(limit=None, file=sys.stdout)
