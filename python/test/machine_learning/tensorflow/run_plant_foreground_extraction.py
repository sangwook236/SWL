#!/usr/bin/env python

# REF [paper] >> "Densely Connected Convolutional Networks", arXiv 2016.
# REF [paper] >> "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation", arXiv 2016.
# REF [site] >> https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation
# REF [site] >> https://github.com/0bserver07/One-Hundred-Layers-Tiramisu

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
sys.path.append(os.path.join(lib_home_dir_path, 'Fully-Connected-DenseNets-Semantic-Segmentation_github'))

#os.chdir(os.path.join(swl_python_home_dir_path, 'test/machine_learning/tensorflow'))

#--------------------
import math, time, datetime
import numpy as np
import tensorflow as tf
from swl.machine_learning.tensorflow.simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.util.util as swl_util
import swl.machine_vision.util as swl_imgproc_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from rda_plant_util import RdaPlantDataset
from fc_densenet_keras import FcDenseNetUsingKeras
from keras import backend as K

#--------------------------------------------------------------------
# Infer full-size images from patches.

from PIL import Image
import matplotlib.pyplot as plt

def resize_image_and_label(img, lbl, patch_height, patch_width):
	if img.shape[0] < patch_height or img.shape[1] < patch_width:
		ratio = max(patch_height / img.shape[0], patch_width / img.shape[1])
		resized_size = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
		#img = np.asarray(Image.fromarray(img).resize((resized_size[1], resized_size[0]), resample=Image.BICUBIC))
		img = np.asarray(Image.fromarray(img).resize((resized_size[1], resized_size[0]), resample=Image.LANCZOS))
		if lbl is not None:
			lbl = np.asarray(Image.fromarray(lbl).resize((resized_size[1], resized_size[0]), resample=Image.NEAREST))
		return img, lbl, resized_size
	else:
		return img, lbl, None

def infer_label_patches(sess, nnInferrer, img, patch_height, patch_width, num_classes, batch_size=None):
	image_size = img.shape[:2]
	if image_size[0] < patch_height or image_size[1] < patch_width:
		ratio = max(patch_height / image_size[0], patch_width / image_size[1])
		resized_size = (int(image_size[0] * ratio), int(image_size[1] * ratio))
		#img = np.asarray(Image.fromarray(img).resize((resized_size[1], resized_size[0]), resample=Image.BICUBIC))
		img = np.asarray(Image.fromarray(img).resize((resized_size[1], resized_size[0]), resample=Image.LANCZOS))
		#lbl = np.asarray(Image.fromarray(lbl).resize((resized_size[1], resized_size[0]), resample=Image.NEAREST))
	else:
		resized_size = None

	image_patches, _, patch_regions = swl_imgproc_util.generate_image_patch_list(img, None, patch_height, patch_width, None)
	if image_patches is not None and patch_regions is not None and len(image_patches) == len(patch_regions):
		image_patches, _ = RdaPlantDataset.preprocess_data(np.array(image_patches), None, num_classes)

		inferred_label_patches = nnInferrer.infer(sess, image_patches, batch_size=batch_size)  # Inferred label patches.
		return inferred_label_patches, image_patches, patch_regions, resized_size
	else:
		return None, None, None, None

def infer_label_from_image_patches(sess, nnInferrer, img, num_classes, patch_height, patch_width, batch_size=None):
	image_size = img.shape[:2]
	inferred_label_patches, _, patch_regions, resized_size = infer_label_patches(sess, nnInferrer, img, patch_height, patch_width, num_classes, batch_size)
	if resized_size is None:
		return swl_imgproc_util.stitch_label_patches(inferred_label_patches, np.array(patch_regions), image_size)
	else:
		inferred_label = swl_imgproc_util.stitch_label_patches(inferred_label_patches, np.array(patch_regions), resized_size)
		return np.asarray(Image.fromarray(inferred_label).resize((image_size[1], image_size[0]), resample=Image.NEAREST))

def infer_full_size_images_from_patches(sess, nnInferrer, image_list, label_list, patch_height, patch_width, num_classes, batch_size, inference_dir_path):
	inferences = []
	start_time = time.time()
	for img in image_list:
		inf = infer_label_from_image_patches(sess, nnInferrer, img, num_classes, patch_height, patch_width, batch_size)
		if inf is not None:
			inferences.append(inf)
	end_time = time.time()

	if len(inferences) == len(label_list):
		total_correct_estimation_count = 0
		total_pixel_count = 0
		inference_accurary_rates = []
		idx = 0
		for (inf, lbl) in zip(inferences, label_list):
			if inf is not None and lbl is not None and np.array_equal(inf.shape, lbl.shape):
				correct_estimation_count = np.count_nonzero(np.equal(inf, lbl))
				total_correct_estimation_count += correct_estimation_count
				total_pixel_count += lbl.size
				inference_accurary_rates.append(correct_estimation_count / lbl.size)

				#plt.imsave((inference_dir_path + '/inference_{}.png').format(idx), inf * 255, cmap='gray')  # Saves images as 32-bit RGBA.
				Image.fromarray(inf * 255).save((inference_dir_path + '/inference_{}.png').format(idx))  # Saves images as grayscale.
				idx += 1
			else:
				print('[SWL] Error: Invalid image or label.')

		print('\tInference time = {}'.format(end_time - start_time))
		print('\tAccurary = {} / {} = {}'.format(total_correct_estimation_count, total_pixel_count, total_correct_estimation_count / total_pixel_count))
		print('\tMin accuracy = {} at index {}, max accuracy = {} at index {}'.format(np.array(inference_accurary_rates).min(), np.argmin(np.array(inference_accurary_rates)), np.array(inference_accurary_rates).max(), np.argmax(np.array(inference_accurary_rates))))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

	if False:
		idx = 67
		inf, lbl = inferences[idx], label_list[idx]
		plt.imshow(inf, cmap='gray')
		plt.imshow(lbl, cmap='gray')
		plt.imshow(np.not_equal(inf, lbl), cmap='gray')

#--------------------------------------------------------------------

# REF [function] >> plot_conv_filters() in ${SWDT_HOME}/sw_dev/python/rnd/test/machine_learning/tensorflow/tensorflow_visualization_filter.py.
def plot_conv_filters(sess, filter_variable, num_columns=5, figsize=None):
	filters = filter_variable.eval(sess)  # Shape = (height, width, input_dim, output_dim).
	input_dim, output_dim = filters.shape[2], filters.shape[3]
	num_columns = num_columns if num_columns > 0 else 1
	num_rows = math.ceil(output_dim / num_columns) + 1
	for odim in range(output_dim):
		plt.figure(figsize=figsize)
		for idim in range(input_dim):
			plt.subplot(num_rows, num_columns, idim + 1)
			#plt.title('Filter {}'.format(idim))
			plt.imshow(filters[:,:,idim,odim], interpolation='nearest', cmap='gray')

# REF [function] >> plot_conv_activations() in ${SWDT_HOME}/sw_dev/python/rnd/test/machine_learning/tensorflow/tensorflow_visualization_activation.py.
def plot_conv_activations(activations, num_columns=5, figsize=None):
	num_filters = activations.shape[3]
	plt.figure(figsize=figsize)
	num_columns = num_columns if num_columns > 0 else 1
	num_rows = math.ceil(num_filters / num_columns) + 1
	for i in range(num_filters):
		plt.subplot(num_rows, num_columns, i + 1)
		plt.title('Filter ' + str(i))
		plt.imshow(activations[0,:,:,i], interpolation='nearest', cmap='gray')

# REF [function] >> compute_layer_activations() in ${SWDT_HOME}/sw_dev/python/rnd/test/machine_learning/tensorflow/tensorflow_visualization_activation.py.
def compute_layer_activations(sess, layer_tensor, feed_dict):
	return sess.run(layer_tensor, feed_dict=feed_dict)

#--------------------------------------------------------------------
# Visualize filters in a convolutional layer.

def visualize_filters(sess):
	#print(tf.global_variables())

	# FIXME [error] >> Not working.
	#	A variable with name 'fc_densenet_using_keras/conv2d_50/kernel:0' does not exist.
	#	The variable might be created by tf.Variable(), not tf.get_variable().
	with tf.variable_scope('fc_densenet_using_keras', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('conv2d_50', reuse=tf.AUTO_REUSE):
			filters = tf.get_variable('kernel')
			#plot_conv_filters(sess, filters)
			print('**************************', filters.op)

#--------------------------------------------------------------------
# Visualize activations(layer ouputs) in a convolutional layer.

def visualize_activations(sess, denseNetModel, nnInferrer, image_list, patch_height, patch_width, num_classes, batch_size, npy_dir_path):
	# TODO [check] >> Check tensors' names.
	layer_before_concat_tensor = sess.graph.get_tensor_by_name('fc_densenet_using_keras/fcn-densenet/up_sampling2d_5/ResizeNearestNeighbor:0')  # Shape = (?, 224, 224, 64).
	layer_after_concat_tensor = sess.graph.get_tensor_by_name('fc_densenet_using_keras/fcn-densenet/concatenate_50/concat:0')  # Shape = (?, 224, 224, 176).

	start_time = time.time()
	#idx = 0
	#for img in image_list:
	if True:
		idx = 3
		img = image_list[idx]
		#plt.imshow(img)

		inferred_label_patches, image_patches, patch_regions, resized_size = infer_label_patches(sess, nnInferrer, img, patch_height, patch_width, num_classes, batch_size)
		#inferred_label_patches = np.argmax(inferred_label_patches, -1)
		if resized_size is None:
			pat_idx = 0
			for (img_pat, lbl_pat) in zip(image_patches, inferred_label_patches):
				feed_dict = denseNetModel.get_feed_dict(img_pat.reshape((-1,) + img_pat.shape), is_training=False)
				activations_before_concat = compute_layer_activations(sess, layer_before_concat_tensor, feed_dict)
				#plot_conv_activations(activations_before_concat, figsize=(40, 40))
				activations_after_concat = compute_layer_activations(sess, layer_after_concat_tensor, feed_dict)
				#plot_conv_activations(activations_after_concat, figsize=(40, 40))

				np.save('{}/image_patch_{}_{}.npy'.format(npy_dir_path, idx, pat_idx), img_pat)
				np.save('{}/label_patch_{}_{}.npy'.format(npy_dir_path, idx, pat_idx), lbl_pat)
				np.save('{}/activations_before_concat_{}_{}.npy'.format(npy_dir_path, idx, pat_idx), activations_before_concat)
				np.save('{}/activations_after_concat_{}_{}.npy'.format(npy_dir_path, idx, pat_idx), activations_after_concat)

				pat_idx += 1

			np.save('{}/patch_ranges_{}.npy'.format(npy_dir_path, idx), np.array(patch_regions))
		else:
			pass
		idx += 1
	end_time = time.time()
	print('\tElapsed time = {}'.format(end_time - start_time))

#--------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'plant_foreground_extraction'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180117T135317'

	initial_epoch = 0

	image_suffix = ''
	image_extension = 'png'
	label_suffix = '_foreground'
	label_extension = 'png'
	patch_height, patch_width = 224, 224

	num_classes = 2
	input_shape = (None, patch_height, patch_width, 3)
	output_shape = (None, patch_height, patch_width, num_classes)

	batch_size = 6  # Number of samples per gradient update.
	num_epochs = 50  # Number of times to iterate over training data.
	shuffle = True

	augmenter = None
	is_output_augmented = False

	sess_config = tf.ConfigProto()
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
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')
	npy_dir_path = os.path.joint(output_dir_path, 'npy')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)
	swl_util.make_dir(npy_dir_path)

	#--------------------
	# Prepare data.

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'

	image_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants'
	label_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants_foreground'

	train_image_patches, test_image_patches, train_label_patches, test_label_patches, image_list, label_list = RdaPlantDataset.load_data(image_dir_path, image_suffix, image_extension, label_dir_path, label_suffix, label_extension, num_classes, patch_height, patch_width)

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	"""
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()
	"""
	default_graph = tf.get_default_graph()

	# Create sessions.
	"""
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)
	"""
	#default_session = tf.get_default_session()
	default_session = tf.Session(graph=default_graph, config=sess_config)
	if does_need_training:
		train_session = default_session
		eval_session = default_session
	infer_session = default_session

	if does_need_training:
		#with train_graph.as_default():
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				K.set_session(sess)
				K.set_learning_phase(1)  # Set the learning phase to 'train'. (Required)

				# Create a model.
				modelForTraining = FcDenseNetUsingKeras(input_shape, output_shape)
				modelForTraining.create_training_model()

				# Create a trainer.
				nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch, augmenter, is_output_augmented)

				# Create a saver.
				#	Save a model every 2 hours and maximum 5 latest models are saved.
				train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

				initializer = tf.global_variables_initializer()

		#with eval_graph.as_default():
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				K.set_session(sess)
				K.set_learning_phase(0)  # Set the learning phase to 'test'. (Required)

				# Create a model.
				"""
				modelForEvaluation = FcDenseNetUsingKeras(input_shape, output_shape)
				modelForEvaluation.create_evaluation_model()
				"""
				modelForEvaluation = modelForTraining

				# Create an evaluator.
				nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

				# Create a saver.
				#eval_saver = tf.train.Saver()
				eval_saver = None  # Do not load a model.
	else:
		modelForTraining = None

	#with infer_graph.as_default():
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			K.set_session(sess)
			K.set_learning_phase(0)  # Set the learning phase to 'test'. (Required)

			# Create a model.
			if does_need_training:
				modelForInference = modelForTraining
			else:
				modelForInference = FcDenseNetUsingKeras(input_shape, output_shape)
				modelForInference.create_inference_model()

			# Create an inferrer.
			nnInferrer = NeuralNetInferrer(modelForInference)

			# Create a saver.
			if does_need_training:
				infer_saver = None  # Do not load a model.
			else:
				infer_saver = tf.train.Saver()

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#--------------------------------------------------------------------
	# Train and evaluate.

	if does_need_training:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				K.set_session(sess)
				K.set_learning_phase(1)  # Set the learning phase to 'train'.
				swl_tf_util.train_neural_net(sess, nnTrainer, train_image_patches, train_label_patches, test_image_patches, test_label_patches, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - start_time))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				K.set_session(sess)
				K.set_learning_phase(0)  # Set the learning phase to 'test'.
				swl_tf_util.evaluate_neural_net(sess, nnEvaluator, test_image_patches, test_label_patches, batch_size, eval_saver, checkpoint_dir_path)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#--------------------------------------------------------------------
	# Infer.

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			K.set_session(sess)
			K.set_learning_phase(0)  # Set the learning phase to 'test'.
			inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_image_patches, batch_size, infer_saver, checkpoint_dir_path)
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

	#--------------------------------------------------------------------

	print('[SWL] Info: Start inferring full-size images using patches...')
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			K.set_session(sess)
			K.set_learning_phase(0)  # Set the learning phase to 'test'.
			infer_full_size_images_from_patches(sess, nnInferrer, image_list, label_list, patch_height, patch_width, num_classes, batch_size, inference_dir_path)
	print('[SWL] Info: End inferrig full-size images using patches...')

	#print('[SWL] Info: Start visualizing filters...')
	#with infer_session.as_default() as sess:
	#	with sess.graph.as_default():
	#		K.set_session(sess)
	#		K.set_learning_phase(0)  # Set the learning phase to 'test'.
	#		visualize_filters(sess)
	#print('[SWL] Info: End visualizing filters...')

	print('[SWL] Info: Start visualizing activations...')
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			K.set_session(sess)
			K.set_learning_phase(0)  # Set the learning phase to 'test'.
			visualize_activations(sess, modelForInference, nnInferrer, image_list, patch_height, patch_width, num_classes, batch_size, npy_dir_path)
	print('[SWL] Info: End visualizing activations...')

	#--------------------
	# Close sessions.

	"""
	if does_need_training:
		train_session.close()
		del train_session
		eval_session.close()
		del eval_session
	infer_session.close()
	del infer_session
	"""
	default_session.close()
	del default_session

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
