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
sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_home_dir_path + '/Fully-Connected-DenseNets-Semantic-Segmentation_github')
#sys.path.append('../../../src')

#os.chdir(swl_python_home_dir_path + '/test/machine_learning/tensorflow')

#--------------------
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from plant_fc_densenet import PlantFcDenseNet
from simple_neural_net_trainer import SimpleNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
from swl.machine_learning.tensorflow.neural_net_trainer import TrainingMode
from swl.machine_learning.util import to_one_hot_encoding
from swl.image_processing.util import load_image_list_by_pil, generate_image_patch_list, stitch_label_patches
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'fc_densenet'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180117T135317'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
inference_dir_path = './result/{}_inference_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

def make_dir(dir_path):
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path)
		except OSError as exception:
			if os.errno.EEXIST != exception.errno:
				raise
make_dir(model_dir_path)
make_dir(inference_dir_path)
make_dir(train_summary_dir_path)
make_dir(val_summary_dir_path)
  
#%%------------------------------------------------------------------
# Load data.

if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

image_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants'
label_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants_foreground'

image_suffix = ''
image_extension = 'png'
label_suffix = '_foreground'
label_extension = 'png'
patch_height, patch_width = 224, 224

image_list = load_image_list_by_pil(image_dir_path, image_suffix, image_extension)
label_list0 = load_image_list_by_pil(label_dir_path, label_suffix, label_extension)
label_list = []
for lbl in label_list0:
	label_list.append(lbl // 255)
label_list0 = None

assert len(image_list) == len(label_list), '[SWL] Error: The numbers of images and labels are not equal.'
for (img, lbl) in zip(image_list, label_list):
	assert img.shape[:2] == lbl.shape[:2], '[SWL] Error: The sizes of every corresponding image and label are not equal.'

# For checking.
if False:
	fg_ratios = []
	for lbl in label_list:
		fg_ratios.append(np.count_nonzero(lbl) / lbl.size)

	small_image_indices = []
	for (idx, img) in enumerate(image_list):
		if img.shape[0] < patch_height or img.shape[1] < patch_width:
			small_image_indices.append(idx)

all_image_patches, all_label_patches = [], []
for (img, lbl) in zip(image_list, label_list):
	if img.shape[0] >= patch_height and img.shape[1] >= patch_width:  # Excludes small-size images.
		img_pats, lbl_pats, _ = generate_image_patch_list(img, lbl, patch_height, patch_width, 0.02)
		if img_pats is not None and lbl_pats is not None:
			all_image_patches += img_pats
			all_label_patches += lbl_pats
			#all_patch_regions += pat_rgns

assert len(all_image_patches) == len(all_label_patches), 'The number of image patches is not equal to that of label patches.'

all_image_patches = np.array(all_image_patches)
all_label_patches = np.array(all_label_patches)
#all_patch_regions = np.array(all_patch_regions)

#--------------------
def preprocess_data(data, labels, num_classes, axis=0):
	if data is not None:
		# Preprocessing (normalization, standardization, etc.).
		data = data.astype(np.float32)
		data /= 255.0
		#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
		#data = np.reshape(data, data.shape + (1,))

	if labels is not None:
		#labels //= 255
		# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
		labels = to_one_hot_encoding(labels, num_classes).astype(np.uint8)

	return data, labels

num_classes = 2
input_shape = (None, patch_height, patch_width, 3)
output_shape = (None, patch_height, patch_width, num_classes)

# Pre-process.
all_image_patches, all_label_patches = preprocess_data(all_image_patches, all_label_patches, num_classes)

train_image_patches, test_image_patches, train_label_patches, test_label_patches = train_test_split(all_image_patches, all_label_patches, test_size=0.2, random_state=None)

#%%------------------------------------------------------------------

def train_neural_net(session, nnTrainer, saver, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, trainingMode, model_dir_path, train_summary_dir_path, val_summary_dir_path):
	if TrainingMode.START_TRAINING == trainingMode:
		print('[SWL] Info: Start training...')
	elif TrainingMode.RESUME_TRAINING == trainingMode:
		print('[SWL] Info: Resume training...')
	elif TrainingMode.USE_SAVED_MODEL == trainingMode:
		print('[SWL] Info: Use a saved model.')
	else:
		assert False, '[SWL] Error: Invalid training mode.'

	if TrainingMode.RESUME_TRAINING == trainingMode or TrainingMode.USE_SAVED_MODEL == trainingMode:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		start_time = time.time()
		history = nnTrainer.train(session, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		end_time = time.time()

		print('\tTraining time = {}'.format(end_time - start_time))

		# Display results.
		nnTrainer.display_history(history)

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, saver, val_images, val_labels, batch_size, model_dir_path):
	num_val_examples = 0
	if val_images is not None and val_labels is not None:
		if val_images.shape[0] == val_labels.shape[0]:
			num_val_examples = val_images.shape[0]

	if num_val_examples > 0:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Loaded a model.')
		print('[SWL] Info: Start evaluation...')

		start_time = time.time()
		val_loss, val_acc = nnEvaluator.evaluate(session, val_images, val_labels, batch_size)
		end_time = time.time()

		print('\tEvaluation time = {}'.format(end_time - start_time))
		print('\tValidation loss = {}, validation accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The number of validation images is not equal to that of validation labels.')

def infer_by_neural_net(session, nnInferrer, saver, test_images, test_labels, batch_size, model_dir_path):
	num_inf_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
			num_inf_examples = test_images.shape[0]

	if num_inf_examples > 0:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Loaded a model.')
		print('[SWL] Info: Start inferring...')

		start_time = time.time()
		inferences = nnInferrer.infer(session, test_image_patches, batch_size)
		end_time = time.time()

		if num_classes <= 2:
			inferences = np.around(inferences)
			groundtruths = test_labels
		else:
			inferences = np.argmax(inferences, -1)
			groundtruths = np.argmax(test_labels, -1)
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))

		print('\tInference time = {}'.format(end_time - start_time))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
		print('[SWL] Info: End inferring...')
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

#%%------------------------------------------------------------------
# Configure tensorflow.

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

#%%------------------------------------------------------------------
# DenseNet models, sessions, and graphs.

from keras import backend as K

train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

train_session = tf.Session(graph=train_graph, config=config)
eval_session = tf.Session(graph=eval_graph, config=config)
infer_session = tf.Session(graph=infer_graph, config=config)

with train_session:
#with train_graph.as_default():
	K.set_learning_phase(1)  # Set the learning phase to 'train'.

	# Create a model.
	denseNetModel = PlantFcDenseNet(input_shape, output_shape)
	denseNetModel.create_training_model()

	# Create a trainer.
	initial_epoch = 0
	nnTrainer = SimpleNeuralNetTrainer(denseNetModel, initial_epoch)

	# Save a model every 2 hours and maximum 5 latest models are saved.
	train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	initializer = tf.global_variables_initializer()

with eval_session:
#with eval_graph.as_default():
	K.set_learning_phase(0)  # Set the learning phase to 'test'.

	# Create a model.
	denseNetModel = PlantFcDenseNet(input_shape, output_shape)
	denseNetModel.create_evaluation_model()

	# Create an evaluator.
	nnEvaluator = NeuralNetEvaluator(denseNetModel)

	# Create a saver.
	eval_saver = tf.train.Saver()

with infer_session:
#with infer_graph.as_default():
	K.set_learning_phase(0)  # Set the learning phase to 'test'.

	# Create a model.
	denseNetModel = PlantFcDenseNet(input_shape, output_shape)
	denseNetModel.create_inference_model()

	# Create an inferrer.
	nnInferrer = NeuralNetInferrer(denseNetModel)

	# Create a saver.
	infer_saver = tf.train.Saver()

train_session.run(initializer)

#--------------------
# FIXME [restore] >>
#batch_size = 12  # Number of samples per gradient update.
batch_size = 5  # Number of samples per gradient update.
num_epochs = 2  # Number of times to iterate over training data.

with train_session as sess:
	K.set_learning_phase(1)  # Set the learning phase to 'train'.
	shuffle = True
	trainingMode = TrainingMode.START_TRAINING
	train_neural_net(sess, nnTrainer, train_saver, train_image_patches, train_label_patches, test_image_patches, test_label_patches, batch_size, num_epochs, shuffle, trainingMode, model_dir_path, train_summary_dir_path, val_summary_dir_path)

with eval_session as sess:
	K.set_learning_phase(0)  # Set the learning phase to 'test'.
	evaluate_neural_net(sess, nnEvaluator, eval_saver, test_image_patches, test_label_patches, batch_size, model_dir_path)

# FIXME [restore] >>
#with infer_session as sess:
if True:
	K.set_learning_phase(0)  # Set the learning phase to 'test'.
	infer_by_neural_net(infer_session, nnInferrer, infer_saver, test_image_patches, test_label_patches, batch_size, model_dir_path)

#%%------------------------------------------------------------------
# Infer full-size images using patches.

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

def infer_label_patches(sess, img, num_classes, patch_height, patch_width, nnInferrer, batch_size=None):
	image_size = img.shape[:2]
	if image_size[0] < patch_height or image_size[1] < patch_width:
		ratio = max(patch_height / image_size[0], patch_width / image_size[1])
		resized_size = (int(image_size[0] * ratio), int(image_size[1] * ratio))
		#img = np.asarray(Image.fromarray(img).resize((resized_size[1], resized_size[0]), resample=Image.BICUBIC))
		img = np.asarray(Image.fromarray(img).resize((resized_size[1], resized_size[0]), resample=Image.LANCZOS))
		#lbl = np.asarray(Image.fromarray(lbl).resize((resized_size[1], resized_size[0]), resample=Image.NEAREST))
	else:
		resized_size = None

	image_patches, _, patch_regions = generate_image_patch_list(img, None, patch_height, patch_width, None)
	if image_patches is not None and patch_regions is not None and len(image_patches) == len(patch_regions):
		image_patches, _ = preprocess_data(np.array(image_patches), None, num_classes)

		inferred_label_patches = nnInferrer.infer(sess, image_patches, batch_size=batch_size)  # Inferred label patches.
		return inferred_label_patches, image_patches, patch_regions, resized_size
	else:
		return None, None, None, None

def infer_label_using_image_patches(sess, img, num_classes, patch_height, patch_width, nnInferror, batch_size=None):
	image_size = img.shape[:2]
	inferred_label_patches, _, patch_regions, resized_size = infer_label_patches(sess, img, num_classes, patch_height, patch_width, nnInferrer, batch_size)
	if resized_size is None:
		return stitch_label_patches(inferred_label_patches, np.array(patch_regions), image_size)
	else:
		inferred_label = stitch_label_patches(inferred_label_patches, np.array(patch_regions), resized_size)
		return np.asarray(Image.fromarray(inferred_label).resize((image_size[1], image_size[0]), resample=Image.NEAREST))

print('[SWL] Info: Start inferring for full-size images using patches...')

with infer_session as sess:
	inferences = []
	start_time = time.time()
	for img in image_list:
		inf = infer_label_using_image_patches(sess, img, num_classes, patch_height, patch_width, nnInferrer, batch_size)
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
		print('\tMin accurary = {} at index {}, max accuracy = {} at index {}'.format(np.array(inference_accurary_rates).min(), np.argmin(np.array(inference_accurary_rates)), np.array(inference_accurary_rates).max(), np.argmax(np.array(inference_accurary_rates))))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

print('[SWL] Info: End inferrig for full-size images using patches...')

if False:
	idx = 67
	inf, lbl = inferences[idx], label_list[idx]
	plt.imshow(inf, cmap='gray')
	plt.imshow(lbl, cmap='gray')
	plt.imshow(np.not_equal(inf, lbl), cmap='gray')

#%%------------------------------------------------------------------

import math

# REF [function] >> plot_conv_filters() in ${SWDT_HOME}/sw_dev/python/rnd/test/machine_learning/tensorflow/tensorflow_layer_filter_visualization.py.
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

# REF [function] >> plot_conv_activations() in ${SWDT_HOME}/sw_dev/python/rnd/test/machine_learning/tensorflow/tensorflow_layer_activation_visualization_1.py.
def plot_conv_activations(activations, num_columns=5, figsize=None):
	num_filters = activations.shape[3]
	plt.figure(figsize=figsize)
	num_columns = num_columns if num_columns > 0 else 1
	num_rows = math.ceil(num_filters / num_columns) + 1
	for i in range(num_filters):
		plt.subplot(num_rows, num_columns, i + 1)
		plt.title('Filter ' + str(i))
		plt.imshow(activations[0,:,:,i], interpolation='nearest', cmap='gray')

# REF [function] >> compute_layer_activations() in ${SWDT_HOME}/sw_dev/python/rnd/test/machine_learning/tensorflow/tensorflow_layer_activation_visualization_1.py.
def compute_layer_activations(sess, layer_tensor, input_stimuli):
	return sess.run(layer_tensor, feed_dict=denseNetModel.get_feed_dict(input_stimuli, is_training=False))

#%%------------------------------------------------------------------
# Visualize filters in a convolutional layer.

global_variables = tf.global_variables()
#print(global_variables)
for var in global_variables:
	print(var)

with infer_session as sess:
	with tf.variable_scope('plant_fc_densenet', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('conv2d_50', reuse=tf.AUTO_REUSE):
			filters = tf.get_variable('kernel')
			#plot_conv_filters(sess, filters)
			print('**************************', filters.op)

#%%------------------------------------------------------------------
# Visualize activations(layer ouputs) in a convolutional layer.

with infer_session as sess:
	layer_before_concat_tensor = sess.graph.get_tensor_by_name('plant_fc_densenet/fcn-densenet/up_sampling2d_5/ResizeNearestNeighbor:0')  # Shape = (?, 224, 224, 64).
	layer_after_concat_tensor = sess.graph.get_tensor_by_name('plant_fc_densenet/fcn-densenet/merge_50/concat:0')  # Shape = (?, 224, 224, 176).

	start_time = time.time()
	#idx = 0
	#for img in image_list:
	if True:
		idx = 3
		img = image_list[idx]
		#plt.imshow(img)

		inferred_label_patches, image_patches, patch_regions, resized_size = infer_label_patches(sess, img, num_classes, patch_height, patch_width, nnInferrer, batch_size)
		#inferred_label_patches = np.argmax(inferred_label_patches, -1)
		if resized_size is None:
			pat_idx = 0
			for (img_pat, lbl_pat) in zip(image_patches, inferred_label_patches):
				activations_before_concat = compute_layer_activations(sess, layer_before_concat_tensor, img_pat.reshape((-1,) + img_pat.shape))
				#plot_conv_activations(activations_before_concat, figsize=(40, 40))
				activations_after_concat = compute_layer_activations(sess, layer_after_concat_tensor, img_pat.reshape((-1,) + img_pat.shape))
				#plot_conv_activations(activations_after_concat, figsize=(40, 40))

				np.save(('./npy/image_patch_{}_{}.npy').format(idx, pat_idx), img_pat)
				np.save(('./npy/label_patch_{}_{}.npy').format(idx, pat_idx), lbl_pat)
				np.save(('./npy/activations_before_concat_{}_{}.npy').format(idx, pat_idx), activations_before_concat)
				np.save(('./npy/activations_after_concat_{}_{}.npy').format(idx, pat_idx), activations_after_concat)
	
				pat_idx += 1

			np.save(('./npy/patch_ranges_{}.npy').format(idx), np.array(patch_regions))
		else:
			pass
		idx += 1
	end_time = time.time()
	print('\tElapsed time = {}'.format(end_time - start_time))
