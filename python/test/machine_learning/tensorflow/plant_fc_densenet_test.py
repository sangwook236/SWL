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
from plant_fc_densenet_trainer import PlantFcDenseNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_predictor import NeuralNetPredictor
from swl.machine_learning.tensorflow.neural_net_trainer import TrainingMode
from swl.machine_learning.util import to_one_hot_encoding
from swl.image_processing.util import load_image_list_by_pil, generate_image_patch_list, stitch_label_patches
import time

#np.random.seed(7)

#%%------------------------------------------------------------------
# Prepare directories.

import datetime

output_dir_prefix = 'fc-densenet'
output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#output_dir_suffix = '20180117T135317'

model_dir_path = './result/{}_model_{}'.format(output_dir_prefix, output_dir_suffix)
prediction_dir_path = './result/{}_prediction_{}'.format(output_dir_prefix, output_dir_suffix)
train_summary_dir_path = './log/{}_train_{}'.format(output_dir_prefix, output_dir_suffix)
val_summary_dir_path = './log/{}_val_{}'.format(output_dir_prefix, output_dir_suffix)

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
input_shape = (patch_height, patch_width, 3)
output_shape = (patch_height, patch_width, num_classes)

# Pre-process.
all_image_patches, all_label_patches = preprocess_data(all_image_patches, all_label_patches, num_classes)

train_image_patches, test_image_patches, train_label_patches, test_label_patches = train_test_split(all_image_patches, all_label_patches, test_size=0.2, random_state=None)

#%%------------------------------------------------------------------
# Create a model.

# TODO [check] >> How does tf.reset_default_graph() work?
#tf.reset_default_graph()

from keras import backend as K
K.set_learning_phase(1)  # Set the learning phase to 'train'.
#K.set_learning_phase(0)  # Set the learning phase to 'test'.

denseNetModel = PlantFcDenseNet(input_shape, output_shape)

print('[SWL] Info: Created a FC-DenseNet model.')

#%%------------------------------------------------------------------
# Configure tensorflow.

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

# REF [site] >> https://stackoverflow.com/questions/45093688/how-to-understand-sess-as-default-and-sess-graph-as-default
#graph = tf.Graph()
#session = tf.Session(graph=graph, config=config)
session = tf.Session(config=config)

#%%------------------------------------------------------------------
# Train the model.

# FIXME [restore] >>
#batch_size = 12  # Number of samples per gradient update.
batch_size = 5  # Number of samples per gradient update.
num_epochs = 50  # Number of times to iterate over training data.

shuffle = True

trainingMode = TrainingMode.START_TRAINING
initial_epoch = 0

#--------------------
if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
	nnTrainer = PlantFcDenseNetTrainer(denseNetModel, initial_epoch)
	print('[SWL] Info: Created a trainer.')
else:
	nnTrainer = None

#--------------------
if TrainingMode.START_TRAINING == trainingMode:
	print('[SWL] Info: Start training...')
elif TrainingMode.RESUME_TRAINING == trainingMode:
	print('[SWL] Info: Resume training...')
elif TrainingMode.USE_SAVED_MODEL == trainingMode:
	print('[SWL] Info: Use a saved model.')
else:
	assert False, '[SWL] Error: Invalid training mode.'

session.run(tf.global_variables_initializer())

with session.as_default() as sess:
	# Save a model every 2 hours and maximum 5 latest models are saved.
	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

	if TrainingMode.RESUME_TRAINING == trainingMode or TrainingMode.USE_SAVED_MODEL == trainingMode:
		# Load a model.
		# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))

		print('[SWL] Info: Restored a model.')

	if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
		start_time = time.time()
		history = nnTrainer.train(sess, train_image_patches, train_label_patches, test_image_patches, test_label_patches, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=model_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
		end_time = time.time()

		print('\tTraining time = {}'.format(end_time - start_time))

		# Display results.
		nnTrainer.display_history(history)

if TrainingMode.START_TRAINING == trainingMode or TrainingMode.RESUME_TRAINING == trainingMode:
	print('[SWL] Info: End training...')

#%%------------------------------------------------------------------
# Evaluate the model.

nnEvaluator = NeuralNetEvaluator()
print('[SWL] Info: Created an evaluator.')

#--------------------
print('[SWL] Info: Start evaluation...')

with session.as_default() as sess:
	if test_image_patches.shape[0] > 0:
		start_time = time.time()
		test_loss, test_acc = nnEvaluator.evaluate(sess, denseNetModel, test_image_patches, test_label_patches, batch_size)
		end_time = time.time()

		print('\tEvaluation time = {}'.format(end_time - start_time))
		print('\tTest loss = {}, test accurary = {}'.format(test_loss, test_acc))
	else:
		print('[SWL] Error: The number of test images is greater than 0.')

print('[SWL] Info: End evaluation...')

#%%------------------------------------------------------------------
# Predict.

nnPredictor = NeuralNetPredictor()
print('[SWL] Info: Created a predictor.')

#--------------------
print('[SWL] Info: Start prediction...')

with session.as_default() as sess:
	if test_image_patches.shape[0] > 0:
		start_time = time.time()
		predictions = nnPredictor.predict(sess, denseNetModel, test_image_patches, batch_size)
		end_time = time.time()

		if num_classes <= 2:
			predictions = np.around(predictions)
			groundtruths = test_label_patches
		else:
			predictions = np.argmax(predictions, -1)
			groundtruths = np.argmax(test_label_patches, -1)
		correct_estimation_count = np.count_nonzero(np.equal(predictions, groundtruths))

		print('\tPrediction time = {}'.format(end_time - start_time))
		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
	else:
		print('[SWL] Error: The number of test images is greater than 0.')

print('[SWL] Info: End prediction...')

#%%------------------------------------------------------------------
# Predict for full-size images using patches.

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

def predict_label_patches(sess, img, num_classes, patch_height, patch_width, nnPredictor, batch_size=None):
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

		predicted_label_patches = nnPredictor.predict(sess, denseNetModel, image_patches, batch_size=batch_size)  # Predicted label patches.
		return predicted_label_patches, image_patches, patch_regions, resized_size
	else:
		return None, None, None, None

def predict_label_using_image_patches(sess, img, num_classes, patch_height, patch_width, nnPredictor, batch_size=None):
	image_size = img.shape[:2]
	predicted_label_patches, _, patch_regions, resized_size = predict_label_patches(sess, img, num_classes, patch_height, patch_width, nnPredictor, batch_size)
	if resized_size is None:
		return stitch_label_patches(predicted_label_patches, np.array(patch_regions), image_size)
	else:
		predicted_label = stitch_label_patches(predicted_label_patches, np.array(patch_regions), resized_size)
		return np.asarray(Image.fromarray(predicted_label).resize((image_size[1], image_size[0]), resample=Image.NEAREST))

print('[SWL] Info: Start prediction for full-size images using patches...')

with session.as_default() as sess:
	predictions = []
	start_time = time.time()
	for img in image_list:
		pred = predict_label_using_image_patches(sess, img, num_classes, patch_height, patch_width, nnPredictor, batch_size)
		if pred is not None:
			predictions.append(pred)
	end_time = time.time()

	if len(predictions) == len(label_list):
		total_correct_estimation_count = 0
		total_pixel_count = 0
		prediction_accurary_rates = []
		idx = 0
		for (pred, lbl) in zip(predictions, label_list):
			if pred is not None and lbl is not None and np.array_equal(pred.shape, lbl.shape):
				correct_estimation_count = np.count_nonzero(np.equal(pred, lbl))
				total_correct_estimation_count += correct_estimation_count
				total_pixel_count += lbl.size
				prediction_accurary_rates.append(correct_estimation_count / lbl.size)

				#plt.imsave((prediction_dir_path + '/prediction_{}.png').format(idx), pred * 255, cmap='gray')  # Saves images as 32-bit RGBA.
				Image.fromarray(pred * 255).save((prediction_dir_path + '/prediction_{}.png').format(idx))  # Saves images as grayscale.
				idx += 1
			else:
				print('[SWL] Error: Invalid image or label.')

		print('\tPrediction time = {}'.format(end_time - start_time))
		print('\tAccurary = {} / {} = {}'.format(total_correct_estimation_count, total_pixel_count, total_correct_estimation_count / total_pixel_count))
		print('\tMin accurary = {} at index {}, max accuracy = {} at index {}'.format(np.array(prediction_accurary_rates).min(), np.argmin(np.array(prediction_accurary_rates)), np.array(prediction_accurary_rates).max(), np.argmax(np.array(prediction_accurary_rates))))
	else:
		print('[SWL] Error: The number of test images is not equal to that of test labels.')

print('[SWL] Info: End prediction for full-size images using patches...')

if False:
	idx = 67
	pred, lbl = predictions[idx], label_list[idx]
	plt.imshow(pred, cmap='gray')
	plt.imshow(lbl, cmap='gray')
	plt.imshow(np.not_equal(pred, lbl), cmap='gray')

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

with session.as_default() as sess:
	with tf.variable_scope('plant_fc_densenet', reuse=tf.AUTO_REUSE):
		with tf.variable_scope('conv2d_50', reuse=tf.AUTO_REUSE):
			filters = tf.get_variable('kernel')
			#plot_conv_filters(sess, filters)
			print('**************************', filters.op)

#%%------------------------------------------------------------------
# Visualize activations(layer ouputs) in a convolutional layer.

with session.as_default() as sess:
	layer_before_concat_tensor = sess.graph.get_tensor_by_name('plant_fc_densenet/fcn-densenet/up_sampling2d_5/ResizeNearestNeighbor:0')  # Shape = (?, 224, 224, 64).
	layer_after_concat_tensor = sess.graph.get_tensor_by_name('plant_fc_densenet/fcn-densenet/merge_50/concat:0')  # Shape = (?, 224, 224, 176).

	start_time = time.time()
	#idx = 0
	#for img in image_list:
	if True:
		idx = 3
		img = image_list[idx]
		#plt.imshow(img)

		predicted_label_patches, image_patches, patch_regions, resized_size = predict_label_patches(sess, img, num_classes, patch_height, patch_width, nnPredictor, batch_size)
		#predicted_label_patches = np.argmax(predicted_label_patches, -1)
		if resized_size is None:
			pat_idx = 0
			for (img_pat, lbl_pat) in zip(image_patches, predicted_label_patches):
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
