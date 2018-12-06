#import keras
import numpy as np
import os, math
import matplotlib.pyplot as plt

def to_one_hot_encoding(label_indexes, num_classes=None):
	if None == num_classes:
		num_classes = np.max(label_indexes) + 1
	elif num_classes <= np.max(label_indexes):
		raise ValueError('num_classes has to be greater than np.max(label_indexes)')
	if 1 == label_indexes.ndim:
		return np.eye(num_classes)[label_indexes]
		#return np.transpose(np.eye(num_classes)[label_indexes])
	elif 1 == label_indexes.shape[-1]:
		return np.eye(num_classes)[label_indexes].reshape(label_indexes.shape[:-1] + (-1,))
		#return np.transpose(np.eye(num_classes)[label_indexes].reshape(label_indexes.shape[:-1] + (-1,)))
	else:
		return np.eye(num_classes)[label_indexes].reshape(label_indexes.shape + (-1,))
		#return np.transpose(np.eye(num_classes)[label_indexes].reshape(label_indexes.shape + (-1,)))

# Time-based learning rate schedule.
# REF [site] >> http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def time_based_learning_rate(epoch, initial_learning_rate, decay_rate):
	return initial_learning_rate / (1.0 + decay_rate * epoch)

# Drop-based learning rate schedule.
# REF [site] >> http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def drop_based_learning_rate(epoch, initial_learning_rate, drop_rate, epoch_drop):
	return initial_learning_rate * math.pow(drop_rate, math.floor((1.0 + epoch) / epoch_drop))

#%%------------------------------------------------------------------

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
def generate_sparse_tuple_from_sequences(sequences, dtype=np.int32):
	"""Create a sparse representention of x.
	Args:
		sequences: A list of lists of type dtype where each element is a sequence.
	Returns:
		A tuple with (indices, values, shape).
	"""
	indices = []
	values = []

	for n, seq in enumerate(sequences):
		indices.extend(zip([n] * len(seq), range(len(seq))))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	dense_shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

	return indices, values, dense_shape   # Refer to tf.SparseTensorValue.

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
def generate_sparse_tuple_from_numpy_array(np_arr, eos_token=0, dtype=np.int32):
	"""Create a sparse representention of x.
	Args:
		np_arr: A numpy array of type dtype.
		eos_token: An integer. It is part of the target label that signifies the end of a sentence.
	Returns:
		A tuple with (indices, values, shape).
	"""
	indices = []
	values = []

	for n, subarr in enumerate(np_arr):
		eos_indices = np.where(subarr == eos_token)[0]
		if 0 != eos_indices.size:
			subarr = subarr[:eos_indices[0]]
		indices.extend(zip([n] * len(subarr), range(len(subarr))))
		values.extend(subarr)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	dense_shape = np.asarray([len(np_arr), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

	return indices, values, dense_shape   # Refer to tf.SparseTensorValue.

def generate_batch_list(data, labels, batch_size, shuffle=True, is_time_major=False, is_sparse_label=False, eos_token=0):
	batch_axis = 1 if is_time_major else 0

	num_examples = 0
	if data is not None and labels is not None:
		if data.shape[batch_axis] == labels.shape[batch_axis]:
			num_examples = data.shape[batch_axis]
		num_steps = ((num_examples - 1) // batch_size + 1) if num_examples > 0 else 0

	data_batch_list, label_batch_list = list(), list()
	#if data is not None and labels is not None:
	if num_examples > 0:
		indices = np.arange(num_examples)
		if shuffle:
			np.random.shuffle(indices)

		for step in range(num_steps):
			start = step * batch_size
			end = start + batch_size
			batch_indices = indices[start:end]
			if batch_indices.size > 0:  # If batch_indices is non-empty.
				data_batch_list.append(data[batch_indices])
				label_batch_list.append(generate_sparse_tuple_from_numpy_array(labels[batch_indices], eos_token=eos_token) if is_sparse_label else labels[batch_indices])

	return data_batch_list, label_batch_list

#%%------------------------------------------------------------------

# REF [site] >> https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def compute_intersection_over_union_of_aabb(aabb1, aabb2):
	# Axis-aligned bounding box = [x_min, y_min, x_max, y_max].

	# Determine the (x, y)-coordinates of the intersection rectangle.
	xA = max(aabb1[0], aabb2[0])
	yA = max(aabb1[1], aabb2[1])
	xB = min(aabb1[2], aabb2[2])
	yB = min(aabb1[3], aabb2[3])

	# Compute the area of intersection rectangle.
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# Compute the area of both the prediction and ground-truth rectangles.
	aabb1Area = (aabb1[2] - aabb1[0] + 1) * (aabb1[3] - aabb1[1] + 1)
	aabb2Area = (aabb2[2] - aabb2[0] + 1) * (aabb2[3] - aabb2[1] + 1)

	# Compute the intersection over union by taking the intersection area and
	#	dividing it by the sum of prediction + ground-truth areas - the interesection area.
	return interArea / float(aabb1Area + aabb2Area - interArea)

#%%------------------------------------------------------------------

import tensorflow as tf

def visualize_activation(session, input_tensor, feed_dict, output_dir_path):
	import tf_cnnvis
	layers = ['r', 'p', 'c']
	return tf_cnnvis.activation_visualization(sess_graph_path=session, value_feed_dict=feed_dict,
			input_tensor=input_tensor, layers=layers,
			path_logdir=os.path.join(output_dir_path, 'vis_log_activation'),
			path_outdir=os.path.join(output_dir_path, 'vis'))

def visualize_by_deconvolution(session, input_tensor, feed_dict, output_dir_path):
	import tf_cnnvis
	layers = ['r', 'p', 'c']
	return tf_cnnvis.deconv_visualization(sess_graph_path=session, value_feed_dict=feed_dict,
			input_tensor=input_tensor, layers=layers,
			path_logdir=os.path.join(output_dir_path, 'vis_log_deconv'),
			path_outdir=os.path.join(output_dir_path, 'vis'))

def visualize_by_partial_occlusion(session, nnInferrer, vis_images, vis_labels, grid_counts, grid_size, occlusion_color, num_classes, batch_size, saver=None, model_dir_path=None):
	"""
	:param grid_point_counts: the numbers of grid points in height and width.
	"""

	if vis_images.shape[0] <= 0:
		return None

	if saver is not None and model_dir_path is not None:
		# Load a model.
		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(session, ckpt.model_checkpoint_path)
		#saver.restore(session, tf.train.latest_checkpoint(model_dir_path))

	img_height, img_width = vis_images.shape[1:3]
	num_grid_height, num_grid_width = grid_counts
	grid_height, grid_width = math.ceil(img_height / num_grid_height), math.ceil(img_width / num_grid_width)
	grid_half_occlusion_height = grid_size[0] * 0.5
	grid_half_occlusion_width = grid_size[1] * 0.5

	occluded_probilities = np.zeros(vis_images.shape[:-1])
	for h in range(num_grid_height):
		h_start = grid_height * h
		h_end = grid_height * (h + 1)
		h_pos = 0.5 * (h_start + h_end)
		h_occlusion_start = math.floor(h_pos - grid_half_occlusion_height)
		if h_occlusion_start < 0:
			h_occlusion_start = 0
		h_occlusion_end = math.ceil(h_pos + grid_half_occlusion_height)
		if h_occlusion_end > img_height:
			h_occlusion_end = img_height
		for w in range(num_grid_width):
			w_start = grid_width * w
			w_end = grid_width * (w + 1)
			w_pos = 0.5 * (w_start + w_end)
			w_occlusion_start = math.floor(w_pos - grid_half_occlusion_width)
			if w_occlusion_start < 0:
				w_occlusion_start = 0
			w_occlusion_end = math.ceil(w_pos + grid_half_occlusion_width)
			if w_occlusion_end > img_width:
				w_occlusion_end = img_width

			images = np.copy(vis_images)  # Deep copy.
			images[:,h_occlusion_start:h_occlusion_end,w_occlusion_start:w_occlusion_end,:] = occlusion_color

			inferences = nnInferrer.infer(session, images, batch_size)

			# Top-1 predicted probability.
			if num_classes >= 2:
				inferences = np.max(inferences * vis_labels, -1)
			else:
				inferences = np.max(inferences * vis_labels)

			#occluded_probilities[:,h_start:h_end,w_start:w_end] = inferences
			for (idx, prob) in enumerate(occluded_probilities):
				prob[h_start:h_end,w_start:w_end] = inferences[idx]

	return occluded_probilities

#%%------------------------------------------------------------------

def display_train_history(history):
	# List all data in history.
	#print(history.keys())

	# Summarize history for accuracy.
	fig = plt.figure()
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('model accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

	# Summarize history for loss.
	fig = plt.figure()
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

def save_train_history(history, dir_path):
	# List all data in history.
	#print(history.keys())

	# Summarize history for accuracy.
	if history['acc'] is not None or history['val_acc'] is not None:
		fig = plt.figure()
		if 'acc' in history:
			plt.plot(history['acc'])
		if 'val_acc' in history:
			plt.plot(history['val_acc'])
		plt.title('Model accuracy')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		if not 'acc' in history:
			plt.legend(['test'], loc='upper left')
		elif not 'val_acc' in history:
			plt.legend(['train'], loc='upper left')
		else:
			plt.legend(['train', 'test'], loc='upper left')
		fig.savefig(dir_path + '/accuracy.png')
		plt.close(fig)

	# Summarize history for loss.
	fig = plt.figure()
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Model loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'test'], loc='upper left')
	fig.savefig(dir_path + '/loss.png')
	plt.close(fig)
