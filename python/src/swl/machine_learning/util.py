import os, math, functools
import numpy as np
import tensorflow as tf
#import keras
import matplotlib.pyplot as plt

def to_one_hot_encoding(label_indexes, num_classes=None):
	if num_classes is None:
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

def standardize_samplewise(data):
	for idx in range(data.shape[0]):
		for ch in range(data.shape[3]):
			mean = np.mean(data[idx,:,:,ch])
			sd = np.std(data[idx,:,:,ch])
			if np.isclose(sd, 0.0) or np.isnan(sd):
				#print('[Warning] sd = 0')
				data[idx,:,:,ch] -= mean
				#data[idx,:,:,ch] = 0
			else:
				data[idx,:,:,ch] = (data[idx,:,:,ch] - mean) / sd
	return data

def standardize_featurewise(data):
	for r in range(data.shape[1]):
		for c in range(data.shape[2]):
			mean = np.mean(data[:,r,c,:], axis=0)
			sd = np.std(data[:,r,c,:], axis=0)
			if np.any(np.isclose(sd, np.zeros(sd.size))) or np.any(np.isnan(sd)):
				#print('[Warning] sd = 0')
				for ch in range(data.shape[3]):
					if np.isclose(sd[ch], 0.0) or np.isnan(sd[ch]):
						data[:,r,c,ch] -= mean[ch]
						#data[:,r,c,ch] = 0
					else:
						data[:,r,c,ch] = (data[:,r,c,ch] - mean[ch]) / sd[ch]
			else:
				data[:,r,c,:] = (data[:,r,c,:] - mean) / sd
	return data

def normalize_samplewise_by_min_max(data):
	for idx in range(data.shape[0]):
		for ch in range(data.shape[3]):
			dmin = np.amin(data[idx,:,:,ch])
			dmax = np.amax(data[idx,:,:,ch])
			if np.isclose(dmin, dmax):
				#print('[Warning] max - min = 0')
				data[idx,:,:,ch] -= dmin
				#data[idx,:,:,ch] = 0
			else:
				data[idx,:,:,ch] = (data[idx,:,:,ch] - dmin) / (dmax - dmin)
	return data

def normalize_featurewise_by_min_max(data):
	for r in range(data.shape[1]):
		for c in range(data.shape[2]):
			dmin = np.amin(data[:,r,c,:], axis=0)
			dmax = np.amax(data[:,r,c,:], axis=0)
			if np.any(np.isclose(dmin, dmax)):
				#print('[Warning] max - min = 0')
				for ch in range(data.shape[3]):
					if np.isclose(dmin[ch], dmax[ch]):
						data[:,r,c,ch] -= dmin[ch]
						#data[:,r,c,ch] = 0
					else:
						data[:,r,c,ch] = (data[:,r,c,ch] - dmin[ch]) / (dmax[ch] - dmin[ch])
			else:
				data[:,r,c,:] = (data[:,r,c,:] - dmin) / (dmax - dmin)
	return data

#%%------------------------------------------------------------------

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
def sequences_to_sparse(sequences, dtype=np.int32):
	"""Change a list of sequences to a 2D sparse representention.
	Inputs:
		sequences(a list of lists): A list of lists of type dtype where each element is a sequence.
		dtype (numpy.dtype): A data type.
	Output:
		A sparse tensor (tuple): A tuple with (indices, values, dense_shape).
			indices (numpy.array): The indices of non-zero elements in a dense tensor.
			values (numpy.array): The values of non-zero elements in a dense tensor.
			dense_shape (numpy.array): The shape of a dense tensor.
	"""

	indices, values = list(), list()
	for idx, seq in enumerate(sequences):
		indices.extend(zip([idx] * len(seq), range(len(seq))))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	dense_shape = np.asarray([len(sequences), indices.max(axis=0)[1] + 1], dtype=np.int64)

	return indices, values, dense_shape   # Refer to tf.SparseTensorValue.

def sparse_to_sequences(indices, values, dense_shape, dtype=np.int32):
	"""Change a 2D sparse representention to a list of sequences.
	Inputs:
		A sparse tensor (tuple): A tuple with (indices, values, dense_shape).
			indices (numpy.array): The indices of non-zero elements in a dense tensor.
			values (numpy.array): The values of non-zero elements in a dense tensor.
			dense_shape (numpy.array): The shape of a dense tensor.
		dtype (numpy.dtype): A data type.
	Output:
		sequences(a list of lists): A list of lists of type dtype where each element is a sequence.
	"""

	default_val = np.max(values) + 1
	dense = sparse_to_dense(indices, values, dense_shape, default_val)

	def extract(x):
		x = x.tolist()
		try:
			return x[:x.index(default_val)]
		except ValueError:
			return x

	return list(map(extract, dense))

def sequences_to_dense(sequences, default_value=0, dtype=np.int32):
	"""Change a list of sequences to a 2D dense tensor.
	Inputs:
		sequences(a list of lists): A list of lists of type dtype where each element is a sequence.
		default_value (int): It is part of the target label that signifies the end of a sentence (EOS).
		dtype (numpy.dtype): A data type.
	Returns:
		A dense tensor (numpy.array): A numpy array of type dtype.
	"""

	max_len = functools.reduce(lambda x, seq: max(x, len(seq)), sequences, 0)
	dense = np.full((len(sequences), max_len), default_value, dtype=dtype)
	for idx, seq in enumerate(sequences):
		dense[idx,:len(seq)] = seq
	return dense

def dense_to_sequences(dense, default_value=0, dtype=np.int32):
	"""Change a 2D dense tensor to a list of sequences.
	Inputs:
		dense (numpy.array): A numpy array of type dtype.
		default_value (int): It is part of the target label that signifies the end of a sentence (EOS).
		dtype (numpy.dtype): A data type.
	Returns:
		sequences(a list of lists): A list of lists of type dtype where each element is a sequence.
	"""

	sequences = list()
	for idx, row in enumerate(dense):
		default_indices = np.where(row == default_value)[0]
		if default_indices.size > 0:
			row = row[:default_indices[0]]
		sequences.append(list(row))

	return sequences

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
def dense_to_sparse(dense, default_value=0, dtype=np.int32):
	"""Change a 2D dense tensor to a 2D sparse representention.
	Inputs:
		dense (numpy.array): A numpy array of type dtype.
		default_value (int): It is part of the target label that signifies the end of a sentence (EOS).
		dtype (numpy.dtype): A data type.
	Returns:
		A sparse tensor (tuple): A tuple with (indices, values, dense_shape).
			indices (numpy.array): The indices of non-zero elements in a dense tensor.
			values (numpy.array): The values of non-zero elements in a dense tensor.
			dense_shape (numpy.array): The shape of a dense tensor.
	"""

	indices, values = list(), list()
	for idx, row in enumerate(dense):
		default_indices = np.where(row == default_value)[0]
		if default_indices.size > 0:
			row = row[:default_indices[0]]
		indices.extend(zip([idx] * len(row), range(len(row))))
		values.extend(row)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	dense_shape = np.asarray([len(dense), indices.max(axis=0)[1] + 1], dtype=np.int64)

	return indices, values, dense_shape   # Refer to tf.SparseTensorValue.

def sparse_to_dense(indices, values, dense_shape, default_value=0, dtype=np.int32):
	"""Change a 2D sparse representation of a tensor to a 2D dense tensor.
	Inputs:
		A sparse tensor (tuple): A tuple with (indices, values, dense_shape).
			indices (numpy.array): The indices of non-zero elements in a dense tensor.
			values (numpy.array): The values of non-zero elements in a dense tensor.
			dense_shape (numpy.array): The shape of a dense tensor.
		default_value (int): It is part of the target label that signifies the end of a sentence (EOS).
		dtype (numpy.dtype): A data type.
	Returns:
		A dense tensor (numpy.array): A numpy array of type dtype.
	"""

	dense = np.full(dense_shape, default_value, dtype=dtype)
	where_indices = tuple([indices[:,idx] for idx in range(indices.shape[1])])
	dense[where_indices] = values
	return dense

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
				label_batch_list.append(dense_to_sparse(labels[batch_indices], default_value=eos_token) if is_sparse_label else labels[batch_indices])

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
	if ('acc' in history and history['acc'] is not None) or ('val_acc' in history and history['val_acc'] is not None):
		fig = plt.figure()
		if 'acc' in history:
			plt.plot(history['acc'])
		if 'val_acc' in history:
			plt.plot(history['val_acc'])
		plt.title('model accuracy')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		if not 'acc' in history:
			plt.legend(['test'], loc='upper left')
		elif not 'val_acc' in history:
			plt.legend(['train'], loc='upper left')
		else:
			plt.legend(['train', 'test'], loc='lower right')
		plt.show()
		plt.close(fig)

	# Summarize history for loss.
	if ('loss' in history and history['loss'] is not None) or ('val_loss' in history and history['val_loss'] is not None):
		fig = plt.figure()
		if 'loss' in history:
			plt.plot(history['loss'])
		if 'val_loss' in history:
			plt.plot(history['val_loss'])
		plt.title('model loss')
		plt.xlabel('epochs')
		plt.ylabel('loss')
		if not 'loss' in history:
			plt.legend(['test'], loc='upper right')
		elif not 'val_loss' in history:
			plt.legend(['train'], loc='upper right')
		else:
			plt.legend(['train', 'test'], loc='upper right')
		plt.show()
		plt.close(fig)

def save_train_history(history, dir_path):
	# List all data in history.
	#print(history.keys())

	# Summarize history for accuracy.
	if ('acc' in history and history['acc'] is not None) or ('val_acc' in history and history['val_acc'] is not None):
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
	if ('loss' in history and history['loss'] is not None) or ('val_loss' in history and history['val_loss'] is not None):
		fig = plt.figure()
		if 'loss' in history:
			plt.plot(history['loss'])
		if 'val_loss' in history:
			plt.plot(history['val_loss'])
		plt.title('Model loss')
		plt.xlabel('epochs')
		plt.ylabel('loss')
		if not 'loss' in history:
			plt.legend(['test'], loc='upper right')
		elif not 'val_loss' in history:
			plt.legend(['train'], loc='upper right')
		else:
			plt.legend(['train', 'test'], loc='upper right')
		fig.savefig(dir_path + '/loss.png')
		plt.close(fig)
