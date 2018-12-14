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
import time, datetime, math, random
import numpy as np
import tensorflow as tf
from PIL import Image
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer, GradientClippingNeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.util.util as swl_util
import swl.machine_learning.util as swl_ml_util
import swl.machine_learning.tensorflow.util as swl_tf_util
from mnist_crnn import MnistCrnnWithCrossEntropyLoss, MnistCrnnWithCtcLoss
import traceback

#%%------------------------------------------------------------------

def create_crnn(image_height, image_width, image_channel, num_classes, num_time_steps, is_time_major, is_sparse_label, label_eos_token):
	if is_sparse_label:
		return MnistCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes, num_time_steps, is_time_major=is_time_major, eos_token=label_eos_token)
	else:
		return MnistCrnnWithCrossEntropyLoss(image_height, image_width, image_channel, num_classes, num_time_steps, is_time_major=is_time_major)

#%%------------------------------------------------------------------

class SimpleCrnnTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			start_learning_rate = 1e-2
			decay_steps = 10000
			decay_rate = 0.96
			#learning_rate = start_learning_rate
			learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.inverse_time_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.natural_exp_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.cosine_decay(start_learning_rate, global_step, decay_steps, alpha=0.0)
			#learning_rate = tf.train.linear_cosine_decay(start_learning_rate, global_step, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.noisy_linear_cosine_decay(start_learning_rate, global_step, decay_steps, initial_variance=1.0, variance_decay=0.55, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.polynomial_decay(start_learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False)
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-8)
			#optimizer = tf.train.AdagradDAOptimizer(learning_rate, global_step=?, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
			#optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1)
			#optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=False)
			#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1e-10)

		super().__init__(neuralNet, optimizer, global_step)

class SimpleCrnnGradientClippingTrainer(GradientClippingNeuralNetTrainer):
	def __init__(self, neuralNet, max_gradient_norm, initial_epoch=0):
		global_step = tf.Variable(initial_epoch, name='global_step', trainable=False)
		with tf.name_scope('learning_rate'):
			start_learning_rate = 1e-2
			decay_steps = 10000
			decay_rate = 0.96
			#learning_rate = start_learning_rate
			learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.inverse_time_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.natural_exp_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
			#learning_rate = tf.train.cosine_decay(start_learning_rate, global_step, decay_steps, alpha=0.0)
			#learning_rate = tf.train.linear_cosine_decay(start_learning_rate, global_step, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.noisy_linear_cosine_decay(start_learning_rate, global_step, decay_steps, initial_variance=1.0, variance_decay=0.55, num_periods=0.5, alpha=0.0, beta=0.001)
			#learning_rate = tf.train.polynomial_decay(start_learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False)
			#tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.95, epsilon=1e-8)
			#optimizer = tf.train.AdagradDAOptimizer(learning_rate, global_step=?, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
			#optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1)
			#optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=False)
			#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1e-10)

		super().__init__(neuralNet, optimizer, max_gradient_norm, global_step)

#%%------------------------------------------------------------------

def visualize_dataset(images, labels, max_example_count=0):
	print('Image shape = {}, label shape = {}'.format(images.shape, labels.shape))
	num_examples, max_time_steps = images.shape[:2]
	max_digit_count = labels.shape[1]
	for idx in range(num_examples if 0 >= max_example_count else min(num_examples, max_example_count)):
		"""
		# Saves each digit.
		for step in range(max_time_steps):
			img_arr = images[idx,step,:,:,0]
			lbl = np.argmax(labels[idx,step,:], axis=-1)
			#img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='L')
			#img.save('./img_I{}_T{}_L{}.png'.format(idx, step, lbl))
			img = Image.fromarray(img_arr.astype(np.float32), mode='F')
			img.save('./img_I{}_T{}_L{}.tif'.format(idx, step, lbl))
		"""

		# Saves each datum with multiple digits.
		comp_img = np.zeros((images.shape[2], images.shape[3] * max_time_steps))
		lbl_list = list()
		for step in range(max_time_steps):
			comp_img[:,step*images.shape[3]:(step+1)*images.shape[3]] = images[idx,step,:,:,0]
		for step in range(max_digit_count):
			lbl_list.append(np.argmax(labels[idx,step,:], axis=-1))
		#img = Image.fromarray((comp_img * 255).astype(np.uint8), mode='L')
		#img.save('./data_I{}_L{}.png'.format(idx, '-'.join(str(lbl) for lbl in lbl_list)))
		img = Image.fromarray(comp_img.astype(np.float32), mode='F')
		img.save('./data_I{}_L{}.tif'.format(idx, '-'.join(str(lbl) for lbl in lbl_list)))

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
		#labels = to_one_hot_encoding(labels, num_classes).astype(np.uint8)
		pass

	return data, labels

def generate_composite_dataset(images, labels, min_digit_count, max_digit_count, max_time_steps, space_label, is_sparse_label):
	num_spaces = 3000
	num_examples = images.shape[0]
	total_count = num_examples + num_spaces
	indices = list(range(total_count))
	random.shuffle(indices)

	image_shape = (max_time_steps,) + images.shape[1:]
	label_shape = (max_digit_count if is_sparse_label else max_time_steps,) + labels.shape[1:]
	space_image = np.zeros(images.shape[1:])
	space_label_arr = np.zeros(labels.shape[1])
	space_label_arr[space_label] = 1

	image_list, label_list = list(), list()
	start_idx = 0
	while start_idx < total_count:
		#end_idx = min(start_idx + random.randint(min_digit_count, max_digit_count), total_count)
		end_idx = start_idx + random.randint(min_digit_count, max_digit_count)
		example_indices = indices[start_idx:end_idx]

		comp_image = np.random.rand(*image_shape)
		comp_label = np.zeros(label_shape)
		comp_label[:,-1] = 1
		for i, idx in enumerate(example_indices):
			if idx >= num_examples:
				comp_image[i,:,:,:] = space_image
				comp_label[i,:] = space_label_arr
			else:
				comp_image[i,:,:,:] = images[idx,:,:,:]
				comp_label[i,:] = labels[idx,:]

		image_list.append(comp_image)
		label_list.append(comp_label)

		start_idx = end_idx

	return np.reshape(image_list, (-1,) + image_list[0].shape), np.reshape(label_list, (-1,) + label_list[0].shape)

def prepare_multiple_character_dataset(image_shape, num_classes, min_digit_count, max_digit_count, max_time_steps, space_label, is_sparse_label):
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

	train_labels = np.pad(train_labels, ((0, 0), (0, num_classes - train_labels.shape[1])), 'constant', constant_values=0)
	test_labels = np.pad(test_labels, ((0, 0), (0, num_classes - test_labels.shape[1])), 'constant', constant_values=0)

	train_images, train_labels = generate_composite_dataset(train_images, train_labels, min_digit_count, max_digit_count, max_time_steps, space_label, is_sparse_label)
	test_images, test_labels = generate_composite_dataset(test_images, test_labels, min_digit_count, max_digit_count, max_time_steps, space_label, is_sparse_label)

	return train_images, train_labels, test_images, test_labels

def prepare_single_character_dataset(image_shape, num_classes, max_time_steps, slice_width, slice_stride, is_sparse_label):
	# Pixel value: [0, 255].
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images /= 255.0
	train_images = np.reshape(train_images, (-1,) + image_shape)
	train_labels = tf.keras.utils.to_categorical(train_labels).astype(np.uint8)
	test_images /= 255.0
	test_images = np.reshape(test_images, (-1,) + image_shape)
	test_labels = tf.keras.utils.to_categorical(test_labels).astype(np.uint8)

	# Pre-process.
	#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
	#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

	train_labels = np.pad(train_labels, ((0, 0), (0, num_classes - train_labels.shape[1])), 'constant', constant_values=0)
	test_labels = np.pad(test_labels, ((0, 0), (0, num_classes - test_labels.shape[1])), 'constant', constant_values=0)

	image_height, image_width, _ = image_shape

	# TODO [improve] >> A more efficient way may exist.
	# (samples, time-steps, features).
	train_sliced_images = np.zeros((train_images.shape[0], max_time_steps, image_height, slice_width, train_images.shape[-1]))
	test_sliced_images = np.zeros((test_images.shape[0], max_time_steps, image_height, slice_width, test_images.shape[-1]))
	for step in range(max_time_steps):
		start_idx, end_idx = step*slice_stride, step*slice_stride+slice_width
		if end_idx > image_width:
			train_sliced_images[:,step,:,:image_width-start_idx,:] = train_images[:,:,start_idx:end_idx,:]
			test_sliced_images[:,step,:,:image_width-start_idx,:] = test_images[:,:,start_idx:end_idx,:]
		else:
			train_sliced_images[:,step,:,:,:] = train_images[:,:,start_idx:end_idx,:]
			test_sliced_images[:,step,:,:,:] = test_images[:,:,start_idx:end_idx,:]

	if is_sparse_label:
		return train_sliced_images, np.reshape(train_labels, (-1, 1, train_labels.shape[-1])), test_sliced_images, np.reshape(test_labels, (-1, 1, test_labels.shape[-1]))
	else:
		train_sliced_labels = np.zeros((train_labels.shape[0], max_time_steps, train_labels.shape[-1]))
		test_sliced_labels = np.zeros((test_labels.shape[0], max_time_steps, test_labels.shape[-1]))
		for step in range(max_time_steps):
			train_sliced_labels[:,step,:] = train_labels
			test_sliced_labels[:,step,:] = test_labels
		return train_sliced_images, train_sliced_labels, test_sliced_images, test_sliced_labels

#%%------------------------------------------------------------------

def main():
	#np.random.seed(7)

	#--------------------
	# Parameters.

	does_need_training = True
	does_resume_training = False

	output_dir_prefix = 'mnist_crnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20181211T172200'

	is_time_major = False  # Fixed.
	is_sparse_label = False
	if is_sparse_label:
		use_batch_list = True  # Fixed.
	else:
		use_batch_list = False

	image_height, image_width, image_channel = 28, 28, 1
	"""
	# For prepare_single_character_dataset().
	slice_width, slice_stride = 14, 7
	min_time_steps = math.ceil((image_width - slice_width) / slice_stride) + 1
	max_time_steps = min_time_steps  # max_time_steps >= min_time_steps.
	"""
	# For prepare_multiple_character_dataset().
	min_digit_count, max_digit_count = 3, 5
	max_time_steps = max_digit_count + 2  # max_time_steps >= max_digit_count.

	num_labels = 10
	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	# 0~9 + space label + blank label.
	num_classes = num_labels + 1 + 1
	space_label = num_classes - 2
	blank_label = num_classes - 1
	label_eos_token = -1

	batch_size = 128  # Number of samples per gradient update.
	if is_sparse_label:
		num_epochs = 500  # Number of times to iterate over training data.
	else:
		num_epochs = 200  # Number of times to iterate over training data.
	shuffle = True

	#max_gradient_norm = 5
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
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	swl_util.make_dir(checkpoint_dir_path)
	swl_util.make_dir(inference_dir_path)
	swl_util.make_dir(train_summary_dir_path)
	swl_util.make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	#train_images, train_labels, test_images, test_labels = prepare_single_character_dataset((image_height, image_width, image_channel), num_classes, max_time_steps, slice_width, slice_stride, is_sparse_label)
	# Images: (samples, time-steps, height, width, channels), labels: (samples, num_digits, one-hot encoding).
	train_images, train_labels, test_images, test_labels = prepare_multiple_character_dataset((image_height, image_width, image_channel), num_classes, min_digit_count, max_digit_count, max_time_steps, space_label, is_sparse_label)

	# Visualize dataset.
	#visualize_dataset(train_images, train_labels, 5)
	#visualize_dataset(test_images, test_labels, 5)

	if is_sparse_label:
		train_labels = np.argmax(train_labels, axis=-1)
		test_labels = np.argmax(test_labels, axis=-1)

	if use_batch_list:
		train_images_list, train_labels_list = swl_ml_util.generate_batch_list(train_images, train_labels, batch_size, shuffle=shuffle, is_time_major=is_time_major, is_sparse_label=is_sparse_label, eos_token=blank_label)
		test_images_list, test_labels_list = swl_ml_util.generate_batch_list(test_images, test_labels, batch_size, shuffle=False, is_time_major=is_time_major, is_sparse_label=is_sparse_label, eos_token=blank_label)

	print('Train images = {}, train labels = {}, test images = {}, test labels = {}'.format(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape))
	if use_batch_list:
		print('Train images list = {}, train labels list = {}, test images list = {}, test labels list = {}'.format(len(train_images_list), len(train_labels_list), len(test_images_list), len(test_labels_list)))

	#--------------------
	# Create models, sessions, and graphs.

	# Create graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			# Create a model.
			cnnModelForTraining = create_crnn(image_height, image_width, image_channel, num_classes, max_time_steps, is_time_major, is_sparse_label, label_eos_token)
			cnnModelForTraining.create_training_model()

			# Create a trainer.
			nnTrainer = SimpleCrnnTrainer(cnnModelForTraining, initial_epoch)
			#nnTrainer = SimpleCrnnGradientClippingTrainer(cnnModelForTraining, max_gradient_norm, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()
			#initializer = tf.variables_initializer(tf.global_variables())
			#initializer = tf.glorot_normal_initializer(tf.global_variables())  # Xavier normal initializer.
			#initializer = tf.glorot_uniform_initializer(tf.global_variables())  # Xavier uniform initializer.
			#initializer = tf.uniform_unit_scaling_initializer(tf.global_variables())
			#initializer = tf.variance_scaling_initializer(tf.global_variables())
			#initializer = tf.orthogonal_initializer(tf.global_variables())
			#initializer = tf.truncated_normal_initializer(tf.global_variables())
			#initializer = tf.random_normal_initializer(tf.global_variables())
			#initializer = tf.random_uniform_initializer(tf.global_variables())

		with eval_graph.as_default():
			# Create a model.
			cnnModelForEvaluation = create_crnn(image_height, image_width, image_channel, num_classes, max_time_steps, is_time_major, is_sparse_label, label_eos_token)
			cnnModelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(cnnModelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		# Create a model.
		cnnModelForInference = create_crnn(image_height, image_width, image_channel, num_classes, max_time_steps, is_time_major, is_sparse_label, label_eos_token)
		cnnModelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(cnnModelForInference)

		# Create a saver.
		infer_saver = tf.train.Saver()

	# Create sessions.
	if does_need_training:
		train_session = tf.Session(graph=train_graph, config=sess_config)
		eval_session = tf.Session(graph=eval_graph, config=sess_config)
	infer_session = tf.Session(graph=infer_graph, config=sess_config)

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#%%------------------------------------------------------------------
	# Train and evaluate.

	if does_need_training:
		start_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				if use_batch_list:
					# Supports lists of dense or sparse labels.
					swl_tf_util.train_neural_net_by_batch_list(sess, nnTrainer, train_images_list, train_labels_list, test_images_list, test_labels_list, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major, is_sparse_label)
				else:
					# Supports a dense label only.
					#swl_tf_util.train_neural_net_after_generating_batch_list(sess, nnTrainer, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path, is_time_major)
					swl_tf_util.train_neural_net(sess, nnTrainer, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - start_time))

		start_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				if use_batch_list:
					# Supports lists of dense or sparse labels.
					swl_tf_util.evaluate_neural_net_by_batch_list(sess, nnEvaluator, test_images_list, test_labels_list, eval_saver, checkpoint_dir_path, is_time_major, is_sparse_label)
				else:
					#test_labels = swl_ml_util.generate_sparse_tuple_from_numpy_array(np.argmax(test_labels, axis=-1), eos_token=label_eos_token)
					# Supports dense or sparse labels.
					#swl_tf_util.evaluate_neural_net(sess, nnEvaluator, test_images, test_labels, batch_size, eval_saver, checkpoint_dir_path, is_time_major, is_sparse_label)
					# Supports dense or sparse labels.
					swl_tf_util.evaluate_neural_net(sess, nnEvaluator, test_images, test_labels, batch_size, eval_saver, checkpoint_dir_path, is_time_major, is_sparse_label)
		print('\tTotal evaluation time = {}'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infer.

	start_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			if is_sparse_label:
				ground_truths = test_labels
				if use_batch_list:
					# Supports lists of dense or sparse labels.
					inferences_list = swl_tf_util.infer_from_batch_list_by_neural_net(sess, nnInferrer, test_images_list, infer_saver, checkpoint_dir_path, is_time_major)
					inferences = None
					for inf in inferences_list:
						#inf = sess.run(tf.sparse_to_dense(inf[0], inf[2], inf[1], default_value=label_eos_token))
						inf = sess.run(tf.sparse_to_dense(inf[0], inf[2], inf[1], default_value=blank_label))
						inferences = inf if inferences is None else np.concatenate((inferences, inf), axis=0)
				else:
					# Supports dense or sparse labels.
					inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_images, batch_size, infer_saver, checkpoint_dir_path, is_time_major, is_sparse_label)
					#inferences = sess.run(tf.sparse_to_dense(inferences[0], inferences[2], inferences[1], default_value=label_eos_token))
					inferences = sess.run(tf.sparse_to_dense(inferences[0], inferences[2], inferences[1], default_value=blank_label))
			else:
				ground_truths = np.argmax(test_labels, axis=-1)
				if use_batch_list:
					# Supports lists of dense or sparse labels.
					inferences_list = swl_tf_util.infer_from_batch_list_by_neural_net(sess, nnInferrer, test_images_list, infer_saver, checkpoint_dir_path, is_time_major)
					inferences = None
					for inf in inferences_list:
						inferences = inf if inferences is None else np.concatenate((inferences, inf), axis=0)
				else:
					# Supports dense or sparse labels.
					inferences = swl_tf_util.infer_by_neural_net(sess, nnInferrer, test_images, batch_size, infer_saver, checkpoint_dir_path, is_time_major, is_sparse_label)
				inferences = np.argmax(inferences, axis=-1)

			# TODO [check] >> Is it correct?
			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))

			for i in range(10):
				print(inferences[i], ground_truths[i])
	print('\tTotal inference time = {}'.format(time.time() - start_time))

	#--------------------
	# Close sessions.

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
