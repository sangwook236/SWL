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
from mnist_crnn import MnistCrnnWithCrossEntropyLoss, MnistCrnnWithCtcLoss
from swl.machine_learning.tensorflow.neural_net_trainer import NeuralNetTrainer
from swl.machine_learning.tensorflow.neural_net_evaluator import NeuralNetEvaluator
from swl.machine_learning.tensorflow.neural_net_inferrer import NeuralNetInferrer
import swl.machine_learning.util as swl_ml_util
import traceback

#%%------------------------------------------------------------------

class SimpleCrnnTrainer(NeuralNetTrainer):
	def __init__(self, neuralNet, initial_epoch=0):
		with tf.name_scope('learning_rate'):
			learning_rate = 1e-2
			tf.summary.scalar('learning_rate', learning_rate)
		with tf.name_scope('optimizer'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
			optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

		super().__init__(neuralNet, optimizer, initial_epoch)

#%%------------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data

def visualize_dataset(images, labels, max_example_count=0):
	print('Image shape = {}, label shape = {}'.format(images.shape, labels.shape))
	num_examples, max_time_steps = images.shape[:2]
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
			lbl_list.append(np.argmax(labels[idx,step,:], axis=-1))
		#img = Image.fromarray((comp_img * 255).astype(np.uint8), mode='L')
		#img.save('./data_I{}_L{}.png'.format(idx, '-'.join(str(lbl) for lbl in lbl_list)))
		img = Image.fromarray(comp_img.astype(np.float32), mode='F')
		img.save('./data_I{}_L{}.tif'.format(idx, '-'.join(str(lbl) for lbl in lbl_list)))

def composite_dataset(images, labels, min_digit_count, max_digit_count, max_time_steps):
	indices = list(range(images.shape[0]))
	random.shuffle(indices)
	num_examples = len(indices)

	start_idx = 0
	image_list, label_list = list(), list()
	while start_idx < num_examples:
		#end_idx = min(start_idx + random.randint(min_digit_count, max_digit_count), num_examples)
		end_idx = start_idx + random.randint(min_digit_count, max_digit_count)
		example_indices = indices[start_idx:end_idx]

		comp_image = np.zeros((max_time_steps,) + images.shape[1:])
		comp_label = np.zeros((max_digit_count,) + labels.shape[1:])
		comp_label[:,-1] = 1
		for i, idx in enumerate(example_indices):
			comp_image[i,:,:,:] = images[idx,:,:,:]
			comp_label[i,:] = labels[idx,:]

		image_list.append(comp_image)
		label_list.append(comp_label)

		start_idx = end_idx

	return np.reshape(image_list, (-1,) + image_list[0].shape), np.reshape(label_list, (-1,) + label_list[0].shape)

def prepare_multiple_character_dataset(data_dir_path, image_shape, num_classes, min_digit_count, max_digit_count, max_time_steps):
	# Pixel value: [0, 1].
	mnist = input_data.read_data_sets(data_dir_path, one_hot=True)

	image_height, image_width, _ = image_shape

	train_images = np.reshape(mnist.train.images, (-1,) + image_shape)
	train_labels = np.round(mnist.train.labels).astype(np.int)
	train_labels = np.pad(train_labels, ((0, 0), (0, num_classes - train_labels.shape[1])), 'constant', constant_values=0)
	test_images = np.reshape(mnist.test.images, (-1,) + image_shape)
	test_labels = np.round(mnist.test.labels).astype(np.int)
	test_labels = np.pad(test_labels, ((0, 0), (0, num_classes - test_labels.shape[1])), 'constant', constant_values=0)

	train_images, train_labels = composite_dataset(train_images, train_labels, min_digit_count, max_digit_count, max_time_steps)
	test_images, test_labels = composite_dataset(test_images, test_labels, min_digit_count, max_digit_count, max_time_steps)

	return train_images, train_labels, test_images, test_labels

def prepare_single_character_dataset(data_dir_path, image_shape, num_classes, max_time_steps, slice_width, slice_stride, use_variable_length_output):
	# Pixel value: [0, 1].
	mnist = input_data.read_data_sets(data_dir_path, one_hot=True)

	image_height, image_width, _ = image_shape

	train_images = np.reshape(mnist.train.images, (-1,) + image_shape)
	train_labels = np.round(mnist.train.labels).astype(np.int)
	train_labels = np.pad(train_labels, ((0, 0), (0, num_classes - train_labels.shape[1])), 'constant', constant_values=0)
	test_images = np.reshape(mnist.test.images, (-1,) + image_shape)
	test_labels = np.round(mnist.test.labels).astype(np.int)
	test_labels = np.pad(test_labels, ((0, 0), (0, num_classes - test_labels.shape[1])), 'constant', constant_values=0)

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

	if use_variable_length_output:
		return train_sliced_images, np.reshape(train_labels, (-1, 1, train_labels.shape[-1])), test_sliced_images, np.reshape(test_labels, (-1, 1, test_labels.shape[-1]))
	else:
		train_sliced_labels = np.zeros((train_labels.shape[0], max_time_steps, train_labels.shape[-1]))
		test_sliced_labels = np.zeros((test_labels.shape[0], max_time_steps, test_labels.shape[-1]))
		for step in range(max_time_steps):
			train_sliced_labels[:,step,:] = train_labels
			test_sliced_labels[:,step,:] = test_labels
		return train_sliced_images, train_sliced_labels, test_sliced_images, test_sliced_labels

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

def train_neural_net(session, nnTrainer, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, does_resume_training, saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path):
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
	history = nnTrainer.train(session, train_images, train_labels, val_images, val_labels, batch_size, num_epochs, shuffle, saver=saver, model_save_dir_path=checkpoint_dir_path, train_summary_dir_path=train_summary_dir_path, val_summary_dir_path=val_summary_dir_path)
	print('\tTraining time = {}'.format(time.time() - start_time))

	# Save a graph.
	#tf.train.write_graph(session.graph_def, output_dir_path, 'crnn_graph.pb', as_text=False)
	##tf.train.write_graph(session.graph_def, output_dir_path, 'crnn_graph.pbtxt', as_text=True)

	# Save a serving model.
	#builder = tf.saved_model.builder.SavedModelBuilder(output_dir_path + '/serving_model')
	#builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
	#builder.save(as_text=False)

	# Display results.
	#swl_ml_util.display_train_history(history)
	if output_dir_path is not None:
		swl_ml_util.save_train_history(history, output_dir_path)
	print('[SWL] Info: End training...')

def evaluate_neural_net(session, nnEvaluator, val_images, val_labels, batch_size, saver=None, checkpoint_dir_path=None):
	num_val_examples = 0
	if val_images is not None and val_labels is not None:
		if val_images.shape[0] == val_labels.shape[0]:
			num_val_examples = val_images.shape[0]

	if num_val_examples > 0:
		if saver is not None and checkpoint_dir_path is not None:
			# Load a model.
			# REF [site] >> https://www.tensorflow.org/programmers_guide/saved_model
			# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			saver.restore(session, ckpt.model_checkpoint_path)
			#saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir_path))
			print('[SWL] Info: Loaded a model.')

		print('[SWL] Info: Start evaluation...')
		start_time = time.time()
		val_loss, val_acc = nnEvaluator.evaluate(session, val_images, val_labels, batch_size)
		print('\tEvaluation time = {}'.format(time.time() - start_time))
		print('\tValidation loss = {}, validation accurary = {}'.format(val_loss, val_acc))
		print('[SWL] Info: End evaluation...')
	else:
		print('[SWL] Error: The number of validation images is not equal to that of validation labels.')

def infer_by_neural_net(session, nnInferrer, test_images, test_labels, num_classes, batch_size, saver=None, checkpoint_dir_path=None):
	num_inf_examples = 0
	if test_images is not None and test_labels is not None:
		if test_images.shape[0] == test_labels.shape[0]:
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

		if num_classes >= 2:
			inferences = np.argmax(inferences, axis=-1)
			groundtruths = np.argmax(test_labels, axis=-1)
		else:
			inferences = np.around(inferences)
			groundtruths = test_labels
		correct_estimation_count = np.count_nonzero(np.equal(inferences, groundtruths))

		print('\tAccurary = {} / {} = {}'.format(correct_estimation_count, groundtruths.size, correct_estimation_count / groundtruths.size))
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

def create_crnn(input_shape, output_shape, is_time_major, use_variable_length_output, label_eos_token):
	if use_variable_length_output:
		return MnistCrnnWithCtcLoss(input_shape, output_shape, is_time_major=is_time_major, eos_token=label_eos_token)
	else:
		return MnistCrnnWithCrossEntropyLoss(input_shape, output_shape, is_time_major=is_time_major)

def main():
	#np.random.seed(7)

	does_need_training = True
	does_resume_training = False

	#--------------------
	# Prepare directories.

	output_dir_prefix = 'mnist_crnn'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20180302T155710'

	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	inference_dir_path = os.path.join(output_dir_path, 'inference')
	train_summary_dir_path = os.path.join(output_dir_path, 'train_log')
	val_summary_dir_path = os.path.join(output_dir_path, 'val_log')

	make_dir(checkpoint_dir_path)
	make_dir(inference_dir_path)
	make_dir(train_summary_dir_path)
	make_dir(val_summary_dir_path)

	#--------------------
	# Prepare data.

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mnist/0_download'

	use_variable_length_output = True

	image_height, image_width = 28, 28
	num_labels = 10
	label_eos_token = -1

	is_time_major = False
	num_classes = num_labels + 1  # num_classes = num_labels + 1. The largest value (num_classes - 1) is reserved for the blank label.

	if False:
		slice_width, slice_stride = 14, 7
		min_time_steps = math.ceil((image_width - slice_width) / slice_stride) + 1
		max_time_steps = min_time_steps  # max_time_steps .= min_time_steps.

		# (samples, time-steps, features).
		input_shape = (None, max_time_steps, image_height, slice_width, 1)
		# The shape of network model output = (None, max_time_steps, num_classes)
		# The shape of ground truth = (None, 1, num_classes) if variable-length output is used.
		#                             (None, max_time_steps, num_classes) otherwise.
		output_shape = (None, 1, num_classes) if use_variable_length_output else (None, max_time_steps, num_classes)

		train_images, train_labels, test_images, test_labels = prepare_single_character_dataset(data_dir_path, (image_height, image_width, 1), num_classes, max_time_steps, slice_width, slice_stride, use_variable_length_output)
	else:
		min_digit_count, max_digit_count = 3, 5
		max_time_steps = max_digit_count + 2  # max_time_steps >= max_digit_count.

		# (samples, time-steps, features).
		input_shape = (None, max_time_steps, image_height, image_width, 1)
		# The shape of network model output = (None, max_time_steps, num_classes)
		# The shape of ground truth = (None, max_digit_count, num_classes) if variable-length output is used.
		#                             (None, max_time_steps, num_classes) otherwise.
		output_shape = (None, max_digit_count, num_classes) if use_variable_length_output else (None, max_time_steps, num_classes)

		train_images, train_labels, test_images, test_labels = prepare_multiple_character_dataset(data_dir_path, (image_height, image_width, 1), num_classes, min_digit_count, max_digit_count, max_time_steps)

	# Pre-process.
	#train_images, train_labels = preprocess_data(train_images, train_labels, num_classes)
	#test_images, test_labels = preprocess_data(test_images, test_labels, num_classes)

	# Visualize dataset.
	#visualize_dataset(train_images, train_labels, 5)
	#visualize_dataset(test_images, test_labels, 5)

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
			cnnModelForTraining = create_crnn(input_shape, output_shape, is_time_major, use_variable_length_output, label_eos_token)
			cnnModelForTraining.create_training_model()

			# Create a trainer.
			initial_epoch = 0
			nnTrainer = SimpleCrnnTrainer(cnnModelForTraining, initial_epoch)

			# Create a saver.
			#	Save a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			# Create a model.
			cnnModelForEvaluation = create_crnn(input_shape, output_shape, is_time_major, use_variable_length_output, label_eos_token)
			cnnModelForEvaluation.create_evaluation_model()

			# Create an evaluator.
			nnEvaluator = NeuralNetEvaluator(cnnModelForEvaluation)

			# Create a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		# Create a model.
		cnnModelForInference = create_crnn(input_shape, output_shape, is_time_major, use_variable_length_output, label_eos_token)
		cnnModelForInference.create_inference_model()

		# Create an inferrer.
		nnInferrer = NeuralNetInferrer(cnnModelForInference)

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
		eval_session = tf.Session(graph=eval_graph, config=config)
	infer_session = tf.Session(graph=infer_graph, config=config)

	# Initialize.
	if does_need_training:
		train_session.run(initializer)

	#%%------------------------------------------------------------------
	# Train and evaluate.

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 500  # Number of times to iterate over training data.
	shuffle = True

	if does_need_training:
		total_elapsed_time = time.time()
		with train_session.as_default() as sess:
			with sess.graph.as_default():
				train_neural_net(sess, nnTrainer, train_images, train_labels, test_images, test_labels, batch_size, num_epochs, shuffle, does_resume_training, train_saver, output_dir_path, checkpoint_dir_path, train_summary_dir_path, val_summary_dir_path)
		print('\tTotal training time = {}'.format(time.time() - total_elapsed_time))

		total_elapsed_time = time.time()
		with eval_session.as_default() as sess:
			with sess.graph.as_default():
				evaluate_neural_net(sess, nnEvaluator, test_images, test_labels, batch_size, eval_saver, checkpoint_dir_path)
		print('\tTotal evaluation time = {}'.format(time.time() - total_elapsed_time))

	#%%------------------------------------------------------------------
	# Infer.

	total_elapsed_time = time.time()
	with infer_session.as_default() as sess:
		with sess.graph.as_default():
			infer_by_neural_net(sess, nnInferrer, test_images, test_labels, num_classes, batch_size, infer_saver, checkpoint_dir_path)
	print('\tTotal inference time = {}'.format(time.time() - total_elapsed_time))

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
