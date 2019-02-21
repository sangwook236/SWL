#!/usr/bin/env python

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import sys
sys.path.append('../../../src')

#--------------------
import os, time, datetime
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
from synth90k_crnn import Synth90kCrnnWithCrossEntropyLoss, Synth90kCrnnWithCtcLoss

#%%------------------------------------------------------------------

def create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output):
	if is_sparse_output:
		return Synth90kCrnnWithCtcLoss(image_height, image_width, image_channel, num_classes, label_eos_token)
	else:
		return Synth90kCrnnWithCrossEntropyLoss(image_height, image_width, image_channel, num_classes)

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
		#labels = to_one_hot_encoding(labels, num_classes).astype(np.uint8)
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

	is_sparse_output = False
	#is_time_major = False  # Fixed.

	image_height, image_width, image_channel = 64, 128, 1
	num_labels = 2350  # KS X 1001.
	if False:
		# num_labels + space label + blank label.
		num_classes = num_labels + 1 + 1
		space_label = num_classes - 2
	else:
		# num_labels + blank label.
		num_classes = num_labels + 1
	# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label.
	blank_label = num_classes - 1
	label_eos_token = -1

	batch_size = 128  # Number of samples per gradient update.
	num_epochs = 20  # Number of times to iterate over training data.
	shuffle = True

	augmenter = ImgaugAugmenter()
	#augmenter = create_imgaug_augmenter()  # If imgaug augmenter is used, data are augmented in background augmentation processes. (faster)
	is_output_augmented = False

	sess_config = tf.ConfigProto()
	#sess_config.device_count = {'GPU': 2}
	#sess_config.allow_soft_placement = True
	sess_config.log_device_placement = True
	sess_config.gpu_options.allow_growth = True
	#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Only allocate 40% of the total memory of each GPU.

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

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'D:/dataset'
	data_dir_path = data_home_dir_path + '/pattern_recognition/language_processing/mjsynth/mnt/ramdisk/max/90kDICT32px'

	# filepath (filename: index_text_lexicon-idx) lexicon-idx.
	data_filepath_list = data_dir_path + '/annotation.txt'  # 8,919,273 files.
	train_data_filepath_list = data_dir_path + '/annotation_train.txt'  # 7,224,612 files.
	val_data_filepath_list = data_dir_path + '/annotation_val.txt'  # 802,734 files.
	test_data_filepath_list = data_dir_path + '/annotation_test.txt'  # 891,927 files.
	lexicon_filepath_list = data_dir_path + '/lexicon.txt'  # 88,172 words.

	#train_images, train_labels, test_images, test_labels = load_data(input_shape[1:])
	train_images, train_labels, test_images, test_labels = None, None, None, None

	#--------------------
	# Creates models, sessions, and graphs.

	# Creates graphs.
	if does_need_training:
		train_graph = tf.Graph()
		eval_graph = tf.Graph()
	infer_graph = tf.Graph()

	if does_need_training:
		with train_graph.as_default():
			#K.set_learning_phase(1)  # Sets the learning phase to 'train'. (Required)

			# Creates a model.
			modelForTraining = create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output)
			modelForTraining.create_training_model()

			# Creates a trainer.
			nnTrainer = SimpleNeuralNetTrainer(modelForTraining, initial_epoch, augmenter, is_output_augmented)

			# Creates a saver.
			#	Saves a model every 2 hours and maximum 5 latest models are saved.
			train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with eval_graph.as_default():
			#K.set_learning_phase(0)  # Sets the learning phase to 'test'. (Required)

			# Creates a model.
			modelForEvaluation = create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output)
			modelForEvaluation.create_evaluation_model()

			# Creates an evaluator.
			nnEvaluator = NeuralNetEvaluator(modelForEvaluation)

			# Creates a saver.
			eval_saver = tf.train.Saver()

	with infer_graph.as_default():
		#K.set_learning_phase(0)  # Sets the learning phase to 'test'. (Required)

		# Creates a model.
		modelForInference = create_synth90k_crnn(image_height, image_width, image_channel, num_classes, label_eos_token, is_sparse_output)
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
	main()
