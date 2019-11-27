#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, time, datetime, functools, itertools, glob, csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, PReLU
from tensorflow.keras.layers import Conv2D, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
import cv2
import swl.machine_learning.util as swl_ml_util
import text_line_data

#--------------------------------------------------------------------

class MyModel(object):
	def __init__(self, hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel):
		self._input_shape = (lr_image_height, lr_image_width, image_channel)

		self._input_ph = tf.placeholder(tf.float32, shape=(None, lr_image_height, lr_image_width, image_channel), name='input_ph')
		self._output_ph = tf.placeholder(tf.float32, shape=(None, hr_image_height, hr_image_width, image_channel), name='output_ph')

	def get_feed_dict(self, data, num_data, *args, **kwargs):	
		len_data = len(data)
		if 1 == len_data:
			feed_dict = {self._input_ph: data[0]}
		elif 2 == len_data:
			feed_dict = {self._input_ph: data[0], self._output_ph: data[1]}
		else:
			raise ValueError('Invalid number of feed data: {}'.format(len_data))
		return feed_dict

	def create_model(self, is_training=False):
		model_output = self._create_model(self._input_ph)

		if is_training:
			loss = self._get_loss(model_output, self._output_ph)
			accuracy = self._get_accuracy(model_output, self._output_ph)
			return model_output, loss, accuracy
		else:
			return model_output

	def _create_model(self, inputs):
		# TODO [decide] >>
		#kernel_initializer = None
		#kernel_initializer = tf.initializers.he_normal()
		#kernel_initializer = tf.initializers.he_uniform()
		#kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
		#kernel_initializer = tf.initializers.uniform_unit_scaling(factor=1.0)
		#kernel_initializer = tf.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
		#kernel_initializer = tf.initializers.glorot_normal()  # Xavier normal initialization.
		#kernel_initializer = tf.initializers.glorot_uniform()  # Xavier uniform initialization.
		kernel_initializer = tf.initializers.orthogonal()

		#--------------------
		# Preprocessing.
		#with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
		#	inputs = tf.nn.local_response_normalization(inputs, depth_radius=5, bias=1, alpha=1, beta=0.5, name='lrn')

		#--------------------
		with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
			cnn_output = MyModel._create_sr_resnet(inputs, kernel_initializer, input_shape=self._input_shape)

		with tf.variable_scope('transcription', reuse=tf.AUTO_REUSE):
			logits = cnn_output

			return logits

	def _get_loss(self, y, t):
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.keras.losses.MSE(y_true=t, y_pred=y))
			tf.summary.scalar('loss', loss)
			return loss

	def _get_accuracy(self, y, t):
		with tf.name_scope('accuracy'):
			accuracy = -tf.reduce_mean(tf.keras.losses.MSE(y_true=t, y_pred=y))
			tf.summary.scalar('accuracy', accuracy)
			return accuracy

	@staticmethod
	def _normalize_01(x):
		"""Normalizes RGB images to [0, 1]."""
		return x / 255.0

	@staticmethod
	def _denormalize_m11(x):
		"""Inverse of normalize_m11."""
		return (x + 1) * 127.5

	@staticmethod
	def _pixel_shuffle(scale):
		return lambda x: tf.nn.depth_to_space(x, scale)

	@staticmethod
	def _upsample(x_in, num_filters):
		x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
		x = Lambda(MyModel._pixel_shuffle(scale=2))(x)
		return PReLU(shared_axes=[1, 2])(x)

	@staticmethod
	def _res_block(x_in, num_filters, momentum=0.8):
		x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
		x = BatchNormalization(momentum=momentum)(x)
		x = PReLU(shared_axes=[1, 2])(x)
		x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
		x = BatchNormalization(momentum=momentum)(x)
		x = Add()([x_in, x])
		return x

	# REF [paper] >> "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network", CVPR 2017.
	# REF [site] >> https://github.com/krasserm/super-resolution
	@staticmethod
	def _create_sr_resnet(inputs, kernel_initializer=None, input_shape=None, num_filters=64, num_res_blocks=16):
		#x_in = Input(shape=(None, None, 3))
		#x = Lambda(MyModel._normalize_01)(x_in)
		x = inputs

		x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
		x = x_1 = PReLU(shared_axes=[1, 2])(x)

		for _ in range(num_res_blocks):
			x = MyModel._res_block(x, num_filters)

		x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Add()([x_1, x])

		x = MyModel._upsample(x, num_filters * 4)
		x = MyModel._upsample(x, num_filters * 4)

		#x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
		#x = Lambda(MyModel._denormalize_m11)(x)
		x = Conv2D(1, kernel_size=9, padding='same', activation='sigmoid')(x)

		#return Model(x_in, x)
		return x

#--------------------------------------------------------------------

def create_corrupter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	corrupter = iaa.Sequential([
		#iaa.Sometimes(0.5, iaa.OneOf([
		#	#iaa.Affine(
		#	#	scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
		#	#	translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent (per axis).
		#	#	rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
		#	#	shear=(-5, 5),  # Shear by -5 to +5 degrees.
		#	#	#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
		#	#	order=0,  # Use nearest neighbour or bilinear interpolation (fast).
		#	#	#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
		#	#	#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
		#	#	#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
		#	#),
		#	#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
		#	#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
		#	iaa.ElasticTransformation(alpha=(10.0, 30.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
		#])),
		iaa.OneOf([
			iaa.OneOf([
				iaa.GaussianBlur(sigma=(0.5, 1.5)),
				iaa.AverageBlur(k=(2, 4)),
				iaa.MedianBlur(k=(3, 3)),
				iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0), order=1),
			]),
			iaa.Sequential([
				iaa.OneOf([
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
					iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
				]),
				iaa.GaussianBlur(sigma=(0.7, 1.0)),
			]),
			#iaa.OneOf([
			#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
			#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
			#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

			#	iaa.Invert(p=1, per_channel=False),

			#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
			#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
			#]),
		]),
	])

	return corrupter

class MyRunner(object):
	def __init__(self):
		# Set parameters.
		sr_ratio = 4
		lr_image_height, lr_image_width, image_channel = 16, 160, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
		hr_image_height, hr_image_width = lr_image_height * sr_ratio, lr_image_width * sr_ratio

		# TODO [modify] >> Depends on a model.
		#	model_output_time_steps = image_width / width_downsample_factor or image_width / width_downsample_factor - 1.
		#	REF [function] >> MyModel.create_model().
		#width_downsample_factor = 4
		model_output_time_steps = 80 #160
		max_label_len = model_output_time_steps  # max_label_len <= model_output_time_steps.

		self._corrupter = create_corrupter()

		#--------------------
		# Create a dataset.
		#word_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
		word_dictionary_filepath = '../../data/language_processing/wordlist_mono_clean.txt'
		#word_dictionary_filepath = '../../data/language_processing/wordlist_bi_clean.txt'

		print('[SWL] Info: Start loading an English dictionary...')
		start_time = time.time()
		with open(word_dictionary_filepath, 'r', encoding='UTF-8') as fd:
			#dictionary_words = fd.readlines()
			#dictionary_words = fd.read().strip('\n')
			dictionary_words = fd.read().splitlines()
		print('[SWL] Info: End loading an English dictionary: {} secs.'.format(time.time() - start_time))

		if False:
			from swl.language_processing.util import draw_character_histogram
			draw_character_histogram(dictionary_words, charset=None)

		#--------------------
		if 'posix' == os.name:
			system_font_dir_path = '/usr/share/fonts'
			font_base_dir_path = '/home/sangwook/work/font'
		else:
			system_font_dir_path = 'C:/Windows/Fonts'
			font_base_dir_path = 'D:/work/font'
		#font_dir_path = font_base_dir_path + '/eng'
		font_dir_path = font_base_dir_path + '/receipt_eng'

		import text_generation_util as tg_util
		font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
		font_list = tg_util.generate_font_list(font_filepaths)
		#handwriting_dict = tg_util.generate_phd08_dict(from_npy=True)
		handwriting_dict = None

		print('[SWL] Info: Start creating an English dataset...')
		start_time = time.time()
		self._dataset = text_line_data.RunTimeSuperResolvedTextLinePairDataset(set(dictionary_words), hr_image_height, hr_image_width, lr_image_height, lr_image_width, image_channel, font_list, handwriting_dict, max_label_len=max_label_len, use_NWHC=False, corrupt_functor=self._corrupt)
		print('[SWL] Info: End creating an English dataset: {} secs.'.format(time.time() - start_time))

		#self._train_examples_per_epoch, self._test_examples_per_epoch = 500000, 10000
		#self._train_examples_per_epoch, self._test_examples_per_epoch = 200000, 10000
		self._train_examples_per_epoch, self._test_examples_per_epoch = 100000, 10000

	@property
	def dataset(self):
		return self._dataset

	def train(self, checkpoint_dir_path, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape)
			model_output, loss, accuracy = model.create_model(is_training=True)

			# Create a trainer.
			learning_rate = 1.0e-4
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
			#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.9, epsilon=1e-10)
			train_op = optimizer.minimize(loss)

			# Create a saver.
			saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

			initializer = tf.global_variables_initializer()

		with tf.Session(graph=graph).as_default() as sess:
			sess.run(initializer)

			# Restore a model.
			if is_training_resumed:
				print('[SWL] Info: Start restoring a model...')
				start_time = time.time()
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
				ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
				#ckpt_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
				if ckpt_filepath:
					initial_epoch = int(ckpt_filepath.split('-')[1])
					saver.restore(sess, ckpt_filepath)
				else:
					print('[SWL] Error: Failed to restore a model from {}.'.format(checkpoint_dir_path))
					return
				print('[SWL] Info: End restoring a model from {}: {} secs.'.format(ckpt_filepath, time.time() - start_time))

			history = {
				'acc': list(),
				'loss': list(),
				'val_acc': list(),
				'val_loss': list()
			}

			#--------------------
			if is_training_resumed:
				print('[SWL] Info: Resume training...')
			else:
				print('[SWL] Info: Start training...')
			start_total_time = time.time()
			train_steps_per_epoch = None if self._train_examples_per_epoch is None else math.ceil(self._train_examples_per_epoch / batch_size)
			test_steps_per_epoch = None if self._test_examples_per_epoch is None else math.ceil(self._test_examples_per_epoch / batch_size)
			final_epoch = num_epochs + initial_epoch
			for epoch in range(initial_epoch + 1, final_epoch + 1):
				print('Epoch {}/{}:'.format(epoch, final_epoch))

				start_time = time.time()
				train_loss, train_acc, num_examples = 0.0, 0.0, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, train_steps_per_epoch, shuffle=True)):
					#batch_corrupted_images, batch_clean_images, batch_labels_str, batch_labels_int = batch_data
					_, batch_loss, batch_acc = sess.run(
						[train_op, loss, accuracy],
						feed_dict=model.get_feed_dict((batch_data[0], batch_data[1]), num_batch_examples)
					)

					train_loss += batch_loss * num_batch_examples
					train_acc += batch_acc * num_batch_examples
					num_examples += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				train_acc /= num_examples
				print('\tTrain:      Loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

				history['loss'].append(train_loss)
				history['acc'].append(train_acc)
				"""
				train_loss, train_acc, num_examples = 0.0, None, 0
				for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_train_batch_generator(batch_size, train_steps_per_epoch, shuffle=True)):
					#batch_corrupted_images, batch_clean_images, batch_labels_str, batch_labels_int = batch_data
					_, batch_loss = sess.run(
						[train_op, loss],
						feed_dict=model.get_feed_dict((batch_data[0], batch_data[1]), num_batch_examples)
					)

					train_loss += batch_loss * num_batch_examples
					num_examples += num_batch_examples

					if (batch_step + 1) % 100 == 0:
						print('\tStep {}: {} secs.'.format(batch_step + 1, time.time() - start_time))
				train_loss /= num_examples
				print('\tTrain:      loss = {:.6f}, accuracy = {}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

				history['loss'].append(train_loss)
				#history['acc'].append(train_acc)
				"""

				#--------------------
				#if epoch % 10 == 0:
				if True:
					start_time = time.time()
					val_loss, val_acc, num_examples = 0.0, 0.0, 0
					for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_test_batch_generator(batch_size, test_steps_per_epoch, shuffle=False)):
						#batch_corrupted_images, batch_clean_images, batch_labels_str, batch_labels_int = batch_data
						batch_loss, batch_acc = sess.run(
							[loss, accuracy],
							feed_dict=model.get_feed_dict((batch_data[0], batch_data[1]), num_batch_examples)
						)

						val_loss += batch_loss * num_batch_examples
						val_acc += batch_acc * num_batch_examples
						num_examples += num_batch_examples
					val_loss /= num_examples
					val_acc /= num_examples
					print('\tValidation: Loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

					history['val_loss'].append(val_loss)
					history['val_acc'].append(val_acc)
				else:
					start_time = time.time()
					val_loss, val_acc, num_examples = 0.0, None, 0
					for batch_step, (batch_data, num_batch_examples) in enumerate(self._dataset.create_test_batch_generator(batch_size, test_steps_per_epoch, shuffle=False)):
						#batch_corrupted_images, batch_clean_images, batch_labels_str, batch_labels_int = batch_data
						batch_loss = sess.run(
							loss,
							feed_dict=model.get_feed_dict((batch_data[0], batch_data[1]), num_batch_examples)
						)

						val_loss += batch_loss * num_batch_examples
						num_examples += num_batch_examples
					val_loss /= num_examples
					print('\tValidation: Loss = {:.6f}, accuracy = {}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

					history['val_loss'].append(val_loss)
					#history['val_acc'].append(val_acc)

				#--------------------
				print('[SWL] Info: Start saving a model...')
				start_time = time.time()
				saved_model_path = saver.save(sess, os.path.join(checkpoint_dir_path, 'model.ckpt'), global_step=epoch - 1)
				print('[SWL] Info: End saving a model to {}: {} secs.'.format(saved_model_path, time.time() - start_time))

				sys.stdout.flush()
				time.sleep(0)
			print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))

			return history

	def test(self, checkpoint_dir_path, test_dir_path, batch_size):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape)
			model_output = model.create_model(is_training=False)

			# Create a saver.
			saver = tf.train.Saver()

		with tf.Session(graph=graph).as_default() as sess:
			# Load a model.
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
			#ckpt_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
			if ckpt_filepath:
				saver.restore(sess, ckpt_filepath)
			else:
				print('[SWL] Error: Failed to load a model from {}.'.format(checkpoint_dir_path))
				return
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(ckpt_filepath, time.time() - start_time))

			#--------------------
			print('[SWL] Info: Start testing...')
			start_time = time.time()
			test_steps_per_epoch = None if self._test_examples_per_epoch is None else math.ceil(self._test_examples_per_epoch / batch_size)
			inferences, inputs, ground_truths = list(), list(), list()
			for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, test_steps_per_epoch, shuffle=False):
				#batch_corrupted_images, batch_clean_images, batch_labels_str, batch_labels_int = batch_data
				batch_clean_images = sess.run(
					model_output,
					feed_dict=model.get_feed_dict((batch_data[0],), num_batch_examples)
				)
				inferences.extend(batch_clean_images)
				inputs.extend(batch_data[0])
				ground_truths.extend(batch_data[1])
			print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

			if inferences and inputs and ground_truths:
				inferences, inputs, ground_truths = np.array(inferences), np.array(inputs), np.array(ground_truths)

				print('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				# Output to image files.
				for idx, (inf, inp, gt) in enumerate(zip(inferences, inputs, ground_truths)):
					inf_filepath, inp_filepath, gt_filepath = os.path.join(test_dir_path, 'inf_{:06}.png'.format(idx)), os.path.join(test_dir_path, 'input_{:06}.png'.format(idx)), os.path.join(test_dir_path, 'gt_{:06}.png'.format(idx))
					cv2.imwrite(inf_filepath, np.round(inf * 255).astype(np.uint8))
					cv2.imwrite(inp_filepath, np.round(inp * 255).astype(np.uint8))
					cv2.imwrite(gt_filepath, np.round(gt * 255).astype(np.uint8))
			else:
				print('[SWL] Warning: Invalid test results.')

	def infer(self, checkpoint_dir_path, image_filepaths, inference_dir_path, batch_size=None):
		graph = tf.Graph()
		with graph.as_default():
			# Create a model.
			model = MyModel(*self._dataset.shape)
			model_output = model.create_model(is_training=False)

			# Create a saver.
			saver = tf.train.Saver()

		with tf.Session(graph=graph).as_default() as sess:
			# Load a model.
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
			ckpt_filepath = ckpt.model_checkpoint_path if ckpt else None
			#ckpt_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
			if ckpt_filepath:
				saver.restore(sess, ckpt_filepath)
			else:
				print('[SWL] Error: Failed to load a model from {}.'.format(checkpoint_dir_path))
				return
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(ckpt_filepath, time.time() - start_time))

			#--------------------
			print('[SWL] Info: Start loading images...')
			inf_images, image_filepaths = self._dataset.load_images_from_files(image_filepaths, is_grayscale=True)
			print('[SWL] Info: End loading images: {} secs.'.format(time.time() - start_time))
			print('[SWL] Info: Loaded images: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inf_images.shape, inf_images.dtype, np.min(inf_images), np.max(inf_images)))

			num_examples = len(inf_images)
			if batch_size is None:
				batch_size = num_examples
			if batch_size <= 0:
				raise ValueError('Invalid batch size: {}'.format(batch_size))

			indices = np.arange(num_examples)

			#--------------------
			print('[SWL] Info: Start inferring...')
			start_time = time.time()
			inferences, inputs = list(), list()
			start_idx = 0
			while True:
				end_idx = start_idx + batch_size
				batch_indices = indices[start_idx:end_idx]
				if batch_indices.size > 0:  # If batch_indices is non-empty.
					# FIXME [fix] >> Does not work correctly in time-major data.
					batch_images = inf_images[batch_indices]
					if batch_images.size > 0:  # If batch_images is non-empty.
						batch_clean_images = sess.run(
							model_output,
							feed_dict=model.get_feed_dict((batch_images,), len(batch_images))
						)
						inferences.extend(batch_clean_images)
						inputs.extend(batch_images)

				if end_idx >= num_examples:
					break
				start_idx = end_idx
			print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

			if inferences and inputs:
				inferences, inputs = np.array(inferences), np.array(inputs)

				print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

				# Output to image files.
				for idx, (inf, inp) in enumerate(zip(inferences, inputs)):
					inf_filepath, inp_filepath = os.path.join(inference_dir_path, 'inf_{:06}.png'.format(idx)), os.path.join(inference_dir_path, 'input_{:06}.png'.format(idx))
					cv2.imwrite(inf_filepath, np.round(inf * 255).astype(np.uint8))
					cv2.imwrite(inp_filepath, np.round(inp * 255).astype(np.uint8))
			else:
				print('[SWL] Warning: Invalid inference results.')

	def _corrupt(self, inputs, *args, **kwargs):
		return self._corrupter.augment_images(inputs)

	def _corrupt2(self, inputs, outputs, *args, **kwargs):
		if outputs is None:
			return self._corrupter.augment_images(inputs), None
		else:
			augmenter_det = self._corrupter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

#--------------------------------------------------------------------

def check_data(num_epochs, batch_size):
	runner = MyRunner()
	default_value = -1

	#train_examples_per_epoch, test_examples_per_epoch = 5000, 1000
	#train_examples_per_epoch, test_examples_per_epoch = 2000, 1000
	train_examples_per_epoch, test_examples_per_epoch = 1000, 1000
	train_steps_per_epoch = None if train_examples_per_epoch is None else math.ceil(train_examples_per_epoch / batch_size)
	test_steps_per_epoch = None if test_examples_per_epoch is None else math.ceil(test_examples_per_epoch / batch_size)

	generator = runner.dataset.create_train_batch_generator(batch_size, train_steps_per_epoch, shuffle=False)
	for batch_step, (batch_data, num_batch_examples) in enumerate(generator):
		#batch_corrupted_images (np.array), batch_clean_images (np.array), batch_labels_str (a list of strings), batch_labels_int (a list of sequences) = batch_data

		if 0 == batch_step:
			print('type(batch_data) = {}, len(batch_data) = {}.'.format(type(batch_data), len(batch_data)))
			print('type(batch_data[0]) = {}.'.format(type(batch_data[0])))
			print('\tShape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_data[0].shape, batch_data[0].dtype, np.min(batch_data[0]), np.max(batch_data[0])))
			print('type(batch_data[1]) = {}.'.format(type(batch_data[1])))
			print('\tShape = {}, dtype = {}, (min, max) = ({}, {}).'.format(batch_data[1].shape, batch_data[1].dtype, np.min(batch_data[1]), np.max(batch_data[1])))
			print('type(batch_data[2]) = {}, len(batch_data[2]) = {}.'.format(type(batch_data[2]), len(batch_data[2])))
			print('type(batch_data[3]) = {}, len(batch_data[3]) = {}.'.format(type(batch_data[3]), len(batch_data[3])))

		if batch_size != batch_data[0].shape[0] or batch_size != batch_data[1].shape[0]:
			print('Invalid image size: {0} != {1} or {0} != {2}.'.format(batch_size, batch_data[0].shape[0], batch_data[1].shape[0]))
		if batch_size != len(batch_data[2]) or batch_size != len(batch_data[3]):
			print('Invalid label size: {0} != {1} or {0} != {2}.'.format(batch_size, len(batch_data[2]), len(batch_data[3])))

		for idx, (lbl, lbl_int) in enumerate(zip(batch_data[2], batch_data[3])):
			if len(lbl) != len(lbl_int):
				print('Unmatched label length: {} != {} ({}: {}).'.format(lbl, lbl_int, idx, batch_data[2]))
			if 0 == len(lbl_int):
				print('Zero-length label: {}, {} ({}: {}).'.format(lbl, lbl_int, idx, batch_data[2]))

		sparse = swl_ml_util.sequences_to_sparse(batch_data[3], dtype=np.int32)
		sequences = swl_ml_util.sparse_to_sequences(*sparse, dtype=np.int32)
		#print('Sparse tensor = {}.'.format(sparse))
		dense = swl_ml_util.sequences_to_dense(batch_data[3], default_value=default_value, dtype=np.int32)
		sequences = swl_ml_util.dense_to_sequences(dense, default_value=default_value, dtype=np.int32)
		#print('Dense tensor = {}.'.format(dense))

		break

	#generator = runner.dataset.create_train_batch_generator(batch_size, train_steps_per_epoch, shuffle=False)
	runner.dataset.visualize(generator, num_examples=10)

#--------------------------------------------------------------------

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#--------------------
	num_epochs, batch_size = 50, 128  # batch_size affects training.
	initial_epoch = 0
	is_trained, is_tested, is_inferred = True, True, True
	is_training_resumed = False

	#--------------------
	if False:
		print('[SWL] Info: Start checking data...')
		start_time = time.time()
		check_data(num_epochs, batch_size)
		print('[SWL] Info: End checking data: {} secs.'.format(time.time() - start_time))
		return

	#--------------------
	output_dir_path = None
	if not output_dir_path:
		output_dir_prefix = 'simple_english_srresnet'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	checkpoint_dir_path = None
	if not checkpoint_dir_path:
		checkpoint_dir_path = os.path.join(output_dir_path, 'tf_checkpoint')
	test_dir_path = None
	if not test_dir_path:
		test_dir_path = os.path.join(output_dir_path, 'test')
	inference_dir_path = None
	if not inference_dir_path:
		inference_dir_path = os.path.join(output_dir_path, 'inference')

	#--------------------
	runner = MyRunner()

	if is_trained:
		if checkpoint_dir_path and checkpoint_dir_path.strip() and not os.path.exists(checkpoint_dir_path):
			os.makedirs(checkpoint_dir_path, exist_ok=True)

		history = runner.train(checkpoint_dir_path, num_epochs, batch_size, initial_epoch, is_training_resumed)

		#print('History =', history)
		#swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if is_tested:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return
		if test_dir_path and test_dir_path.strip() and not os.path.exists(test_dir_path):
			os.makedirs(test_dir_path, exist_ok=True)

		runner.test(checkpoint_dir_path, test_dir_path, batch_size)

	if is_inferred:
		if not checkpoint_dir_path or not os.path.exists(checkpoint_dir_path):
			print('[SWL] Error: Model directory, {} does not exist.'.format(checkpoint_dir_path))
			return
		if inference_dir_path and inference_dir_path.strip() and not os.path.exists(inference_dir_path):
			os.makedirs(inference_dir_path, exist_ok=True)

		image_filepaths = glob.glob('./icdar2019_sroie/task1_test_text_line/image/*.jpg', recursive=False)
		image_filepaths.sort()
		runner.infer(checkpoint_dir_path, image_filepaths, inference_dir_path, batch_size)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
