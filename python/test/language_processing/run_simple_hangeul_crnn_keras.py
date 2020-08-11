#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, time, datetime, functools, glob, csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LSTM, MaxPooling2D, Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import Input, Dense, Activation, add, concatenate
import swl.language_processing.util as swl_langproc_util
import swl.machine_learning.util as swl_ml_util
import text_line_data
import TextRecognitionDataGenerator_data

#--------------------------------------------------------------------

def create_augmenter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	"""
	augmenter = iaa.Sequential([
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
			iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
			#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
			#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.Affine(
				scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent (per axis).
				rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
				shear=(-5, 5),  # Shear by -5 to +5 degrees.
				#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(15.0, 30.0), sigma=5.0),  # Move pixels locally around (with random strengths).
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.OneOf([
				iaa.GaussianBlur(sigma=(1.5, 2.5)),
				iaa.AverageBlur(k=(3, 6)),
				iaa.MedianBlur(k=(3, 5)),
				iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0), order=1),
			]),
			iaa.OneOf([
				iaa.AdditiveGaussianNoise(loc=0, scale=(0.1 * 255, 0.3 * 255), per_channel=False),
				#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.1 * 255, 0.3 * 255), per_channel=False),
				#iaa.AdditivePoissonNoise(lam=(32, 64), per_channel=False),
				iaa.CoarseSaltAndPepper(p=(0.1, 0.3), size_percent=(0.2, 0.9), per_channel=False),
				iaa.CoarseSalt(p=(0.1, 0.3), size_percent=(0.2, 0.9), per_channel=False),
				#iaa.CoarsePepper(p=(0.1, 0.3), size_percent=(0.2, 0.9), per_channel=False),
				iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.05, 0.3), per_channel=False),
			]),
			#iaa.OneOf([
			#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
			#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
			#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

			#	iaa.Invert(p=1, per_channel=False),

			#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
			#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
			#]),
		])),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])
	"""
	augmenter = iaa.Sequential([
		#iaa.Sometimes(0.5, iaa.OneOf([
		#	iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
		#	iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
		#	#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
		#	#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
		#])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent along x-axis and -10 to +10 percent along y-axis.
				rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				shear=(-2, 2),  # Shear by -2 to +2 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
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
		])),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

	return augmenter

def generate_font_colors(image_depth):
	import random
	#font_color = (255,) * image_depth  # White font color.
	font_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB font color.
	#font_color = (random.randrange(256),) * image_depth  # A grayscale font color.
	#gray_val = random.randrange(255)
	#font_color = (gray_val,) * image_depth  # A lighter grayscale font color.
	#font_color = (random.randrange(gray_val, 256),) * image_depth  # A darker grayscale font color.
	#font_color = (random.randrange(128, 256),) * image_depth  # A light grayscale font color.
	#font_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale font color.
	#bg_color = (0,) * image_depth  # Black background color.
	bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # An RGB background color.
	#bg_color = (random.randrange(256),) * image_depth  # A grayscale background color.
	#bg_color = (random.randrange(gray_val, 256),) * image_depth  # A lighter grayscale background color.
	#bg_color = (gray_val,) * image_depth  # A darker grayscale background color.
	#bg_color = (random.randrange(0, 128),) * image_depth  # A dark grayscale background color.
	#bg_color = (random.randrange(128, 256),) * image_depth  # A light grayscale background color.
	return font_color, bg_color

class MyRunTimeAlphaMatteTextLineDataset(text_line_data.RunTimeAlphaMatteTextLineDataset):
	def __init__(self, label_converter, text_set, image_height, image_width, image_channel, font_list, char_images_dict, alpha_matte_mode='1', use_NWHC=True):
		super().__init__(label_converter, text_set, image_height, image_width, image_channel, font_list, char_images_dict, functools.partial(generate_font_colors, image_depth=image_channel), alpha_matte_mode, use_NWHC)

		self._train_data, self._test_data = None, None
		self._augmenter = create_augmenter()

	@property
	def train_examples(self):
		return self._train_data

	@property
	def test_examples(self):
		return self._test_data

	@property
	def num_train_examples(self):
		return 0 if self._train_data is None else len(self._train_data)

	@property
	def num_test_examples(self):
		return 0 if self._test_data is None else len(self._test_data)

	def augment(self, inputs, outputs, *args, **kwargs):
		if outputs is None:
			return self._augmenter.augment_images(inputs), None
		else:
			augmenter_det = self._augmenter.to_deterministic()  # Call this for each batch again, NOT only once at the start.
			return augmenter_det.augment_images(inputs), augmenter_det.augment_images(outputs)

class MyHangeulTextLineDataset(TextRecognitionDataGenerator_data.HangeulTextRecognitionDataGeneratorTextLineDataset):
	def __init__(self, label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle=True, use_NWHC=True):
		super().__init__(label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len, shuffle, use_NWHC)

		self._train_data, self._test_data = None, None

	@property
	def train_examples(self):
		return self._train_data

	@property
	def test_examples(self):
		return self._test_data

	@property
	def num_train_examples(self):
		return 0 if self._train_data is None else len(self._train_data)

	@property
	def num_test_examples(self):
		return 0 if self._test_data is None else len(self._test_data)

	#def augment(self, inputs, outputs, *args, **kwargs):
	#	raise NotImplementedError

#--------------------------------------------------------------------

"""
class MyDataSequence(tf.keras.utils.Sequence):
	def __init__(self, batch_generator, label_converter, steps_per_epoch, max_label_len, model_output_time_steps):
		self._batch_generator = batch_generator
		self._label_converter = label_converter
		self._steps_per_epoch = steps_per_epoch
		self._max_label_len, self._model_output_time_steps = max_label_len, model_output_time_steps

	def __len__(self):
		return self._steps_per_epoch if self._steps_per_epoch is not None else 0

	def __getitem__(self, idx):
		for batch_data, num_batch_examples in self._batch_generator:
			#batch_images, batch_labels_str, batch_labels_int (sparse tensor) = batch_data
			batch_images, batch_labels_str, _ = batch_data
			batch_labels_int = self._encode_labels(batch_labels_str)  # Dense tensor.

			if batch_labels_int.shape[1] < self._max_label_len:
				labels = np.full((num_batch_examples, self._max_label_len), self._label_converter.pad_id)
				labels[:,:batch_labels_int.shape[1]] = batch_labels_int
				batch_labels_int = labels
			elif batch_labels_int.shape[1] > self._max_label_len:
				print('[SWL] Warning: Invalid label length, {} > {}.'.format(batch_labels_int.shape[1], self._max_label_len))

			model_output_length = np.full((num_batch_examples, 1), self._model_output_time_steps)
			label_length = np.array(list(len(lbl) for lbl in batch_labels_int))

			inputs = {
				'inputs': batch_images,
				'outputs': batch_labels_int,
				'model_output_length': model_output_length,
				'output_length': label_length
			}
			outputs = {'ctc_loss': np.zeros([num_batch_examples])}  # Dummy.
			yield (inputs, outputs)
"""

class MyDataSequence(tf.keras.utils.Sequence):
	def __init__(self, examples, label_converter, max_label_len, model_output_time_steps, batch_size=None, shuffle=False):
		self._examples = np.array(examples)
		self._label_converter = label_converter
		self._max_label_len, self._model_output_time_steps = max_label_len, model_output_time_steps
		self._batch_size = batch_size

		self._num_examples = len(self._examples)
		if self._batch_size is None:
			self._batch_size = self._num_examples
		if self._batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(self._batch_size))

		self._indices = np.arange(self._num_examples)
		if shuffle:
			np.random.shuffle(self._indices)

	def __len__(self):
		return math.ceil(self._num_examples / self._batch_size)

	def __getitem__(self, idx):
		start_idx = idx * self._batch_size
		end_idx = start_idx + self._batch_size
		batch_indices = self._indices[start_idx:end_idx]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			batch_data = self._examples[batch_indices]
			num_batch_examples = len(batch_indices)
			if batch_data.size > 0:  # If batch_data is non-empty.
				#batch_images, batch_labels_str, batch_labels_int (sparse tensor) = batch_data
				batch_images, batch_labels_str, _ = zip(*batch_data)
				batch_labels_int = self._encode_labels(batch_labels_str)  # Dense tensor.

				if batch_labels_int.shape[1] < self._max_label_len:
					labels = np.full((num_batch_examples, self._max_label_len), self._label_converter.pad_id)
					labels[:,:batch_labels_int.shape[1]] = batch_labels_int
					batch_labels_int = labels
				elif batch_labels_int.shape[1] > self._max_label_len:
					print('[SWL] Warning: Invalid label length, {} > {}.'.format(batch_labels_int.shape[1], self._max_label_len))

				model_output_length = np.full((num_batch_examples, 1), self._model_output_time_steps)
				label_length = np.array(list(len(lbl) for lbl in batch_labels_int))

				inputs = {
					'inputs': batch_images,
					'outputs': batch_labels_int,
					'model_output_length': model_output_length,
					'output_length': label_length
				}
				outputs = {'ctc_loss': np.zeros((num_batch_examples,))}  # Dummy.
				return (inputs, outputs)
		return (None, None)

	# String sequences -> Integer sequences.
	def _encode_labels(self, labels_str, dtype=np.int16):
		max_label_len = functools.reduce(lambda x, y: max(x, len(y)), labels_str, 0)
		labels_int = np.full((len(labels_str), max_label_len), self._label_converter.pad_id, dtype=dtype)
		for (idx, lbl) in enumerate(labels_str):
			"""
			try:
				labels_int[idx,:len(lbl)] = np.array(list(self._labels.index(ch) for ch in lbl))
			except ValueError:
				pass
			"""
			labels_int[idx,:len(lbl)] = self._label_converter.encode_label(lbl)
		return labels_int

	# Integer sequences -> string sequences.
	def _decode_labels(self, labels_int):
		"""
		def int2str(label):
			try:
				label = list(self._labels[id] for id in label if id != self._label_converter.pad_id)
				return ''.join(label)
			except ValueError:
				return None
		return list(map(int2str, labels_int))
		"""
		return list(map(self._label_converter.decode_label, labels_int))

#--------------------------------------------------------------------

class MyModel(object):
	@classmethod
	def create_model(cls, input_shape, num_classes, max_label_len, is_training=False):
		kernel_initializer = 'he_normal'

		inputs = Input(shape=input_shape, dtype='float32', name='inputs')

		# (None, width, height, 1).

		#--------------------
		# CNN.
		x = Conv2D(64, (3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv1')(inputs)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(2, 2), name='max1')(x)

		# (None, width/2, height/2, 64).

		x = Conv2D(128, (3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv2')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(2, 2), name='max2')(x)

		# (None, width/4, height/4, 128).

		x = Conv2D(256, (3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv3_1')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv2D(256, (3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv3_2')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(1, 2), name='max3')(x)

		# (None, width/4, height/8, 256).

		x = Conv2D(512, (3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv4_1')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		# TODO [decide] >>
		x = Conv2D(512, (3, 3), padding='same', kernel_initializer=None, name='conv4_2')(x)
		#x = Conv2D(512, (3, 3), padding='same', kernel_initializer=kernel_initializer, name='conv4_2')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(1, 2), name='max4')(x)

		# (None, width/4, height/16, 512).

		x = Conv2D(512, (2, 2), padding='same', kernel_initializer=kernel_initializer, name='conv5')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		# (None, width/4, height/16, 512).

		#--------------------
		rnn_input_shape = x.shape
		x = Reshape(target_shape=((rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3])), name='reshape')(x)
		# TODO [decide] >>
		x = Dense(64, activation='relu', kernel_initializer=kernel_initializer, name='dense6')(x)

		#--------------------
		# RNN.
		lstm_fw_1 = LSTM(256, return_sequences=True, kernel_initializer=kernel_initializer, name='lstm_fw_1')(x)
		lstm_bw_1 = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer=kernel_initializer, name='lstm_bw_1')(x)
		x = concatenate([lstm_fw_1, lstm_bw_1])  # add -> concatenate.
		x = BatchNormalization()(x)
		lstm_fw_2 = LSTM(256, return_sequences=True, kernel_initializer=kernel_initializer, name='lstm_fw_2')(x)
		lstm_bw_2 = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer=kernel_initializer, name='lstm_bw_2')(x)
		x = concatenate([lstm_fw_2, lstm_bw_2])
		# TODO [decide] >>
		#x = BatchNormalization()(x)

		#--------------------
		# Transcription.
		model_outputs = Dense(num_classes, activation='softmax', kernel_initializer=kernel_initializer, name='dense9')(x)

		#--------------------
		labels = Input(shape=(max_label_len,), dtype='float32', name='outputs')  # (None, max_label_len).
		model_output_length = Input(shape=(1,), dtype='int64', name='model_output_length')  # (None, 1).
		label_length = Input(shape=(1,), dtype='int64', name='output_length')  # (None, 1).

		# Currently Keras does not support loss functions with extra parameters so CTC loss is implemented in a lambda layer.
		loss = Lambda(MyModel.compute_ctc_loss, output_shape=(1,), name='ctc_loss')([model_outputs, labels, model_output_length, label_length])  # (None, 1).

		if is_training:
			return tf.keras.models.Model(inputs=[inputs, labels, model_output_length, label_length], outputs=loss)
		else:
			return tf.keras.models.Model(inputs=[inputs], outputs=model_outputs)

	@staticmethod
	def compute_ctc_loss(args):
		model_outputs, labels, model_output_length, label_length = args
		# TODO [check] >> The first couple of RNN outputs tend to be garbage. (???)
		model_outputs = model_outputs[:, 2:, :]
		return tf.keras.backend.ctc_batch_cost(labels, model_outputs, model_output_length, label_length)

	@staticmethod
	def decode_label(labels, blank_label):
		labels = np.argmax(labels, axis=-1)
		return list(map(lambda lbl: list(k for k, g in itertools.groupby(lbl) if k < blank_label), labels))  # Removes repetitive labels.

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, image_height, image_width, image_channel, max_label_len, train_test_ratio=0.8, is_dataset_generated_at_runtime=True):
		# Set parameters.
		# TODO [modify] >> Depends on a model.
		#	model_output_time_steps = image_width / width_downsample_factor or image_width / width_downsample_factor - 1.
		#	REF [function] >> MyModel.create_model().
		width_downsample_factor = 4
		self._model_output_time_steps = image_width // width_downsample_factor
		# TODO [check] >> The first couple of RNN outputs tend to be garbage. (???)
		self._model_output_time_steps -= 2
		self._max_label_len = min(max_label_len, self._model_output_time_steps)

		self._max_queue_size, self._num_workers = 10, 8
		self._use_multiprocessing = True

		#self._sess = tf.Session(config=config)
		#tf.keras.backend.set_session(self._sess)
		#tf.keras.backend.set_learning_phase(0)  # Sets the learning phase to 'test'.
		#tf.keras.backend.set_learning_phase(1)  # Sets the learning phase to 'train'.

		labels = list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝')
		self._label_converter = swl_langproc_util.TokenConverter(labels, pad=None)
		# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
		print('[SWL] Info: Labels = {}.'.format(self._label_converter.tokens))
		print('[SWL] Info: #labels = {}.'.format(self._label_converter.num_tokens))

		#--------------------
		# Create a dataset.

		if is_dataset_generated_at_runtime:
			import text_generation_util as tg_util

			word_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'

			print('[SWL] Info: Start loading a Korean dictionary...')
			start_time = time.time()
			with open(word_dictionary_filepath, 'r', encoding='UTF-8') as fd:
				#dictionary_words = fd.read().strip('\n')
				#dictionary_words = fd.readlines()
				dictionary_words = fd.read().splitlines()
			print('[SWL] Info: End loading a Korean dictionary, {} words loaded: {} secs.'.format(len(dictionary_words), time.time() - start_time))

			print('[SWL] Info: Start generating text lines...')
			texts = tg_util.generate_random_text_lines(dictionary_words, min_word_len=1, max_word_len=5)
			print('[SWL] Info: End generating text lines, {} text lines generated: {} secs.'.format(len(texts), time.time() - start_time))

			if max_label_len > 0:
				texts = set(filter(lambda txt: len(txt) <= max_label_len, texts))

			if False:
				from swl.language_processing.util import draw_character_histogram
				draw_character_histogram(texts, charset=None)

			"""
			labels = functools.reduce(lambda x, txt: x.union(txt), texts, set())
			labels = sorted(labels)
			#labels = ''.join(sorted(labels))

			self._label_converter = swl_langproc_util.TokenConverter(labels, pad=None)
			# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
			print('[SWL] Info: Labels = {}.'.format(self._label_converter.tokens))
			print('[SWL] Info: #labels = {}.'.format(self._label_converter.num_tokens))
			"""

			#--------------------
			if 'posix' == os.name:
				system_font_dir_path = '/usr/share/fonts'
				font_base_dir_path = '/home/sangwook/work/font'
			else:
				system_font_dir_path = 'C:/Windows/Fonts'
				font_base_dir_path = 'D:/work/font'
			font_dir_path = font_base_dir_path + '/kor'
			#font_dir_path = font_base_dir_path + '/kor_small'
			#font_dir_path = font_base_dir_path + '/kor_large'

			font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
			font_list = tg_util.generate_hangeul_font_list(font_filepaths)
			#char_images_dict = tg_util.generate_phd08_dict(from_npy=True)
			char_images_dict = None

			print('[SWL] Info: Start creating a Hangeul dataset...')
			start_time = time.time()
			self._dataset = MyRunTimeAlphaMatteTextLineDataset(self._label_converter, texts, image_height, image_width, image_channel, font_list, char_images_dict, alpha_matte_mode='1')
			print('[SWL] Info: End creating a Hangeul dataset: {} secs.'.format(time.time() - start_time))

			self._train_examples_per_epoch, self._val_examples_per_epoch, self._test_examples_per_epoch = 200000, 10000, 10000 #500000, 10000, 10000  # Uses a subset of texts per epoch.
			#self._train_examples_per_epoch, self._val_examples_per_epoch, self._test_examples_per_epoch = None, None, None  # Uses the whole set of texts per epoch.
		else:
			"""
			hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
			#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
			#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
			with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
				#hangeul_charset = fd.read().strip('\n')  # A string.
				hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
				#hangeul_charset = fd.readlines()  # A list of strings.
				#hangeul_charset = fd.read().splitlines()  # A list of strings.
			#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
			#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
			hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

			import string
			labels = \
				hangeul_charset + \
				hangeul_jamo_charset + \
				string.ascii_uppercase + \
				string.ascii_lowercase + \
				string.digits + \
				string.punctuation + \
				' '
			# There are words of Unicode Hangeul letters besides KS X 1001.
			#labels = functools.reduce(lambda x, fpath: x.union(fpath.split('_')[0]), os.listdir(data_dir_path), set(labels))
			labels = sorted(labels)
			#labels = ''.join(sorted(labels))

			self._label_converter = swl_langproc_util.TokenConverter(labels, pad=None)
			# NOTE [info] >> The ID of the blank label is reserved as label_converter.num_tokens.
			print('[SWL] Info: Labels = {}.'.format(self._label_converter.tokens))
			print('[SWL] Info: #labels = {}.'.format(self._label_converter.num_tokens))
			"""

			# Data generation.
			#	REF [function] >> HangeulTextRecognitionDataGeneratorTextLineDataset_test() in TextRecognitionDataGenerator_data_test.py.

			# REF [directory] >> ${TextRecognitionDataGenerator_HOME}/TextRecognitionDataGenerator
			data_dir_path = './text_line_samples_kr_train'

			self._dataset = MyHangeulTextLineDataset(self._label_converter, data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len)

			self._train_examples_per_epoch, self._val_examples_per_epoch, self._test_examples_per_epoch = self._dataset.num_train_examples, self._dataset.num_test_examples, self._dataset.num_test_examples

	@property
	def label_converter(self):
		return self._label_converter

	def train(self, model_filepath, model_checkpoint_filepath, batch_size, final_epoch, initial_epoch=0, is_training_resumed=False):
		if is_training_resumed:
			# Restore a model.
			try:
				print('[SWL] Info: Start restoring a model...')
				start_time = time.time()
				"""
				# Load only the architecture of a model.
				model = tf.keras.models.model_from_json(json_string)
				#model = tf.keras.models.model_from_yaml(yaml_string)
				# Load only the weights of a model.
				model.load_weights(model_weight_filepath)
				"""
				# Load a model.
				model = tf.keras.models.load_model(model_filepath)
				print('[SWL] Info: End restoring a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except (ImportError, IOError):
				print('[SWL] Error: Failed to restore a model from {}.'.format(model_filepath))
				return
		else:
			# Create a model.
			model = MyModel.create_model(self._dataset.shape, self._label_converter.num_tokens + 1, self._max_label_len, is_training=True)
			#print('Model summary =', model.summary())

		# Create a trainer.
		optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

		# Loss is computed as the model output, so a dummy loss is assigned here.
		model.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=['accuracy'])

		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, mode='min', verbose=1)
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=1)

		train_steps_per_epoch = None if self._train_examples_per_epoch is None else math.ceil(self._train_examples_per_epoch / batch_size)
		val_steps_per_epoch = None if self._val_examples_per_epoch is None else math.ceil(self._val_examples_per_epoch / batch_size)

		num_epochs = final_epoch - initial_epoch

		#--------------------
		if is_training_resumed:
			print('[SWL] Info: Resume training...')
		else:
			print('[SWL] Info: Start training...')
		start_time = time.time()
		# NOTE [error] >> TypeError("can't pickle generator objects").
		#train_sequence = MyDataSequence(self._dataset.create_train_batch_generator(batch_size, train_steps_per_epoch, shuffle=True), self._label_converter, train_steps_per_epoch, self._max_label_len, self._model_output_time_steps)
		#val_sequence = MyDataSequence(self._dataset.create_test_batch_generator(batch_size, val_steps_per_epoch, shuffle=False), self._label_converter, val_steps_per_epoch, self._max_label_len, self._model_output_time_steps)
		train_sequence = MyDataSequence(self._dataset.train_examples, self._label_converter, self._max_label_len, self._model_output_time_steps, batch_size, shuffle=True)
		val_sequence = MyDataSequence(self._dataset.test_examples, self._label_converter, self._max_label_len, self._model_output_time_steps, batch_size, shuffle=False)
		history = model.fit_generator(train_sequence, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch, validation_data=val_sequence, validation_steps=val_steps_per_epoch, shuffle=True, initial_epoch=initial_epoch, class_weight=None, max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing, callbacks=[early_stopping_callback, model_checkpoint_callback])
		print('[SWL] Info: End training: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('[SWL] Info: Start evaluating...')
		start_time = time.time()
		# NOTE [error] >> TypeError("can't pickle generator objects").
		#val_sequence = MyDataSequence(self._dataset.create_test_batch_generator(batch_size, val_steps_per_epoch, shuffle=False), self._label_converter, self._max_label_len, self._model_output_time_steps)
		val_sequence = MyDataSequence(self._dataset.test_examples, self._label_converter, self._max_label_len, self._model_output_time_steps, batch_size, shuffle=False)
		score = model.evaluate_generator(val_sequence, steps=val_steps_per_epoch, max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		print('\tValidation: Loss = {:.6f}, accuracy = {:.6f}.'.format(*score))
		print('[SWL] Info: End evaluating: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('[SWL] Info: Start saving a model to {}...'.format(model_filepath))
		start_time = time.time()
		"""
		# Save only the architecture of a model.
		json_string = model.to_json()
		#yaml_string = model.to_yaml()
		# Save only the weights of a model.
		model.save_weights(model_weight_filepath)
		"""
		# Save a model.
		model.save(model_filepath)
		print('[SWL] Info: End saving a model: {} secs.'.format(time.time() - start_time))

		return history.history

	def test(self, model_filepath, test_dir_path, batch_size, shuffle=False):
		# Load a model.
		try:
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			"""
			# Load only the architecture of a model.
			model = tf.keras.models.model_from_json(json_string)
			#model = tf.keras.models.model_from_yaml(yaml_string)
			# Load only the weights of a model.
			model.load_weights(model_weight_filepath)
			"""
			# Load a model.
			model = tf.keras.models.load_model(model_filepath)
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except (ImportError, IOError):
			print('[SWL] Error: Failed to load a model from {}.'.format(model_filepath))
			return

		test_steps_per_epoch = None if self._test_examples_per_epoch is None else math.ceil(self._test_examples_per_epoch / batch_size)
		batch_images_list, batch_labels_list = list(), list()
		for batch_data, num_batch_examples in self._dataset.create_test_batch_generator(batch_size, test_steps_per_epoch, shuffle=shuffle):
			#batch_images, batch_labels_str, batch_labels_int = batch_data
			batch_images, batch_labels_str, _ = batch_data
			batch_images_list.append(batch_images)
			batch_labels_list.append(batch_labels_str)

		#--------------------
		print('[SWL] Info: Start testing...')
		start_time = time.time()
		inferences, ground_truths = list(), list()
		for batch_images, batch_labels in zip(batch_images_list, batch_labels_list):
			batch_outputs = model.predict(batch_images, batch_size=batch_size)
			batch_outputs = MyModel.decode_label(batch_outputs, self._label_converter.num_tokens)

			inferences.extend(batch_outputs)
			ground_truths.extend(list(batch_labels))
		print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

		if inferences is not None and ground_truths is not None:
			#print('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			# REF [function] >> compute_simple_text_recognition_accuracy() in ${SWL_PYTHON_HOME}/src/swl/language_processing/util.py.
			correct_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0, 0
			total_text_count = max(len(inferences), len(ground_truths))
			for inf_lbl, gt_lbl in zip(inferences, ground_truths):
				inf_lbl = self._label_converter.decode(inf_lbl)

				if inf_lbl == gt_lbl:
					correct_text_count += 1

				inf_words, gt_words = inf_lbl.split(' '), gt_lbl.split(' ')
				total_word_count += max(len(inf_words), len(gt_words))
				#correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(inf_words, gt_words))))
				correct_word_count += len(list(filter(lambda x: x[0].lower() == x[1].lower(), zip(inf_words, gt_words))))

				total_char_count += max(len(inf_lbl), len(gt_lbl))
				#correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(inf_lbl, gt_lbl))))
				correct_char_count += len(list(filter(lambda x: x[0].lower() == x[1].lower(), zip(inf_lbl, gt_lbl))))
			print('\tTest: Text accuracy = {} / {} = {}.'.format(correct_text_count, total_text_count, correct_text_count / total_text_count))
			print('\tTest: Word accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
			print('\tTest: Char accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

			# Output to a file.
			csv_filepath = os.path.join(test_dir_path, 'test_results.csv')
			with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')

				for inf, gt in zip(inferences, ground_truths):
					inf = self._label_converter.decode(inf)
					writer.writerow([gt, inf])
		else:
			print('[SWL] Warning: Invalid test results.')

	def infer(self, model_filepath, inputs, batch_size=None, shuffle=False):
		# Load a model.
		try:
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			"""
			# Load only the architecture of a model.
			model = tf.keras.models.model_from_json(json_string)
			#model = tf.keras.models.model_from_yaml(yaml_string)
			# Load only the weights of a model.
			model.load_weights(model_weight_filepath)
			"""
			# Load a model.
			model = tf.keras.models.load_model(model_filepath)
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except (ImportError, IOError):
			print('[SWL] Error: Failed to load a model from {}.'.format(model_filepath))
			return

		#--------------------
		num_examples = len(inputs)
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		#--------------------
		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		inferences = model.predict(inputs, batch_size=batch_size)
		inferences = MyModel.decode_label(inferences, self._label_converter.num_tokens)
		print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))
		return inferences

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#--------------------
	if False:
		image_height, image_width, image_channel = 32, 160, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
		max_label_len = 40
	else:
		image_height, image_width, image_channel = 64, 320, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
		max_label_len = 80

	is_trained, is_tested, is_inferred = True, True, False
	is_training_resumed = False
	initial_epoch, final_epoch, batch_size = 0, 100, 64
	train_test_ratio = 0.8
	is_dataset_generated_at_runtime = False

	#--------------------
	model_filepath, output_dir_path = None, None
	if model_filepath:
		if not output_dir_path:
			output_dir_path = os.path.dirname(model_filepath)
	else:
		if not output_dir_path:
			output_dir_prefix = 'simple_hangeul_crnn_keras'
			output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		model_filepath = os.path.join(output_dir_path, 'model.hdf5')
		#model_weight_filepath = os.path.join(output_dir_path, 'model_weights.hdf5')

	test_dir_path = None
	if not test_dir_path:
		test_dir_path = os.path.join(output_dir_path, 'test')
	inference_dir_path = None
	if not inference_dir_path:
		inference_dir_path = os.path.join(output_dir_path, 'inference')

	#--------------------
	runner = MyRunner(image_height, image_width, image_channel, max_label_len, train_test_ratio, is_dataset_generated_at_runtime)

	if is_trained:
		model_checkpoint_filepath = os.path.join(output_dir_path, 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		# Copy the model file to the output directory.
		new_model_filepath = os.path.join(output_dir_path, os.path.basename(model_filepath))
		if os.path.exists(model_filepath) and not os.path.samefile(model_filepath, new_model_filepath):
			try:
				shutil.copyfile(model_filepath, new_model_filepath)
			except (FileNotFoundError, PermissionError) as ex:
				print('[SWL] Error: Failed to copy a model, {}: {}.'.format(model_filepath, ex))
				return
		model_filepath = new_model_filepath

		history = runner.train(model_filepath, model_checkpoint_filepath, batch_size, final_epoch, initial_epoch, is_training_resumed)

		#print('History =', history)
		swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if is_tested:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return
		if test_dir_path and test_dir_path.strip() and not os.path.exists(test_dir_path):
			os.makedirs(test_dir_path, exist_ok=True)

		runner.test(model_filepath, test_dir_path, batch_size)

	if is_inferred:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return
		if inference_dir_path and inference_dir_path.strip() and not os.path.exists(inference_dir_path):
			os.makedirs(inference_dir_path, exist_ok=True)

		image_filepaths = glob.glob('./text_line_samples_kr_test/**/*.jpg', recursive=False)
		if not image_filepaths:
			print('[SWL] Error: No image file for inference.')
			return
		image_filepaths.sort()

		print('[SWL] Info: Start loading images...')
		start_time = time.time()
		inf_images, image_filepaths = runner.dataset.load_images_from_files(image_filepaths, is_grayscale=True)
		print('[SWL] Info: End loading images: {} secs.'.format(time.time() - start_time))
		print('[SWL] Info: Loaded images: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inf_images.shape, inf_images.dtype, np.min(inf_images), np.max(inf_images)))

		inferences = runner.infer(model_filepath, inf_images)

		if inferences is not None:
			#print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			inferences = list(map(lambda x: runner.label_converter.decode(x), inferences))

			# Output to a file.
			csv_filepath = os.path.join(inference_dir_path, 'inference_results.csv')
			with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')

				for fpath, inf in zip(image_filepaths, inferences):
					writer.writerow([fpath, inf])
		else:
			print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
