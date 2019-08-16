#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, time, datetime, glob, csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LSTM, MaxPooling2D, Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import Input, Dense, Activation, add, concatenate
import swl.machine_learning.util as swl_ml_util
import text_line_data
from TextRecognitionDataGenerator_data import HangeulTextRecognitionDataGeneratorTextLineDataset as TextLineDataset

#--------------------------------------------------------------------

class MyDataSequence(tf.keras.utils.Sequence):
	def __init__(self, batch_generator, steps_per_epoch, max_label_len, model_output_time_steps, default_value, encode_labels_functor):
		self._batch_generator = batch_generator
		self._steps_per_epoch = steps_per_epoch
		self._max_label_len, self._model_output_time_steps, self._default_value = max_label_len, model_output_time_steps, default_value
		self._encode_labels_functor = encode_labels_functor

	def __len__(self):
		return self._steps_per_epoch if self._steps_per_epoch is not None else -1

	def __getitem__(self, idx):
		for batch_data, num_batch_examples in self._batch_generator:
			#batch_images, batch_labels_str, batch_labels_int = batch_data
			batch_images, batch_labels_str, _ = batch_data
			batch_labels_int = self._encode_labels_functor(batch_labels_str)

			if batch_labels_int.shape[1] < self._max_label_len:
				labels = np.full((num_batch_examples, self._max_label_len), self._default_value)
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
		labels = Input(shape=[max_label_len], dtype='float32', name='outputs')  # (None, max_label_len).
		model_output_length = Input(shape=[1], dtype='int64', name='model_output_length')  # (None, 1).
		label_length = Input(shape=[1], dtype='int64', name='output_length')  # (None, 1).

		# Currently Keras does not support loss functions with extra parameters so CTC loss is implemented in a lambda layer.
		loss = Lambda(MyModel.compute_ctc_loss, output_shape=(1,), name='ctc_loss')([model_outputs, labels, model_output_length, label_length])  # (None, 1).

		if is_training:
			return Model(inputs=[inputs, labels, model_output_length, label_length], outputs=loss)
		else:
			return Model(inputs=[inputs], outputs=model_outputs)

	@staticmethod
	def compute_ctc_loss(args):
		model_outputs, labels, model_output_length, label_length = args
		# TODO [check] >> The first couple of RNN outputs tend to be garbage. (???)
		model_outputs = model_outputs[:, 2:, :]
		return K.ctc_batch_cost(labels, model_outputs, model_output_length, label_length)

	@staticmethod
	def decode_label(labels, blank_label):
		labels = np.argmax(labels, axis=-1)
		return list(map(lambda lbl: list(k for k, g in itertools.groupby(lbl) if k < blank_label), labels))  # Removes repetitive labels.

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, is_dataset_generated_at_runtime, data_dir_path=None, train_test_ratio=0.8):
		# Set parameters.
		self._max_queue_size, self._num_workers = 10, 8
		self._use_multiprocessing = True

		#sess = tf.Session(config=config)
		#K.set_session(sess)
		#K.set_learning_phase(0)  # Sets the learning phase to 'test'.
		#K.set_learning_phase(1)  # Sets the learning phase to 'train'.

		# TODO [modify] >> Depends on a model.
		#	model_output_time_steps = image_width / width_downsample_factor or image_width / width_downsample_factor - 1.
		#	REF [function] >> MyModel.create_model().
		#width_downsample_factor = 4
		if False:
			image_height, image_width, image_channel = 32, 160, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
			self._model_output_time_steps = 40
		else:
			image_height, image_width, image_channel = 64, 320, 1  # TODO [modify] >> image_height is hard-coded and image_channel is fixed.
			self._model_output_time_steps = 80
		# TODO [check] >> The first couple of RNN outputs tend to be garbage. (???)
		self._model_output_time_steps -= 2
		self._max_label_len = self._model_output_time_steps  # max_label_len <= model_output_time_steps.

		#--------------------
		# Create a dataset.

		if is_dataset_generated_at_runtime:
			print('[SWL] Info: Start loading a Korean dictionary...')
			start_time = time.time()
			korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'
			with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
				#korean_words = fd.readlines()
				#korean_words = fd.read().strip('\n')
				korean_words = fd.read().splitlines()
			print('[SWL] Info: End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))

			print('[SWL] Info: Start creating a Hangeul dataset...')
			start_time = time.time()
			self._dataset = text_line_data.RunTimeTextLineDataset(set(korean_words), image_height, image_width, image_channel, max_label_len=self._max_label_len)
			print('[SWL] Info: End creating a Hangeul dataset: {} secs.'.format(time.time() - start_time))

			self._train_examples_per_epoch, self._val_examples_per_epoch, self._test_examples_per_epoch = 200000, 10000, 10000 #500000, 10000, 10000
		else:
			# When using TextRecognitionDataGenerator_data.HangeulTextRecognitionDataGeneratorTextLineDataset.
			self._dataset = TextLineDataset(data_dir_path, image_height, image_width, image_channel, train_test_ratio, max_label_len=self._max_label_len)

			self._train_examples_per_epoch, self._val_examples_per_epoch, self._test_examples_per_epoch = None, None, None

	def train(self, model_filepath, model_checkpoint_filepath, num_epochs, batch_size, initial_epoch=0, is_training_resumed=False):
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
			model = MyModel.create_model(self._dataset.shape, self._dataset.num_classes, self._max_label_len, is_training=True)
			#print('Model summary =', model.summary())

		# Create a trainer.
		optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

		# Loss is computed as the model output, so a dummy loss is assigned here.
		model.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=['accuracy'])

		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, mode='min', verbose=1)
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=1)

		train_steps_per_epoch = None if self._train_examples_per_epoch is None else math.ceil(self._train_examples_per_epoch / batch_size)
		val_steps_per_epoch = None if self._val_examples_per_epoch is None else math.ceil(self._val_examples_per_epoch / batch_size)

		#--------------------
		if is_training_resumed:
			print('[SWL] Info: Resume training...')
		else:
			print('[SWL] Info: Start training...')
		start_time = time.time()
		train_sequence = MyDataSequence(self._dataset.create_train_batch_generator(batch_size, train_steps_per_epoch, shuffle=True), train_steps_per_epoch, self._max_label_len, self._model_output_time_steps, self._dataset.default_value, self._dataset.encode_labels)
		val_sequence = MyDataSequence(self._dataset.create_test_batch_generator(batch_size, val_steps_per_epoch, shuffle=False), val_steps_per_epoch, self._max_label_len, self._model_output_time_steps, self._dataset.default_value, self._dataset.encode_labels)
		history = model.fit_generator(train_sequence, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch, validation_data=val_sequence, validation_steps=val_steps_per_epoch, shuffle=True, initial_epoch=initial_epoch, class_weight=None, max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing, callbacks=[early_stopping_callback, model_checkpoint_callback])
		print('[SWL] Info: End training: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('[SWL] Info: Start evaluating...')
		start_time = time.time()
		val_sequence = MyDataSequence(self._dataset.create_test_batch_generator(batch_size, val_steps_per_epoch, shuffle=False), val_steps_per_epoch, self._max_label_len, self._model_output_time_steps, self._dataset.default_value, self._dataset.encode_labels)
		score = model.evaluate_generator(val_sequence, steps=val_steps_per_epoch, max_queue_size=self._max_queue_size, workers=self._num_workers, use_multiprocessing=self._use_multiprocessing)
		print('\tValidation: loss = {:.6f}, accuracy = {:.6f}.'.format(*score))
		print('[SWL] Info: End evaluating: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('[SWL] Info: Start saving a model...')
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
		print('[SWL] Info: End saving a model to {}: {} secs.'.format(model_filepath, time.time() - start_time))

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
			batch_outputs = MyModel.decode_label(batch_outputs, self._dataset.num_classes - 1)

			inferences.extend(batch_outputs)
			ground_truths.extend(list(batch_labels))
		print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

		if inferences is not None and ground_truths is not None:
			#print('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0
			for pred, gt in zip(inferences, ground_truths):
				pred = np.array(list(map(lambda x: self._dataset.decode_label(x), pred)))

				correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(pred, gt))))
				total_word_count += len(gt)
				for ps, gs in zip(pred, gt):
					correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(ps, gs))))
					total_char_count += max(len(ps), len(gs))
				#correct_char_count += functools.reduce(lambda l, pgs: l + len(list(filter(lambda pg: pg[0] == pg[1], zip(pgs[0], pgs[1])))), zip(pred, gt), 0)
				#total_char_count += functools.reduce(lambda l, pg: l + max(len(pg[0]), len(pg[1])), zip(pred, gt), 0)
			print('Test: word accuracy = {} / {} = {}.'.format(correct_word_count, total_word_count, correct_word_count / total_word_count))
			print('Test: character accuracy = {} / {} = {}.'.format(correct_char_count, total_char_count, correct_char_count / total_char_count))

			# Output to a file.
			csv_filepath = os.path.join(test_dir_path, 'test_results.csv')
			with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')

				for pred, gt in zip(inferences, ground_truths):
					pred = np.array(list(map(lambda x: self._dataset.decode_label(x), pred)))
					writer.writerow([gt, pred])
		else:
			print('[SWL] Warning: Invalid test results.')

	def infer(self, model_filepath, image_filepaths, inference_dir_path, batch_size=None, shuffle=False):
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
		print('[SWL] Info: Start loading images...')
		inf_images = self._dataset.load_images_from_files(image_filepaths)
		print('[SWL] Info: End loading images: {} secs.'.format(time.time() - start_time))

		num_examples = len(inf_images)
		if batch_size is None:
			batch_size = num_examples
		if batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		#--------------------
		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		inferences = model.predict(inf_images, batch_size=batch_size)
		inferences = MyModel.decode_label(inferences, self._dataset.num_classes - 1)
		print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

		if inferences is not None:
			#print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			inferences = list(map(lambda x: self._dataset.decode_label(x), inferences))

			# Output to a file.
			csv_filepath = os.path.join(inference_dir_path, 'inference_results.csv')
			with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')

				for fpath, inf in zip(image_filepaths, inferences):
					writer.writerow([fpath, inf])
		else:
			print('[SWL] Warning: Invalid test results.')

#--------------------------------------------------------------------

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	#--------------------
	num_epochs, batch_size = 100, 64
	initial_epoch = 0
	is_trained, is_tested, is_inferred = True, True, False
	is_training_resumed = False

	is_dataset_generated_at_runtime = False
	if not is_dataset_generated_at_runtime and (is_trained or is_tested):
		#data_dir_path = './kr_samples_100000'
		data_dir_path = './kr_samples_200000'
	else:
		data_dir_path = None
	train_test_ratio = 0.8

	#--------------------
	model_filepath = None
	if model_filepath:
		output_dir_path = os.path.dirname(model_filepath)
	else:
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
	runner = MyRunner(is_dataset_generated_at_runtime, data_dir_path, train_test_ratio)

	if is_trained:
		model_checkpoint_filepath = os.path.join(output_dir_path, 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		history = runner.train(model_filepath, model_checkpoint_filepath, num_epochs, batch_size, initial_epoch, is_training_resumed)

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

		runner.test(model_filepath, test_dir_path)

	if is_inferred:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return
		if inference_dir_path and inference_dir_path.strip() and not os.path.exists(inference_dir_path):
			os.makedirs(inference_dir_path, exist_ok=True)

		image_filepaths = glob.glob('./kr_samples_1000/*.jpg')  # TODO [modify] >>
		runner.infer(model_filepath, image_filepaths, inference_dir_path)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
