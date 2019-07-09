#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time, datetime, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import backend as K
#from sklearn import preprocessing
import cv2
import matplotlib.pyplot as plt

#--------------------------------------------------------------------

def preprocess_data(inputs, outputs, image_height, image_width, image_channel, num_classes):
	if inputs is not None:
		# Contrast limited adaptive histogram equalization (CLAHE).
		#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		#inputs = np.array([clahe.apply(inp) for inp in inputs])

		# Normalization, standardization, etc.
		inputs = inputs.astype(np.float32)

		if False:
			inputs = preprocessing.scale(inputs, axis=0, with_mean=True, with_std=True, copy=True)
			#inputs = preprocessing.minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
			#inputs = preprocessing.maxabs_scale(inputs, axis=0, copy=True)  # [-1, 1].
			#inputs = preprocessing.robust_scale(inputs, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
		elif True:
			inputs = (inputs - np.mean(inputs, axis=None)) / np.std(inputs, axis=None)  # Standardization.
		elif False:
			in_min, in_max = 0, 255 #np.min(inputs), np.max(inputs)
			out_min, out_max = 0, 1 #-1, 1
			inputs = (inputs - in_min) * (out_max - out_min) / (in_max - in_min) + out_min  # Normalization.
		elif False:
			inputs /= 255.0  # Normalization.

		# Reshaping.
		inputs = np.reshape(inputs, (-1, image_height, image_width, image_channel))

	if outputs is not None:
		# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
		#outputs = swl_ml_util.to_one_hot_encoding(outputs, num_classes).astype(np.uint8)
		outputs = tf.keras.utils.to_categorical(outputs).astype(np.uint8)

	return inputs, outputs

def load_data(image_height, image_width, image_channel, num_classes):
	# Pixel value: [0, 255].
	(train_inputs, train_outputs), (test_inputs, test_outputs) = tf.keras.datasets.mnist.load_data()

	# Preprocessing.
	train_inputs, train_outputs = preprocess_data(train_inputs, train_outputs, image_height, image_width, image_channel, num_classes)
	test_inputs, test_outputs = preprocess_data(test_inputs, test_outputs, image_height, image_width, image_channel, num_classes)

	return train_inputs, train_outputs, test_inputs, test_outputs

def generate_data(inputs, outputs, batch_size=None, shuffle=False):
	num_examples = len(inputs)
	if len(outputs) != num_examples:
		raise ValueError('Invalid data size: {} != {}'.format(num_examples, len(outputs)))
	if batch_size is None:
		batch_size = num_examples
	if batch_size <= 0:
		raise ValueError('Invalid batch size: {}'.format(batch_size))

	indices = np.arange(num_examples)
	if shuffle:
		np.random.shuffle(indices)

	start_idx = 0
	while True:
		end_idx = start_idx + batch_size
		batch_indices = indices[start_idx:end_idx]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			batch_input, batch_output = inputs[batch_indices], outputs[batch_indices]
			if batch_input.size > 0 and batch_output.size > 0:  # If batch_input and batch_output are non-empty.
				yield (batch_input, batch_output)
			else:
				yield (None, None)
		else:
			yield (None, None)

		#start_idx = 0 if end_idx >= num_examples else end_idx
		if end_idx >= num_examples:
			indices = np.arange(num_examples)
			if shuffle:
				np.random.shuffle(indices)

			start_idx = 0
		else:
			start_idx = end_idx

class DataSequence(tf.keras.utils.Sequence):
	def __init__(self, inputs, outputs, batch_size=None, shuffle=False):
		self.inputs, self.outputs = inputs, outputs
		self.batch_size = batch_size

		self.num_examples = len(self.inputs)
		if len(self.outputs) != self.num_examples:
			raise ValueError('Invalid data size: {} != {}'.format(self.num_examples, len(self.outputs)))
		if self.batch_size is None:
			self.batch_size = self.num_examples
		if self.batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(self.batch_size))

		self.indices = np.arange(self.num_examples)
		if shuffle:
			np.random.shuffle(self.indices)

	def __len__(self):
		return math.ceil(self.num_examples / self.batch_size)

	def __getitem__(self, idx):
		start_idx = idx * self.batch_size
		end_idx = start_idx + self.batch_size
		batch_indices = self.indices[start_idx:end_idx]
		if batch_indices.size > 0:  # If batch_indices is non-empty.
			# FIXME [fix] >> Does not work correctly in time-major data.
			batch_input, batch_output = self.inputs[batch_indices], self.outputs[batch_indices]
			if batch_input.size > 0 and batch_output.size > 0:  # If batch_input and batch_output are non-empty.
				return (batch_input, batch_output)
		return (None, None)

#--------------------------------------------------------------------

def create_model(input_shape, num_classes):
	model = Sequential()

	# Layer 1.
	model.add(Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=2, strides=2))
	# Layer 2.
	model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
	model.add(MaxPooling2D(pool_size=2, strides=2))
	model.add(Flatten())
	# Layer 3.
	model.add(Dense(units=1024, activation='relu'))
	# Layer 4.
	model.add(Dense(units=num_classes, activation='softmax'))

	return model

def draw_history(history):
	# Plot training & validation accuracy values.
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values.
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

#--------------------------------------------------------------------

def main():
	image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
	num_classes = 10

	BATCH_SIZE, NUM_EPOCHS = 128, 30
	initial_epoch = 0
	max_queue_size, num_workers = 10, 8
	use_multiprocessing = True

	output_dir_prefix = 'simple_training'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
	os.makedirs(output_dir_path, exist_ok=True)

	checkpoint_filepath = os.path.join(output_dir_path, 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5')

	#sess = tf.Session(config=config)
	#K.set_session(sess)
	#K.set_learning_phase(0)  # Sets the learning phase to 'test'.
	#K.set_learning_phase(1)  # Sets the learning phase to 'train'.

	#%%------------------------------------------------------------------
	# Load data.

	print('Start loading dataset...')
	start_time = time.time()
	train_images, train_labels, test_images, test_labels = load_data(image_height, image_width, image_channel, num_classes)
	print('End loading dataset: {} secs.'.format(time.time() - start_time))

	num_train_images, num_test_images = len(train_images), len(test_images)

	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_images.shape, train_images.dtype, np.min(train_images), np.max(train_images)))
	print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_labels.shape, train_labels.dtype, np.min(train_labels), np.max(train_labels)))
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_images.shape, test_images.dtype, np.min(test_images), np.max(test_images)))
	print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_labels.shape, test_labels.dtype, np.min(test_labels), np.max(test_labels)))

	#%%------------------------------------------------------------------
	# Create a model.

	model = create_model((image_height, image_width, image_channel), num_classes)
	#print('Model summary =', model.summary())

	#%%------------------------------------------------------------------
	# Train and evaluate.

	if True:
		loss = tf.keras.losses.categorical_crossentropy
		optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)

		model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

		#--------------------
		print('Start training...')
		start_time = time.time()
		if False:
			history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2, shuffle=True, initial_epoch=0, class_weight=None, sample_weight=None, callbacks=[early_stopping_callback, model_checkpoint_callback])
		elif False:
			# Use generators.
			train_generator = generate_data(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
			val_generator = generate_data(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)
			history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, steps_per_epoch=math.ceil(num_train_images / BATCH_SIZE), validation_data=val_generator, validation_steps=math.ceil(num_test_images / BATCH_SIZE), shuffle=True, initial_epoch=initial_epoch, class_weight=None, max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing, callbacks=[early_stopping_callback, model_checkpoint_callback])
		else:
			# Use Keras sequences.
			train_sequence = DataSequence(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
			val_sequence = DataSequence(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)
			history = model.fit_generator(train_sequence, epochs=NUM_EPOCHS, steps_per_epoch=math.ceil(num_train_images / BATCH_SIZE), validation_data=val_sequence, validation_steps=math.ceil(num_test_images / BATCH_SIZE), shuffle=True, initial_epoch=initial_epoch, class_weight=None, max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing, callbacks=[early_stopping_callback, model_checkpoint_callback])
		print('End training: {} secs.'.format(time.time() - start_time))

		#print('History =', history.history)
		draw_history(history)

		#--------------------
		print('Start evaluating...')
		start_time = time.time()
		if False:
			score = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, sample_weight=None)
		elif False:
			# Use a generator.
			val_generator = generate_data(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)
			score = model.evaluate_generator(val_generator, steps=math.ceil(num_test_images / BATCH_SIZE), max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing)
		else:
			# Use a Keras sequence.
			val_sequence = DataSequence(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)
			score = model.evaluate_generator(val_sequence, steps=math.ceil(num_test_images / BATCH_SIZE), max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing)
		print('\tValidation: loss = {}, accuracy = {}.'.format(*score))
		print('End evaluating: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('Start saving a model...')
		start_time = time.time()
		if False:
			# Save only a model's architecture.
			json_string = model.to_json()
			#yaml_string = model.to_yaml()
			# Save only a model's weights.
			model.save_weights(os.path.join(output_dir_path, 'model_weights.hdf5'))
		else:
			model.save(os.path.join(output_dir_path, 'model.hdf5'))
		del model
		print('End saving a model: {} secs.'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infer.

	print('Start loading a model...')
	start_time = time.time()
	if False:
		# Load only a model's architecture.
		loaded_model = keras.models.model_from_json(json_string)
		#loaded_model = keras.models.model_from_yaml(yaml_string)
		# Load only a model's weights.
		loaded_model.load_weights(os.path.join(output_dir_path, 'model_weights.hdf5'))
	else:
		loaded_model = tf.keras.models.load_model(os.path.join(output_dir_path, 'model.hdf5'))
	print('End loading a model: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start inferring...')
	start_time = time.time()
	if False:
		inferences = loaded_model.predict(test_images, batch_size=BATCH_SIZE)
	elif False:
		# Use a generator.
		test_generator = generate_data(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)
		inferences = loaded_model.predict_generator(test_generator, steps=math.ceil(num_test_images / BATCH_SIZE), max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing)
	else:
		# Use a Keras sequence.
		test_sequence = DataSequence(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)
		inferences = loaded_model.predict_generator(test_sequence, steps=math.ceil(num_test_images / BATCH_SIZE), max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing)
	print('End inferring: {} secs.'.format(time.time() - start_time))

	if inferences is not None:
		if num_classes > 2:
			inferences = np.argmax(inferences, -1)
			ground_truths = np.argmax(test_labels, -1)
		elif 2 == num_classes:
			inferences = np.around(inferences)
			ground_truths = test_labels
		else:
			raise ValueError('Invalid number of classes')
		correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
		print('Inference: accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
	else:
		print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
