#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
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

def generate_data(image_height, image_width, image_channel, num_classes, batch_size, is_training=False, shuffle=False):
	# Pixel value: [0, 255].
	if is_training:
		(inputs, outputs), (_, _) = tf.keras.datasets.mnist.load_data()
	else:
		(_, _), (inputs, outputs) = tf.keras.datasets.mnist.load_data()

	# Preprocessing.
	inputs, outputs = preprocess_data(inputs, outputs, image_height, image_width, image_channel, num_classes)

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

		#start_idx = 0 if end_idx >= num_examples else end_idx
		if end_idx >= num_examples:
			indices = np.arange(num_examples)
			if shuffle:
				np.random.shuffle(indices)

			start_idx = 0
		else:
			start_idx = end_idx

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

	checkpoint_filepath = './mnist_cnn_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

	#sess = tf.Session(config=config)
	#K.set_session(sess)
	#K.set_learning_phase(0)  # Sets the learning phase to 'test'.
	#K.set_learning_phase(1)  # Sets the learning phase to 'train'.

	#--------------------
	# Load data.

	train_images, train_labels, test_images, test_labels = load_data(image_height, image_width, image_channel, num_classes)
	num_train_images, num_test_images = len(train_images), len(test_images)

	print("Train image's shape = {}, train label's shape = {}.".format(train_images.shape, train_labels.shape))
	print("Test image's shape = {}, test label's shape = {}.".format(test_images.shape, test_labels.shape))

	#--------------------
	# Create a model.

	model = create_model((image_height, image_width, image_channel), num_classes)
	#print('Model summary =', model.summary())

	loss = tf.keras.losses.categorical_crossentropy
	optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)

	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	#--------------------
	# Train and evaluate.

	if True:
		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

		print('Start training...')
		start_time = time.time()
		if False:
			history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2, shuffle=True, initial_epoch=0, class_weight=None, sample_weight=None, callbacks=[early_stopping_callback, model_checkpoint_callback])
		else:
			train_generator = generate_data(image_height, image_width, image_channel, num_classes, BATCH_SIZE, is_training=True, shuffle=True)
			val_generator = generate_data(image_height, image_width, image_channel, num_classes, BATCH_SIZE, is_training=False, shuffle=False)
			history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, steps_per_epoch=num_train_images // BATCH_SIZE + 1, validation_data=val_generator, validation_steps=num_test_images // BATCH_SIZE + 1, shuffle=True, initial_epoch=0, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=True, callbacks=[early_stopping_callback, model_checkpoint_callback])
		print('End training: {} secs.'.format(time.time() - start_time))

		#print('History =', history.history)
		draw_history(history)

		print('Start evaluating...')
		start_time = time.time()
		if False:
			score = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, sample_weight=None)
		else:
			val_generator = generate_data(image_height, image_width, image_channel, num_classes, BATCH_SIZE, is_training=False, shuffle=False)
			score = model.evaluate_generator(val_generator, steps=num_test_images // BATCH_SIZE + 1, max_queue_size=10, workers=1, use_multiprocessing=True)
		print('\tEvaluation: loss = {}, accuracy = {}.'.format(*score))
		print('End evaluating: {} secs.'.format(time.time() - start_time))

		if False:
			# Save only a model's architecture.
			json_string = model.to_json()
			#yaml_string = model.to_yaml()
			# Save only a model's weights.
			model.save_weights('./mnist_cnn_weights.h5')
		else:
			model.save('./mnist_cnn.h5')
		del model

	#--------------------
	# Infer.

	if False:
		# Load only a model's architecture.
		loaded_model = keras.models.model_from_json(json_string)
		#loaded_model = keras.models.model_from_yaml(yaml_string)
		# Load only a model's weights.
		loaded_model.load_weights('./mnist_cnn_weights.h5')
	else:
		loaded_model = tf.keras.models.load_model('./mnist_cnn.h5')

	print('Start inferring...')
	start_time = time.time()
	if False:
		inferences = loaded_model.predict(test_images, batch_size=BATCH_SIZE)
	else:
		test_generator = generate_data(image_height, image_width, image_channel, num_classes, BATCH_SIZE, is_training=False, shuffle=False)
		inferences = loaded_model.predict_generator(test_generator, steps=num_test_images // BATCH_SIZE + 1, max_queue_size=10, workers=1, use_multiprocessing=True)
	print('End inferring: {} secs.'.format(time.time() - start_time))

	if inferences is not None:
		if num_classes >= 2:
			inferences = np.argmax(inferences, -1)
			ground_truths = np.argmax(test_labels, -1)
		else:
			inferences = np.around(inferences)
			ground_truths = test_labels
		correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
		print('Accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
	else:
		print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
