#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
import numpy as np
import tensorflow as tf
#from sklearn import preprocessing
import cv2

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

#--------------------------------------------------------------------

def create_model(input_tensor, num_classes):
	# Preprocessing.
	with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
		input_tensor = tf.nn.local_response_normalization(input_tensor, depth_radius=5, bias=1, alpha=1, beta=0.5, name='lrn')

	#--------------------
	with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
		conv1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, name='conv')
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool')

	with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
		conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv')
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool')

	with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		fc1 = tf.layers.flatten(conv2, name='flatten')

		fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='dense')

	with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
		if 1 == num_classes:
			fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, name='dense')
			#fc2 = tf.layers.dense(fc1, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
		elif num_classes >= 2:
			fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='dense')
			#fc2 = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='dense')
		else:
			assert num_classes > 0, 'Invalid number of classes.'

		return fc2

def get_loss(y, t):
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=t, logits=y))
		return loss

def get_accuracy(y, t):
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y, axis=-1), tf.argmax(t, axis=-1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

#--------------------------------------------------------------------

def main():
	image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
	num_classes = 10
	BATCH_SIZE, NUM_EPOCHS = 128, 30

	checkpoint_dir_path = './tf_checkpoint'
	os.makedirs(checkpoint_dir_path, exist_ok=True)

	#--------------------
	# Load data.

	print('Start loading dataset...')
	start_time = time.time()
	train_images, train_labels, test_images, test_labels = load_data(image_height, image_width, image_channel, num_classes)
	print('End loading dataset: {} secs.'.format(time.time() - start_time))

	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_images.shape, train_images.dtype, np.min(train_images), np.max(train_images)))
	print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_labels.shape, train_labels.dtype, np.min(train_labels), np.max(train_labels)))
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_images.shape, test_images.dtype, np.min(test_images), np.max(test_images)))
	print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_labels.shape, test_labels.dtype, np.min(test_labels), np.max(test_labels)))

	#--------------------
	input_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel])
	output_ph = tf.placeholder(tf.float32, shape=[None, num_classes])

	use_reinitializable_iterator = False
	if not use_reinitializable_iterator:
		# Use an initializable iterator.
		dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph)).batch(BATCH_SIZE)

		iter = dataset.make_initializable_iterator()
	else:
		# Use a reinitializable iterator.
		train_dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph)).batch(BATCH_SIZE)
		test_dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph)).batch(BATCH_SIZE)

		iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

		train_init_op = iter.make_initializer(train_dataset)
		val_init_op = iter.make_initializer(test_dataset)
		test_init_op = iter.make_initializer(test_dataset)
	input_elem, output_elem = iter.get_next()

	#--------------------
	# Create a model.

	model_output = create_model(input_elem, num_classes)

	loss = get_loss(model_output, output_elem)
	accuracy = get_accuracy(model_output, output_elem)

	learning_rate = 0.001
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
	#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)

	train_op = optimizer.minimize(loss)

	#--------------------
	# Train and evaluate.

	if True:
		saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				print('Epoch {}:'.format(epoch + 1))
				# Initialize iterator with train data.
				if not use_reinitializable_iterator:
					sess.run(iter.initializer, feed_dict={input_ph: train_images, output_ph: train_labels})
				else:
					sess.run(train_init_op, feed_dict={input_ph: train_images, output_ph: train_labels})
				start_time = time.time()
				train_loss, train_accuracy = 0, 0
				while True:
					try:
						#_, loss_value, accuracy_value = sess.run([train_op, loss, accuracy])
						_, loss_value, accuracy_value, elem_value = sess.run([train_op, loss, accuracy, input_elem])
						train_loss += loss_value * elem_value.shape[0]
						train_accuracy += accuracy_value * elem_value.shape[0]
					except tf.errors.OutOfRangeError:
						break
				train_loss /= train_images.shape[0]
				train_accuracy /= train_images.shape[0]
				print('\tTrain loss = {:.6f}, Train accuracy = {:.6f}: {} secs.'.format(train_loss, train_accuracy, time.time() - start_time))

				# Switch to validation data.
				if not use_reinitializable_iterator:
					sess.run(iter.initializer, feed_dict={input_ph: test_images, output_ph: test_labels})
				else:
					sess.run(val_init_op, feed_dict={input_ph: test_images, output_ph: test_labels})
				start_time = time.time()
				val_loss, val_accuracy = 0, 0
				while True:
					try:
						#loss_value, accuracy_value = sess.run([loss, accuracy])
						loss_value, accuracy_value, elem_value = sess.run([loss, accuracy, input_elem])
						val_loss += loss_value * elem_value.shape[0]
						val_accuracy += accuracy_value * elem_value.shape[0]
					except tf.errors.OutOfRangeError:
						break
				val_loss /= test_images.shape[0]
				val_accuracy /= test_images.shape[0]
				print('\tVal loss   = {:.6f}, Val accuracy   = {:.6f}: {} secs.'.format(val_loss, val_accuracy, time.time() - start_time))

			# Save a model.
			saved_model_path = saver.save(sess, checkpoint_dir_path + '/model.ckpt')

	#--------------------
	# Infer.

	with tf.Session() as sess:
		# Load a model.
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))

		# Switch to test data.
		if not use_reinitializable_iterator:
			sess.run(iter.initializer, feed_dict={input_ph: test_images, output_ph: test_labels})
			#sess.run(iter.initializer, feed_dict={input_ph: test_images})  # Error.
		else:
			sess.run(test_init_op, feed_dict={input_ph: test_images, output_ph: test_labels})
			#sess.run(test_init_op, feed_dict={input_ph: test_images})  # Error.
		start_time = time.time()
		inferences = list()
		while True:
			try:
				inferences.append(sess.run(model_output))
			except tf.errors.OutOfRangeError:
				break
		print('Inference time: {} secs.'.format(time.time() - start_time))

		inferences = np.vstack(inferences)
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
			print('Accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
