#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import cv2

#--------------------------------------------------------------------

def text_dataset_to_numpy(dataset_json_filepath, image_height, image_width, image_channel, eos_token):
	with open(dataset_json_filepath, 'r', encoding='UTF8') as json_file:
		dataset = json.load(json_file)

	"""
	print(dataset['charset'])
	for datum in dataset['data']:
		print('file =', datum['file'])
		print('size =', datum['size'])
		print('text =', datum['text'])
		print('char IDs =', datum['char_id'])
	"""

	num_examples = len(dataset['data'])
	max_height, max_width, max_channel, max_label_len = 0, 0, 0, 0
	for datum in dataset['data']:
		sz = datum['size']
		if len(sz) != 3:
			print('[Warning] Invalid data size: {}.'.format(datum['file']))
			continue

		if sz[0] > max_height:
			max_height = sz[0]
		if sz[1] > max_width:
			max_width = sz[1]
		if sz[2] > max_channel:
			max_channel = sz[2]
		if len(datum['char_id']) > max_label_len:
			max_label_len = len(datum['char_id'])

	if 0 == max_height or 0 == max_width or 0 == max_channel or 0 == max_label_len:
		raise ValueError('[Error] Invalid dataset size')

	#data = np.zeros((num_examples, max_height, max_width, max_channel))
	data = np.zeros((num_examples, image_height, image_width, image_channel))
	#labels = np.zeros((num_examples, max_label_len))
	labels = np.full((num_examples, max_label_len), eos_token)
	for idx, datum in enumerate(dataset['data']):
		img = cv2.imread(datum['file'], cv2.IMREAD_GRAYSCALE)
		sz = datum['size']
		if sz[0] != image_height or sz[1] != image_width:
			img = cv2.resize(img, (image_width, image_height))
		#data[idx,:sz[0],:sz[1],:sz[2]] = img.reshape(img.shape + (-1,))
		data[idx,:,:,0] = img
		labels[idx,:len(datum['char_id'])] = datum['char_id']

	return data, labels

def load_data(image_height, image_width, image_channel):
	label_eos_token = -1

	print('Start loading train dataset to numpy...')
	start_time = time.time()
	train_data, train_labels = text_dataset_to_numpy('./text_train_dataset_tmp/text_dataset.json', image_height, image_width, image_channel, eos_token=label_eos_token)
	print('End loading train dataset: {} secs.'.format(time.time() - start_time))
	print('Start loading test dataset to numpy...')
	start_time = time.time()
	test_data, test_labels = text_dataset_to_numpy('./text_test_dataset_tmp/text_dataset.json', image_height, image_width, image_channel, eos_token=label_eos_token)
	print('End loading test dataset: {} secs.'.format(time.time() - start_time))

	# Preprocessing.
	train_data = (train_data.astype(np.float32) / 255.0) * 2 - 1  # [-1, 1].
	#train_labels = tf.keras.utils.to_categorical(train_labels).astype(np.int32)
	train_labels = train_labels.astype(np.int32)
	test_data = (test_data.astype(np.float32) / 255.0) * 2 - 1  # [-1, 1].
	#test_labels = tf.keras.utils.to_categorical(test_labels).astype(np.int32)
	test_labels = test_labels.astype(np.int32)

	# (samples, height, width, channels) -> (samples, width, height, channels).
	train_data = train_data.transpose((0, 2, 1, 3))
	test_data = test_data.transpose((0, 2, 1, 3))

	return train_data, train_labels, test_data, test_labels

#--------------------------------------------------------------------

def create_model(input_tensor, num_classes):
	#inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, width, height, 1)

	# Convolution layer (VGG).
	inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_tensor)  # (None, width, height, 64)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None, width/2, height/2, 64)

	inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, width/2, height/2, 128)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, width/4, height/4, 128)

	inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, width/4, height/4, 256)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, width/4, height/4, 256)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, width/4, height/8, 256)

	inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, width/4, height/8, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, width/4, height/8, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, width/4, height/16, 512)

	inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, width/4, height/16, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)

	# CNN to RNN.
	rnn_input_shape = inner.shape #inner.shape.as_list()
	inner = Reshape(target_shape=((rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3])), name='reshape')(inner)  # (None, width/4, height/16 * 512)
	if True:
		inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256)

		# RNN layer.
		lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
		lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
		lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 512)
		lstm1_merged = BatchNormalization()(lstm1_merged)
		lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
		lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
		lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 512)
		lstm2_merged = BatchNormalization()(lstm2_merged)
	elif False:
		inner = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256)

		# RNN layer.
		lstm_1 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
		lstm_1b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
		lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 1024)
		lstm1_merged = BatchNormalization()(lstm1_merged)
		lstm_2 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
		lstm_2b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
		lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 1024)
		lstm2_merged = BatchNormalization()(lstm2_merged)
	elif False:
		inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256)

		# RNN layer.
		lstm_1 = LSTM(1024, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
		lstm_1b = LSTM(1024, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
		lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 2048)
		lstm1_merged = BatchNormalization()(lstm1_merged)
		lstm_2 = LSTM(1024, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
		lstm_2b = LSTM(1024, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
		lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 2048)
		lstm2_merged = BatchNormalization()(lstm2_merged)  # NOTE [check] >> Different from the original implementation.

	# Transforms RNN output to character activations.
	inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, width/4, num_classes)
	y_pred = Activation('softmax', name='softmax')(inner)

	return y_pred

def get_loss(y, t, y_length, t_length):
	with tf.name_scope('loss'):
		# The 2 is critical here since the first couple outputs of the RNN tend to be garbage.
		y = y[:, 2:, :]

		# Connectionist temporal classification (CTC) loss.
		loss = K.ctc_batch_cost(t, y, y_length, t_length)  # Output shape: [batch_size, 1].
		loss = tf.reduce_mean(loss)

		return loss

#--------------------------------------------------------------------

def main():
	image_height, image_width, image_channel = 64, 320, 1
	#num_classes = 2350
	num_classes = 2350 + 1  # Includes EOS token.
	BATCH_SIZE, NUM_EPOCHS = 128, 1000

	checkpoint_dir_path = './tf_checkpoint'
	try:
		os.makedirs(checkpoint_dir_path)
	except FileExistsError:
		pass

	#--------------------
	# Load data.

	train_images, train_labels, test_images, test_labels = load_data(image_height, image_width, image_channel)
	train_label_lengths, test_label_lengths = np.full((train_labels.shape[0], 1), train_labels.shape[-1]), np.full((test_labels.shape[0], 1), test_labels.shape[-1])
	# FIXME [improve] >> Stupid implementation.
	#model_output_lengths = np.full((train_images.shape[0], 1), image_width / 4)
	model_output_lengths = np.full((train_images.shape[0], 1), image_width / 4 - 2)  # See get_loss().

	max_output_len = max(train_labels.shape[-1], test_labels.shape[-1])

	print("Train image's shape = {}, train label's shape = {}.".format(train_images.shape, train_labels.shape))
	print("Test image's shape = {}, test label's shape = {}.".format(test_images.shape, test_labels.shape))

	#--------------------

	# (samples, height, width, channels) -> (samples, width, height, channels).
	#input_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_ph')  # NOTE [caution] >> (?, image_height, image_width, ?)
	input_ph = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='input_ph')  # NOTE [caution] >> (?, image_width, image_height, ?)
	output_ph = tf.placeholder(tf.float32, shape=[None, max_output_len], name='output_ph')
	output_length_ph = tf.placeholder(tf.int32, shape=[None, 1], name='output_length_ph')
	model_output_length_ph = tf.placeholder(tf.int32, shape=[None, 1], name='model_output_length_ph')

	dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph, output_length_ph, model_output_length_ph)).batch(BATCH_SIZE)

	iter = dataset.make_initializable_iterator()
	input_elem, output_elem, output_length_elem, model_output_length_elem = iter.get_next()

	#--------------------
	# Create a model.

	model_output = create_model(input_elem, num_classes)

	loss = get_loss(model_output, output_elem, model_output_length_elem, output_length_elem)

	learning_rate = 0.001
	optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08)

	train_op = optimizer.minimize(loss)

	#--------------------
	# Train.

	if True:
		saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				print('Epoch {}:'.format(epoch + 1))
				start_time = time.time()
				# Initialise iterator with train data.
				sess.run(iter.initializer, feed_dict={input_ph: train_images, output_ph: train_labels, output_length_ph: train_label_lengths, model_output_length_ph: model_output_lengths})
				train_loss = 0
				batch_idx = 0
				while True:
					try:
						#_, loss_value = sess.run([train_op, loss])
						_, loss_value, elem_value = sess.run([train_op, loss, input_elem])
						train_loss += loss_value * elem_value.shape[0]
						if (batch_idx + 1) % 100 == 0:
							print('\tBatch = {}: {:.6f} secs.'.format(batch_idx + 1, time.time() - start_time))
					except tf.errors.OutOfRangeError:
						break
					batch_idx += 1
				train_loss /= train_images.shape[0]
				print('\tLoss = {:.4f}: {:.6f} secs.'.format(train_loss, time.time() - start_time))

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
		sess.run(iter.initializer, feed_dict={input_ph: test_images, output_ph: test_labels, output_length_ph: test_label_lengths, model_output_length_ph: model_output_lengths})
		#sess.run(iter.initializer, feed_dict={input_ph: test_images})  # Error.
		start_time = time.time()
		inferences = list()
		while True:
			try:
				inferences.append(sess.run(model_output))
			except tf.errors.OutOfRangeError:
				break
		print('Inference time: {:.6f} secs.'.format(time.time() - start_time))

		inferences = np.vstack(inferences)
		if inferences is not None:
			# TODO [implement] >>
			pass
		else:
			print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
