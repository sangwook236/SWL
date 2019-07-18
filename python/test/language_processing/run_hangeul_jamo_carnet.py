#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time, random
import numpy as np
import tensorflow as tf
import cv2
import text_generation_util as tg_util
import hangeul_util as hg_util

#--------------------------------------------------------------------

# REF [function] >> generate_text_lines() in text_generation_util.py.
def generate_text_lines(word_set, textGenerator, font_size_interval, char_space_ratio_interval, image_height, image_width, image_channel, batch_size, font_color=None, bg_color=None, eoc_str='<EOC>'):
	sceneTextGenerator = tg_util.MySceneTextGenerator(tg_util.IdentityTransformer())

	scene_list, scene_text_mask_list, text_list = list(), list(), list()
	step = 0
	while True:
		font_size = random.randint(*font_size_interval)
		char_space_ratio = random.uniform(*char_space_ratio_interval)

		text = random.sample(word_set, 1)[0]

		char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space_ratio, font_size)
		text_line, text_line_alpha = tg_util.MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

		if bg_color is None:
			# Grayscale background.
			bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
		else:
			bg = np.full_like(text_line, bg_color, dtype=np.uint8)

		scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
		scene = cv2.resize(scene, (image_height, image_width), interpolation=cv2.INTER_AREA)
		#scene_text_mask = cv2.resize(scene_text_mask, (image_height, image_width), interpolation=cv2.INTER_AREA)
		scene_list.append(scene)
		#scene_text_mask_list.append(scene_text_mask)
		if True:
			text = hg_util.hangeul2jamo(text, eoc_str, use_separate_consonants=False, use_separate_vowels=True)  # Hangeul letters -> Hangeul jamos.
		text_list.append(text)

		step += 1
		if 0 == (idx + 1) % batch_size:
			#yield scene_list, scene_text_mask_list, text_list
			yield scene_list, text_list
			scene_list, scene_text_mask_list, text_list = list(), list(), list()
			step = 0

def create_text_line_generator(image_height, image_width, image_channel, min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color=None, bg_color=None):
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#data = fd.readlines()  # A string.
		#data = fd.read().strip('\n')  # A list of strings.
		#data = fd.read().splitlines()  # A list of strings.
		data = fd.read().replace(' ', '').replace('\n', '')  # A string.
	count = 80
	hangeul_charset = str()
	for idx in range(0, len(data), count):
		txt = ''.join(data[idx:idx+count])
		#hangeul_charset += ('' if 0 == idx else '\n') + txt
		hangeul_charset += txt
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	#print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
	#print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
	#print('Digit charset =', len(digit_charset), digit_charset)
	#print('Symbol charset =', len(symbol_charset), symbol_charset)

	num_char_repetitions = 2000
	min_char_count, max_char_count = 5, 5
	word_set = tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset, min_char_count, max_char_count)
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, alphabet_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, digit_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + digit_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + symbol_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, alphabet_charset + digit_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, alphabet_charset + symbol_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + alphabet_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + alphabet_charset + digit_charset, min_char_count, max_char_count))
	#word_set = word_set.union(tg_util.generate_repetitive_word_set(num_char_repetitions, hangeul_charset + alphabet_charset + symbol_charset, min_char_count, max_char_count))

	#--------------------
	characterTransformer = tg_util.IdentityTransformer()
	#characterTransformer = tg_util.RotationTransformer(-30, 30)
	#characterTransformer = tg_util.ImgaugAffineTransformer()
	characterAlphaMattePositioner = tg_util.MyCharacterAlphaMattePositioner()
	textGenerator = tg_util.MySimplePrintedHangeulTextGenerator(characterTransformer, characterAlphaMattePositioner)

	#--------------------
	batch_size = 30
	generator = generate_text_lines(word_set, textGenerator, (min_font_size, max_font_size), (min_char_space_ratio, max_char_space_ratio), image_height, image_width, image_channel, batch_size, font_color, bg_color)

	return generator

def load_data(image_height, image_width, image_channel, num_classes, eos_token, blank_label):
	min_font_size, max_font_size = 32, 32
	min_char_space_ratio, max_char_space_ratio = 0.8, 1.25

	#font_color = (255, 255, 255)
	#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
	font_color = None  # Uses random font colors.
	#bg_color = (0, 0, 0)
	bg_color = None  # Uses random colors.

	print('Start creating a train text line generator...')
	start_time = time.time()
	train_generator = create_text_line_generator(min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color, bg_color)
	print('End creating a train text line generator: {} secs.'.format(time.time() - start_time))
	print('Start creating a test text line generator...')
	start_time = time.time()
	test_generator = create_text_line_generator(min_font_size, max_font_size, min_char_space_ratio, max_char_space_ratio, font_color, bg_color)
	print('End creating a test text line generator: {} secs.'.format(time.time() - start_time))

	step = 1
	for scene_list, jamo_text_list in train_generator:
		for scene, jamo_text in zip(scene_list, jamo_text_list):
			if 'posix' == os.name:
				cv2.imwrite('./scene.png', scene)
				cv2.imwrite('./scene_text_mask.png', scene_text_mask)
			else:
				#scene_text_mask[scene_text_mask > 0] = 255
				#scene_text_mask = scene_text_mask.astype(np.uint8)
				minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
				#scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)

				print('Jamo text =', jamo_text)
				cv2.imshow('Scene', scene)
				#cv2.imshow('Scene Mask', scene_text_mask)
				cv2.waitKey(0)

		if step >= 3:
			break
		step += 1
	raise ValueError

	# Preprocessing.
	train_data = (train_data.astype(np.float32) / 255.0) * 2 - 1  # [-1, 1].
	#train_labels = tf.keras.utils.to_categorical(train_labels, num_classes, np.int16)
	train_labels = train_labels.astype(np.int16)
	test_data = (test_data.astype(np.float32) / 255.0) * 2 - 1  # [-1, 1].
	#test_labels = tf.keras.utils.to_categorical(test_labels, num_classes, np.int16)
	test_labels = test_labels.astype(np.int16)

	# (samples, height, width, channels) -> (samples, width, height, channels).
	train_data = train_data.transpose((0, 2, 1, 3))
	test_data = test_data.transpose((0, 2, 1, 3))

	return train_data, train_labels, test_data, test_labels

#--------------------------------------------------------------------

def create_model(input_tensor, num_classes):
	#inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, width, height, 1).

	# Convolution layer (VGG).
	inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_tensor)  # (None, width, height, 64).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None, width/2, height/2, 64).

	inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, width/2, height/2, 128).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, width/4, height/4, 128).

	inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, width/4, height/4, 256).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, width/4, height/4, 256).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, width/4, height/8, 256).

	inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, width/4, height/8, 512).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, width/4, height/8, 512).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, width/4, height/16, 512).

	inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, width/4, height/16, 512).
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)

	# CNN to RNN.
	rnn_input_shape = inner.shape #inner.shape.as_list()
	inner = Reshape(target_shape=((rnn_input_shape[1], rnn_input_shape[2] * rnn_input_shape[3])), name='reshape')(inner)  # (None, width/4, height/16 * 512).
	if True:
		inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256).

		# RNN layer.
		lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
		lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
		lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 512).
		lstm1_merged = BatchNormalization()(lstm1_merged)
		lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
		lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
		lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 512).
		lstm2_merged = BatchNormalization()(lstm2_merged)
	elif False:
		inner = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256).

		# RNN layer.
		lstm_1 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
		lstm_1b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
		lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 1024).
		lstm1_merged = BatchNormalization()(lstm1_merged)
		lstm_2 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
		lstm_2b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
		lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 1024).
		lstm2_merged = BatchNormalization()(lstm2_merged)
	elif False:
		inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, width/4, 256).

		# RNN layer.
		lstm_1 = LSTM(1024, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
		lstm_1b = LSTM(1024, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
		lstm1_merged = add([lstm_1, lstm_1b])  # (None, width/4, 2048).
		lstm1_merged = BatchNormalization()(lstm1_merged)
		lstm_2 = LSTM(1024, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
		lstm_2b = LSTM(1024, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
		lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, width/4, 2048).
		lstm2_merged = BatchNormalization()(lstm2_merged)  # NOTE [check] >> Different from the original implementation.

	# Transforms RNN output to character activations.
	inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, width/4, num_classes).
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
	num_labels = 2350
	if False:
		num_classes = num_labels
	elif False:
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
		num_classes = num_labels + 1  # #labels + blank label.
		blank_label = num_classes - 1
	else:
		# NOTE [info] >> The largest value (num_classes - 1) is reserved for the blank label in case of tf.nn.ctc_loss().
		num_classes = num_labels + 1 + 1  # #labels + EOS + blank label.
		eos_token_label = num_classes - 2
		blank_label = num_classes - 1
	width_downsample_factor = 4  # Fixed.

	BATCH_SIZE, NUM_EPOCHS = 128, 1000

	checkpoint_dir_path = './tf_checkpoint'
	os.makedirs(checkpoint_dir_path, exist_ok=True)

	#%%------------------------------------------------------------------
	# Load data.

	print('Start loading dataset...')
	start_time = time.time()
	train_images, train_labels, test_images, test_labels = load_data(image_height, image_width, image_channel, num_classes, eos_token_label, blank_label)
	print('End loading dataset: {} secs.'.format(time.time() - start_time))

	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_images.shape, train_images.dtype, np.min(train_images), np.max(train_images)))
	print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_labels.shape, train_labels.dtype, np.min(train_labels), np.max(train_labels)))
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_images.shape, test_images.dtype, np.min(test_images), np.max(test_images)))
	print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_labels.shape, test_labels.dtype, np.min(test_labels), np.max(test_labels)))

	train_label_lengths, test_label_lengths = np.full((train_labels.shape[0], 1), train_labels.shape[-1]), np.full((test_labels.shape[0], 1), test_labels.shape[-1])

	# FIXME [improve] >> Stupid implementation.
	if False:
		train_model_output_lengths = np.full((train_images.shape[0], 1), image_width // width_downsample_factor)
		test_model_output_lengths = np.full((test_images.shape[0], 1), image_width // width_downsample_factor)
	else:
		train_model_output_lengths = np.full((train_images.shape[0], 1), image_width // width_downsample_factor - 2)  # See get_loss().
		test_model_output_lengths = np.full((test_images.shape[0], 1), image_width // width_downsample_factor - 2)  # See get_loss().

	max_output_len = max(train_labels.shape[-1], test_labels.shape[-1])

	#--------------------

	# (samples, height, width, channels) -> (samples, width, height, channels).
	#input_ph = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel], name='input_ph')  # NOTE [caution] >> (?, image_height, image_width, ?)
	input_ph = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channel], name='input_ph')  # NOTE [caution] >> (?, image_width, image_height, ?)
	output_ph = tf.placeholder(tf.float32, shape=[None, max_output_len], name='output_ph')
	output_length_ph = tf.placeholder(tf.int32, shape=[None, 1], name='output_length_ph')
	model_output_length_ph = tf.placeholder(tf.int32, shape=[None, 1], name='model_output_length_ph')

	use_reinitializable_iterator = False
	if not use_reinitializable_iterator:
		# Use an initializable iterator.
		dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph, output_length_ph, model_output_length_ph)).batch(BATCH_SIZE)

		iter = dataset.make_initializable_iterator()
	else:
		# Use a reinitializable iterator.
		train_dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph, output_length_ph, model_output_length_ph)).batch(BATCH_SIZE)
		test_dataset = tf.data.Dataset.from_tensor_slices((input_ph, output_ph, output_length_ph, model_output_length_ph)).batch(BATCH_SIZE)

		iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

		train_init_op = iter.make_initializer(train_dataset)
		test_init_op = iter.make_initializer(test_dataset)
	input_elem, output_elem, output_length_elem, model_output_length_elem = iter.get_next()

	#%%------------------------------------------------------------------
	# Create a model.

	model_output = create_model(input_elem, num_classes)

	#%%------------------------------------------------------------------
	# Train.

	if True:
		loss = get_loss(model_output, output_elem, model_output_length_elem, output_length_elem)

		learning_rate = 0.001
		#optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1.0e-6, momentum=0.0, nesterov=True)
		#optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon=1.0e-7, centered=False)
		#optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1.0e-7, amsgrad=False)
		#optimizer = tf.keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1.0e-7)
		optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1.0e-7, use_locking=False)

		train_op = optimizer.minimize(loss)

		saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

		#--------------------
		print('Start training...')
		start_total_time = time.time()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				print('Epoch {}:'.format(epoch + 1))

				#--------------------
				start_time = time.time()
				# Initialize iterator with train data.
				if not use_reinitializable_iterator:
					sess.run(iter.initializer, feed_dict={input_ph: train_images, output_ph: train_labels, output_length_ph: train_label_lengths, model_output_length_ph: train_model_output_lengths})
				else:
					sess.run(train_init_op, feed_dict={input_ph: train_images, output_ph: train_labels, output_length_ph: train_label_lengths, model_output_length_ph: train_model_output_lengths})
				train_loss = 0
				batch_idx = 0
				while True:
					try:
						#_, loss_value = sess.run([train_op, loss])
						_, loss_value, elem_value = sess.run([train_op, loss, input_elem])
						train_loss += loss_value * elem_value.shape[0]
						if (batch_idx + 1) % 100 == 0:
							print('\tBatch = {}: {} secs.'.format(batch_idx + 1, time.time() - start_time))
					except tf.errors.OutOfRangeError:
						break
					batch_idx += 1
				train_loss /= train_images.shape[0]
				print('\tTrain: loss = {:.6f}: {} secs.'.format(train_loss, time.time() - start_time))

				#--------------------
				print('Start saving a model...')
				start_time = time.time()
				saved_model_path = saver.save(sess, checkpoint_dir_path + '/model.ckpt')
				print('End saving a model: {} secs.'.format(time.time() - start_time))
		print('End training: {} secs.'.format(time.time() - start_total_time))

	#%%------------------------------------------------------------------
	# Infer.

	with tf.Session() as sess:
		print('Start loading a model...')
		start_time = time.time()
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))
		print('End loading a model: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('Start inferring...')
		start_time = time.time()
		# Switch to test data.
		if not use_reinitializable_iterator:
			sess.run(iter.initializer, feed_dict={input_ph: test_images, output_ph: test_labels, output_length_ph: test_label_lengths, model_output_length_ph: test_model_output_lengths})
			#sess.run(iter.initializer, feed_dict={input_ph: test_images})  # Error.
		else:
			sess.run(test_init_op, feed_dict={input_ph: test_images, output_ph: test_labels, output_length_ph: test_label_lengths, model_output_length_ph: test_model_output_lengths})
		start_time = time.time()
		inferences = list()
		while True:
			try:
				inferences.append(sess.run(model_output))
			except tf.errors.OutOfRangeError:
				break
		print('End inferring: {} secs.'.format(time.time() - start_time))

		inferences = np.vstack(inferences)
		if inferences is not None:
			print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))
			inferences = np.argmax(inferences, -1)

			print('**********', inferences[:10])
			print('**********', test_labels[:10])
		else:
			print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
