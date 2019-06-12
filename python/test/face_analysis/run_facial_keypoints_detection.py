#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, time, csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import cv2

# REF [function] >> data_loading_test() in facial_keypoints_detection_dataset.py
def load_facial_keypoints_detection_dataset():
	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/my_dataset'
	else:
		data_home_dir_path = 'E:/dataset'
	facial_keypoints_detection_dir_path = data_home_dir_path + '/pattern_recognition/kaggle/facial_keypoints_detection/facial-keypoints-detection'
	train_filepath = facial_keypoints_detection_dir_path + '/training.csv'
	test_filepath = facial_keypoints_detection_dir_path + '/test.csv'
	image_height, image_width = 96, 96

	def convert_coordinates(elem):
		try:
			return -1 if not elem else float(elem)
		except:
			return -1

	#--------------------
	print('Loading training data...')
	start_time = time.time()
	face_train_keypoints, face_train_images = list(), list()
	with open(train_filepath, newline='', encoding='UTF-8') as csvfile:
		# left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y,Image
		reader = csv.reader(csvfile, delimiter=',', quotechar=None)
		idx = 0
		for row in reader:
			if idx > 0:
				if 31 != len(row):
					print('Invalid length in the {}-th row: {}.'.format(idx, row))
				#keypoints = list(float(elem) for elem in row[:30] if not elem else -1)
				keypoints = list(map(convert_coordinates, row[:30]))
				if 30 != len(keypoints):
					print('Invalid keypoints in the {}-th row: {}.'.format(idx, keypoints))
				face_train_keypoints.append(np.array(keypoints, dtype=np.float32).reshape(15, 2))
				pixels = list(int(px) for px in row[30].split(' '))
				if image_height * image_width != len(pixels):
					print('Invalid pixels in the {}-th row: {}.'.format(idx, pixels))
				face_train_images.append(np.array(pixels, dtype=np.uint8).reshape(image_height, image_width))
			idx += 1
	print('\tElapsed time = {}'.format(time.time() - start_time))

	#--------------------
	print('Loading test data...')
	start_time = time.time()
	face_test_images = list()
	with open(test_filepath, newline='', encoding='UTF-8') as csvfile:
		# ImageId,Image
		reader = csv.reader(csvfile, delimiter=',', quotechar=None)
		idx = 0
		for row in reader:
			if idx > 0:
				if 2 != len(row):
					print('Invalid length in the {}-th row: {}.'.format(idx, row))
				id = int(row[0])
				pixels = list(int(px) for px in row[1].split(' '))
				if image_height * image_width != len(pixels):
					print('Invalid pixels in the {}-th row: {}.'.format(idx, pixels))
				face_test_images.append(np.array(pixels, dtype=np.uint8).reshape(image_height, image_width))
			idx += 1
	print('\tElapsed time = {}'.format(time.time() - start_time))

	X_train = np.array(face_train_images)
	X_train = X_train.reshape(X_train.shape + (-1,))
	y_train = np.array(face_train_keypoints).reshape(len(face_train_keypoints), -1)
	X_test = np.array(face_test_images)
	X_test = X_test.reshape(X_test.shape + (-1,))
	y_test = None

	return X_train, y_train, X_test, y_test

# REF [site] >> https://towardsdatascience.com/detecting-facial-features-using-deep-learning-2e23c8660a7a
def create_facial_keypoints_detection_model_1():
	model = Sequential()
	model.add(BatchNormalization(input_shape=(96, 96, 1)))
	model.add(Conv2D(24, (5, 5), padding='same', kernel_initializer='he_normal'))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Conv2D(36, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Conv2D(48, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(GlobalAveragePooling2D());
	model.add(Dense(500, activation='relu'))
	model.add(Dense(90, activation='relu'))
	model.add(Dense(30))

	return model

# REF [site] >> https://hackernoon.com/key-point-detection-in-flower-images-using-deep-learning-66a06aadc765
def create_facial_keypoints_detection_model_2():
	model = Sequential()
	model.add(Conv2D(64, (3, 3), padding='valid', kernel_initializer='he_normal', input_shape=(96, 96, 1)))
	model.add(Conv2D(64, (3, 3), padding='valid', kernel_initializer='he_normal'))
	model.add(Conv2D(64, (3, 3), padding='valid', kernel_initializer='he_normal'))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.5))
	model.add(Conv2D(128, (3, 3), padding='valid', kernel_initializer='he_normal'))
	model.add(Conv2D(128, (3, 3), padding='valid', kernel_initializer='he_normal'))
	model.add(Conv2D(128, (3, 3), padding='valid', kernel_initializer='he_normal'))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.5))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.5))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.5))
	model.add(Dense(30))

	return model

def train_facial_keypoints_detection_using_keras(X_train, y_train):
	model_filepath = './facial_keypoints_detection_model.h5'

	if True:
		# Better.
		model = create_facial_keypoints_detection_model_1()
	else:
		model = create_facial_keypoints_detection_model_2()

	#--------------------
	model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

	checkpoint_callback = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)
	num_epochs = 100
	batch_size = 20
	validation_split = 0.2
	shuffle = True
	history = model.fit(X_train, y_train, validation_split=validation_split, shuffle=shuffle, epochs=num_epochs, batch_size=batch_size, callbacks=[checkpoint_callback], verbose=1)

	#--------------------
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

def detect_facial_keypoints_using_keras(X_test, y_test):
	model_filepath = './facial_keypoints_detection_model.h5'

	#model.save(model_filepath)
	#json_string = model.to_json()
	#yaml_string = model.to_yaml()

	model = tf.keras.models.load_model(model_filepath)
	#model = tf.keras.models.model_from_json(json_string)
	#model = tf.keras.models.model_from_yaml(yaml_string)

	y_pred = model.predict(X_test, batch_size=1)
	y_pred = y_pred.reshape(y_pred.shape[0], -1, 2)

	print('Facial Keypoints Detection =', y_pred.shape)
	for img, keypts in zip(X_test, y_pred):
		rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		keypts = np.round(keypts).astype(np.int)
		for pt in keypts:
			cv2.circle(rgb, tuple(pt), 1, (0, 0, 255), 1, cv2.LINE_8)

		cv2.imshow('Facial Keypoints', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	#--------------------
	# Load data.
	X_train, y_train, X_test, y_test = load_facial_keypoints_detection_dataset()

	train_facial_keypoints_detection_using_keras(X_train, y_train)
	detect_facial_keypoints_using_keras(X_test, y_test)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
