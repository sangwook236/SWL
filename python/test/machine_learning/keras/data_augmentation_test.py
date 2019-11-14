# REF [paper] >> "Densely Connected Convolutional Networks", arXiv 2016.
# REF [paper] >> "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation", arXiv 2016.
# REF [site] >> https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation

#--------------------------------------------------------------------

import os
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
os.chdir(swl_python_home_dir_path + '/test/machine_learning/keras')

import sys
sys.path.append('../../../src')

#--------------------------------------------------------------------

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from swl.machine_learning.keras.preprocessing import ImageDataGeneratorWithCrop

#--------------------------------------------------------------------

# REF [site] >> https://keras.io/preprocessing/image/

#train_datagen = ImageDataGenerator(
#	rescale=1./255,
#	shear_range=0.2,
#	zoom_range=0.2,
#	horizontal_flip=True)
#
#test_datagen = ImageDataGenerator(rescale=1./255)
#
#train_generator = train_datagen.flow_from_directory(
#	data_dir_path,
#	target_size=(150, 150),
#	batch_size=32,
#	class_mode='binary')
#
#validation_generator = test_datagen.flow_from_directory(
#	data_dir_path,
#	target_size=(150, 150),
#	batch_size=32,
#	class_mode='binary')
#
#model.fit_generator(
#	train_generator,
#	steps_per_epoch=2000,
#	epochs=50,
#	validation_data=validation_generator,
#	validation_steps=800)


#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Example 1.
#	Labels are given to each image.
# REF [fie] >> ${KERAS_HOME}/examples/cifar10_cnn.py

#num_examples = x_train.shape[0]
num_classes = 10
batch_size = 37  # Number of samples per gradient update.
num_epochs = 1  # Number of times to iterate over the training data arrays.
#steps_per_epoch = num_examples // batch_size
steps_per_epoch = 1

# The data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Initiate RMSprop optimizer.
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop.
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# NOTICE [info] >>
#	The generator output: (data, labels) -> tuple.
#	The shape of the generator output:
#		data: (batch_size, heigth, width, channel) = (37, 32, 32, 3).
#		labels: (batch_size, num_classes) = (37, 10).
#	The range of the generator output:
#		data: [0.0, 1.0] -> float32.
#		labels: one-hot encoding.
datagen = ImageDataGenerator(
	rescale=1./255.,
	preprocessing_function=None,
	featurewise_center=True,  # Set input mean to 0 over the dataset.
	featurewise_std_normalization=True,  # Divide inputs by std of the dataset.
	samplewise_center=False,  # Set each sample mean to 0.
	samplewise_std_normalization=False,  # Divide each input by its std.
	zca_whitening=False,  # Apply ZCA whitening.
	zca_epsilon=1e-6,
	rotation_range=0,  # Randomly rotate images in the range (degrees, 0 to 180).
	width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width).
	height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height).
	horizontal_flip=True,  # Randomly flip images.
	vertical_flip=False,  # Randomly flip images.
	zoom_range=0.2,
	#shear_range=0.,
	#channel_shift_range=0.,
	fill_mode='nearest',
	cval=0.)

# Compute quantities required for feature-wise normalization (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Train the model on the batches generated by datagen.flow().
model.fit_generator(
	datagen.flow(x_train, y_train, batch_size=batch_size),
	steps_per_epoch=steps_per_epoch,
	epochs=num_epochs,
	validation_data=(x_test, y_test))


#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Example 2.
#	Images are only given.
#	Labels are given to each image.
#	Labels are automatically inferred from the subdirectory names/structure.
# REF [site] >> https://keras.io/preprocessing/image/

# All images are of the same size, 300x200.
data_dir_path = '../../../data/machine_learning/data_only1'
# All images have different sizes.
#data_dir_path = '../../../data/machine_learning/data_only2'

data_save_dir_path = '../../../data/machine_learning/generated/data_only'

# Parameters.
class_labels = ['forest', 'fruit', 'house', 'street']
num_classes = len(class_labels)
num_examples = 9
batch_size = 6  # Number of samples per gradient update.
num_epochs = 1  # Number of times to iterate over the training data arrays.
#steps_per_epoch = num_examples / batch_size
steps_per_epoch = 1

resized_input_size = (200, 300)  # (height, width).
cropped_input_size = (100, 100)  # (height, width).
crop_dataset_flag = False
save_to_dir_flag = True

# Create a data generator.
if crop_dataset_flag:
	# REF [site] >> https://github.com/fchollet/keras/issues/3338

	data_gen_with_crop_args = dict(
		rescale=1./255.,
		preprocessing_function=None,
		featurewise_center=False,
		featurewise_std_normalization=False,
		samplewise_center=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		zca_epsilon=1e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=0.2,
		#shear_range=0.,
		#channel_shift_range=0.,
		random_crop_size=cropped_input_size,
		center_crop_size=None,
		fill_mode='constant',
		cval=0.)
	train_data_generator = ImageDataGeneratorWithCrop(**data_gen_with_crop_args)
else:
	data_gen_args = dict(
		rescale=1./255.,
		preprocessing_function=None,
		featurewise_center=False,
		featurewise_std_normalization=False,
		samplewise_center=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		zca_epsilon=1e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=0.2,
		#shear_range=0.,
		#channel_shift_range=0.,
		fill_mode='constant',
		cval=0.)
	train_data_generator = ImageDataGenerator(**data_gen_args)

# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
#data_generator.fit(dataset.data, augment=True)

# NOTICE [info] >>
#	The generator output: (data, labels) -> tuple.
#	The shape of the generator output:
#		data: (batch_size, heigth, width, channel) = (6, 200, 300, 3).
#		labels: (batch_size, num_classes) = (6, 4).
#	The range of the generator output:
#		data: [0.0, 1.0] -> float32.
#		labels: one-hot encoding.
train_data_gen = train_data_generator.flow_from_directory(
	data_dir_path,
	target_size=resized_input_size,
	color_mode='rgb',
	classes=class_labels,
	class_mode='categorical',  # (batch_size, num_classes).
	#class_mode='sparse',  # (batch_size,).
	batch_size=batch_size,
	shuffle=True,
	save_to_dir=(data_save_dir_path if save_to_dir_flag else None))

# Create a model.
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(cropped_input_size + (3,) if crop_dataset_flag else resized_input_size + (3,))))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on batches with real-time data augmentation.
model.fit_generator(
	train_data_gen,
	#validation_data=validation_data_gen,
	#validation_steps=800,
	steps_per_epoch=steps_per_epoch,
	epochs=num_epochs)


#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Example 3.
#	The pair of images & label(mask) images is given.
#	Labels are given to each pixel.
# REF [site] >> https://keras.io/preprocessing/image/

# All images are of the same size, 2048x1024.
data_dir_path = '../../../data/machine_learning/data_label/data'
label_dir_path = '../../../data/machine_learning/data_label/label'

data_save_dir_path = '../../../data/machine_learning/generated/data'
label_save_dir_path = '../../../data/machine_learning/generated/label'

# Parameters.
# NOTICE [caution] >>
#	['forest', 'fruit', 'house', 'street'] may not be corret. To be exact, the number of labels equals the number of object classes in images.
#class_labels = ???
#num_classes = len(class_labels)
num_classes = 34
num_examples = 10
batch_size = 4  # Number of samples per gradient update.
num_epochs = 1  # Number of times to iterate over the training data arrays.
#steps_per_epoch = num_examples / batch_size
steps_per_epoch = 1

resized_input_size = (1024, 2048)  # (height, width).
cropped_input_size = (300, 300)  # (height, width).
crop_dataset_flag = False
save_to_dir_flag = True

# Create a data generator.
if crop_dataset_flag:
	# REF [site] >> https://github.com/fchollet/keras/issues/3338

	data_gen_with_crop_args = dict(
		#rescale=1./255.,
		preprocessing_function=None,
		featurewise_center=False,
		featurewise_std_normalization=False,
		samplewise_center=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		zca_epsilon=1e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=False,
		zoom_range=0.2,
		#shear_range=0.,
		#channel_shift_range=0.,
		random_crop_size=cropped_input_size,
		center_crop_size=None,
		fill_mode='reflect',
		cval=0.)
	data_generator = ImageDataGeneratorWithCrop(rescale=1./255., **data_gen_with_crop_args)
	label_generator = ImageDataGeneratorWithCrop(rescale=None, **data_gen_with_crop_args)  # Need no scaling.
else:
	data_gen_args = dict(
		#rescale=1./255.,
		preprocessing_function=None,
		featurewise_center=False,
		featurewise_std_normalization=False,
		samplewise_center=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		zca_epsilon=1e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=False,
		zoom_range=0.2,
		#shear_range=0.,
		#channel_shift_range=0.,
		fill_mode='reflect',
		cval=0.)
	data_generator = ImageDataGenerator(rescale=1./255., **data_gen_args)
	label_generator = ImageDataGenerator(rescale=None, **data_gen_args)  # Need no scaling.

# NOTICE [caustion] >> Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
#data_generator.fit(dataset.data, augment=True, seed=seed)
#label_generator.fit(dataset.labels, augment=True, seed=seed)

data_gen = data_generator.flow_from_directory(
	data_dir_path,
	target_size=resized_input_size,
	color_mode='rgb',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=batch_size,
	shuffle=True,
	seed=seed,
	save_to_dir=(data_save_dir_path if save_to_dir_flag else None))
label_gen = label_generator.flow_from_directory(
	label_dir_path,
	target_size=resized_input_size,
	color_mode='grayscale',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=batch_size,
	shuffle=True,
	seed=seed,
	save_to_dir=(label_save_dir_path if save_to_dir_flag else None))

# Combine generators into one which yields image and labels.
# NOTICE [info] >>
#	The generator output: (data, labels) -> tuple.
#	The shape of the generator output:
#		data: (batch_size, heigth, width, channel) = (4, 200, 300, 3).
#		labels: (batch_size, heigth, width, channel) = (4, 200, 300, 1).
#	The range of the generator output:
#		data: [0.0, 1.0] -> float32.
#		labels: [0.0, 255.0] -> float32.
# TODO [convert] >>
#	Labels must be converted from grayvalues to class labels (if possibly, one-hot encoding).
#		(batch_size, heigth, width, channel=1) -> (batch_size, heigth, width, num_classes)
dataset_gen = zip(data_gen, label_gen)

# Create a model.
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(cropped_input_size + (3,) if crop_dataset_flag else resized_input_size + (3,))))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
final_block_layer_shape = model.get_layer(index=-1).output_shape
model.add(Conv2D(1024, (final_block_layer_shape[1], final_block_layer_shape[2]), activation='relu', padding='valid'))
model.add(Dense(final_block_layer_shape[1] * final_block_layer_shape[2] * 512, activation='relu'))
model.add(Reshape((final_block_layer_shape[1:3] + (512,))))
model.add(UpSampling2D((2, 2)))
model.add(Conv2DTranspose(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(512, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(num_classes, (1, 1), activation='softmax'))
#model.add(Conv2D(1, (1, 1), activation='linear'))

model.summary()
#print('model output shape =', model.output_shape)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on batches with real-time data augmentation.

# Method 1: Not correctly working.
# NOTICE [important] >>
#   The dimension of the 2nd item of the generator output here is different from that of the 2nd example described above.
#model.fit_generator(
#	dataset_gen,
#	#validation_data=validation_dataset_generator,
#	#validation_steps=800,
#	steps_per_epoch=steps_per_epoch,
#	epochs=num_epochs)

# Method 2: Not correctly working.
for epoch in range(num_epochs):
	print('Epoch %d/%d' % (epoch + 1, num_epochs))
	steps = 0
	# NOTICE [error] >> 'zip' object has no attribute 'flow'.
	#for data_batch, label_batch in dataset_gen.flow(x_train, y_train, batch_size=batch_size):
	for data_batch, label_batch in dataset_gen:
		model.fit(data_batch, keras.utils.to_categorical(label_batch.astype(np.int8)).reshape(label_batch.shape[:-1] + (-1,)), batch_size=batch_size, epochs=1, verbose=0)
		#model.fit(data_batch, keras.utils.to_categorical(label_batch).reshape(label_batch.shape[:-1] + (-1,)), batch_size=batch_size, epochs=1, verbose=0)
		steps += 1
		if steps >= steps_per_epoch:
			break

# Method 3: Not working.
# REF [site] >> https://keras.io/models/model/
#def generate_dataset_from_generators(data_gen, label_gen, steps_per_epoch):
#    for _ in range(steps_per_epoch):
#        #yield ({'input_1': data_gen}, {'output': label_gen})
#        yield (data_gen.next(), label_gen.next())
#
#model.fit_generator(
#	generate_dataset_from_generators(data_gen, label_gen, steps_per_epoch),
#	#validation_data=validation_dataset_generator,
#	#validation_steps=800,
#	steps_per_epoch=steps_per_epoch,
#	epochs=num_epochs)
