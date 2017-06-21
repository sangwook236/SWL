# REF [paper] >> "Densely Connected Convolutional Networks", arXiv 2016.
# REF [paper] >> "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation", arXiv 2016.
# REF [site] >> https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation

#%%------------------------------------------------------------------

import os
os.chdir('D:/work/swl_github/python/test/machine_learning')

import sys
sys.path.insert(0, '../../src/machine_learning')

#%%------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from preprocessing import ImageDataGeneratorWithCrop

#%%------------------------------------------------------------------

data_dir_path = '../../data/machine_learning/dataset1/data'
label_dir_path = '../../data/machine_learning/dataset1/label'
#data_dir_path = '../../data/machine_learning/dataset2/data'
#label_dir_path = '../../data/machine_learning/dataset2/label'

data_save_dir_path = '../../data/machine_learning/generated/data'
label_save_dir_path = '../../data/machine_learning/generated/label'

num_examples = 9
batch_size = 4
num_epochs = 50
#steps_per_epoch = num_examples / batch_size
steps_per_epoch = 10

resized_input_size = (200, 300)  # (height, width).
cropped_input_size = (100, 100)  # (height, width).
crop_dataset_flag = False
save_to_dir_flag = True

#%%------------------------------------------------------------------
# Create a data generator.

# REF [site] >> https://keras.io/preprocessing/image/

if crop_dataset_flag:
	# REF [site] >> https://github.com/fchollet/keras/issues/3338

	data_gen_with_crop_args = dict(
		rescale=1./255.,
		#preprocessing_function=None,
		#featurewise_center=True,
		#featurewise_std_normalization=True,
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#zca_whitening=True,
		#zca_epsilon=1e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		#shear_range=0.,
		zoom_range=0.2,
		#channel_shift_range=0.,
		horizontal_flip=True,
		vertical_flip=True,
		random_crop_size=cropped_input_size,
		center_crop_size=None,
		fill_mode='reflect',
		cval=0.)

	data_generator = ImageDataGeneratorWithCrop(**data_gen_with_crop_args)
	label_generator = ImageDataGeneratorWithCrop(**data_gen_with_crop_args)
else:
	data_gen_args = dict(
		rescale=1./255.,
		#preprocessing_function=None,
		#featurewise_center=True,
		#featurewise_std_normalization=True,
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#zca_whitening=True,
		#zca_epsilon=1e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		#shear_range=0.,
		zoom_range=0.2,
		#channel_shift_range=0.,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode='reflect',
		cval=0.)

	data_generator = ImageDataGenerator(**data_gen_args)
	label_generator = ImageDataGenerator(**data_gen_args)

# NOTICE [caustion] >> Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
#data_generator.fit(dataset.data, augment=True, seed=seed)
#label_generator.fit(dataset.labels, augment=True, seed=seed)

data_gen = data_generator.flow_from_directory(
	data_dir_path,
	target_size=resized_input_size,
	batch_size=batch_size,
	class_mode=None,
	shuffle=True,
	seed=seed,
	save_to_dir=(data_save_dir_path if save_to_dir_flag else None))
label_gen = label_generator.flow_from_directory(
	label_dir_path,
	target_size=resized_input_size,
	batch_size=batch_size,
	class_mode=None,
	shuffle=True,
	seed=seed,
	save_to_dir=(label_save_dir_path if save_to_dir_flag else None))

# Combine generators into one which yields image and labels.
dataset_generator = zip(data_gen, label_gen)

#%%------------------------------------------------------------------
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
model.add(Dropout(0.25))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#%%------------------------------------------------------------------
# Train the model

# Fit the model on batches with real-time data augmentation.
model.fit_generator(
	#dataset_generator,
	data_gen,
	#validation_data=validation_dataset_generator,
	#validation_steps=800,
	steps_per_epoch=steps_per_epoch,
	epochs=num_epochs)

#%%------------------------------------------------------------------

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        data_dir_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        data_dir_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
