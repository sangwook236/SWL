import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#%%------------------------------------------------------------------

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from swl.machine_learning.keras.preprocessing import ImageDataGeneratorWithCrop
from swl.machine_learning.data_loader import DataLoader
from swl.image_processing.util import load_images_by_pil, load_labels_by_pil

#%%------------------------------------------------------------------
# Create a CamVid data generator.

# REF [site] >> https://keras.io/preprocessing/image/
# REF [site] >> https://github.com/fchollet/keras/issues/3338

# Provide the same seed and keyword arguments to the fit and flow methods.
#seed = 1

#resized_image_size = (height, width)
#cropped_image_size = (height, width)

def create_camvid_generator(train_data_dir_path, train_label_dir_path, val_data_dir_path, val_label_dir_path, test_data_dir_path, test_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', batch_size=32, resized_image_size=None, cropped_image_size=None, use_loaded_dataset=True, shuffle=True, seed=None):
	if cropped_image_size is None:
		train_data_generator = ImageDataGenerator(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			#featurewise_center=False,
			#featurewise_std_normalization=False,
			samplewise_center=True,
			samplewise_std_normalization=True,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			#rotation_range=20,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			#horizontal_flip=True,
			#vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			fill_mode='reflect',
			cval=0.0)
		train_label_generator = ImageDataGenerator(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			#featurewise_center=False,
			#featurewise_std_normalization=False,
			#samplewise_center=False,
			#samplewise_std_normalization=False,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			#rotation_range=20,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			#horizontal_flip=True,
			#vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			fill_mode='reflect',
			cval=0.0)
		test_data_generator = ImageDataGenerator(rescale=1.0/255.0)
		test_label_generator = ImageDataGenerator()
	else:
		train_data_generator = ImageDataGeneratorWithCrop(
			rescale=1.0/255.0,
			#preprocessing_function=None,
			#featurewise_center=False,
			#featurewise_std_normalization=False,
			#samplewise_center=False,
			#samplewise_std_normalization=False,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			#rotation_range=20,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			#horizontal_flip=True,
			vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			random_crop_size=cropped_image_size,
			center_crop_size=None,
			fill_mode='reflect',
			cval=0.0)
		train_label_generator = ImageDataGeneratorWithCrop(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			#featurewise_center=False,
			#featurewise_std_normalization=False,
			#samplewise_center=False,
			#samplewise_std_normalization=False,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			#rotation_range=20,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			#horizontal_flip=True,
			vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			random_crop_size=cropped_image_size,
			center_crop_size=None,
			fill_mode='reflect',
			cval=0.0)
		test_data_generator = ImageDataGeneratorWithCrop(rescale=1.0/255.0)
		test_label_generator = ImageDataGeneratorWithCrop()

	if use_loaded_dataset == True:
		data_loader = DataLoader() if resized_image_size is None else DataLoader(width=resized_image_size[1], height=resized_image_size[0])
		train_dataset = data_loader.load(data_dir_path=train_data_dir_path, label_dir_path=train_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)
		val_dataset = data_loader.load(data_dir_path=val_data_dir_path, label_dir_path=val_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)
		test_dataset = data_loader.load(data_dir_path=test_data_dir_path, label_dir_path=test_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)

		# Change the dimension of labels.
		if train_dataset.data.ndim == train_dataset.labels.ndim:
			pass
		elif 1 == train_dataset.data.ndim - train_dataset.labels.ndim:
			train_dataset.labels = train_dataset.labels.reshape(train_dataset.labels.shape + (1,))
		else:
			raise ValueError('train_dataset.data.ndim or train_dataset.labels.ndim is invalid.')
		if val_dataset.data.ndim == val_dataset.labels.ndim:
			pass
		elif 1 == val_dataset.data.ndim - val_dataset.labels.ndim:
			val_dataset.labels = val_dataset.labels.reshape(val_dataset.labels.shape + (1,))
		else:
			raise ValueError('val_dataset.data.ndim or val_dataset.labels.ndim is invalid.')
		if test_dataset.data.ndim == test_dataset.labels.ndim:
			pass
		elif 1 == test_dataset.data.ndim - test_dataset.labels.ndim:
			test_dataset.labels = test_dataset.labels.reshape(test_dataset.labels.shape + (1,))
		else:
			raise ValueError('test_dataset.data.ndim or test_dataset.labels.ndim is invalid.')

		# One-hot encoding.
		num_classes = np.max([np.max(np.unique(train_dataset.labels)), np.max(np.unique(val_dataset.labels)), np.max(np.unique(test_dataset.labels))]) + 1
		train_dataset.labels = np.uint8(keras.utils.to_categorical(train_dataset.labels, num_classes).reshape(train_dataset.labels.shape[:-1] + (-1,)))
		val_dataset.labels = np.uint8(keras.utils.to_categorical(val_dataset.labels, num_classes).reshape(val_dataset.labels.shape[:-1] + (-1,)))
		test_dataset.labels = np.uint8(keras.utils.to_categorical(test_dataset.labels, num_classes).reshape(test_dataset.labels.shape[:-1] + (-1,)))

		# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
		# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
		train_data_generator.fit(train_dataset.data, augment=True, seed=seed)
		#train_label_generator.fit(train_dataset.labels, augment=True, seed=seed)
		test_data_generator.fit(test_dataset.data, augment=True, seed=seed)
		#test_label_generator.fit(test_dataset.labels, augment=True, seed=seed)

		train_dataset_gen = train_data_generator.flow(
			train_dataset.data, train_dataset.labels,
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		val_dataset_gen = test_data_generator.flow(
			val_dataset.data, val_dataset.labels,
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		test_dataset_gen = test_data_generator.flow(
			test_dataset.data, test_dataset.labels,
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
	else:
		train_data_gen = train_data_generator.flow_from_directory(
			train_data_dir_path,
			target_size=resized_image_size,
			color_mode='rgb',
			#classes=None,
			class_mode=None,  # NOTICE [important] >>
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		train_label_gen = train_label_generator.flow_from_directory(
			train_label_dir_path,
			target_size=resized_image_size,
			color_mode='grayscale',
			#classes=None,
			class_mode=None,  # NOTICE [important] >>
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		val_data_gen = test_data_generator.flow_from_directory(
			val_data_dir_path,
			target_size=resized_image_size,
			color_mode='rgb',
			#classes=None,
			class_mode=None,  # NOTICE [important] >>
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		val_label_gen = test_label_generator.flow_from_directory(
			val_label_dir_path,
			target_size=resized_image_size,
			color_mode='grayscale',
			#classes=None,
			class_mode=None,  # NOTICE [important] >>
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		test_data_gen = test_data_generator.flow_from_directory(
			test_data_dir_path,
			target_size=resized_image_size,
			color_mode='rgb',
			#classes=None,
			class_mode=None,  # NOTICE [important] >>
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)
		test_label_gen = test_label_generator.flow_from_directory(
			test_label_dir_path,
			target_size=resized_image_size,
			color_mode='grayscale',
			#classes=None,
			class_mode=None,  # NOTICE [important] >>
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=False,
			save_prefix='',
			save_format='png',
			seed=seed)

		# FIXME [implement] >>
		# One-hot encoding.

		# Combine generators into one which yields image and labels.
		train_dataset_gen = zip(train_data_gen, train_label_gen)
		val_dataset_gen = zip(val_data_gen, val_label_gen)
		test_dataset_gen = zip(test_data_gen, test_label_gen)

	return train_dataset_gen, val_dataset_gen, test_dataset_gen

#%%------------------------------------------------------------------
# Load a CamVid dataset.

# REF [file] >> ${SWL_PYTHON_HOME}/test/image_processing/util_test.py

def load_camvid_dataset(train_data_dir_path, train_label_dir_path, val_data_dir_path, val_label_dir_path, test_data_dir_path, test_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', width=None, height=None):
	train_data = load_images_by_pil(train_data_dir_path, data_suffix, data_extension, width=width, height=height)
	train_labels = load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=width, height=height)
	val_data = load_images_by_pil(val_data_dir_path, data_suffix, data_extension, width=width, height=height)
	val_labels = load_labels_by_pil(val_label_dir_path, label_suffix, label_extension, width=width, height=height)
	test_data = load_images_by_pil(test_data_dir_path, data_suffix, data_extension, width=width, height=height)
	test_labels = load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=width, height=height)

	# One-hot encoding.
	num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
	train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
	val_labels = np.uint8(keras.utils.to_categorical(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
	test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

	# FIXME [implement] >> Preprocessing (normalization, standardization, etc).

	return train_data, train_labels, val_data, val_labels, test_data, test_labels
