import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = "/home/sangwook/lib_repo/python"
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	lib_home_dir_path = "D:/lib_repo/python"
	#lib_home_dir_path = "D:/lib_repo/python/rnd"
lib_dir_path = lib_home_dir_path + "/imgaug_github"

sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_dir_path)

#%%------------------------------------------------------------------

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from swl.machine_learning.keras.preprocessing import ImageDataGeneratorWithCrop
from swl.machine_learning.data_loader import DataLoader
from swl.machine_learning.data_preprocessing import standardize_samplewise, standardize_featurewise
from swl.image_processing.util import load_images_by_pil, load_labels_by_pil

#%%------------------------------------------------------------------
# Create a CamVid data generator.

# REF [site] >> https://keras.io/preprocessing/image/
# REF [site] >> https://github.com/fchollet/keras/issues/3338

# Provide the same seed and keyword arguments to the fit and flow methods.
#seed = 1

#resized_image_size = (height, width)
#random_crop_size = (height, width)
#center_crop_size = (height, width)

# NOTICE [caution] >> Not correctly working.
#	Use create_camvid_generator2.
def create_camvid_generator(train_data_dir_path, train_label_dir_path, val_data_dir_path, val_label_dir_path, test_data_dir_path, test_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', batch_size=32, resized_image_size=None, random_crop_size=None, center_crop_size=None, use_loaded_dataset=True, shuffle=True, seed=None):
	if random_crop_size is None and center_crop_size is None:
		train_data_generator = ImageDataGenerator(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			featurewise_center=True,
			featurewise_std_normalization=True,
			#samplewise_center=True,
			#samplewise_std_normalization=True,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			#rotation_range=20,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			horizontal_flip=True,
			vertical_flip=True,
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
			horizontal_flip=True,
			vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			fill_mode='reflect',
			cval=0.0)
		test_data_generator = ImageDataGenerator(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			featurewise_center=True,
			featurewise_std_normalization=True,
			#samplewise_center=True,
			#samplewise_std_normalization=True,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			fill_mode='reflect',
			cval=0.0)
		test_label_generator = ImageDataGenerator()
	else:
		train_data_generator = ImageDataGeneratorWithCrop(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			featurewise_center=True,
			featurewise_std_normalization=True,
			#samplewise_center=True,
			#samplewise_std_normalization=True,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			#rotation_range=20,
			#width_shift_range=0.2,
			#height_shift_range=0.2,
			horizontal_flip=True,
			vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			random_crop_size=random_crop_size,
			center_crop_size=center_crop_size,
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
			horizontal_flip=True,
			vertical_flip=True,
			#zoom_range=0.2,
			#shear_range=0.0,
			#channel_shift_range=0.0,
			random_crop_size=random_crop_size,
			center_crop_size=center_crop_size,
			fill_mode='reflect',
			cval=0.0)
		test_data_generator = ImageDataGeneratorWithCrop(
			#rescale=1.0/255.0,
			#preprocessing_function=None,
			featurewise_center=True,
			featurewise_std_normalization=True,
			#samplewise_center=True,
			#samplewise_std_normalization=True,
			#zca_whitening=False,
			#zca_epsilon=1.0e-6,
			fill_mode='reflect',
			cval=0.0)
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
			save_to_dir=None,
			save_prefix='',
			save_format='png',
			seed=seed)
		val_dataset_gen = test_data_generator.flow(
			val_dataset.data, val_dataset.labels,
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=None,
			save_prefix='',
			save_format='png',
			seed=seed)
		test_dataset_gen = test_data_generator.flow(
			test_dataset.data, test_dataset.labels,
			batch_size=batch_size,
			shuffle=shuffle,
			save_to_dir=None,
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
			save_to_dir=None,
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
			save_to_dir=None,
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
			save_to_dir=None,
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
			save_to_dir=None,
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
			save_to_dir=None,
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
			save_to_dir=None,
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
# Create a CamVid data generator using imgaug in https://github.com/aleju/imgaug.

# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/image_augmentation_test.py

import imgaug as ia
from imgaug import augmenters as iaa

# REF [function] >> generate_batch_from_image_augmentation_sequence() in ${SWL_PYTHON_HOME}/src/swl/machine_learning/util/py
def generate_batch_from_image_augmentation_sequence(seq, X, Y, num_classes, batch_size, shuffle=False):
	while True:
		seq_det = seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		X_aug = seq_det.augment_images(X)
		Y_aug = seq_det.augment_images(Y)

		# One-hot encoding.
		Y_aug = np.uint8(keras.utils.to_categorical(Y_aug, num_classes).reshape(Y_aug.shape + (-1,)))

		num_steps = np.ceil(len(X) / batch_size).astype(np.int)
		if shuffle is True:
			indexes = np.arange(len(X_aug))
			np.random.shuffle(indexes)
			for idx in range(num_steps):
				batch_x = X_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				batch_y = Y_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				#yield({'input': batch_x}, {'output': batch_y})
				yield(batch_x, batch_y)
		else:
			for idx in range(num_steps):
				batch_x = X_aug[idx*batch_size:(idx+1)*batch_size]
				batch_y = Y_aug[idx*batch_size:(idx+1)*batch_size]
				#yield({'input': batch_x}, {'output': batch_y})
				yield(batch_x, batch_y)

def create_camvid_generator2(train_data_dir_path, train_label_dir_path, val_data_dir_path, val_label_dir_path, test_data_dir_path, test_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', batch_size=32, width=None, height=None, shuffle=True):
	train_data = load_images_by_pil(train_data_dir_path, data_suffix, data_extension, width=None, height=None)
	train_labels = load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=None, height=None)
	val_data = load_images_by_pil(val_data_dir_path, data_suffix, data_extension, width=None, height=None)
	val_labels = load_labels_by_pil(val_label_dir_path, label_suffix, label_extension, width=None, height=None)
	test_data = load_images_by_pil(test_data_dir_path, data_suffix, data_extension, width=None, height=None)
	test_labels = load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=None, height=None)

	# Preprocessing (normalization, standardization, etc).
	train_data = train_data.astype(np.float)
	#train_data /= 255.0
	#train_data = standardize_samplewise(train_data)
	train_data = standardize_featurewise(train_data)

	val_data = val_data.astype(np.float)
	#val_data /= 255.0
	#val_data = standardize_samplewise(val_data)
	val_data = standardize_featurewise(val_data)

	test_data = test_data.astype(np.float)
	#test_data /= 255.0
	#test_data = standardize_samplewise(test_data)
	test_data = standardize_featurewise(test_data)

	for idx in range(train_data.shape[0]):
		train_data[idx] = (train_data[idx] - np.min(train_data[idx])) / (np.max(train_data[idx]) - np.min(train_data[idx])) * 255
	train_data = train_data.astype(np.uint8)
	for idx in range(val_data.shape[0]):
		val_data[idx] = (val_data[idx] - np.min(val_data[idx])) / (np.max(val_data[idx]) - np.min(val_data[idx])) * 255
	val_data = val_data.astype(np.uint8)
	for idx in range(test_data.shape[0]):
		test_data[idx] = (test_data[idx] - np.min(test_data[idx])) / (np.max(test_data[idx]) - np.min(test_data[idx])) * 255
	test_data = test_data.astype(np.uint8)

	if height is not None and width is not None:
		seq = iaa.Sequential([
			iaa.SomeOf(1, [
				#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.25))), # Crop images by 0-10% of their height/width.
				iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
				iaa.Flipud(0.5),  # Vertically flip 50% of the images.
				iaa.Sometimes(0.5, iaa.Affine(
					scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
					rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
					shear=(-16, 16),  # Shear by -16 to +16 degrees.
					#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
					order=0,  # Use nearest neighbour or bilinear interpolation (fast).
					#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
					#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
					mode='reflect'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				))
				#iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
			]),
			iaa.Scale(size={'height': height, 'width': width}, interpolation='nearest')  # Resize.
		])
	else:
		seq = iaa.Sequential(
			iaa.SomeOf(1, [
				#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))), # Crop images by 0-10% of their height/width.
				iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
				iaa.Flipud(0.5),  # Vertically flip 50% of the images.
				iaa.Sometimes(0.5, iaa.Affine(
					scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
					rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
					shear=(-16, 16),  # Shear by -16 to +16 degrees.
					#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
					order=0,  # Use nearest neighbour or bilinear interpolation (fast).
					#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
					#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
					mode='reflect'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				))
				#iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
			])
		)

	# One-hot encoding.
	num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
	#train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
	#val_labels = np.uint8(keras.utils.to_categorical(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
	#test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

	return generate_batch_from_image_augmentation_sequence(seq, train_data, train_labels, num_classes, batch_size, shuffle), generate_batch_from_image_augmentation_sequence(seq, val_data, val_labels, num_classes, batch_size, shuffle), generate_batch_from_image_augmentation_sequence(seq, test_data, test_labels, num_classes, batch_size, shuffle)

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

	# Preprocessing (normalization, standardization, etc).
	train_data = train_data.astype(np.float)
	#train_data /= 255.0
	#train_data = standardize_samplewise(train_data)
	train_data = standardize_featurewise(train_data)

	val_data = val_data.astype(np.float)
	#val_data /= 255.0
	#val_data = standardize_samplewise(val_data)
	val_data = standardize_featurewise(val_data)

	test_data = test_data.astype(np.float)
	#test_data /= 255.0
	#test_data = standardize_samplewise(test_data)
	test_data = standardize_featurewise(test_data)

	return train_data, train_labels, val_data, val_labels, test_data, test_labels
