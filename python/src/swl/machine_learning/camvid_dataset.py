import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	lib_home_dir_path = 'D:/lib_repo/python'
	#lib_home_dir_path = 'D:/lib_repo/python/rnd'

sys.path.append(swl_python_home_dir_path + '/src')

#%%------------------------------------------------------------------

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from swl.machine_learning.keras.preprocessing import ImageDataGeneratorWithCrop
from swl.machine_learning.data_loader import DataLoader
from swl.machine_learning.data_preprocessing import standardize_samplewise, standardize_featurewise
from swl.image_processing.util import load_images_by_pil, load_labels_by_pil

#%%------------------------------------------------------------------
# Prepare dataset.

def prepare_camvid_dataset(X=None, Y=None):
	if X is None:
		pass
	else:
		# Do something on X.
		pass
	if Y is None:
		pass
	else:
		# Do something on Y.
		pass
	return X, Y

def preprocess_camvid_dataset(X, Y, num_classes):
	# Preprocessing (normalization, standardization, etc).
	X = X.astype(np.float)
	#X /= 255.0
	X = standardize_samplewise(X)
	#X = standardize_featurewise(X)

	# One-hot encoding. (num_examples, height, width) -> (num_examples, height, width, num_classes).
	#if num_classes > 2:
	#	Y = np.uint8(keras.utils.to_categorical(Y, num_classes).reshape(Y.shape + (-1,)))
	Y = np.uint8(keras.utils.to_categorical(Y, num_classes).reshape(Y.shape + (-1,)))

	return X, Y

#%%------------------------------------------------------------------
# Load a CamVid dataset.

# REF [file] >> ${SWL_PYTHON_HOME}/test/image_processing/util_test.py

def load_camvid_dataset(train_data_dir_path, train_label_dir_path, val_data_dir_path, val_label_dir_path, test_data_dir_path, test_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', width=None, height=None):
	# Method 1: Use load_images_by_pil().
	train_data = load_images_by_pil(train_data_dir_path, data_suffix, data_extension, width=width, height=height)
	train_labels = load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=width, height=height)
	val_data = load_images_by_pil(val_data_dir_path, data_suffix, data_extension, width=width, height=height)
	val_labels = load_labels_by_pil(val_label_dir_path, label_suffix, label_extension, width=width, height=height)
	test_data = load_images_by_pil(test_data_dir_path, data_suffix, data_extension, width=width, height=height)
	test_labels = load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=width, height=height)
	# Method 2: Use DataLoader class.
	#data_loader = DataLoader() if width is None or height is None else DataLoader(width=width, height=height)
	#train_dataset = data_loader.load(data_dir_path=train_data_dir_path, label_dir_path=train_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)
	#val_dataset = data_loader.load(data_dir_path=val_data_dir_path, label_dir_path=val_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)
	#test_dataset = data_loader.load(data_dir_path=test_data_dir_path, label_dir_path=test_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)

	# Prepare dataset.
	#train_data, train_labels = prepare_camvid_dataset(train_data, train_labels)
	#val_data, val_labels = prepare_camvid_dataset(val_data, val_labels)
	#test_data, test_labels = prepare_camvid_dataset(test_data, test_labels)

	num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
	#num_classes = np.max([np.unique(train_labels).size, np.unique(val_labels).size, np.unique(test_labels).size])

	return train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes

#%%------------------------------------------------------------------

def get_camvid_dataset_generator(data_preprocessing_function, label_preprocessing_function, num_classes):
	train_data_generator = ImageDataGenerator(
		#rescale=1.0/255.0,
		preprocessing_function=data_preprocessing_function,
		#featurewise_center=True,
		#featurewise_std_normalization=True,
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
		preprocessing_function=label_preprocessing_function,
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
		preprocessing_function=data_preprocessing_function,
		#featurewise_center=True,
		#featurewise_std_normalization=True,
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#zca_whitening=False,
		#zca_epsilon=1.0e-6,
		fill_mode='reflect',
		cval=0.0)
	test_label_generator = ImageDataGenerator(
		#rescale=1.0/255.0,
		preprocessing_function=label_preprocessing_function,
		#featurewise_center=False,
		#featurewise_std_normalization=False,
		#samplewise_center=False,
		#samplewise_std_normalization=False,
		#zca_whitening=False,
		#zca_epsilon=1.0e-6,
		fill_mode='reflect',
		cval=0.0)
	return train_data_generator, train_label_generator, test_data_generator, test_label_generator

def get_camvid_dataset_generator_with_crop(data_preprocessing_function, label_preprocessing_function, num_classes, random_crop_size, center_crop_size):
	train_data_generator = ImageDataGeneratorWithCrop(
		#rescale=1.0/255.0,
		preprocessing_function=data_preprocessing_function,
		#featurewise_center=True,
		#featurewise_std_normalization=True,
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
		preprocessing_function=label_preprocessing_function,
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
		preprocessing_function=data_preprocessing_function,
		#featurewise_center=True,
		#featurewise_std_normalization=True,
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#zca_whitening=False,
		#zca_epsilon=1.0e-6,
		fill_mode='reflect',
		cval=0.0)
	test_label_generator = ImageDataGeneratorWithCrop(
		#rescale=1.0/255.0,
		preprocessing_function=label_preprocessing_function,
		#featurewise_center=False,
		#featurewise_std_normalization=False,
		#samplewise_center=False,
		#samplewise_std_normalization=False,
		#zca_whitening=False,
		#zca_epsilon=1.0e-6,
		fill_mode='reflect',
		cval=0.0)
	return train_data_generator, train_label_generator, test_data_generator, test_label_generator

#%%------------------------------------------------------------------
# Create a CamVid data generator.

# REF [site] >> https://keras.io/preprocessing/image/
# REF [site] >> https://github.com/fchollet/keras/issues/3338

# Provide the same seed and keyword arguments to the fit and flow methods.
#seed = 1

#resized_image_size = (height, width)
#random_crop_size = (height, width)
#center_crop_size = (height, width)

# A dataset generator for images(data) and labels per image.
#	- Images are only transformed, but labels are not transformed.
def create_camvid_generator_from_array(train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes, batch_size=32, random_crop_size=None, center_crop_size=None, shuffle=True, seed=None):
	train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocess_camvid_dataset(train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes)

	if random_crop_size is None and center_crop_size is None:
		train_data_generator, _, test_data_generator, _ = get_camvid_dataset_generator(None, None, num_classes)
	else:
		train_data_generator, _, test_data_generator, _ = get_camvid_dataset_generator_with_crop(None, None, num_classes, random_crop_size, center_crop_size)

	# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
	# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
	train_data_generator.fit(train_data, augment=True, rounds=1, seed=seed)
	#train_label_generator.fit(train_labels, augment=True, rounds=1, seed=seed)
	test_data_generator.fit(test_data, augment=True, rounds=1, seed=seed)
	#test_label_generator.fit(test_labels, augment=True, rounds=1, seed=seed)

	train_dataset_gen = train_data_generator.flow(
		train_data, train_labels,
		batch_size=batch_size,
		shuffle=shuffle,
		save_to_dir=None,
		save_prefix='',
		save_format='png',
		seed=seed)
	val_dataset_gen = test_data_generator.flow(
		val_data, val_labels,
		batch_size=batch_size,
		shuffle=shuffle,
		save_to_dir=None,
		save_prefix='',
		save_format='png',
		seed=seed)
	test_dataset_gen = test_data_generator.flow(
		test_data, test_labels,
		batch_size=batch_size,
		shuffle=shuffle,
		save_to_dir=None,
		save_prefix='',
		save_format='png',
		seed=seed)
	return train_dataset_gen, val_dataset_gen, test_dataset_gen

def create_camvid_generator_from_directory(train_data_dir_path, train_label_dir_path, val_data_dir_path, val_label_dir_path, test_data_dir_path, test_label_dir_path, num_classes, batch_size=32, resized_image_size=None, random_crop_size=None, center_crop_size=None, shuffle=True, seed=None):
	# Prepare dataset & one-hot encoding.
	if random_crop_size is None and center_crop_size is None:
		train_data_generator, train_label_generator, test_data_generator, test_label_generator = get_camvid_dataset_generator(None, None, num_classes)
	else:
		train_data_generator, train_label_generator, test_data_generator, test_label_generator = get_camvid_dataset_generator_with_crop(None, None, num_classes, random_crop_size, center_crop_size)

	# FIXME [implement] >>
	# Preprocessing (normalization, standardization, etc).

	# FIXME [implement] >>
	# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
	# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
	#train_data_generator.fit(train_dataset.data, augment=True, rounds=1, seed=seed)
	##train_label_generator.fit(train_dataset.labels, augment=True, rounds=1, seed=seed)
	#test_data_generator.fit(test_dataset.data, augment=True, rounds=1, seed=seed)
	##test_label_generator.fit(test_dataset.labels, augment=True, rounds=1, seed=seed)

	train_data_gen = train_data_generator.flow_from_directory(
		train_data_dir_path,
		target_size=resized_image_size,
		color_mode='rgb',  # Load images of size (resized_image_size, 3).
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
		color_mode='grayscale',  # Load images of size (resized_image_size, 1).
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
import threading
from swl.util.threading import ThreadSafeGenerator

def get_imgaug_sequence_for_camvid(width=None, height=None):
	if height is not None and width is not None:
		seq = iaa.Sequential([
			iaa.SomeOf(1, [
				#iaa.Sometimes(0.5, iaa.Crop(px=(0, 100))),  # Crop images from each side by 0 to 16px (randomly chosen).
				iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.25))), # Crop images by 0-10% of their height/width.
				iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
				iaa.Flipud(0.5),  # Vertically flip 50% of the images.
				iaa.Sometimes(0.5, iaa.Affine(
					scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
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
					scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
					translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
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

	return seq

# REF [function] >> generate_batch_using_imgaug() in ${SWL_PYTHON_HOME}/src/swl/machine_learning/util/py
# NOTICE [info] >> This is not thread-safe. To make it thread-safe, use ThreadSafeGenerator.
def generate_batch_using_imgaug(seq, X, Y, num_classes, batch_size, shuffle=True):
	while True:
		seq_det = seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		X_aug = seq_det.augment_images(X)
		Y_aug = seq_det.augment_images(Y)

		# Preprocessing (normalization, standardization, etc).
		X_aug, Y_aug = preprocess_camvid_dataset(X_aug, Y_aug, num_classes)

		num_steps = np.ceil(len(X) / batch_size).astype(np.int)
		#num_steps = len(X) // batch_size + (0 if len(X) % batch_size == 0 else 1)
		if shuffle is True:
			indexes = np.arange(len(X_aug))
			np.random.shuffle(indexes)
			for idx in range(num_steps):
				batch_x = X_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				batch_y = Y_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				#yield {'input': batch_x}, {'output': batch_y}
				yield batch_x, batch_y
		else:
			for idx in range(num_steps):
				batch_x = X_aug[idx*batch_size:(idx+1)*batch_size]
				batch_y = Y_aug[idx*batch_size:(idx+1)*batch_size]
				#yield {'input': batch_x}, {'output': batch_y}
				yield batch_x, batch_y

class DatasetGeneratorUsingImgaug:
	def __init__(self, seq, X, Y, num_classes, batch_size, shuffle=True):
		self.seq = seq
		self.X = X
		self.Y = Y
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.shuffle = shuffle

		self.num_steps = np.ceil(len(self.X) / self.batch_size).astype(np.int)
		#self.num_steps = len(self.X) // self.batch_size + (0 if len(self.X) % self.batch_size == 0 else 1)
		self.idx = 0
		self.X_aug = None
		self.Y_aug = None

		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			if 0 == self.idx:
				seq_det = self.seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
				self.X_aug = seq_det.augment_images(self.X)
				self.Y_aug = seq_det.augment_images(self.Y)

				# Preprocessing (normalization, standardization, etc).
				self.X_aug, self.Y_aug = preprocess_camvid_dataset(self.X_aug, self.Y_aug, self.num_classes)

				indexes = np.arange(len(self.X_aug))
				if self.shuffle is True:
					np.random.shuffle(indexes)

			if self.X_aug is None or self.Y_aug is None:
				assert False, 'Both X_aug and Y_aug are not None.'

			if self.shuffle is True:
				batch_x = self.X_aug[indexes[self.idx*self.batch_size:(self.idx+1)*self.batch_size]]
				batch_y = self.Y_aug[indexes[self.idx*self.batch_size:(self.idx+1)*self.batch_size]]
			else:
				batch_x = self.X_aug[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
				batch_y = self.Y_aug[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
			self.idx = (self.idx + 1) % self.num_steps
			return batch_x, batch_y

def create_camvid_generator_using_imgaug_sequence(seq, train_data, train_labels, val_data, val_labels, test_data, test_labels, num_classes, batch_size=32, shuffle=True):
	#return DatasetGeneratorUsingImgaug(seq, train_data, train_labels, num_classes, batch_size, shuffle), DatasetGeneratorUsingImgaug(seq, val_data, val_labels, num_classes, batch_size, shuffle), DatasetGeneratorUsingImgaug(seq, test_data, test_labels, num_classes, batch_size, shuffle)
	return ThreadSafeGenerator(generate_batch_using_imgaug(seq, train_data, train_labels, num_classes, batch_size, shuffle)), ThreadSafeGenerator(generate_batch_using_imgaug(seq, val_data, val_labels, num_classes, batch_size, shuffle)), ThreadSafeGenerator(generate_batch_using_imgaug(seq, test_data, test_labels, num_classes, batch_size, shuffle))
	#return generate_batch_using_imgaug(seq, train_data, train_labels, num_classes, batch_size, shuffle), generate_batch_using_imgaug(seq, val_data, val_labels, num_classes, batch_size, shuffle), generate_batch_using_imgaug(seq, test_data, test_labels, num_classes, batch_size, shuffle)

