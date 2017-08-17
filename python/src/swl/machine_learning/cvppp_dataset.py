import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	lib_home_dir_path = 'D:/lib_repo/python'
	#lib_home_dir_path = 'D:/lib_repo/python/rnd'
lib_dir_path = lib_home_dir_path + '/imgaug_github'

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
# Prepare dataset.

def prepare_dataset(X, Y=None):
	if X is None:
		pass
	else:
		# RGBA -> RGB.
		X = X[:,:,:,:-1]
	if Y is None:
		pass
	else:
		# Do something on Y.
		pass
	return X, Y

#%%------------------------------------------------------------------

def get_cvppp_dataset_generator(data_preprocessing_function, label_preprocessing_function, num_classes):
	train_data_generator = ImageDataGenerator(
		#rescale=1.0/255.0,
		preprocessing_function=data_preprocessing_function,
		featurewise_center=True,
		featurewise_std_normalization=True,
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#zca_whitening=False,
		#zca_epsilon=1.0e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=0.2,
		#shear_range=0.0,
		#channel_shift_range=0.0,
		fill_mode='constant',
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
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=0.2,
		#shear_range=0.0,
		#channel_shift_range=0.0,
		fill_mode='constant',
		cval=0.0)
	return train_data_generator, train_label_generator

def get_cvppp_dataset_generator_with_crop(data_preprocessing_function, label_preprocessing_function, num_classes, random_crop_size, center_crop_size):
	train_data_generator = ImageDataGeneratorWithCrop(
		#rescale=1.0/255.0,
		preprocessing_function=data_preprocessing_function,
		featurewise_center=True,
		featurewise_std_normalization=True,
		#samplewise_center=True,
		#samplewise_std_normalization=True,
		#zca_whitening=False,
		#zca_epsilon=1.0e-6,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=0.2,
		#shear_range=0.0,
		#channel_shift_range=0.0,
		random_crop_size=random_crop_size,
		center_crop_size=center_crop_size,
		fill_mode='constant',
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
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=0.2,
		#shear_range=0.0,
		#channel_shift_range=0.0,
		random_crop_size=random_crop_size,
		center_crop_size=center_crop_size,
		fill_mode='constant',
		cval=0.0)
	return train_data_generator, train_label_generator

#%%------------------------------------------------------------------
# Create a CVPPP data generator.

# REF [site] >> https://keras.io/preprocessing/image/
# REF [site] >> https://github.com/fchollet/keras/issues/3338

# Provide the same seed and keyword arguments to the fit and flow methods.
#seed = 1

#resized_image_size = (height, width)
#random_crop_size = (height, width)
#random_crop_size = (height, width)

def create_cvppp_generator_from_data_loader(train_data_dir_path, train_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', batch_size=32, resized_image_size=None, random_crop_size=None, center_crop_size=None, shuffle=True, seed=None):
	data_loader = DataLoader() if resized_image_size is None else DataLoader(width=resized_image_size[1], height=resized_image_size[0])
	train_dataset = data_loader.load(data_dir_path=train_data_dir_path, label_dir_path=train_label_dir_path, data_suffix=data_suffix, data_extension=data_extension, label_suffix=label_suffix, label_extension=label_extension)

	# Prepare dataset.
	train_dataset.data, _ = prepare_dataset(train_dataset.data)

	num_classes = np.max(np.unique(train_dataset.labels)) + 1
	#num_classes = np.unique(train_dataset.labels).size

	# Preprocessing (normalization, standardization, etc).
	train_dataset.data = train_dataset.data.astype(np.float)
	#train_dataset.data /= 255.0
	train_dataset.data = standardize_samplewise(train_dataset.data)
	#train_dataset.data = standardize_featurewise(train_dataset.data)

	# One-hot encoding. (num_examples, height, width) -> (num_examples, height, width, num_classes).
	#if num_classes > 2:
	#	train_dataset.labels = np.uint8(keras.utils.to_categorical(train_dataset.labels, num_classes).reshape(train_dataset.labels.shape + (-1,)))
	train_dataset.labels = np.uint8(keras.utils.to_categorical(train_dataset.labels, num_classes).reshape(train_dataset.labels.shape + (-1,)))

	if random_crop_size is None and center_crop_size is None:
		train_data_generator, _ = get_cvppp_dataset_generator(None, None, num_classes)
	else:
		train_data_generator, _ = get_cvppp_dataset_generator_with_crop(None, None, num_classes, random_crop_size, center_crop_size)

	# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
	# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
	train_data_generator.fit(train_dataset.data, augment=True, rounds=1, seed=seed)
	#train_label_generator.fit(train_dataset.labels, augment=True, rounds=1, seed=seed)

	train_dataset_gen = train_data_generator.flow(
		train_dataset.data, train_dataset.labels,
		batch_size=batch_size,
		shuffle=shuffle,
		save_to_dir=None,
		save_prefix='',
		save_format='png',
		seed=seed)
	return train_dataset_gen

def create_cvppp_generator_from_directory(train_data_dir_path, train_label_dir_path, num_classes, batch_size=32, resized_image_size=None, random_crop_size=None, center_crop_size=None, shuffle=True, seed=None):
	# Prepare dataset & one-hot encoding.
	if random_crop_size is None and center_crop_size is None:
		train_data_generator, train_label_generator = get_cvppp_dataset_generator(
				#lambda img: img = img[:,:,:,:-1],  # RGBA -> RGB.
				None,
				None, num_classes)
	else:
		train_data_generator, train_label_generator = get_cvppp_dataset_generator_with_crop(
				#lambda img: img = img[:,:,:,:-1],  # RGBA -> RGB.
				None,
				None, num_classes, random_crop_size, center_crop_size)

	# FIXME [implement] >>
	# Preprocessing (normalization, standardization, etc).

	# FIXME [implement] >>
	# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
	# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
	#train_data_generator.fit(train_dataset.data, augment=True, rounds=1, seed=seed)
	#train_label_generator.fit(train_dataset.labels, augment=True, rounds=1, seed=seed)

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

	# Combine generators into one which yields image and labels.
	train_dataset_gen = zip(train_data_gen, train_label_gen)
	return train_dataset_gen

#%%------------------------------------------------------------------
# Create a CVPPP data generator using imgaug in https://github.com/aleju/imgaug.

# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/image_augmentation_test.py

import imgaug as ia
from imgaug import augmenters as iaa
import threading
from swl.util.threading import ThreadSafeGenerator

# REF [function] >> generate_batch_from_image_augmentation_sequence() in ${SWL_PYTHON_HOME}/src/swl/machine_learning/util/py
# NOTICE [info] >> This is not thread-safe. To make it thread-safe, use ThreadSafeGenerator.
def generate_batch_from_imgaug_sequence(seq, X, Y, num_classes, batch_size, shuffle=True):
	while True:
		seq_det = seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		X_aug = seq_det.augment_images(X)
		Y_aug = seq_det.augment_images(Y)

		# Preprocessing (normalization, standardization, etc).
		X_aug = X_aug.astype(np.float)
		#X_aug /= 255.0
		X_aug = standardize_samplewise(X_aug)
		#X_aug = standardize_featurewise(X_aug)

		# One-hot encoding. (num_examples, height, width) -> (num_examples, height, width, num_classes).
		#if num_classes > 2:
		#	Y_aug = np.uint8(keras.utils.to_categorical(Y_aug, num_classes).reshape(Y_aug.shape + (-1,)))
		Y_aug = np.uint8(keras.utils.to_categorical(Y_aug, num_classes).reshape(Y_aug.shape + (-1,)))

		num_steps = np.ceil(len(X_aug) / batch_size).astype(np.int)
		#num_steps = len(X_aug) // batch_size + (0 if len(X_aug) % batch_size == 0 else 1)
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

class DatasetGenerator:
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
				self.X_aug = self.X_aug.astype(np.float)
				#self.X_aug /= 255.0
				self.X_aug = standardize_samplewise(self.X_aug)
				#self.X_aug = standardize_featurewise(self.X_aug)

				# One-hot encoding. (num_examples, height, width) -> (num_examples, height, width, num_classes).
				#if self.num_classes > 2:
				#	self.Y_aug = np.uint8(keras.utils.to_categorical(self.Y_aug, self.num_classes).reshape(self.Y_aug.shape + (-1,)))
				self.Y_aug = np.uint8(keras.utils.to_categorical(self.Y_aug, self.num_classes).reshape(self.Y_aug.shape + (-1,)))

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

def create_cvppp_generator_from_imgaug(train_data_dir_path, train_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', batch_size=32, width=None, height=None, shuffle=True):
	train_data = load_images_by_pil(train_data_dir_path, data_suffix, data_extension, width=None, height=None)
	train_labels = load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=None, height=None)
	#test_data = load_images_by_pil(test_data_dir_path, data_suffix, data_extension, width=None, height=None)
	#test_labels = load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=None, height=None)

	# Prepare dataset.
	train_data, _ = prepare_dataset(train_data, train_labels)
	#test_data, _ = prepare_dataset(test_data, test_labels)

	num_classes = np.max(np.unique(train_labels)) + 1
	#num_classes = np.unique(train_labels).size

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
					#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
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
					#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				))
				#iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0)))  # Blur images with a sigma of 0 to 3.0.
			])
		)

	#return DatasetGenerator(seq, train_data, train_labels, num_classes, batch_size, shuffle)
	return ThreadSafeGenerator(generate_batch_from_imgaug_sequence(seq, train_data, train_labels, num_classes, batch_size, shuffle))
	#return generate_batch_from_imgaug_sequence(seq, train_data, train_labels, num_classes, batch_size, shuffle)
	##return DatasetGenerator(seq, train_data, train_labels, num_classes, batch_size, shuffle), DatasetGenerator(seq, test_data, test_labels, num_classes, batch_size, shuffle)
	##return ThreadSafeGenerator(generate_batch_from_imgaug_sequence(seq, train_data, train_labels, num_classes, batch_size, shuffle)), ThreadSafeGenerator(generate_batch_from_imgaug_sequence(seq, test_data, test_labels, num_classes, batch_size, shuffle))
	##return generate_batch_from_imgaug_sequence(seq, train_data, train_labels, num_classes, batch_size, shuffle), generate_batch_from_imgaug_sequence(seq, test_data, test_labels, num_classes, batch_size, shuffle)

#%%------------------------------------------------------------------
# Load a CVPPP dataset.

# REF [file] >> ${SWL_PYTHON_HOME}/test/image_processing/util_test.py

def load_cvppp_dataset(train_data_dir_path, train_label_dir_path, data_suffix='', data_extension='png', label_suffix='', label_extension='png', width=None, height=None):
	train_data = load_images_by_pil(train_data_dir_path, data_suffix, data_extension, width=width, height=height)
	train_labels = load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=width, height=height)
	#test_data = load_images_by_pil(test_data_dir_path, data_suffix, data_extension, width=width, height=height)
	#test_labels = load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=width, height=height)

	# Prepare dataset.
	train_data, _ = prepare_dataset(train_data, train_labels)
	#test_data, _ = prepare_dataset(test_data, test_labels)

	num_classes = np.max(np.unique(train_labels)) + 1
	#num_classes = np.unique(train_labels).size

	# Preprocessing (normalization, standardization, etc).
	train_data = train_data.astype(np.float)
	#train_data /= 255.0
	train_data = standardize_samplewise(train_data)
	#train_data = standardize_featurewise(train_data)

	#test_data = test_data.astype(np.float)
	#test_data /= 255.0
	#test_data = standardize_samplewise(train_data)
	#test_data = standardize_featurewise(train_data)

	# One-hot encoding. (num_examples, height, width) -> (num_examples, height, width, num_classes).
	#if num_classes > 2:
	#	train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
	#	#test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))
	train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
	#test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

	return train_data, train_labels
	#return train_data, train_labels, test_data, test_labels
