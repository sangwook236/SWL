import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

#os.chdir(swl_python_home_dir_path + '/test/machine_learning')

#%%------------------------------------------------------------------
# Prepare dataset.

from swl.image_processing.util import load_images_by_pil, load_labels_by_pil
from swl.machine_learning.data_preprocessing import standardize_samplewise, standardize_featurewise
import numpy as np
import keras

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

image_dir_path = dataset_home_dir_path + '/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1'
label_dir_path = dataset_home_dir_path + '/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1'

image_suffix = '_rgb'
image_extension = 'png'
label_suffix = '_fg'
label_extension = 'png'

image_width, image_height = None, None
#image_width, image_height = 500, 530
#image_width, image_height = 224, 224

# REF [file] >> ${SWL_PYTHON_HOME}/test/image_processing/util_test.py
images = load_images_by_pil(image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
labels = load_labels_by_pil(label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)

# RGBA -> RGB.
images = images[:,:,:,:-1]

#%%------------------------------------------------------------------
# Transform randomly.

if 'posix' == os.name:
	lib_home_dir_path = "/home/sangwook/lib_repo/python"
else:
	lib_home_dir_path = "D:/lib_repo/python"
	#lib_home_dir_path = "D:/lib_repo/python/rnd"
lib_dir_path = lib_home_dir_path + "/imgaug_github"
sys.path.append(lib_dir_path)

import imgaug as ia
from imgaug import augmenters as iaa

# FIXME [decide] >> Before or after random transformation?
# Preprocessing (normalization, standardization, etc).
images_pp = images.astype(np.float)
#images_pp /= 255.0
images_pp = standardize_samplewise(images_pp)
#images_pp = standardize_featurewise(images_pp)

seq = iaa.Sequential([
	iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
	iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
	iaa.Flipud(0.5),  # Vertically flip 50% of the images.
	iaa.Affine(
		scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis).
		rotate=(-45, 45),  # Rotate by -45 to +45 degrees.
		shear=(-16, 16),  # Shear by -16 to +16 degrees.
		#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
		order=0,  # Use nearest neighbour or bilinear interpolation (fast).
		#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
		#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
		mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
	)
	#iaa.GaussianBlur(sigma=(0, 3.0))  # Blur images with a sigma of 0 to 3.0.
])

for idx in range(images.shape[0]):
	images_pp[idx] = (images_pp[idx] - np.min(images_pp[idx])) / (np.max(images_pp[idx]) - np.min(images_pp[idx])) * 255

seq_det = seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
#images_aug = seq_det.augment_images(images)
images_aug = seq_det.augment_images(images_pp)
labels_aug = seq_det.augment_images(labels)

# One-hot encoding.
#num_classes = np.unique(labels).shape[0]
#labels = np.uint8(keras.utils.to_categorical(labels, num_classes).reshape(labels.shape + (-1,)))
#labels_aug = np.uint8(keras.utils.to_categorical(labels_aug, num_classes).reshape(labels_aug.shape + (-1,)))

#%%------------------------------------------------------------------

import matplotlib.pyplot as plt

for idx in range(images.shape[0]):
	fig = plt.figure(figsize=(10,10))

	plt.subplot(221)
	#plt.imshow((images[idx] - np.min(images[idx])) / (np.max(images[idx]) - np.min(images[idx])))
	plt.imshow((images_pp[idx] - np.min(images_pp[idx])) / (np.max(images_pp[idx]) - np.min(images_pp[idx])))
	plt.subplot(222)
	#plt.imshow(np.argmax(labels[idx], axis=-1), cmap='gray')
	plt.imshow(labels[idx], cmap='gray')

	plt.subplot(223)
	plt.imshow((images_aug[idx] - np.min(images_aug[idx])) / (np.max(images_aug[idx]) - np.min(images_aug[idx])))
	plt.subplot(224)
	#plt.imshow(np.argmax(labels_aug[idx], axis=-1), cmap='gray')
	plt.imshow(labels_aug[idx], cmap='gray')

	fig.savefig('./generated/img' + str(idx) + '.jpg')

#%%------------------------------------------------------------------

# NOTICE [info] >> Keras does not support keras.utils.data_utils.Sequence.

#from keras.utils.data_utils import Sequence

#class CvpppSequence(Sequence):
#	def __init__(self, images, labels, batch_size, shuffle=False):
#		self.X, self.y = images, labels
#		self.batch_size = batch_size
#		self.shuffle = shuffle
#
#	def __len__(self):
#		return len(self.X) // self.batch_size
#
#	def __getitem__(self, idx):
#		batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
#		batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
#
#		return np.array(batch_x), np.array(batch_y)

#ii = 0
#for batch_images, batch_labels in CvpppSequence(images, labels, 5):
#	print('**************', batch_images.shape, batch_labels.shape)
#	ii = ii + 1

#%%------------------------------------------------------------------

def generate_batch_from_dataset(X, Y, batch_size, shuffle=False):
	num_steps = np.ceil(len(X) / batch_size).astype(np.int)
	if shuffle is True:
		indexes = np.arange(len(X))
		np.random.shuffle(indexes)
		for idx in range(num_steps):
			batch_x = X[indexes[idx*batch_size:(idx+1)*batch_size]]
			batch_y = Y[indexes[idx*batch_size:(idx+1)*batch_size]]
			#yield({'input': batch_x}, {'output': batch_y})
			yield(batch_x, batch_y)
	else:
		for idx in range(num_steps):
			batch_x = X[idx*batch_size:(idx+1)*batch_size]
			batch_y = Y[idx*batch_size:(idx+1)*batch_size]
			#yield({'input': batch_x}, {'output': batch_y})
			yield(batch_x, batch_y)

ii = 0
for batch_images, batch_labels in generate_batch_from_dataset(images, labels, 5):
	#print('**************', ii, type(batch_images), type(batch_labels))
	print('**************', ii, batch_images.shape, batch_labels.shape)
	ii = ii + 1
