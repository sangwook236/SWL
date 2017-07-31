import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

from swl.machine_learning.keras.cvppp_dataset import create_cvppp_generator, load_cvppp_dataset

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

train_image_dir_path = dataset_home_dir_path + '/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1'
train_label_dir_path = train_image_dir_path

image_suffix = '_rgb'
image_extension = 'png'
label_suffix = '_fg'
label_extension = 'png'

#%%------------------------------------------------------------------

batch_size = 32
use_loaded_dataset = True
shuffle = False

original_image_size = (530, 500)  # (height, width).
resized_image_size = None
#resized_image_size = original_image_size
random_crop_size = None
#random_crop_size = (224, 224)  # (height, width).
center_crop_size = None

# Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

train_dataset_gen = create_cvppp_generator(
		train_image_dir_path, train_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		batch_size=batch_size, resized_image_size=resized_image_size, random_crop_size=random_crop_size, center_crop_size=center_crop_size, use_loaded_dataset=use_loaded_dataset, shuffle=shuffle, seed=seed)

# Usage.
#num_examples = 128
#num_epochs = 10
#steps_per_epoch = num_examples / batch_size
#model.fit_generator(train_dataset_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs)
#for epoch in range(num_epochs):
#	print('Epoch', epoch)
#	num_batches = 0
#	for data_batch, label_batch in train_dataset_gen:
#		model.fit(data_batch, label_batch)
#		num_batches += 1
#		if num_batches >= steps_per_epoch:
#			break

#%%------------------------------------------------------------------
# Load images and convert them to numpy.array.

width, height = None, None
#width, height = 500, 530

train_images, train_labels = load_cvppp_dataset(
		train_image_dir_path, train_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		width=width, height=height)

# Usage.
#num_examples = 128
#batch_size = 32
#num_epochs = 10
#history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs)

#%%------------------------------------------------------------------
# For comparison.

import numpy as np

width, height = 500, 530
num_examples = 128
num_classes = 2
num_epochs = 1
steps_per_epoch = num_examples / batch_size

data_list = []
labels_list = []
for epoch in range(num_epochs):
	print('Epoch', epoch)
	num_batches = 0
	for data_batch, label_batch in train_dataset_gen:
		data_list.append(data_batch)
		labels_list.append(label_batch)
		num_batches += 1
		if num_batches >= steps_per_epoch:
			break

assert len(data_list) == len(labels_list), '[Error] len(data_list) == len(labels_list).'
generated_images = np.ndarray(shape=(num_examples, height, width, 3))
generated_labels = np.ndarray(shape=(num_examples, height, width, num_classes))
for idx in range(len(data_list)):
    start_idx = idx * batch_size
    end_idx = start_idx + data_list[idx].shape[0]
    generated_images[start_idx:end_idx] = data_list[idx]
    generated_labels[start_idx:end_idx] = labels_list[idx]
generated_labels = generated_labels.astype(np.uint8)

np.sum(train_images - generated_images)
np.sum(train_labels - generated_labels)

#%%------------------------------------------------------------------
# Plot.

import matplotlib.pyplot as plt

idx = 0
plt.figure()
plt.imshow((train_images[idx] - np.min(train_images[idx])) / (np.max(train_images[idx]) - np.min(train_images[idx])))
plt.figure()
plt.imshow((generated_images[idx] - np.min(generated_images[idx])) / (np.max(generated_images[idx]) - np.min(generated_images[idx])))
