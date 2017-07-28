import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

from swl.machine_learning.keras.camvid_dataset import create_camvid_generator, load_camvid_dataset

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

train_image_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/train/image'
train_label_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/trainannot/image'
val_image_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/val/image'
val_label_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/valannot/image'
test_image_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/test/image'
test_label_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/testannot/image'

image_suffix = ''
image_extension = 'png'
label_suffix = ''
label_extension = 'png'

#%%------------------------------------------------------------------

batch_size = 32
use_loaded_dataset = True
shuffle = False

original_image_size = (360, 480)  # (height, width).
resized_image_size = None
#resized_image_size = original_image_size
cropped_image_size = None
#cropped_image_size = (224, 224)  # (height, width).

# Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

train_dataset_gen, val_dataset_gen, test_dataset_gen = create_camvid_generator(
		train_image_dir_path, train_label_dir_path, val_image_dir_path, val_label_dir_path, test_image_dir_path, test_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		batch_size=batch_size, resized_image_size=resized_image_size, cropped_image_size=cropped_image_size, use_loaded_dataset=use_loaded_dataset, shuffle=shuffle, seed=seed)

# Usage.
#num_examples = 367
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
#width, height = 480, 360

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_camvid_dataset(
		train_image_dir_path, train_label_dir_path, val_image_dir_path, val_label_dir_path, test_image_dir_path, test_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		width=width, height=height)

# Usage.
#num_examples = 367
#batch_size = 32
#num_epochs = 10
#history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs)

#%%------------------------------------------------------------------
# For comparison.

import numpy as np

data_list = []
labels_list = []
num_examples = 367
num_epochs = 1
steps_per_epoch = num_examples / batch_size
for epoch in range(num_epochs):
	print('Epoch', epoch)
	num_batches = 0
	for data_batch, label_batch in train_dataset_gen:
		data_list.append(data_batch)
		labels_list.append(label_batch)
		num_batches += 1
		if num_batches >= steps_per_epoch:
			break

data = np.ndarray(shape=(367,360,480,3))
for idx in range(len(data_list)):
	start_idx = idx * batch_size
	end_idx = start_idx + data_list[idx].shape[0]
	data[start_idx:end_idx] = data_list[idx]
labels = np.ndarray(shape=(367,360,480,12))
for idx in range(len(labels_list)):
	start_idx = idx * batch_size
	end_idx = start_idx + labels_list[idx].shape[0]
	labels[start_idx:end_idx] = labels_list[idx]
labels = labels.astype(np.uint8)

np.sum(train_images - data)
np.sum(train_labels - labels)
