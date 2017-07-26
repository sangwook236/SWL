# REF [file] >> ${SWL_PYTHON_HOME}/test/image_processing/util_test.py

import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

os.chdir(swl_python_home_dir_path + '/test/machine_learning/keras')

#%%------------------------------------------------------------------

import numpy as np
import keras
import swl

#%%------------------------------------------------------------------
# Load data.

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

#%%------------------------------------------------------------------
# Convert image to array.

image_suffix = ''
image_extension = 'png'
label_suffix = ''
label_extension = 'png'

train_images = swl.image_processing.util.load_images_by_pil(train_image_dir_path, image_suffix, image_extension, width=224, height=224)
train_labels = swl.image_processing.util.load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=224, height=224)
val_images = swl.image_processing.util.load_images_by_pil(val_image_dir_path, image_suffix, image_extension, width=224, height=224)
val_labels = swl.image_processing.util.load_labels_by_pil(val_label_dir_path, label_suffix, label_extension, width=224, height=224)
test_images = swl.image_processing.util.load_images_by_pil(test_image_dir_path, image_suffix, image_extension, width=224, height=224)
test_labels = swl.image_processing.util.load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=224, height=224)

#%%------------------------------------------------------------------

num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
val_labels = np.uint8(keras.utils.to_categorical(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

#%%------------------------------------------------------------------
# Save array to npy file.

np.save('camvid_data/train_images.npy', train_images)
np.save('camvid_data/train_labels.npy', train_labels)
np.save('camvid_data/val_images.npy', val_images)
np.save('camvid_data/val_labels.npy', val_labels)
np.save('camvid_data/test_images.npy', test_images)
np.save('camvid_data/test_labels.npy', test_labels)
