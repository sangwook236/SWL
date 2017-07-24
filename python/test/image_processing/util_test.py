import os
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
os.chdir(swl_python_home_dir_path + '/test/image_processing')

import sys
sys.path.append('../../src')

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
# Image to array.

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
#train_images = swl.image_processing.util.load_images_by_scipy(train_image_dir_path, image_suffix, image_extension, width=224, height=224)
#train_labels = swl.image_processing.util.load_labels_by_scipy(train_label_dir_path, label_suffix, label_extension, width=224, height=224)
#val_images = swl.image_processing.util.load_images_by_scipy(val_image_dir_path, image_suffix, image_extension, width=224, height=224)
#val_labels = swl.image_processing.util.load_labels_by_scipy(val_label_dir_path, label_suffix, label_extension, width=224, height=224)
#test_images = swl.image_processing.util.load_images_by_scipy(test_image_dir_path, image_suffix, image_extension, width=224, height=224)
#test_labels = swl.image_processing.util.load_labels_by_scipy(test_label_dir_path, label_suffix, label_extension, width=224, height=224)

#%%------------------------------------------------------------------

num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
val_labels = np.uint8(keras.utils.to_categorical(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

#%%------------------------------------------------------------------
# Save array to npy file.

np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('val_images.npy', val_images)
np.save('val_labels.npy', val_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
#np.savez('train_images.npz', train_images)
#np.savez('train_labels.npz', train_labels)
#np.save('val_images.npz', val_images)
#np.save('val_labels.npz', val_labels)
#np.save('test_images.npz', test_images)
#np.save('test_labels.npz', test_labels)

#%%------------------------------------------------------------------
# Load array from npy file.

train_images0 = np.load('train_images.npy')
train_labels0 = np.load('train_labels.npy')
val_images0 = np.load('val_images.npy')
val_labels0 = np.load('val_labels.npy')
test_images0 = np.load('test_images.npy')
test_labels0 = np.load('test_labels.npy')
#train_images0 = np.load('train_images.npz')
#train_labels0 = np.load('train_labels.npz')
#val_images0 = np.load('val_images.npz')
#val_labels0 = np.load('val_labels.npz')
#test_images0 = np.load('test_images.npz')
#test_labels0 = np.load('test_labels.npz')
