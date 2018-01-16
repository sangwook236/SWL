import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')
#sys.path.append('../../src')

#--------------------
import numpy as np
import keras
import swl

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

train_image_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/train/image'
train_label_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/trainannot/image'
val_image_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/val/image'
val_label_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/valannot/image'
test_image_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/test/image'
test_label_dir_path = data_home_dir_path + '/pattern_recognition/camvid/tmp/testannot/image'

#%%------------------------------------------------------------------
# Convert image to array.

image_suffix = ''
image_extension = 'png'
label_suffix = ''
label_extension = 'png'

#image_width, image_height = None, None
#image_width, image_height = 480, 360
image_width, image_height = 224, 224

train_images = swl.image_processing.util.load_images_by_pil(train_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
train_labels = swl.image_processing.util.load_labels_by_pil(train_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
val_images = swl.image_processing.util.load_images_by_pil(val_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
val_labels = swl.image_processing.util.load_labels_by_pil(val_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
test_images = swl.image_processing.util.load_images_by_pil(test_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
test_labels = swl.image_processing.util.load_labels_by_pil(test_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
#train_images = swl.image_processing.util.load_images_by_scipy(train_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
#train_labels = swl.image_processing.util.load_labels_by_scipy(train_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
#val_images = swl.image_processing.util.load_images_by_scipy(val_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
#val_labels = swl.image_processing.util.load_labels_by_scipy(val_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)
#test_images = swl.image_processing.util.load_images_by_scipy(test_image_dir_path, image_suffix, image_extension, width=image_width, height=image_height)
#test_labels = swl.image_processing.util.load_labels_by_scipy(test_label_dir_path, label_suffix, label_extension, width=image_width, height=image_height)

#%%------------------------------------------------------------------

num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
train_labels = np.uint8(keras.utils.to_categorical(train_labels, num_classes).reshape(train_labels.shape + (-1,)))
val_labels = np.uint8(keras.utils.to_categorical(val_labels, num_classes).reshape(val_labels.shape + (-1,)))
test_labels = np.uint8(keras.utils.to_categorical(test_labels, num_classes).reshape(test_labels.shape + (-1,)))

#%%------------------------------------------------------------------
# Save a numpy.array to an npy file.

np.save('camvid_data/train_images.npy', train_images)
np.save('camvid_data/train_labels.npy', train_labels)
np.save('camvid_data/val_images.npy', val_images)
np.save('camvid_data/val_labels.npy', val_labels)
np.save('camvid_data/test_images.npy', test_images)
np.save('camvid_data/test_labels.npy', test_labels)
#np.savez('camvid_data/train_images.npz', train_images)
#np.savez('camvid_data/train_labels.npz', train_labels)
#np.savez('camvid_data/val_images.npz', val_images)
#np.savez('camvid_data/val_labels.npz', val_labels)
#np.savez('camvid_data/test_images.npz', test_images)
#np.savez('camvid_data/test_labels.npz', test_labels)

#%%------------------------------------------------------------------
# Load a numpy.array from an npy file.

train_images0 = np.load('camvid_data/train_images.npy')
train_labels0 = np.load('camvid_data/train_labels.npy')
val_images0 = np.load('camvid_data/val_images.npy')
val_labels0 = np.load('camvid_data/val_labels.npy')
test_images0 = np.load('camvid_data/test_images.npy')
test_labels0 = np.load('camvid_data/test_labels.npy')
#train_images0 = np.load('camvid_data/train_images.npz')
#train_labels0 = np.load('camvid_data/train_labels.npz')
#val_images0 = np.load('camvid_data/val_images.npz')
#val_labels0 = np.load('camvid_data/val_labels.npz')
#test_images0 = np.load('camvid_data/test_images.npz')
#test_labels0 = np.load('camvid_data/test_labels.npz')
