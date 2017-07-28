import os, sys
from swl.machine_learning.keras.camvid_dataset import create_camvid_generator, load_camvid_dataset

if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')

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
#resized_image_size = None
resized_image_size = (360, 480)  # (height, width).
cropped_image_size = None
#cropped_image_size = (224, 224)  # (height, width).
use_loaded_dataset = True

# Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

train_dataset_gen, val_dataset_gen, test_dataset_gen = create_camvid_generator(
		train_image_dir_path, train_label_dir_path, val_image_dir_path, val_label_dir_path, test_image_dir_path, test_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		batch_size=batch_size, resized_image_size=resized_image_size, cropped_image_size=cropped_image_size, use_loaded_dataset=use_loaded_dataset, seed=seed)

#%%------------------------------------------------------------------
# Load images and convert them to numpy.array.

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_camvid_dataset(
		train_image_dir_path, train_label_dir_path, val_image_dir_path, val_label_dir_path, test_image_dir_path, test_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		resized_image_size=resized_image_size)
