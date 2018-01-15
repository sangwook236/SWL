import numpy as np
from PIL import Image
import os, re, math
import matplotlib.pyplot as plt

#%%------------------------------------------------------------------

# REF [function] >> load_images_by_pil() in ${SWL_PYTHON_HOME}/src/swl/image_processing/util.py.
def load_images_by_pil(dir_path, file_suffix, file_extension):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = Image.open(filepath)
					images.append(np.asarray(image))
			break  # Do not include subdirectories.
	#return np.array(images)
	return images

# REF [function] >> load_labels_by_pil() in ${SWL_PYTHON_HOME}/src/swl/image_processing/util.py.
def load_labels_by_pil(dir_path, file_suffix, file_extension):
	labels = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					label = Image.open(filepath)
					labels.append(np.asarray(label))
			break  # Do not include subdirectories.
	#return np.array(labels)
	return labels

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

image_dir_path = dataset_home_dir_path + '/phenotyping/RDA/all_plants'
label_dir_path = dataset_home_dir_path + '/phenotyping/RDA/all_plants_foreground'

image_suffix = ''
image_extension = 'png'
label_suffix = '_foreground'
label_extension = 'png'

patch_height, patch_width = 224, 224

image_list = load_images_by_pil(image_dir_path, image_suffix, image_extension)
label_list = load_labels_by_pil(label_dir_path, label_suffix, label_extension)

assert len(image_list) == len(label_list), '[SWL] Error: The numbers of images and labels are not equal.'
for idx in range(len(image_list)):
	assert image_list[idx].shape[:2] == label_list[idx].shape[:2], '[SWL] Error: The sizes of every corresponding image and label are not equal.'

fg_ratios = []
for idx in range(len(label_list)):
	fg_ratios.append(np.count_nonzero(label_list[idx]) / label_list[idx].size)

small_image_indices = []
for idx in range(len(image_list)):
	if image_list[idx].shape[0] < patch_height or image_list[idx].shape[1] < patch_width:
		small_image_indices.append(idx)

#%%------------------------------------------------------------------

def generate_patches(img, lbl, patch_height, patch_width, fg_ratio_threshold):
	rows, cols = math.ceil(img.shape[0] / patch_height), math.ceil(img.shape[1] / patch_width)
	image_patch_list, label_patch_list = [], []
	"""
	for r in range(rows):
		r_start = r * patch_height
		if r_start > img.shape[0]:
			break
		if rows - 1 == r:
			r_end = img.shape[0]
			r_start = r_end - patch_height
		else:
			r_end = r_start + patch_height
			#if r_end > img.shape[0]:
			#	r_end = img.shape[0]
			#	r_start = r_end - patch_height

		for c in range(cols):
			c_start = c * patch_width
			if c_start > img.shape[1]:
				break
			if cols - 1 == c:
				c_end = img.shape[1]
				c_start = c_end - patch_width
			else:
				c_end = c_start + patch_width
				#if c_end > img.shape[1]:
				#	c_end = img.shape[1]
				#	c_start = c_end - patch_width

			lbl_pat = lbl[r_start:r_end,c_start:c_end]
			if np.count_nonzero(lbl_pat) / lbl_pat.size >= fg_ratio_threshold:
				image_patch_list.append(img[r_start:r_end,c_start:c_end])
				label_patch_list.append(lbl_pat)
	return image_patch_list, label_patch_list
	"""
	r_stride = (img.shape[0] - patch_height) / (rows - 1)
	c_stride = (img.shape[1] - patch_width) / (cols - 1)
	for r in range(rows):
		r_start = int(r * r_stride)
		if r_start > img.shape[0]:
			break
		if rows - 1 == r:
			r_end = img.shape[0]
			r_start = r_end - patch_height
		else:
			r_end = r_start + patch_height
			#if r_end > img.shape[0]:
			#	r_end = img.shape[0]
			#	r_start = r_end - patch_height
		print('#############################', r_start, r_end)

		for c in range(cols):
			c_start = int(c * c_stride)
			if c_start > img.shape[1]:
				break
			if cols - 1 == c:
				c_end = img.shape[1]
				c_start = c_end - patch_width
			else:
				c_end = c_start + patch_width
				#if c_end > img.shape[1]:
				#	c_end = img.shape[1]
				#	c_start = c_end - patch_width
			print('***********************************', c_start, c_end)

			lbl_pat = lbl[r_start:r_end,c_start:c_end]
			if np.count_nonzero(lbl_pat) / lbl_pat.size >= fg_ratio_threshold:
				image_patch_list.append(img[r_start:r_end,c_start:c_end])
				label_patch_list.append(lbl_pat)
	return image_patch_list, label_patch_list

#%%------------------------------------------------------------------

image_patch_list = []
label_patch_list = []
for idx in range(len(image_list)):
	if image_list[idx].shape[0] >= patch_height and image_list[idx].shape[1] >= patch_width:
		img_pats, lbl_pats = generate_patches(image_list[idx], label_list[idx], patch_height, patch_width, 0.02)
		image_patch_list += img_pats
		label_patch_list += lbl_pats
