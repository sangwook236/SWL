import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
sys.path.append(swl_python_home_dir_path + '/src')
#sys.path.append('../../src')

#--------------------
import numpy as np
from swl.image_processing.util import load_image_list_by_pil, generate_image_patch_list

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

image_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants'
label_dir_path = data_home_dir_path + '/phenotyping/RDA/all_plants_foreground'

image_suffix = ''
image_extension = 'png'
label_suffix = '_foreground'
label_extension = 'png'

patch_height, patch_width = 224, 224

image_list = load_image_list_by_pil(image_dir_path, image_suffix, image_extension)
label_list = load_image_list_by_pil(label_dir_path, label_suffix, label_extension)

assert len(image_list) == len(label_list), '[SWL] Error: The numbers of images and labels are not equal.'
for idx in range(len(image_list)):
	assert image_list[idx].shape[:2] == label_list[idx].shape[:2], '[SWL] Error: The sizes of every corresponding image and label are not equal.'

if False:
	fg_ratios = []
	for idx in range(len(label_list)):
		fg_ratios.append(np.count_nonzero(label_list[idx]) / label_list[idx].size)

	small_image_indices = []
	for idx in range(len(image_list)):
		if image_list[idx].shape[0] < patch_height or image_list[idx].shape[1] < patch_width:
			small_image_indices.append(idx)

#%%------------------------------------------------------------------

image_patch_list, label_patch_list, patch_region_list = [], [], []
for idx in range(len(image_list)):
	if image_list[idx].shape[0] >= patch_height and image_list[idx].shape[1] >= patch_width:
		img_pats, lbl_pats, pat_rgns = generate_image_patch_list(image_list[idx], label_list[idx], patch_height, patch_width, 0.02)
		if img_pats is not None and lbl_pats is not None and pat_rgns is not None:
			image_patch_list += img_pats
			label_patch_list += lbl_pats
			patch_region_list += pat_rgns

image_patches = np.array(image_patch_list)
label_patches = np.array(label_patch_list)
patch_regions = np.array(patch_region_list)

#%%------------------------------------------------------------------
# Save a numpy.array to an npy file.

np.save('./image_patches.npy', image_patches)
np.save('./label_patches.npy', label_patches)

#%%------------------------------------------------------------------
# Load a numpy.array from an npy file.

image_patches0 = np.load('./image_patches.npy')
label_patches0 = np.load('./label_patches.npy')
