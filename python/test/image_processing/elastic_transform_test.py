# REF [site] >> https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

%matplotlib inline

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
import cv2
import matplotlib.pyplot as plt
import swl

# Draw a grid.
def draw_grid(img, grid_size):
	# Draw grid lines.
	for i in range(0, img.shape[1], grid_size):
		cv2.line(img, (i, 0), (i, img.shape[0]), color=(255,))
	for j in range(0, img.shape[0], grid_size):
		cv2.line(img, (0, j), (img.shape[1], j), color=(255,))

# Load images.
img = cv2.imread("../../data/image_processing/em_train_00.tif", -1)
mask = cv2.imread("../../data/image_processing/em_train_00_mask.tif", -1)

# Draw grid lines.
draw_grid(img, 50)
draw_grid(mask, 50)

# Merge images into separete channels (shape will be (cols, rols, 2)).
img_merged = np.concatenate((img[...,None], mask[...,None]), axis=2)

# Apply transformation on image.
img_merged_transformed = swl.image_processing.elastic_transform.elastic_transform(img_merged, img_merged.shape[1] * 2, img_merged.shape[1] * 0.08, img_merged.shape[1] * 0.08)

# Split image and mask.
img_transformed = img_merged_transformed[...,0]
mask_transformed = img_merged_transformed[...,1]

# Display result.
plt.figure(figsize = (16,14))
plt.imshow(np.c_[np.r_[img, mask], np.r_[img_transformed, mask_transformed]], cmap='gray')
