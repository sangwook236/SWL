# REF [site] >> https://github.com/fchollet/keras/issues/3338

import math
import numpy as np
from PIL import Image
from scipy import ndimage, misc
import os, re

#%%------------------------------------------------------------------

# REF [site] >> http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
def to_rgb(gray):
    return np.dstack([gray.astype(np.uint8)] * 3)

#%%------------------------------------------------------------------

def center_crop(x, center_crop_size, **kwargs):
	centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
	halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
	return x[:, centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
	np.random.seed(sync_seed)
	w, h = x.shape[1], x.shape[2]
	rangew = (w - random_crop_size[0]) // 2
	rangeh = (h - random_crop_size[1]) // 2
	offsetw = 0 if rangew == 0 else np.random.randint(rangew)
	offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
	return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

#%%------------------------------------------------------------------

# Load images or label images as a list by PIL.
def load_image_list_by_pil(dir_path, file_suffix, file_extension):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = Image.open(filepath)
					images.append(np.asarray(image))
			break  # Do not include subdirectories.
	return images

# Load images as numpy.array by PIL.
def load_images_by_pil(dir_path, file_suffix, file_extension, width=None, height=None):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = Image.open(filepath)
					if (height is not None and height > 0 and image.size[1] != height) or (width is not None and width > 0 and image.size[0] != width):
						#images.append(np.asarray(image.resize((width, height), resample=Image.NEAREST)))
						#images.append(np.asarray(image.resize((width, height), resample=Image.BICUBIC)))
						images.append(np.asarray(image.resize((width, height), resample=Image.NEAREST)))
					else:
						images.append(np.asarray(image))
			break  # Do not include subdirectories.
	return np.array(images)

# Load label images as numpy.array by PIL.
def load_labels_by_pil(dir_path, file_suffix, file_extension, width=None, height=None):
	labels = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					label = Image.open(filepath)
					if (height is not None and height > 0 and label.size[1] != height) or (width is not None and width > 0 and label.size[0] != width):
						labels.append(np.asarray(label.resize((width, height), resample=Image.NEAREST)))
					else:
						labels.append(np.asarray(label))
			break  # Do not include subdirectories.
	return np.array(labels)

# Load images or label images as a list by scipy.
def load_image_list_by_scipy(dir_path, file_suffix, file_extension):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = ndimage.imread(filepath, mode='RGB')  # RGB image.
					#image = ndimage.imread(filepath)
					images.append(image)
			break  # Do not include subdirectories.
	return images

# Load images as numpy.array by scipy.
def load_images_by_scipy(dir_path, file_suffix, file_extension, width=None, height=None):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = ndimage.imread(filepath, mode='RGB')  # RGB image.
					#image = ndimage.imread(filepath)
					if (height is not None and height > 0 and image.shape[0] != height) or (width is not None and width > 0 and image.shape[1] != width):
						images.append(misc.imresize(image, (height, width)))
					else:
						images.append(image)
			break  # Do not include subdirectories.
	return np.array(images)

# Load label images as numpy.array by scipy.
def load_labels_by_scipy(dir_path, file_suffix, file_extension, width=None, height=None):
	labels = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					label = np.uint16(ndimage.imread(filepath, flatten=True))  # Gray image.
					#label = np.uint16(ndimage.imread(filepath))  # Do not correctly work.
					if (height is not None and height > 0 and label.shape[0] != height) or (width is not None and width > 0 and label.shape[1] != width):
						#labels.append(misc.imresize(label, (height, width)))  # Do not work.
						labels.append(misc.imresize(label, (height, width), interp='nearest'))  # Do not correctly work.
					else:
						labels.append(label)
			break  # Do not include subdirectories.
	return np.array(labels)

#%%------------------------------------------------------------------

# Generate fixed-size image & label
def generate_image_patch_list(image, label, patch_height, patch_width, nonzero_label_ratio_threshold=None):
	if label is not None and False == np.array_equal(image.shape[:2], label.shape[:2]):
		return None, None, None

	rows, cols = math.ceil(image.shape[0] / patch_height), math.ceil(image.shape[1] / patch_width)
	if rows < 1 or cols < 1:
		return None, None, None

	"""
	# Patches without overlap except patches in the last row and column.
	r_stride = patch_height
	c_stride = patch_width
	"""
	# Patches with equal overlap.
	r_stride = (image.shape[0] - patch_height) / (rows - 1)
	c_stride = (image.shape[1] - patch_width) / (cols - 1)

	r_intervals = []
	for r in range(rows):
		r_start = int(r * r_stride)
		if r_start > image.shape[0]:
			break
		if rows - 1 == r:
			r_end = image.shape[0]
			r_start = r_end - patch_height
		else:
			r_end = r_start + patch_height
			#if r_end > image.shape[0]:
			#	r_end = image.shape[0]
			#	r_start = r_end - patch_height
		r_intervals.append((r_start, r_end))

	c_intervals = []
	for c in range(cols):
		c_start = int(c * c_stride)
		if c_start > image.shape[1]:
			break
		if cols - 1 == c:
			c_end = image.shape[1]
			c_start = c_end - patch_width
		else:
			c_end = c_start + patch_width
			#if c_end > image.shape[1]:
			#	c_end = image.shape[1]
			#	c_start = c_end - patch_width
		c_intervals.append((c_start, c_end))

	path_region_list = []
	for r_itv in r_intervals:
		for c_itv in c_intervals:
			path_region_list.append((r_itv[0], c_itv[0], r_itv[1], c_itv[1]))  # (top, left, bottom, right).

	image_patch_list, label_patch_list = [], []
	if label is None or nonzero_label_ratio_threshold is None:
		for rgn in path_region_list:
			image_patch_list.append(image[rgn[0]:rgn[2],rgn[1]:rgn[3]])
	else:
		"""
		for rgn in path_region_list:
			lbl_pat = label[rgn[0]:rgn[2],rgn[1]:rgn[3]]
			if np.count_nonzero(lbl_pat) / lbl_pat.size >= nonzero_label_ratio_threshold:
				image_patch_list.append(image[rgn[0]:rgn[2],rgn[1]:rgn[3]])
				label_patch_list.append(lbl_pat)
			else:
				path_region_list.remove(rgn)  # Do not correctly work.
		"""
		for idx in reversed(range(len(path_region_list))):
			rgn = path_region_list[idx]
			lbl_pat = label[rgn[0]:rgn[2],rgn[1]:rgn[3]]
			if np.count_nonzero(lbl_pat) / lbl_pat.size >= nonzero_label_ratio_threshold:
				image_patch_list.insert(0, image[rgn[0]:rgn[2],rgn[1]:rgn[3]])
				label_patch_list.insert(0, lbl_pat)
			else:
				del path_region_list[idx]
	return image_patch_list, label_patch_list, path_region_list

#%%------------------------------------------------------------------

# REF [size] >> https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def stack_images_horzontally(images):
	'''
	images: PIL image list.
	'''

	widths, heights = zip(*(img.size for img in images))

	total_width = sum(widths)
	max_height = max(heights)

	stacked_img = Image.new('RGB', (total_width, max_height))

	x_offset = 0
	for img in images:
		stacked_img.paste(img, (x_offset, 0))
		x_offset += img.size[0]

	return stacked_img

# REF [size] >> https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def stack_images_vertically(images):
	'''
	images: PIL image list.
	'''

	widths, heights = zip(*(img.size for img in images))

	max_width = max(widths)
	total_height = sum(heights)

	stacked_img = Image.new('RGB', (max_width, total_height))

	y_offset = 0
	for img in images:
		stacked_img.paste(img, (0, y_offset))
		y_offset += img.size[1]

	return stacked_img
