# REF [site] >> https://github.com/fchollet/keras/issues/3338

import numpy as np
from PIL import Image
from scipy import ndimage, misc
import os, re

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

# Image to numpy.array by PIL.
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
						images.append(np.asarray(image.resize((width, height), resample=Image.BICUBIC)))
					else:
						images.append(np.asarray(image))
			break  # Do not include subdirectories.
	return np.array(images)

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

# Image to numpy.array by scipy.
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

# REF [site] >> http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
def to_rgb(gray):
    return np.dstack([gray.astype(np.uint8)] * 3)

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
