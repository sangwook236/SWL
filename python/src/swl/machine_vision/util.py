# REF [site] >> https://github.com/fchollet/keras/issues/3338

import os, re, math, csv
import numpy as np
from PIL import Image
from scipy import ndimage, misc
import cv2
import swl.util.util as swl_util

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

def load_images_from_files(image_filepaths, height, width, channels):
	images, valid_indices = list(), list()
	for idx, filepath in enumerate(image_filepaths):
		if 'grayscale' == channels or 1 == channels:
			img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
		elif 'RGB' == channels or 'BGR' == channels or 3 == channels:
			img = cv2.imread(filepath, cv2.IMREAD_COLOR)
		else:
			img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
		#try:
		#	img = Image.open(filepath)
		#except IOError as ex:
		#	print('Failed to load an image:', filepath)
		#	continue
		#img = np.asarray(img, dtype=np.uint8)

		if img is None:
			print('Failed to load an image:', filepath)
			continue
		if img.shape[0] != height or img.shape[1] != width:
			img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
		images.append(img)
		valid_indices.append(idx)
	return np.array(images), (np.array(valid_indices, dtype=np.int) if valid_indices else None)
	#return images, valid_indices

def save_images_to_npy_files(image_filepaths, labels, image_height, image_width, image_channels, num_files_to_load_at_a_time, save_dir_path, input_filename_format, output_filename_format, npy_file_csv_filename, data_processing_functor=None):
	if image_height is None or image_width is None or image_height <= 0 or image_width <= 0:
		raise ValueError('Invalid image width or height')

	num_files = len(image_filepaths)
	if num_files <= 0 or len(labels) != num_files:
		raise ValueError('Invalid image filepaths or labels')

	swl_util.make_dir(save_dir_path)

	with open(os.path.join(save_dir_path, npy_file_csv_filename), mode='w', encoding='UTF8', newline='') as csvfile:
		writer = csv.writer(csvfile)

		npy_file_idx = 0
		for start_idx in range(0, num_files, num_files_to_load_at_a_time):
			inputs, valid_input_indices = load_images_from_files(image_filepaths[start_idx:start_idx+num_files_to_load_at_a_time], image_height, image_width, image_channels)
			if valid_input_indices is None:
				print('No valid data in npy file #{}.'.format(npy_file_idx))
				continue
			outputs = np.array(labels[start_idx:start_idx+num_files_to_load_at_a_time])

			if len(valid_input_indices) != len(outputs):
				outputs = outputs[valid_input_indices]
			if len(inputs) != len(outputs):
				print('The number of inputs is not equal to that of outputs in npy file #{}: input size = {}, output shape = {}.'.format(npy_file_idx, inputs.shape, outputs.shape))
				continue

			if data_processing_functor:
				inputs, outputs = data_processing_functor(inputs, outputs)

			input_filepath, output_filepath = os.path.join(save_dir_path, input_filename_format.format(npy_file_idx)), os.path.join(save_dir_path, output_filename_format.format(npy_file_idx))
			np.save(input_filepath, inputs)
			np.save(output_filepath, outputs)
			writer.writerow((input_filepath, output_filepath, len(inputs)))

			npy_file_idx += 1

#%%------------------------------------------------------------------

# Load images or label images as a list by PIL.
def load_image_list_by_pil(dir_path, file_suffix, file_extension, is_recursive=False):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = Image.open(filepath)
					images.append(np.asarray(image))
			if not is_recursive:
				break  # Do not include subdirectories.
	return images

# Load images as numpy.array by PIL.
def load_images_by_pil(dir_path, file_suffix, file_extension, width=None, height=None, is_recursive=False):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
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
			if not is_recursive:
				break  # Do not include subdirectories.
	return np.array(images)

# Load label images as numpy.array by PIL.
def load_labels_by_pil(dir_path, file_suffix, file_extension, width=None, height=None, is_recursive=False):
	labels = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					label = Image.open(filepath)
					if (height is not None and height > 0 and label.size[1] != height) or (width is not None and width > 0 and label.size[0] != width):
						labels.append(np.asarray(label.resize((width, height), resample=Image.NEAREST)))
					else:
						labels.append(np.asarray(label))
			if not is_recursive:
				break  # Do not include subdirectories.
	return np.array(labels)

# Load images or label images as a list by scipy.
def load_image_list_by_scipy(dir_path, file_suffix, file_extension, is_recursive=False):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = ndimage.imread(filepath, mode='RGB')  # RGB image.
					#image = ndimage.imread(filepath)
					images.append(image)
			if not is_recursive:
				break  # Do not include subdirectories.
	return images

# Load images as numpy.array by scipy.
def load_images_by_scipy(dir_path, file_suffix, file_extension, width=None, height=None, is_recursive=False):
	images = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
			for filename in filenames:
				if re.search(file_suffix + '\.' + file_extension + '$', filename):
					filepath = os.path.join(root, filename)
					image = ndimage.imread(filepath, mode='RGB')  # RGB image.
					#image = ndimage.imread(filepath)
					if (height is not None and height > 0 and image.shape[0] != height) or (width is not None and width > 0 and image.shape[1] != width):
						images.append(misc.imresize(image, (height, width)))
					else:
						images.append(image)
			if not is_recursive:
				break  # Do not include subdirectories.
	return np.array(images)

# Load label images as numpy.array by scipy.
def load_labels_by_scipy(dir_path, file_suffix, file_extension, width=None, height=None, is_recursive=False):
	labels = []
	if dir_path is not None:
		for root, dirnames, filenames in os.walk(dir_path):
			filenames.sort()
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
			if not is_recursive:
				break  # Do not include subdirectories.
	return np.array(labels)

#%%------------------------------------------------------------------

def generate_image_patch_list(image, label, patch_height, patch_width, nonzero_label_ratio_threshold=None):
	"""
	Generate fixed-size image & label patches.
	Args:
	Returns:
	"""
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
	r_stride = (image.shape[0] - patch_height) / (rows - 1) if rows > 1 else patch_height
	c_stride = (image.shape[1] - patch_width) / (cols - 1) if cols > 1 else patch_width

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
	if label is None:
		for rgn in path_region_list:
			image_patch_list.append(image[rgn[0]:rgn[2],rgn[1]:rgn[3]])
	elif nonzero_label_ratio_threshold is None:
		for rgn in path_region_list:
			image_patch_list.append(image[rgn[0]:rgn[2],rgn[1]:rgn[3]])
			label_patch_list.append(label[rgn[0]:rgn[2],rgn[1]:rgn[3]])
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

def stitch_label_patches(patches, regions, shape):
	"""
	Stitches partially overlapped label patches to create a label.
	Args:
		patches (np.array): Label patches have labels expressed in one-hot encoding.
		regions (np.array): Each row is (top, right, bottom, left).
		shape (tuple): The size of the output array.
	Returns:
		np.array: array of shape 'shape'
	"""
	if patches.shape[0] != regions.shape[0]:
		return None

	stitched = np.zeros(shape + (patches.shape[-1],))
	for idx in range(patches.shape[0]):
		rgn = regions[idx]
		stitched[rgn[0]:rgn[2],rgn[1]:rgn[3]] += patches[idx]
	return np.argmax(stitched, -1).astype(np.uint8)  # Its shape = shape.

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
