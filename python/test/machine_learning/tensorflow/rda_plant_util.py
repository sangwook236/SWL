import numpy as np
from sklearn.model_selection import train_test_split
from swl.machine_vision.util import load_image_list_by_pil, generate_image_patch_list
from swl.machine_learning.util import to_one_hot_encoding
import os, json
from PIL import Image

#%%------------------------------------------------------------------

class RdaPlantDataset(object):
	@staticmethod
	def load_data(image_dir_path, image_suffix, image_extension, label_dir_path, label_suffix, label_extension, num_classes, patch_height, patch_width):
		image_list = load_image_list_by_pil(image_dir_path, image_suffix, image_extension)
		label_list0 = load_image_list_by_pil(label_dir_path, label_suffix, label_extension)
		label_list = []
		for lbl in label_list0:
			label_list.append(lbl // 255)
		label_list0 = None

		assert len(image_list) == len(label_list), '[SWL] Error: The numbers of images and labels are not equal.'
		for (img, lbl) in zip(image_list, label_list):
			assert img.shape[:2] == lbl.shape[:2], '[SWL] Error: The sizes of every corresponding image and label are not equal.'

		# For checking.
		if False:
			fg_ratios = []
			for lbl in label_list:
				fg_ratios.append(np.count_nonzero(lbl) / lbl.size)

			small_image_indices = []
			for (idx, img) in enumerate(image_list):
				if img.shape[0] < patch_height or img.shape[1] < patch_width:
					small_image_indices.append(idx)

		all_image_patches, all_label_patches = [], []
		for (img, lbl) in zip(image_list, label_list):
			if img.shape[0] >= patch_height and img.shape[1] >= patch_width:  # Excludes small-size images.
				img_pats, lbl_pats, _ = generate_image_patch_list(img, lbl, patch_height, patch_width, 0.02)
				if img_pats is not None and lbl_pats is not None:
					all_image_patches += img_pats
					all_label_patches += lbl_pats
					#all_patch_regions += pat_rgns

		assert len(all_image_patches) == len(all_label_patches), 'The number of image patches is not equal to that of label patches.'

		all_image_patches = np.array(all_image_patches)
		all_label_patches = np.array(all_label_patches)
		#all_patch_regions = np.array(all_patch_regions)

		# Pre-process.
		all_image_patches, all_label_patches = RdaPlantDataset.preprocess_data(all_image_patches, all_label_patches, num_classes)

		train_image_patches, test_image_patches, train_label_patches, test_label_patches = train_test_split(all_image_patches, all_label_patches, test_size=0.2, random_state=None)

		return train_image_patches, test_image_patches, train_label_patches, test_label_patches, image_list, label_list

	@staticmethod
	def preprocess_data(data, labels, num_classes, axis=0):
		if data is not None:
			# Preprocessing (normalization, standardization, etc.).
			data = data.astype(np.float32)
			data /= 255.0
			#data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
			#data = np.reshape(data, data.shape + (1,))

		if labels is not None:
			#labels //= 255
			# One-hot encoding (num_examples, height, width) -> (num_examples, height, width, num_classes).
			labels = to_one_hot_encoding(labels, num_classes).astype(np.uint8)

		return data, labels

	@staticmethod
	def load_masks_from_json(data_dir_path, json_file_name):
		with open(data_dir_path + json_file_name) as json_file:
			plant_mask_filenames = json.load(json_file)

		plant_mask_list = []
		max_size = 0, 0
		for pm_file in plant_mask_filenames:
			filepath = os.path.join(data_dir_path, pm_file['plant'])
			plant = np.asarray(Image.open(filepath))
			height, width, _ = plant.shape
			masks = []
			for mask_file in pm_file['masks']:
				filepath = os.path.join(data_dir_path, mask_file)
				mask = np.asarray(Image.open(filepath))
				# For checking.
				mask_height, mask_width = mask.shape
				assert height == mask_height and width == mask_width, 'The size of mask is not equal to that of image.'
				masks.append(mask)
			plant_mask_list.append([plant, masks])
			max_size = max(max_size, (height, width))
		return plant_mask_list, max_size
