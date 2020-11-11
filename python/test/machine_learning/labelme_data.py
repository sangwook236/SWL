import os, math, json
import numpy as np
import torch
from PIL import Image
import cv2

#--------------------------------------------------------------------

# REF [site] >> https://github.com/wkentaro/labelme
class LabelMeDataset(torch.utils.data.Dataset):
	def __init__(self, json_filepaths, image_channel, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__()

		self.is_preloaded_image_used = is_preloaded_image_used
		self.transform = transform
		self.target_transform = target_transform

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		self.patches, self.labels = self._load_from_json(json_filepaths, image_channel, is_preloaded_image_used)
		assert (self.patches is None and self.labels is None) or len(self.patches) == len(self.labels)

	def __len__(self):
		raise NotImplementedError

	def __getitem__(self, idx):
		raise NotImplementedError

	def _load_from_json(self, json_filepaths, image_channel, is_preloaded_image_used):
		"""
		version
		flags
		shapes *
			label
			line_color
			fill_color
			points
			shape_type
		lineColor
		fillColor
		imagePath
		imageData
		imageWidth
		imageHeight
		"""

		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		for json_filepath in json_filepaths:
			try:
				with open(json_filepath, 'r') as fd:
					json_data = json.load(fd)
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
				continue
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
				continue

			#version = json_data['version']
			#flags = json_data['flags']
			#line_color, fill_color = json_data['lineColor'], json_data['fillColor']

			dir_path = os.path.dirname(json_filepath)
			image_filepath = os.path.join(dir_path, json_data['imagePath'])
			#image_data = os.path.join(dir_path, json_data['imageData'])
			image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

			img = cv2.imread(image_filepath, flag)
			if img is None:
				print('[SWL] Error: Failed to load an image, {}.'.format(image_filepath))
				continue

			for shape in json_data['shapes']:
				label, points, shape_type = shape['label'], shape['points'], shape['shape_type']
				#shape_line_color, shape_fill_color = shape['line_color'], shape['fill_color']

		return None, None

#--------------------------------------------------------------------

class FigureLabelMeDataset(LabelMeDataset):
	def __init__(self, json_filepaths, image_channel, is_preloaded_image_used=True, transform=None, target_transform=None):
		super().__init__(json_filepaths, image_channel, is_preloaded_image_used, transform, target_transform)

	def __len__(self):
		return len(self.patches)

	def __getitem__(self, idx):
		if self.is_preloaded_image_used:
			image = Image.fromarray(self.patches[idx])
		else:
			img_fpath, rct = self.images[idx]  # (left, top, right, bottom).
			try:
				image = Image.open(img_fpath).crop(rct)
			except IOError as ex:
				print('[SWL] Error: Failed to load an image, {}: {}.'.format(img_fpath, ex))
				image = None

		if image and image.mode != self.mode:
			image = image.convert(self.mode)
		#image = np.array(image, np.uint8)
		label = self.labels[idx]

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)

		return image, label

	def _load_from_json(self, json_filepaths, image_channel, is_preloaded_image_used):
		"""
		version
		flags
		shapes *
			label
			line_color
			fill_color
			points
			shape_type
		lineColor
		fillColor
		imagePath
		imageData
		imageWidth
		imageHeight
		"""

		if 1 == image_channel:
			flag = cv2.IMREAD_GRAYSCALE
		elif 3 == image_channel:
			flag = cv2.IMREAD_COLOR
		elif 4 == image_channel:
			flag = cv2.IMREAD_ANYCOLOR  # ?
		else:
			flag = cv2.IMREAD_UNCHANGED

		#valid_labels = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear']
		valid_labels = ['table-all', 'table-partial', 'table-hv', 'table-bare', 'picture', 'diagram', 'unclear', 'table-all_unclear', 'table-partial_unclear', 'table-hv_unclear', 'table-bare_unclear', 'picture_unclear', 'diagram_unclear']
		patches, labels = list(), list()
		for json_filepath in json_filepaths:
			try:
				with open(json_filepath, 'r') as fd:
					json_data = json.load(fd)
			except UnicodeDecodeError as ex:
				print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
				continue
			except FileNotFoundError as ex:
				print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
				continue

			dir_path = os.path.dirname(json_filepath)
			image_filepath = os.path.join(dir_path, json_data['imagePath'])
			image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

			img = cv2.imread(image_filepath, flag)
			if img is None:
				print('[SWL] Warning: Failed to load an image, {}.'.format(image_filepath))
				continue
			if img.shape[0] != image_height or img.shape[1] != image_width:
				print('[SWL] Warning: Invalid image shape, ({}, {}) != ({}, {}).'.format(img.shape[0], img.shape[1], image_height, image_width))
				continue

			for shape in json_data['shapes']:
				label, shape_type = shape['label'], shape['shape_type']
				if label not in valid_labels:
					print('[SWL] Warning: Invalid label, {}.'.format(label))
					continue
				if shape_type != 'rectangle':
					print('[SWL] Warning: Invalid shape type, {}.'.format(shape_type))
					continue

				(left, top), (right, bottom) = shape['points']
				assert left >= 0 and top >= 0 and right <= img.shape[1] and bottom <= img.shape[0]
				left, top, right, bottom = math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
				patches.append(img[top:bottom,left:right] if is_preloaded_image_used else (image_filepath, (left, top, right, bottom)))
				labels.append([valid_labels.index(label)])

		##patches = list(map(lambda patch: self.resize(patch), patches))
		#patches = self._transform_images(np.array(patches), use_NWHC=self._use_NWHC)
		#patches, _ = self.preprocess(patches, None)

		return patches, labels
