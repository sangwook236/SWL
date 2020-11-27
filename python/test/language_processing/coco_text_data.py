import os
import numpy as np
import torch
from PIL import Image

#--------------------------------------------------------------------

# REF [site] >> https://bgshih.github.io/cocotext/
#	images = 53686, annotations = 201126.
# REF [class] >> PubLayNetDetectronDataset in ${DataAnalysis_HOME}/app/document_image_processing/pubtabnet_data.py
"""
class CocoTextDataset(torch.utils.data.Dataset):
	def __init__(self, tag, pickle_filepath, data_dir_path, label_converter, image_channel, max_text_len, transform=None, target_transform=None):
		super().__init__()

		if tag not in ['train', 'val']:
			raise ValueError('Invalid tag, {}.'.format(tag))

		self.data_dir_path = data_dir_path
		self.label_converter = label_converter
		self.max_text_len = max_text_len
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_text_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		# Load from a pickle file.
		import pickle
		try:
			with open(pickle_filepath, 'rb') as fd:
				self.data_dicts = pickle.load(fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(pickle_filepath, ex))
			self.data_dicts = None
		assert self.data_dicts

	def __len__(self):
		return len(self.data_dicts)

	def __getitem__(self, idx):
		#return self.data_dicts[idx]

		data_dict = self.data_dicts[idx]

		img_fpath = data_dict['file_name']
		image_height, image_width = data_dict['height'], data_dict['width']
		#image_id = data_dict['image_id']
		annotations = data_dict['annotations']

		# FIXME [fix] >>
		ann_info = annotations

		img_fpath = os.path.join(self.data_dir_path, img_fpath)
		try:
			image = Image.open(img_fpath)
		except IOError as ex:
			print('[SWL] Error: Failed to load an image, {}: {}.'.format(img_fpath, ex))
			image = None
		assert image_height == image.size[1] and image_width == image.size[0], ((image_width, image_height), image.size)

		bbox = ann_info['bbox']
		patch = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

		target = [self.pad_id] * self.max_time_steps
		#target_len = len(text)
		#target[:target_len] = self.label_converter.encode(ann_info['utf8_string'])  # Undecorated label ID.
		textline_id = self.label_converter.encode(ann_info['utf8_string'])  # Decorated/undecorated label ID.
		target_len = min(len(textline_id), self.max_text_len)
		target[:target_len] = textline_id[:target_len]

		if patch and patch.mode != self.mode:
			patch = patch.convert(self.mode)
		#patch = np.array(patch, np.uint8)

		if self.transform:
			patch = self.transform(patch)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return patch, target, target_len
"""
class CocoTextDataset(torch.utils.data.Dataset):
	def __init__(self, tag, json_filepath, data_dir_path, label_converter, image_channel, max_text_len, transform=None, target_transform=None):
		super().__init__()

		if tag not in ['train', 'val']:
			raise ValueError('Invalid tag, {}.'.format(tag))

		self.data_dir_path = data_dir_path
		self.label_converter = label_converter
		self.max_text_len = max_text_len
		self.transform = transform
		self.target_transform = target_transform

		self.pad_id = label_converter.pad_id
		self.max_time_steps = max_text_len + label_converter.num_affixes

		if image_channel == 1:
			self.mode = 'L'
			#self.mode = '1'
		elif image_channel == 3:
			self.mode = 'RGB'
		elif image_channel == 4:
			self.mode = 'RGBA'
		else:
			raise ValueError('Invalid image channel, {}'.format(image_channel))

		# Load from a JSON file.
		import json
		try:
			with open(json_filepath, encoding='UTF8') as fd:
				self.coco_text = json.load(fd)
		except UnicodeDecodeError as ex:
			print('Unicode decode error in {}: {}.'.format(json_filepath, ex))
			self.coco_text = None
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(json_filepath, ex))
			self.coco_text = None
		assert self.coco_text

		image_keys = list(key for key, val in self.coco_text['imgs'].items() if val['set'] == tag)

		self.annotation_ids = set()
		for img_key in image_keys:
			self.annotation_ids.update(self.coco_text['imgToAnns'][img_key])
		self.annotation_ids = list(self.annotation_ids)
		assert self.annotation_ids

	def __len__(self):
		return len(self.annotation_ids)

	def __getitem__(self, idx):
		ann_info = self.coco_text['anns'][str(self.annotation_ids[idx])]

		image_id = ann_info['image_id']
		img_info = self.coco_text['imgs'][str(image_id)]

		img_fpath = img_info['file_name']
		image_height, image_width = img_info['height'], img_info['width']

		img_fpath = os.path.join(self.data_dir_path, img_fpath)
		try:
			image = Image.open(img_fpath)
		except IOError as ex:
			print('[SWL] Error: Failed to load an image, {}: {}.'.format(img_fpath, ex))
			image = None
		assert image_height == image.size[1] and image_width == image.size[0], ((image_width, image_height), image.size)

		bbox = ann_info['bbox']
		patch = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

		target = [self.pad_id] * self.max_time_steps
		#target_len = len(text)
		#target[:target_len] = self.label_converter.encode(ann_info['utf8_string'])  # Undecorated label ID.
		textline_id = self.label_converter.encode(ann_info['utf8_string'])  # Decorated/undecorated label ID.
		target_len = min(len(textline_id), self.max_text_len)
		target[:target_len] = textline_id[:target_len]

		if patch and patch.mode != self.mode:
			patch = patch.convert(self.mode)
		#patch = np.array(patch, np.uint8)

		if self.transform:
			patch = self.transform(patch)
		if self.target_transform:
			target = self.target_transform(target)
		target_len = torch.tensor(target_len, dtype=torch.int32)

		return patch, target, target_len
