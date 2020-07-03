#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, time
import numpy as np
import torch, torchvision
import cv2
import swl.language_processing.util as swl_langproc_util
import text_generation_util as tg_util
import aihub_data

def aihub_printed_text_data_loading_test():
	import json

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
	aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

	try:
		print('Start loading AI Hub dataset info...')
		start_time = time.time()
		with open(aihub_data_json_filepath, encoding='UTF8') as fd:
			json_data = json.load(fd)
		print('End loading AI Hub dataset info: {} secs.'.format(time.time() - start_time))
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(aihub_data_json_filepath))
		return
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(aihub_data_json_filepath))
		return

	print('#images = {}, #annotations = {}.'.format(len(json_data['images']), len(json_data['annotations'])))
	assert len(json_data['images']) == len(json_data['annotations'])

	#--------------------
	if False:
		print('Info:', json_data['info'])
		for idx, img in enumerate(json_data['images']):
			print('id: {}, width: {}, height: {}, file_name: {}, date_captured: {}.'.format(int(img['id']), img['width'], img['height'], img['file_name'], img['date_captured']))
			if idx >= 9: break
		for idx, anno in enumerate(json_data['annotations']):
			print('attributes: (font: {}, type: {}, is_aug: {}), id: {}, image_id: {}, text: {}.'.format(anno['attributes']['font'], anno['attributes']['type'], anno['attributes']['is_aug'], int(anno['id']), int(anno['image_id']), anno['text']))
			if idx >= 9: break
		print('Licenses:', json_data['licenses'])

	if False:
		id_set, file_name_set = set(), set()
		for info in json_data['images']:
			id_set.add(int(info['id']))
			file_name_set.add(info['file_name'])

		print('images - id: {} IDs.'.format(len(id_set)))
		print('images - file_name: {} files.'.format(len(file_name_set)))

	if False:
		font_set, type_set, is_aug_set = set(), set(), set()
		id_set, image_id_set, text_set = set(), set(), set()
		for info in json_data['annotations']:
			font_set.add(info['attributes']['font'])  # 50 fonts.
			type_set.add(info['attributes']['type'])  # {'글자(음절)', '문장', '단어(어절)'}.
			is_aug_set.add(info['attributes']['is_aug'])  # {False}.

			id_set.add(int(info['id']))
			image_id_set.add(int(info['image_id']))
			text_set.add(info['text'])

		print('annotations - attributes - font: {} fonts.'.format(len(font_set)))
		print('annotations - attributes - type: {}.'.format(type_set))
		print('annotations - attributes - is_aug: {}.'.format(is_aug_set))

		print('annotations - id: {} IDs.'.format(len(id_set)))
		print('annotations - image_id: {} image IDs.'.format(len(image_id_set)))
		print('annotations - text: {} texts.'.format(len(text_set)))

	#--------------------
	image_type_mapper = {'글자(음절)': 'syllable', '단어(어절)': 'word', '문장': 'sentence'}
	additional_data_dir_path = '01_printed_{}_images'
	image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.

	print('Start loading AI Hub dataset...')
	start_time = time.time()
	image_infos = dict()
	for info in json_data['images']:
		image_infos[int(info['id'])] = {'width': info['width'], 'height': info['height'], 'file_name': info['file_name']}

	for info in json_data['annotations']:
		img_type = image_type_mapper[info['attributes']['type']]
		if img_type not in image_types_to_load: continue

		img_id = int(info['image_id'])
		label = info['text']
		img_height, img_width = image_infos[img_id]['height'], image_infos[img_id]['width']
		img_fname = os.path.join(additional_data_dir_path.format(img_type), image_infos[img_id]['file_name'])
		
		if False:
			img_fpath = os.path.join(aihub_data_dir_path, img_fname)
			img = cv2.imread(img_fpath)
			if img is None:
				print('Failed to load an image, {}.'.format(img_fpath))
				continue

			assert img.shape[0] == img_height and img.shape[1] == img_width
	print('End loading AI Hub dataset: {} secs.'.format(time.time() - start_time))

# REF [function] >> visualize_data_with_length() in test_data_test.py.
def visualize_data_with_length(dataloader, label_converter, num_data=10):
	data_iter = iter(dataloader)
	images, labels, label_lens = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	images = images.transpose(0, 2, 3, 1)
	for idx, (img, lbl, l) in enumerate(zip(images, labels, label_lens)):
		print('Label (len={}): {} (int), {} (str).'.format(l, [ll for ll in lbl if ll != label_converter.pad_value], label_converter.decode(lbl)))
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		if idx >= (num_data - 1): break
	cv2.destroyAllWindows()

# REF [class] >> ResizeImage in test_data_test.py.
class ResizeImage(object):
	def __init__(self, height, width, is_pil=True):
		self.height, self.width = height, width
		self.resize_functor = self._resize_by_pil if is_pil else self._resize_by_opencv

	def __call__(self, x):
		return self.resize_functor(x, self.height, self.width)

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_opencv() in text_line_data.py.
	def _resize_by_opencv(self, input, height, width, *args, **kwargs):
		interpolation = cv2.INTER_AREA
		"""
		hi, wi = input.shape[:2]
		if wi >= width:
			return cv2.resize(input, (width, height), interpolation=interpolation)
		else:
			aspect_ratio = height / hi
			min_width = min(width, int(wi * aspect_ratio))
			input = cv2.resize(input, (min_width, height), interpolation=interpolation)
			if min_width < width:
				image_zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
				image_zeropadded[:,:min_width] = input[:,:min_width]
				return image_zeropadded
			else:
				return input
		"""
		hi, wi = input.shape[:2]
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = np.zeros((height, width) + input.shape[2:], dtype=input.dtype)
		zeropadded[:,:min_width] = cv2.resize(input, (min_width, height), interpolation=interpolation)
		return zeropadded
		"""
		return cv2.resize(input, (width, height), interpolation=interpolation)
		"""

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_pil() in text_line_data.py.
	def _resize_by_pil(self, input, height, width, *args, **kwargs):
		import PIL.Image

		interpolation = PIL.Image.BICUBIC
		wi, hi = input.size
		aspect_ratio = height / hi
		min_width = min(width, int(wi * aspect_ratio))
		zeropadded = PIL.Image.new(input.mode, (width, height), color=0)
		zeropadded.paste(input.resize((min_width, height), resample=interpolation), (0, 0, min_width, height))
		return zeropadded
		"""
		return input.resize((width, height), resample=interpolation)
		"""

# REF [class] >> ToIntTensor in test_data_test.py.
class ToIntTensor(object):
	def __call__(self, lst):
		return torch.IntTensor(lst)

# REF [class] >> MySubsetDataset in test_data_test.py.
class MySubsetDataset(torch.utils.data.Dataset):
	def __init__(self, subset, transform=None, target_transform=None):
		self.subset = subset
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, idx):
		inp, outp, outp_len = self.subset[idx]
		if self.transform:
			inp = self.transform(inp)
		if self.target_transform:
			outp = self.target_transform(outp)
		return inp, outp, outp_len

	def __len__(self):
		return len(self.subset)

# REF [function] >> SimpleWordDataset_test() in test_data_test.py.
def AiHubPrintedTextDataset_test():
	image_height, image_width, image_channel = 64, 640, 3
	#image_height_before_crop, image_width_before_crop = int(image_height * 1.1), int(image_width * 1.1)
	image_height_before_crop, image_width_before_crop = image_height, image_width

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	aihub_data_json_filepath = data_base_dir_path + '/ai_hub/korean_font_image/printed/printed_data_info.json'
	aihub_data_dir_path = data_base_dir_path + '/ai_hub/korean_font_image/printed'

	image_types_to_load = ['word']  # {'syllable', 'word', 'sentence'}.
	max_label_len = 10
	is_image_used = False

	charset = tg_util.construct_charset(space=False)

	train_test_ratio = 0.8
	batch_size = 64
	shuffle = True
	num_workers = 4

	#--------------------
	train_transform = torchvision.transforms.Compose([
		ResizeImage(image_height_before_crop, image_width_before_crop),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = ToIntTensor()
	test_transform = torchvision.transforms.Compose([
		ResizeImage(image_height, image_width),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = ToIntTensor()

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	label_converter = swl_langproc_util.TokenConverter(list(charset), pad_value=None)
	#label_converter = swl_langproc_util.TokenConverter(list(charset), use_sos=True, use_eos=True, pad_value=None)
	dataset = aihub_data.AiHubPrintedTextDataset(label_converter, aihub_data_json_filepath, aihub_data_dir_path, image_types_to_load, image_height, image_width, image_channel, max_label_len, is_image_used)

	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))
	print('#classes = {}.'.format(label_converter.num_tokens))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Show data info.
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels, label_lens = data_iter.next()
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Train label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels, label_lens = data_iter.next()
	images, labels, label_lens = images.numpy(), labels.numpy(), label_lens.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
	print('Test label length: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(label_lens.shape, label_lens.dtype, np.min(label_lens), np.max(label_lens)))

	#--------------------
	# Visualize.
	visualize_data_with_length(train_dataloader, label_converter, num_data=10)
	visualize_data_with_length(test_dataloader, label_converter, num_data=10)

def main():
	#aihub_printed_text_data_loading_test()

	AiHubPrintedTextDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
