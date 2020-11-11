#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, glob, time
import numpy as np
import torch, torchvision
import cv2
import labelme_data

def create_augmenter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	augmenter = iaa.Sequential([
		iaa.Sometimes(0.25,
			iaa.Grayscale(alpha=(0.0, 1.0)),  # Requires RGB images.
		),
		#iaa.Sometimes(0.5, iaa.OneOf([
		#	iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
		#	iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
		#	#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
		#	#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
		#])),
		iaa.Sometimes(0.8, iaa.Sequential([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # Translate by -5 to +5 percent along x-axis and -5 to +5 percent along y-axis.
				#rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				#shear=(-2, 2),  # Shear by -2 to +2 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				#order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			iaa.Sometimes(0.75, iaa.OneOf([
				iaa.Sequential([
					iaa.ShearX((-45, 45)),
					iaa.ShearY((-5, 5))
				]),
				iaa.PiecewiseAffine(scale=(0.01, 0.03)),  # Move parts of the image around. Slow.
				#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
				iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
			])),
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
			iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
			iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
			#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.GaussianBlur(sigma=(0.5, 1.5)),
			iaa.AverageBlur(k=(2, 4)),
			iaa.MedianBlur(k=(3, 3)),
			iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0), order=1),
		])),
		#iaa.Sometimes(0.8, iaa.OneOf([
		#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
		#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
		#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

		#	iaa.Invert(p=1, per_channel=False),

		#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
		#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
		#])),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

	return augmenter

class MySubsetDataset(torch.utils.data.Dataset):
	def __init__(self, subset, transform=None, target_transform=None):
		self.subset = subset
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, idx):
		inp, outp = self.subset[idx]
		if self.transform:
			inp = self.transform(inp)
		if self.target_transform:
			outp = self.target_transform(outp)
		return inp, outp

	def __len__(self):
		return len(self.subset)

def visualize_data(dataloader, num_data=10):
	data_iter = iter(dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	images = images.transpose(0, 2, 3, 1)
	for idx, (img, lbl) in enumerate(zip(images, labels)):
		print('Label: {}.'.format(lbl))
		cv2.imshow('Image', img)
		cv2.waitKey(0)
		if idx >= (num_data - 1): break
	cv2.destroyAllWindows()

def FigureLabelMeDataset_test():
	image_height, image_width, image_channel = 320, 320, 3

	is_preloaded_image_used = True
	train_test_ratio = 0.8
	batch_size = 64
	shuffle = True
	num_workers = 4

	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/text/table/sminds'

	json_filepaths = glob.glob(data_dir_path + '/**/*.json', recursive=False)
	print('#loaded JSON files = {}.'.format(len(json_filepaths)))

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(create_augmenter()),
		#ConvertPILMode(mode='RGB'),
		torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_target_transform = torch.IntTensor
	test_transform = torchvision.transforms.Compose([
		#ConvertPILMode(mode='RGB'),
		torchvision.transforms.Resize((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_target_transform = torch.IntTensor

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	dataset = labelme_data.FigureLabelMeDataset(json_filepaths, image_channel, is_preloaded_image_used)

	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)

	train_subset, test_subset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset = MySubsetDataset(train_subset, transform=train_transform, target_transform=train_target_transform)
	test_dataset = MySubsetDataset(test_subset, transform=test_transform, target_transform=test_target_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))
	print('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	#--------------------
	# Show data info.
	data_iter = iter(train_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	#--------------------
	# Visualize.
	visualize_data(train_dataloader, num_data=10)
	visualize_data(test_dataloader, num_data=10)

def main():
	FigureLabelMeDataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
