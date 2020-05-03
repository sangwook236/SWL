#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, functools, glob, time
import numpy as np
import torch, torchvision
from PIL import Image, ImageOps
import cv2
#import mixup.vgg, mixup.resnet

def save_model(model_filepath, model):
	#torch.save(model.state_dict(), model_filepath)
	torch.save({'state_dict': model.state_dict()}, model_filepath)
	print('Saved a model to {}.'.format(model_filepath))

def load_model(model_filepath, model, device='cpu'):
	loaded_data = torch.load(model_filepath, map_location=device)
	#model.load_state_dict(loaded_data)
	model.load_state_dict(loaded_data['state_dict'])
	print('Loaded a model from {}.'.format(model_filepath))
	return model

class Net(torch.nn.Module):
	def __init__(self, num_classes, input_channels=1):
		super(Net, self).__init__()
		self.conv1 = torch.nn.Conv2d(input_channels, 32, 3, 1)
		self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = torch.nn.Dropout2d(0.25)
		self.dropout2 = torch.nn.Dropout2d(0.5)
		self.fc1 = torch.nn.Linear(9216, 128)
		self.fc2 = torch.nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = torch.nn.functional.relu(x)
		x = self.conv2(x)
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = torch.nn.functional.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		#x = torch.nn.functional.log_softmax(x, dim=1)
		return x

def train(model, train_dataloader, criterion, optimizer, epoch, log_interval, device='cpu'):
	model.train()
	for batch_idx, (data, target) in enumerate(train_dataloader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if (batch_idx + 1) % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))

def train_nll(model, train_dataloader, criterion, optimizer, epoch, log_interval, device='cpu'):
	model.train()
	m = torch.nn.LogSoftmax(dim=1)
	for batch_idx, (data, target) in enumerate(train_dataloader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		#loss = torch.nn.functional.nll_loss(output, target)
		loss = criterion(m(output), target)
		loss.backward()
		optimizer.step()
		if (batch_idx + 1) % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))

def train_mixup(model, train_dataloader, criterion, optimizer, epoch, log_interval, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device='cpu'):
	model.train()
	for batch_idx, (data, target) in enumerate(train_dataloader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output, target = model(data, target, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
		loss = criterion(output, torch.argmax(target, dim=1))
		loss.backward()
		optimizer.step()
		if (batch_idx + 1) % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))

def train_nll_mixup(model, train_dataloader, criterion, optimizer, epoch, log_interval, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device='cpu'):
	model.train()
	m = torch.nn.LogSoftmax(dim=1)
	for batch_idx, (data, target) in enumerate(train_dataloader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output, target = model(data, target, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
		#loss = torch.nn.functional.nll_loss(output, torch.argmax(target, dim=1))
		loss = criterion(m(output), torch.argmax(target, dim=1))
		loss.backward()
		optimizer.step()
		if (batch_idx + 1) % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))

def test(model, test_dataloader, criterion, device='cpu'):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_dataloader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item()  # Sum up batch loss.
			pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_dataloader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))

def test_nll(model, test_dataloader, criterion, device='cpu'):
	model.eval()
	m = torch.nn.LogSoftmax(dim=1)
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_dataloader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			#test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss.
			test_loss += criterion(m(output), target).item()  # Sum up batch loss.
			pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_dataloader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_dataloader.dataset), 100. * correct / len(test_dataloader.dataset)))

def create_augmenter():
	#import imgaug as ia
	from imgaug import augmenters as iaa

	augmenter = iaa.Sequential([
		#iaa.Sometimes(0.5, iaa.OneOf([
		#	iaa.Crop(px=(0, 100)),  # Crop images from each side by 0 to 16px (randomly chosen).
		#	iaa.Crop(percent=(0, 0.1)),  # Crop images by 0-10% of their height/width.
		#	#iaa.Fliplr(0.5),  # Horizontally flip 50% of the images.
		#	#iaa.Flipud(0.5),  # Vertically flip 50% of the images.
		#])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.Affine(
				#scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (0.0, 0.1), 'y': (-0.05, 0.05)},  # Translate by 0 to +10 percent along x-axis and -5 to +5 percent along y-axis.
				rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				shear=(-10, 10),  # Shear by -10 to +10 degrees.
				#order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				order=0,  # Use nearest neighbour or bilinear interpolation (fast).
				#cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				#mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
				#mode='edge'  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			),
			#iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # Move parts of the image around. Slow.
			#iaa.PerspectiveTransform(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(20.0, 40.0), sigma=(6.0, 8.0)),  # Move pixels locally around (with random strengths).
		])),
		iaa.Sometimes(0.5, iaa.OneOf([
			iaa.OneOf([
				iaa.GaussianBlur(sigma=(0.5, 1.5)),
				iaa.AverageBlur(k=(2, 4)),
				iaa.MedianBlur(k=(3, 3)),
				iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0), order=1),
			]),
			iaa.Sequential([
				iaa.OneOf([
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					#iaa.AdditiveLaplaceNoise(loc=0, scale=(0.05 * 255, 0.2 * 255), per_channel=False),
					iaa.AdditivePoissonNoise(lam=(20, 30), per_channel=False),
					iaa.CoarseSaltAndPepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarseSalt(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					iaa.CoarsePepper(p=(0.01, 0.1), size_percent=(0.2, 0.9), per_channel=False),
					#iaa.CoarseDropout(p=(0.1, 0.3), size_percent=(0.8, 0.9), per_channel=False),
				]),
				iaa.GaussianBlur(sigma=(0.7, 1.0)),
			]),
			#iaa.OneOf([
			#	#iaa.MultiplyHueAndSaturation(mul=(-10, 10), per_channel=False),
			#	#iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=False),
			#	#iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False),

			#	iaa.Invert(p=1, per_channel=False),

			#	#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
			#	iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
			#]),
		])),
		#iaa.Scale(size={'height': image_height, 'width': image_width})  # Resize.
	])

	return augmenter

def create_data(batch_size, shuffle, num_workers, train_transform, test_transform):
	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=train_transform)
	test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	return train_dataloader, test_dataloader

def show_data_info(train_dataloader, test_dataloader):
	print('#train steps per epoch = {}.'.format(len(train_dataloader)))
	data_iter = iter(train_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

	print('#test steps per epoch = {}.'.format(len(test_dataloader)))
	data_iter = iter(test_dataloader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))

def visualize_data(train_dataloader, test_dataloader):
	for dataloader in [train_dataloader, test_dataloader]:
		data_iter = iter(dataloader)
		images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
		images, labels = images.numpy(), labels.numpy()
		for idx, (img, lbl) in enumerate(zip(images, labels)):
			print('Label: (int) = {}.'.format(lbl))
			cv2.imshow('Image', img[0])
			cv2.waitKey(0)
			if idx >= 9: break
	cv2.destroyAllWindows()

class RandomAugment(object):
	def __init__(self):
		self.augmenter = create_augmenter()

	def __call__(self, x):
		return Image.fromarray(self.augmenter.augment_images(np.array(x)))

class RandomInvert(object):
	def __call__(self, x):
		return ImageOps.invert(x) if random.randrange(2) else x

class ConvertChannel(object):
	def __call__(self, x):
		return x.convert('RGB')
		#return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
		#return torch.repeat_interleave(x, 3, dim=0)
		#return torch.repeat_interleave(torch.unsqueeze(x, dim=3), 3, dim=0)

def mnist_test():
	image_height, image_width = 28, 28
	#image_height_before_crop, image_width_before_crop = 32, 32
	image_height_before_crop, image_width_before_crop = image_height, image_width
	num_classes = 10

	num_epochs = 10
	batch_size = 256
	shuffle = True
	num_workers = 4

	learning_rate = 1.0
	gamma = 0.7
	log_interval = 100

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './mnist_cnn.pth'

	#torch.manual_seed(1)

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(),
		#RandomInvert(),
		#ConvertChannel(),
		#torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		#ConvertChannel(),
		#torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))
	])

	# Create datasets.
	train_dataloader, test_dataloader = create_data(batch_size, shuffle, num_workers, train_transform, test_transform)
	# Show data info.
	show_data_info(train_dataloader, test_dataloader)
	if False:
		# Visualize data.
		visualize_data(train_dataloader, test_dataloader)

	#--------------------
	# Create a model.
	model = Net(num_classes=num_classes, input_channels=1)

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Create a trainer.
	optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

	# Train a model.
	if False:
		criterion = torch.nn.CrossEntropyLoss()
		test_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

		for epoch in range(1, num_epochs + 1):
			train(model, train_dataloader, criterion, optimizer, epoch, log_interval, device)
			test(model, test_dataloader, test_criterion, device)
			scheduler.step()
	else:
		criterion = torch.nn.NLLLoss()
		test_criterion = torch.nn.NLLLoss(reduction='sum')

		for epoch in range(1, num_epochs + 1):
			train_nll(model, train_dataloader, criterion, optimizer, epoch, log_interval, device)
			test_nll(model, test_dataloader, test_criterion, device)
			scheduler.step()

	# Save a model.
	save_model(model_filepath, model)

def mnist_predefined_test():
	image_height, image_width = 32, 32
	#image_height_before_crop, image_width_before_crop = 36, 36
	image_height_before_crop, image_width_before_crop = image_height, image_width
	num_classes = 10

	num_epochs = 10
	batch_size = 256
	shuffle = True
	num_workers = 4

	learning_rate = 1.0
	gamma = 0.7
	log_interval = 100

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './mnist_cnn_predefined.pth'

	#torch.manual_seed(1)

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(),
		#RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
	])

	# Create datasets.
	train_dataloader, test_dataloader = create_data(batch_size, shuffle, num_workers, train_transform, test_transform)
	# Show data info.
	show_data_info(train_dataloader, test_dataloader)
	if False:
		# Visualize data.
		visualize_data(train_dataloader, test_dataloader)

	#--------------------
	# Create a model.
	if False:
		# NOTE [info] >> Hard to train.
		model = torchvision.models.vgg16(pretrained=False, num_classes=num_classes)
		#model = torchvision.models.vgg16_bn(pretrained=False, num_classes=num_classes)
	elif False:
		#model = torchvision.models.vgg16(pretrained=True, progress=True)
		model = torchvision.models.vgg16_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
	else:
		model = torchvision.models.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Create a trainer.
	optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

	# Train a model.
	if True:
		criterion = torch.nn.CrossEntropyLoss()
		test_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

		for epoch in range(1, num_epochs + 1):
			train(model, train_dataloader, criterion, optimizer, epoch, log_interval, device)
			test(model, test_dataloader, test_criterion, device)
			scheduler.step()
	else:
		criterion = torch.nn.NLLLoss()
		test_criterion = torch.nn.NLLLoss(reduction='sum')

		for epoch in range(1, num_epochs + 1):
			train_nll(model, train_dataloader, criterion, optimizer, epoch, log_interval, device)
			test_nll(model, test_dataloader, test_criterion, device)
			scheduler.step()

	# Save a model.
	save_model(model_filepath, model)

def mnist_predefined_mixup_test():
	image_height, image_width = 32, 32
	#image_height_before_crop, image_width_before_crop = 36, 36
	image_height_before_crop, image_width_before_crop = image_height, image_width
	num_classes = 10

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	num_epochs = 10
	batch_size = 256
	shuffle = True
	num_workers = 4

	learning_rate = 1.0
	gamma = 0.7
	log_interval = 100

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './mnist_cnn_predefined_mixup.pth'

	#torch.manual_seed(1)

	#--------------------
	train_transform = torchvision.transforms.Compose([
		#RandomAugment(),
		#RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
	])

	# Create datasets.
	train_dataloader, test_dataloader = create_data(batch_size, shuffle, num_workers, train_transform, test_transform)
	# Show data info.
	show_data_info(train_dataloader, test_dataloader)
	if False:
		# Visualize data.
		visualize_data(train_dataloader, test_dataloader)

	#--------------------
	# Create a model.
	if False:
		import mixup.vgg
		# NOTE [info] >> Hard to train.
		model = mixup.vgg.vgg16(pretrained=False, num_classes=num_classes)
		#model = mixup.vgg.vgg16_bn(pretrained=False, num_classes=num_classes)
	elif False:
		import mixup.vgg
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = mixup.vgg.vgg16(pretrained=True, progress=True)
		model = mixup.vgg.vgg16_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=False, num_classes=num_classes)
	else:
		import mixup.resnet
		model = mixup.resnet.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Create a trainer.
	optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

	# Train a model.
	if True:
		criterion = torch.nn.CrossEntropyLoss()
		test_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

		for epoch in range(1, num_epochs + 1):
			train_mixup(model, train_dataloader, criterion, optimizer, epoch, log_interval, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
			test(model, test_dataloader, test_criterion, device)
			scheduler.step()
	else:
		criterion = torch.nn.NLLLoss()
		test_criterion = torch.nn.NLLLoss(reduction='sum')

		for epoch in range(1, num_epochs + 1):
			train_nll_mixup(model, train_dataloader, criterion, optimizer, epoch, log_interval, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
			test_nll(model, test_dataloader, test_criterion, device)
			scheduler.step()

	# Save a model.
	save_model(model_filepath, model)

def main():
	#mnist_test()  # User-defined model.
	#mnist_predefined_test()  # Predefined models (VGG or ResNet).
	mnist_predefined_mixup_test()  # Predefined models (VGG or ResNet) + (Mixup, Manifold Mixup, Cutout).

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
