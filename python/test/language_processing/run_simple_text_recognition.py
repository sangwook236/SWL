#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, random, glob, time
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import text_data
import text_generation_util as tg_util
#import vgg_mixup, resnet_mixup

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

def construct_charset():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_dir_path = font_base_dir_path + '/kor'
	#font_dir_path = font_base_dir_path + '/eng'

	font_filepaths = glob.glob(os.path.join(font_dir_path, '*.ttf'))
	#font_list = tg_util.generate_hangeul_font_list(font_filepaths)
	font_list = tg_util.generate_font_list(font_filepaths)

	#--------------------
	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#hangeul_charset = fd.read().strip('\n')  # A string.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of strings.
		#hangeul_charset = fd.read().splitlines()  # A list of strings.

	#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
	#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	import string
	alphabet_charset = string.ascii_uppercase + string.ascii_lowercase
	digit_charset = string.digits
	symbol_charset = string.punctuation
	#symbol_charset = string.punctuation + ' '

	#charset = alphabet_charset + digit_charset + symbol_charset + hangeul_charset + hangeul_jamo_charset
	charset = alphabet_charset + digit_charset + symbol_charset + hangeul_charset
	return charset, font_list

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

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_single_character():
	image_height, image_width = 64, 64
	#image_height_before_crop, image_width_before_crop = 72, 72
	image_height_before_crop, image_width_before_crop = image_height, image_width

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	font_size_interval = (10, 100)
	font_overlap_interval = (0.8, 1.25)

	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './simple_text_recognition.pth'

	#--------------------
	# Load and normalize datasets.
	charset, font_list = construct_charset()

	train_transform = torchvision.transforms.Compose([
		RandomAugment(),
		RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	print('Start creating datasets...')
	start_time = time.time()
	if False:
		train_dataset = text_data.SingleCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, transform=train_transform)
		test_dataset = text_data.SingleCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, transform=test_transform)
	else:
		train_dataset = text_data.SingleNoisyCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, font_overlap_interval, transform=train_transform)
		test_dataset = text_data.SingleNoisyCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, font_overlap_interval, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	classes = train_dataset.classes
	num_classes = train_dataset.num_classes

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(train_dataloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a convolutional neural network.

	if False:
		model = torchvision.models.vgg19(pretrained=False, num_classes=num_classes)
		#model = torchvision.models.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		#model = torchvision.models.vgg19(pretrained=True, progress=True)
		model = torchvision.models.vgg19_bn(pretrained=True, progress=True)
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
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	if True:
		model.train()
		for epoch in range(num_epochs):  # Loop over the dataset multiple times.
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(train_dataloader, 0):
				# Get the inputs.
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if i % 1000 == 999:  # Print every 2000 mini-batches.
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
			#scheduler.step()

			# Save a checkpoint.
			save_model(model_filepath, model)

		print('Finished training.')

		# Save a model.
		save_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the %i test images: %d %%' % (len(test_dataset), 100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	#for i in range(num_classes):
	#	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
	accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	print('Accuracy frequency: {}.'.format(hist))
	valid_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] > 0]
	print('Accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
	accuracy_threshold = 98
	for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]):
		if acc < accuracy_threshold:
			print('\tChar = {}: accuracy = {}.'.format(classes[idx], acc))

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_single_character_using_mixup():
	image_height, image_width = 64, 64
	#image_height_before_crop, image_width_before_crop = 72, 72
	image_height_before_crop, image_width_before_crop = image_height, image_width

	mixup_input, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	font_size_interval = (10, 100)
	font_overlap_interval = (0.8, 1.25)

	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './simple_text_recognition_mixup.pth'

	#--------------------
	# Load and normalize datasets.
	charset, font_list = construct_charset()

	train_transform = torchvision.transforms.Compose([
		RandomAugment(),
		RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height_before_crop, image_width_before_crop)),
		#torchvision.transforms.RandomCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		#RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	print('Start creating datasets...')
	start_time = time.time()
	if False:
		train_dataset = text_data.SingleCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, transform=train_transform)
		test_dataset = text_data.SingleCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, transform=test_transform)
	else:
		train_dataset = text_data.SingleNoisyCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, font_overlap_interval, transform=train_transform)
		test_dataset = text_data.SingleNoisyCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, font_overlap_interval, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	assert train_dataset.classes == test_dataset.classes, 'Unmatched classes, {} != {}'.format(train_dataset.classes, test_dataset.classes)
	#assert train_dataset.num_classes == test_dataset.num_classes, 'Unmatched number of classes, {} != {}'.format(train_dataset.num_classes, test_dataset.num_classes)

	#--------------------
	print('Start creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating data loaders: {} secs.'.format(time.time() - start_time))

	classes = train_dataset.classes
	num_classes = train_dataset.num_classes

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(train_dataloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a convolutional neural network.

	if False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/vgg_mixup.py
		import vgg_mixup
		# NOTE [info] >> Hard to train.
		model = vgg_mixup.vgg19(pretrained=False, num_classes=num_classes)
		#model = vgg_mixup.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/vgg_mixup.py
		import vgg_mixup
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = vgg_mixup.vgg19(pretrained=True, progress=True)
		model = vgg_mixup.vgg19_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/resnet_mixup.py
		import resnet_mixup
		model = resnet_mixup.resnet18(pretrained=False, num_classes=num_classes)
	else:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/resnet_mixup.py
		import resnet_mixup
		model = resnet_mixup.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	#model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	if True:
		model.train()
		for epoch in range(num_epochs):  # Loop over the dataset multiple times.
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(train_dataloader, 0):
				# Get the inputs.
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				outputs, labels = model(inputs, labels, mixup_input, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
				loss = criterion(outputs, torch.argmax(labels, dim=1))
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if i % 1000 == 999:  # Print every 2000 mini-batches.
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
			#scheduler.step()

			# Save a checkpoint.
			save_model(model_filepath, model)

		print('Finished training.')

		# Save a model.
		save_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the %i test images: %d %%' % (len(test_dataset), 100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for images, labels in test_dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	#for i in range(num_classes):
	#	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
	accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]
	#print('Accuracy: {}.'.format(accuracies))
	hist, bin_edges = np.histogram(accuracies, bins=range(-1, 101), density=False)
	print('Accuracy frequency: {}.'.format(hist))
	valid_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] > 0]
	print('Accuracy: min = {}, max = {}.'.format(np.min(valid_accuracies), np.max(valid_accuracies)))
	accuracy_threshold = 98
	for idx, acc in sorted(enumerate(valid_accuracies), key=lambda x: x[1]):
		if acc < accuracy_threshold:
			print('\tChar = {}: accuracy = {}.'.format(classes[idx], acc))

# REF [function] >> test_net() in https://github.com/clovaai/CRAFT-pytorch/blob/master/test.py
def test_net_sangwook(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net=None, show_time=False):
	from torch.autograd import Variable
	import craft.craft_utils as craft_utils
	import craft.imgproc as imgproc

	t0 = time.time()

	# Resize.
	img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
	ratio_h = ratio_w = 1 / target_ratio

	# Preprocessing.
	x = imgproc.normalizeMeanVariance(img_resized)
	x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
	x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
	if cuda:
		x = x.cuda()

	# Forward pass.
	with torch.no_grad():
		y, feature = net(x)

	# Make score and link map.
	score_text = y[0,:,:,0].cpu().data.numpy()
	score_link = y[0,:,:,1].cpu().data.numpy()

	# Refine link.
	if refine_net is not None:
		with torch.no_grad():
			y_refiner = refine_net(y, feature)
		score_link = y_refiner[0,:,:,0].cpu().data.numpy()

	t0 = time.time() - t0
	t1 = time.time()

	# Post-processing.
	boxes, labels, ch_bboxes_lst, mapper = craft_utils.getDetBoxes_core_sangwook(score_text, score_link, text_threshold, link_threshold, low_text, img_resized)

	# Coordinate adjustment.
	boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
	ch_bboxes_lst = [craft_utils.adjustResultCoordinates(ch_bboxes, ratio_w, ratio_h) for ch_bboxes in ch_bboxes_lst]

	t1 = time.time() - t1

	# Render results (optional).
	render_img = score_text.copy()
	render_img = np.hstack((render_img, score_link))
	ret_score_text = imgproc.cvt2HeatmapImg(render_img)

	if show_time : print('\ninfer/postproc time : {:.3f}/{:.3f}'.format(t0, t1))

	return boxes, ch_bboxes_lst, ret_score_text

# REF [site] >> https://github.com/clovaai/CRAFT-pytorch/blob/master/test.py
def run_craft(image):
	import torch.backends.cudnn as cudnn
	import craft.craft as craft

	def copyStateDict(state_dict):
		import collections

		if list(state_dict.keys())[0].startswith('module'):
			start_idx = 1
		else:
			start_idx = 0
		new_state_dict = collections.OrderedDict()
		for k, v in state_dict.items():
			name = '.'.join(k.split('.')[start_idx:])
			new_state_dict[name] = v
		return new_state_dict

	trained_model = './craft/craft_mlt_25k.pth'
	text_threshold = 0.7  # Text confidence threshold.
	low_text = 0.4  # Text low-bound score.
	link_threshold = 0.4  # Link confidence threshold.
	cuda = True  # Use cuda for inference.
	canvas_size = 1280  # Image size for inference.
	mag_ratio = 1.5  # Image magnification ratio.
	poly = False  # Enable polygon type.
	show_time = False  # Show processing time.
	test_folder = './data'  # Folder path to input images.
	refine = False  # Enable link refiner.
	refiner_model = './craft/craft_refiner_CTW1500.pth'  # Pretrained refiner model.

	# Load a net.
	net = craft.CRAFT()  # Initialize.

	print('Loading weights from checkpoint (' + trained_model + ')')
	if cuda:
		net.load_state_dict(copyStateDict(torch.load(trained_model)))
	else:
		net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

	if cuda:
		net = net.cuda()
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = False

	net.eval()

	# LinkRefiner.
	refine_net = None
	if refine:
		from craft.refinenet import RefineNet
		refine_net = RefineNet()
		print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
		if cuda:
			refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
			refine_net = refine_net.cuda()
			refine_net = torch.nn.DataParallel(refine_net)
		else:
			refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

		refine_net.eval()
		poly = True

	t = time.time()

	#--------------------
	bboxes, ch_bboxes_lst, score_text = test_net_sangwook(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net, show_time)
	assert len(bboxes) == len(ch_bboxes_lst)

	print('Elapsed time : {}s'.format(time.time() - t))
	return bboxes, ch_bboxes_lst, score_text

def recognize_text_using_craft_and_single_character_recognizer():
	import craft.imgproc as imgproc
	#import craft.file_utils as file_utils

	image_height, image_width = 64, 64

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	#model_filepath = './craft/simple_text_recognition.pth'
	model_filepath = './craft/simple_text_recognition_mixup.pth'

	#--------------------
	# Construct a charset.
	charset, _ = construct_charset()

	classes = charset
	num_classes = len(classes)

	#--------------------
	# Define a convolutional neural network.

	if False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/vgg_mixup.py
		import vgg_mixup
		# NOTE [info] >> Hard to train.
		model = vgg_mixup.vgg19(pretrained=False, num_classes=num_classes)
		#model = vgg_mixup.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/vgg_mixup.py
		import vgg_mixup
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = vgg_mixup.vgg19(pretrained=True, progress=True)
		model = vgg_mixup.vgg19_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/resnet_mixup.py
		import resnet_mixup
		model = resnet_mixup.resnet18(pretrained=False, num_classes=num_classes)
	else:
		# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch/resnet_mixup.py
		import resnet_mixup
		model = resnet_mixup.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes

	# Load a model.
	model = load_model(model_filepath, model, device=device)

	model = model.to(device)

	#--------------------
	#image_filepath = './craft/images/I3.jpg'
	image_filepath = './craft/images/book_1.png'
	#image_filepath = './craft/images/book_2.png'
	image = imgproc.loadImage(image_filepath)

	print('Start running CRAFT...')
	start_time = time.time()
	bboxes, ch_bboxes_lst, score_text = run_craft(image)
	print('End running CRAFT: {} secs.'.format(time.time() - start_time))

	if len(bboxes) > 0:
		print('\tbboxes:', bboxes.shape, bboxes.dtype)
		print('\tch_bboxes_lst:', len(ch_bboxes_lst), ch_bboxes_lst[0].shape, ch_bboxes_lst[0].dtype)
		print('\tscore_text:', score_text.shape, score_text.dtype)

		"""
		import random
		cv2.imshow('Input', image)
		rgb1, rgb2 = image.copy(), image.copy()
		for bbox, ch_bboxes in zip(bboxes, ch_bboxes_lst):
			color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
			cv2.drawContours(rgb1, [np.round(np.expand_dims(bbox, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
			for bbox in ch_bboxes:
				cv2.drawContours(rgb2, [np.round(np.expand_dims(bbox, axis=1)).astype(np.int32)], 0, color, 1, cv2.LINE_AA)
		cv2.imshow('Word BBox', rgb1)
		cv2.imshow('Char BBox', rgb2)
		cv2.waitKey(0)
		"""

		os.makedirs('./char_recog_results', exist_ok=True)

		print('Start inferring...')
		start_time = time.time()
		ch_images = []
		rgb = image.copy()
		for i, ch_bboxes in enumerate(ch_bboxes_lst):
			imgs = []
			color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
			for j, bbox in enumerate(ch_bboxes):
				(x1, y1), (x2, y2) = np.min(bbox, axis=0), np.max(bbox, axis=0)
				x1, y1, x2, y2 = round(float(x1)), round(float(y1)), round(float(x2)), round(float(y2))
				img = image[y1:y2+1,x1:x2+1]
				img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img = np.repeat(np.expand_dims(img, axis=0), 3, axis=0)
				imgs.append(img)
				cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)
				cv2.imwrite('./char_recog_results/ch_{}_{}.png'.format(i, j), np.transpose(img, (1, 2, 0)))
			ch_images.append(np.array(imgs))
		cv2.imwrite('./char_recog_results/char_bbox.png', rgb)

		#--------------------
		with torch.no_grad():
			for i, imgs in enumerate(ch_images):
				imgs = imgs.astype(np.float32) * 2 / 255 - 1  # [-1, 1].
				imgs = torch.from_numpy(imgs).to(device)
				outputs = model(imgs)
				_, predicted = torch.max(outputs, 1)
				predicted = predicted.cpu().numpy()
				print('Prediction {}: {} (int), {} (str).'.format(i, predicted, ''.join([classes[id] for id in predicted])))
		print('End inferring: {} secs.'.format(time.time() - start_time))
	else:
		print('No text detected.')

def main():
	#recognize_single_character()
	#recognize_single_character_using_mixup()

	# Recognize text using CRAFT (scene text detector) + single character recognizer.
	recognize_text_using_craft_and_single_character_recognizer()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
