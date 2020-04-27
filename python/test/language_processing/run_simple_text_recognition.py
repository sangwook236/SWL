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
# REF [directory] >> ${SWL_PYTHON_HOME}/test/machine_learning/pytorch
#import vgg_mixup, resnet_mixup

def save_model(model_filepath, model):
	#torch.save(model.state_dict(), model_filepath)
	torch.save({'state_dict': model.state_dict()}, model_filepath)
	print('Saved a model to {}.'.format(model_filepath))

def load_model(model_filepath, model):
	loaded_data = torch.load(model_filepath)
	#model.load_state_dict(loaded_data)
	model.load_state_dict(loaded_data['state_dict'])
	print('Loaded a model from {}.'.format(model_filepath))
	return model

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

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_single_character():
	image_height, image_width = 64, 64
	#image_height_before_crop, image_width_before_crop = 72, 72
	image_height_before_crop, image_width_before_crop = image_height, image_width

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	font_size_interval = (10, 100)

	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './simple_text_recognition.pt'

	#--------------------
	# Load and normalize datasets.
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

	#--------------------
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
		RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, transform=train_transform)
	test_dataset = text_data.SingleCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, transform=test_transform)
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
	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	model.train()
	for epoch in range(num_epochs):  # Loop over the dataset multiple times.
		running_loss = 0.0
		for i, data in enumerate(train_dataloader, 0):
			# Get the inputs; data is a list of [inputs, labels].
			inputs, labels = data[0].to(device), data[1].to(device)

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

		save_model(model_filepath, model)

	print('Finished Training')

	#save_model(model_filepath, model)
	#model = load_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_dataloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the %i test images: %d %%' % (len(test_dataset), 100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for data in test_dataloader:
			images, labels = data[0].to(device), data[1].to(device)
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

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_single_character_using_mixup():
	image_height, image_width = 64, 64
	#image_height_before_crop, image_width_before_crop = 72, 72
	image_height_before_crop, image_width_before_crop = image_height, image_width

	mixup, mixup_hidden, mixup_alpha = True, True, 2.0
	cutout, cutout_size = True, 4

	num_train_examples_per_class, num_test_examples_per_class = 500, 50
	font_size_interval = (10, 100)

	num_epochs = 100
	batch_size = 256
	shuffle = True
	num_workers = 4

	gpu = 0
	device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
	print('Device =', device)

	model_filepath = './simple_text_recongnition_mixup.pt'

	#--------------------
	# Load and normalize datasets.
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

	#--------------------
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
		RandomInvert(),
		ConvertChannel(),
		torchvision.transforms.Resize((image_height, image_width)),
		#torchvision.transforms.CenterCrop((image_height, image_width)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleCharacterDataset(num_train_examples_per_class, charset, font_list, font_size_interval, transform=train_transform)
	test_dataset = text_data.SingleCharacterDataset(num_test_examples_per_class, charset, font_list, font_size_interval, transform=test_transform)
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
		import vgg_mixup
		# NOTE [info] >> Hard to train.
		model = vgg_mixup.vgg19(pretrained=False, num_classes=num_classes)
		#model = vgg_mixup.vgg19_bn(pretrained=False, num_classes=num_classes)
	elif False:
		import vgg_mixup
		# NOTE [error] >> Cannot load the pretrained model weights because the model is slightly changed.
		#model = vgg_mixup.vgg19(pretrained=True, progress=True)
		model = vgg_mixup.vgg19_bn(pretrained=True, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	elif False:
		import resnet_mixup
		model = resnet_mixup.resnet18(pretrained=False, num_classes=num_classes)
	else:
		import resnet_mixup
		model = resnet_mixup.resnet18(pretrained=True, progress=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, num_classes)
		model.num_classes = num_classes
	model = model.to(device)

	#--------------------
	# Define a loss function and optimizer.

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

	#--------------------
	# Train the network.

	model.train()
	for epoch in range(num_epochs):  # Loop over the dataset multiple times.
		running_loss = 0.0
		for i, data in enumerate(train_dataloader, 0):
			# Get the inputs; data is a list of [inputs, labels].
			inputs, labels = data[0].to(device), data[1].to(device)

			# Zero the parameter gradients.
			optimizer.zero_grad()

			# Forward + backward + optimize.
			outputs, labels = model(inputs, labels, mixup, mixup_hidden, mixup_alpha, cutout, cutout_size, device)
			loss = criterion(outputs, torch.argmax(labels, dim=1))
			loss.backward()
			optimizer.step()

			# Print statistics.
			running_loss += loss.item()
			if i % 1000 == 999:  # Print every 2000 mini-batches.
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0
		#scheduler.step()

		save_model(model_filepath, model)

	print('Finished Training')

	#save_model(model_filepath, model)
	#model = load_model(model_filepath, model)

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	model.eval()
	outputs = model(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_dataloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the %i test images: %d %%' % (len(test_dataset), 100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0 for i in range(num_classes))
	class_total = list(0 for i in range(num_classes))
	with torch.no_grad():
		for data in test_dataloader:
			images, labels = data[0].to(device), data[1].to(device)
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

def main():
	#recognize_single_character()
	recognize_single_character_using_mixup()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
