#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, glob, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch, torchvision
import cv2
import text_data
import text_generation_util as tg_util

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def recognize_single_character():
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
		#hangeul_charset = fd.read().strip('\n')  # A strings.
		hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
		#hangeul_charset = fd.readlines()  # A list of string.
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
	class ChannelChanger(object):
		def __call__(self, x):
			return x.convert('RGB')
			#return np.repeat(np.expand_dims(x, axis=0), 3, axis=0)
			#return torch.repeat_interleave(x, 3, dim=0)
			#return torch.repeat_interleave(torch.unsqueeze(x, dim=3), 3, dim=0)
	image_size = (64, 64)
	train_transform = torchvision.transforms.Compose([
		ChannelChanger(),
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_transform = torchvision.transforms.Compose([
		ChannelChanger(),
		torchvision.transforms.Resize(image_size),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	#--------------------
	font_size_interval = (10, 100)
	num_train_examples, num_test_examples = int(1e6), int(1e4)

	print('Start creating datasets...')
	start_time = time.time()
	train_dataset = text_data.SingleCharacterDataset(charset, font_list, font_size_interval, num_train_examples, transform=train_transform)
	test_dataset = text_data.SingleCharacterDataset(charset, font_list, font_size_interval, num_test_examples, transform=test_transform)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))

	assert train_dataset.num_classes == test_dataset.num_classes
	print('#classes = {}.'.format(train_dataset.num_classes))

	#--------------------
	batch_size = 256
	num_epochs = 10
	shuffle = True
	num_workers = 4

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
	print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a Convolutional Neural Network.

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# Assuming that we are on a CUDA machine, this should print a CUDA device.
	print('Device =', device)

	if True:
		model = torchvision.models.vgg19(pretrained=False, progress=True)
		num_features = model.classifier[6].in_features
		model.classifier[6] = torch.nn.Linear(num_features, train_dataset.num_classes)
	else:
		model = torchvision.models.resnet18(pretrained=True)
		num_features = model.fc.in_features
		model.fc = torch.nn.Linear(num_features, train_dataset.num_classes)
	model = model.to(device)

	#--------------------
	# Define a Loss function and optimizer.

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	#--------------------
	# Train the network.

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
			if i % 2000 == 1999:  # Print every 2000 mini-batches.
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

	#--------------------
	# Test the network on the test data.

	dataiter = iter(test_dataloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
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

	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

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
	print('Accuracy: {}.'.format([100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1 for i in range(num_classes)]))

def main():
	recognize_single_character()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
