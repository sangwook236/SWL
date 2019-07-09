#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2

#--------------------------------------------------------------------

def load_data(batch_size, num_workers=4):
	# Preprocessing.
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return train_loader, test_loader

#--------------------------------------------------------------------

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		#self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
		self.pool = nn.MaxPool2d(2, 2)
		#self.conv2 = nn.Conv2d(32, 64, 3)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		#self.fc1 = nn.Linear(64 * 5 * 5, 1024)
		self.fc1 = nn.Linear(64 * 7 * 7, 1024)
		self.fc2 = nn.Linear(1024, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		#x = x.view(-1, 64 * 5 * 5)
		x = x.view(-1, 64 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), dim=1)
		return x

#--------------------------------------------------------------------

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	#image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
	num_classes = 10

	BATCH_SIZE, NUM_EPOCHS = 128, 30

	model_file_path = './mnist_cnn.pt'

	#%%------------------------------------------------------------------
	# Load data.

	print('Start loading dataset...')
	start_time = time.time()
	train_loader, test_loader = load_data(BATCH_SIZE)
	print('End loading dataset: {} secs.'.format(time.time() - start_time))

	data_iter = iter(train_loader)
	images, labels = data_iter.next()
	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_images.shape, train_images.dtype, np.min(train_images), np.max(train_images)))
	print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(train_labels.shape, train_labels.dtype, np.min(train_labels), np.max(train_labels)))

	data_iter = iter(test_loader)
	images, labels = data_iter.next()
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_images.shape, test_images.dtype, np.min(test_images), np.max(test_images)))
	print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(test_labels.shape, test_labels.dtype, np.min(test_labels), np.max(test_labels)))

	#%%------------------------------------------------------------------
	# Create a model.

	model = Model()
	model = model.to(device)

	#%%------------------------------------------------------------------
	# Train.

	if True:
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

		#--------------------
		print('Start training...')
		start_total_time = time.time()
		for epoch in range(NUM_EPOCHS):
			print('Epoch {}:'.format(epoch + 1))

			start_time = time.time()
			running_loss = 0.0
			for idx, data in enumerate(train_loader):
				inputs, outputs = data

				"""
				# One-hot encoding.
				outputs_onehot = torch.LongTensor(outputs.shape[0], num_classes)
				outputs_onehot.zero_()
				outputs_onehot.scatter_(1, outputs.view(outputs.shape[0], -1), 1)
				"""

				inputs, outputs = inputs.to(device), outputs.to(device)
				#inputs, outputs, outputs_onehot = inputs.to(device), outputs.to(device), outputs_onehot.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				model_outputs = model(inputs)
				loss = criterion(model_outputs, outputs)
				loss.backward()
				optimizer.step()

				# Print statistics.
				running_loss += loss.item()
				if (idx + 1) % 100 == 0:
					print('\tStep {}: loss = {:.6f}.'.format(idx + 1, running_loss / 100))
					running_loss = 0.0
			print('\tTrain: time = {} secs.'.format(time.time() - start_time))
		print('End training: {} secs.'.format(time.time() - start_total_time))

		#--------------------
		print('Start saving a model...')
		start_time = time.time()
		torch.save(model, model_file_path)
		print('End saving a model: {} secs.'.format(time.time() - start_time))

	#%%------------------------------------------------------------------
	# Infer.

	print('Start loading a model...')
	start_time = time.time()
	model = torch.load(model_file_path)
	model.eval()
	print('End loading a model: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start inferring...')
	start_time = time.time()
	inferences, ground_truths = list(), list()
	for idx, data in enumerate(test_loader):
		inputs, outputs = data
		#inputs, outputs = inputs.to(device), outputs.to(device)
		inputs = inputs.to(device)

		model_outputs = model(inputs)

		_, model_outputs = torch.max(model_outputs, 1)
		inferences.extend(model_outputs.cpu().numpy())
		ground_truths.extend(outputs.numpy())
	print('End inferring: {} secs.'.format(time.time() - start_time))

	inferences = np.array(inferences)
	ground_truths = np.array(ground_truths)
	if inferences is not None:
		correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
		print('Inference: accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
	else:
		print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
