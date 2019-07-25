#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self):
		# Preprocessing.
		self._transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	def create_train_data_loader(self, batch_size, shuffle=True, num_workers=4):
		train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self._transform)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

		return train_loader

	def create_test_data_loader(self, batch_size, shuffle=False, num_workers=4):
		test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self._transform)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

		return test_loader

#--------------------------------------------------------------------

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()

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

class MyRunner(object):
	def __init__(self, batch_size, output_dir_path):
		self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		#image_height, image_width, image_channel = 28, 28, 1  # 784 = 28 * 28.
		self._num_classes = 10

		self._model_filepath = os.path.join(output_dir_path, 'mnist_cnn.pt')
		os.makedirs(output_dir_path, exist_ok=True)

		#--------------------
		# Create a dataset.

		print('Start loading dataset...')
		start_time = time.time()
		dataset = MyDataset()
		self._train_loader = dataset.create_train_data_loader(batch_size, shuffle=True, num_workers=4)
		self._test_loader = dataset.create_test_data_loader(batch_size, shuffle=False, num_workers=4)
		print('End loading dataset: {} secs.'.format(time.time() - start_time))

		data_iter = iter(self._train_loader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
		print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

		data_iter = iter(self._test_loader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
		print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

		#--------------------
		# Create a model.

		self._model = MyModel()
		self._model = self._model.to(self._device)

	def train(self, num_epochs):
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self._model.parameters(), lr=0.001, momentum=0.9)

		#--------------------
		print('Start training...')
		start_total_time = time.time()
		for epoch in range(num_epochs):
			print('Epoch {}:'.format(epoch + 1))

			start_time = time.time()
			running_loss = 0.0
			for idx, data in enumerate(self._train_loader):
				inputs, outputs = data

				"""
				# One-hot encoding.
				outputs_onehot = torch.LongTensor(outputs.shape[0], self._num_classes)
				outputs_onehot.zero_()
				outputs_onehot.scatter_(1, outputs.view(outputs.shape[0], -1), 1)
				"""

				inputs, outputs = inputs.to(self._device), outputs.to(self._device)
				#inputs, outputs, outputs_onehot = inputs.to(self._device), outputs.to(self._device), outputs_onehot.to(self._device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				model_outputs = self._model(inputs)
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
		torch.save(self._model, self._model_filepath)
		print('End saving a model: {} secs.'.format(time.time() - start_time))

	def infer(self):
		print('Start loading a model...')
		start_time = time.time()
		self._model = torch.load(self._model_filepath)
		self._model.eval()
		print('End loading a model: {} secs.'.format(time.time() - start_time))

		#--------------------
		print('Start inferring...')
		start_time = time.time()
		inferences, ground_truths = list(), list()
		for idx, data in enumerate(self._test_loader):
			inputs, outputs = data
			#inputs, outputs = inputs.to(self._device), outputs.to(self._device)
			inputs = inputs.to(self._device)

			model_outputs = self._model(inputs)

			_, model_outputs = torch.max(model_outputs, 1)
			inferences.extend(model_outputs.cpu().numpy())
			ground_truths.extend(outputs.numpy())
		print('End inferring: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.array(inferences), np.array(ground_truths)
		if inferences is not None:
			print('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			print('Inference: accurary = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

def main():
	num_epochs, batch_size = 30, 128

	output_dir_prefix = 'simple_training'
	output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	#output_dir_suffix = '20190724T231604'
	output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))

	#--------------------
	runner = MyRunner(batch_size, output_dir_path)

	runner.train(num_epochs)
	runner.infer()

#--------------------------------------------------------------------

if '__main__' == __name__:
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	main()
