#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, argparse, logging, time, datetime
import numpy as np
import torch
import torchvision
import cv2
#import swl.machine_learning.util as swl_ml_util

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self):
		#self._image_height, self._image_width, self._image_channel = 28, 28, 1  # 784 = 28 * 28.
		self._num_classes = 10

		# Preprocess.
		self._transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	@property
	def num_classes(self):
		return self._num_classes

	def create_train_data_loader(self, batch_size, shuffle=True, num_workers=4):
		train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self._transform)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

		return train_loader

	def create_test_data_loader(self, batch_size, shuffle=False, num_workers=4):
		test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self._transform)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

		return test_loader

#--------------------------------------------------------------------

class MyModel(torch.nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()

		#self.conv1 = torch.nn.Conv2d(1, 32, 5)
		self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2)
		self.pool = torch.nn.MaxPool2d(2, 2)
		#self.conv2 = torch.nn.Conv2d(32, 64, 3)
		self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
		#self.fc1 = torch.nn.Linear(64 * 5 * 5, 1024)
		self.fc1 = torch.nn.Linear(64 * 7 * 7, 1024)
		self.fc2 = torch.nn.Linear(1024, 10)

	def forward(self, x):
		x = self.pool(torch.nn.functional.relu(self.conv1(x)))
		x = self.pool(torch.nn.functional.relu(self.conv2(x)))
		#x = x.view(-1, 64 * 5 * 5)
		x = x.view(-1, 64 * 7 * 7)
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.softmax(self.fc2(x), dim=1)
		return x

#--------------------------------------------------------------------

class MyRunner(object):
	def __init__(self, batch_size):
		# Create a dataset.
		print('Start loading dataset...')
		start_time = time.time()
		self._dataset = MyDataset()
		self._train_loader = self._dataset.create_train_data_loader(batch_size, shuffle=True, num_workers=4)
		self._test_loader = self._dataset.create_test_data_loader(batch_size, shuffle=False, num_workers=4)
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

	def train(self, model_filepath, num_epochs, device, initial_epoch=0, is_training_resumed=False):
		if is_training_resumed:
			# Restore a model.
			try:
				print('[SWL] Info: Start restoring a model...')
				start_time = time.time()
				model = torch.load(model_filepath)
				model.eval()
				print('[SWL] Info: End restoring a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except:
				print('[SWL] Error: Failed to restore a model from {}.'.format(model_filepath))
				return
		else:
			# Create a model.
			model = MyModel()

		model = model.to(device)

		# Create a trainer.
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

		history = {
			'acc': list(),
			'loss': list(),
			'val_acc': list(),
			'val_loss': list()
		}

		#--------------------
		if is_training_resumed:
			print('[SWL] Info: Resume training...')
		else:
			print('[SWL] Info: Start training...')
		start_total_time = time.time()
		initial_epoch = 0
		final_epoch = initial_epoch + num_epochs
		for epoch in range(initial_epoch, final_epoch):
			print('Epoch {}/{}'.format(epoch, final_epoch - 1))

			start_time = time.time()
			running_loss = 0.0
			for batch_step, batch_data in enumerate(self._train_loader):
				batch_inputs, batch_outputs = batch_data

				"""
				# One-hot encoding.
				batch_outputs_onehot = torch.LongTensor(batch_outputs.shape[0], self._dataset.num_classes)
				batch_outputs_onehot.zero_()
				batch_outputs_onehot.scatter_(1, batch_outputs.view(batch_outputs.shape[0], -1), 1)
				"""

				batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
				#batch_inputs, batch_outputs, batch_outputs_onehot = batch_inputs.to(device), batch_outputs.to(device), batch_outputs_onehot.to(device)

				# Zero the parameter gradients.
				optimizer.zero_grad()

				# Forward + backward + optimize.
				model_outputs = model(batch_inputs)
				loss = criterion(model_outputs, batch_outputs)
				loss.backward()
				"""
				# Gradient clipping.
				max_gradient_norm = 5
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm, norm_type=2)
				#for p in model.parameters():
				#	if p.grad is not None:
				#		p.grad.data.clamp_(min=-max_gradient_norm, max=max_gradient_norm)
				"""
				# Update weights.
				optimizer.step()
				#for p in model.parameters():
				#	p.data.add_(-lr, p.grad.data)  # p.data = p.data + (-lr * p.grad.data).

				# Print statistics.
				running_loss += loss.item()
				if (batch_step + 1) % 100 == 0:
					print('\tStep {}: loss = {:.6f}: {} secs.'.format(batch_step + 1, running_loss / 100, time.time() - start_time))
					running_loss = 0.0
			print('\tTrain: {} secs.'.format(time.time() - start_time))

			# TODO [implement] >>
			#history['loss'].append(train_loss)
			#history['acc'].append(train_acc)
			#history['val_loss'].append(val_loss)
			#history['val_acc'].append(val_acc)

			sys.stdout.flush()
			time.sleep(0)
		print('[SWL] Info: End training: {} secs.'.format(time.time() - start_total_time))

		#--------------------
		print('[SWL] Info: Start saving a model...')
		start_time = time.time()
		torch.save(model, model_filepath)  # Saves a model using either a .pt or .pth file extension.
		print('[SWL] Info: End saving a model to {}: {} secs.'.format(model_filepath, time.time() - start_time))

		return history

	def test(self, model_filepath, device):
		# Load a model.
		try:
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			model = torch.load(model_filepath)
			model.eval()
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except:
			print('[SWL] Error: Failed to load a model from {}.'.format(model_filepath))
			return

		model = model.to(device)

		#--------------------
		print('[SWL] Info: Start testing...')
		start_time = time.time()
		inferences, ground_truths = list(), list()
		for batch_data in self._test_loader:
			batch_inputs, batch_outputs = batch_data
			#batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
			batch_inputs = batch_inputs.to(device)

			model_outputs = model(batch_inputs)

			_, model_outputs = torch.max(model_outputs, 1)
			inferences.extend(model_outputs.cpu().numpy())
			ground_truths.extend(batch_outputs.numpy())
		print('[SWL] Info: End testing: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.array(inferences), np.array(ground_truths)
		if inferences is not None and ground_truths is not None:
			print('\tTest: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			print('\tTest: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			print('[SWL] Warning: Invalid test results.')

	def infer(self, model_filepath, device, batch_size, shuffle=False):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		# Load a model.
		try:
			print('[SWL] Info: Start loading a model...')
			start_time = time.time()
			model = torch.load(model_filepath)
			model.eval()
			print('[SWL] Info: End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except:
			print('[SWL] Error: Failed to load a model from {}.'.format(model_filepath))
			return

		model = model.to(device)

		#--------------------
		inf_loader = self._dataset.create_test_data_loader(batch_size, shuffle=shuffle, num_workers=4)

		inf_images = list(batch_data[0] for batch_data in inf_loader)

		#--------------------
		print('[SWL] Info: Start inferring...')
		start_time = time.time()
		inferences = list()
		for batch_images in inf_images:
			batch_images = batch_images.to(device)
			model_outputs = model(batch_images)

			_, model_outputs = torch.max(model_outputs, 1)
			inferences.extend(model_outputs.cpu().numpy())
		print('[SWL] Info: End inferring: {} secs.'.format(time.time() - start_time))

		inferences = np.array(inferences)
		if inferences is not None:
			print('\tInference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			print('\tInference results: index,inference')
			for idx, inf in enumerate(inferences):
				print('{},{}'.format(idx, inf))
				if (idx + 1) >= 10:
					break
		else:
			print('[SWL] Warning: Invalid inference results.')

#--------------------------------------------------------------------

def parse_command_line_options():
	parser = argparse.ArgumentParser(description='Train, test, or infer a CNN model for MNIST dataset.')

	parser.add_argument(
		'--train',
		action='store_true',
		help='Specify whether to train a model'
	)
	parser.add_argument(
		'--test',
		action='store_true',
		help='Specify whether to test a trained model'
	)
	parser.add_argument(
		'--infer',
		action='store_true',
		help='Specify whether to infer by a trained model'
	)
	parser.add_argument(
		'-r',
		'--resume',
		action='store_true',
		help='Specify whether to resume training'
	)
	parser.add_argument(
		'-m',
		'--model_file',
		type=str,
		#nargs='?',
		help='The model file path where a trained model is saved or a pretrained model is loaded',
		#required=True,
		default=None
	)
	parser.add_argument(
		'-tr',
		'--train_data_dir',
		type=str,
		#nargs='?',
		help='The directory path of training data',
		default='./train_data'
	)
	parser.add_argument(
		'-te',
		'--test_data_dir',
		type=str,
		#nargs='?',
		help='The directory path of test data',
		default='./test_data'
	)
	parser.add_argument(
		'-e',
		'--epoch',
		type=int,
		help='Number of epochs',
		default=30
	)
	parser.add_argument(
		'-b',
		'--batch_size',
		type=int,
		help='Batch size',
		default=128
	)
	parser.add_argument(
		'-g',
		'--gpu',
		type=str,
		help='Specify GPU to use',
		default='0'
	)
	parser.add_argument(
		'-l',
		'--log_level',
		type=int,
		help='Log level, [0, 50]',  # {NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL}.
		default=None
	)

	return parser.parse_args()

def set_logger(log_level):
	"""
	# When log_level is string.
	if log_level is not None:
		log_level = getattr(logging, log_level.upper(), None)
		if not isinstance(log_level, int):
			raise ValueError('Invalid log level: {}'.format(log_level))
	else:
		log_level = logging.WARNING
	"""
	print('[SWL] Info: Log level = {}.'.format(log_level))

	handler = logging.handlers.RotatingFileHandler('./simple_training.log', maxBytes=5000, backupCount=10)
	formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	handler.setFormatter(formatter)

	#logger = logging.getLogger(__name__)
	logger = logging.getLogger('simple_training_logger')
	logger.addHandler(handler) 
	logger.setLevel(log_level)

	return logger

def main():
	args = parse_command_line_options()

	if not args.train and not args.test:
		print('[SWL] Error: At least one of command line options "--train", "--test", and "--infer" has to be specified.')
		return

	#if args.gpu:
	#	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	#logger = set_logger(args.log_level)

	#--------------------
	num_epochs, batch_size = args.epoch, args.batch_size
	initial_epoch = 0
	is_training_resumed = False

	train_device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')
	test_device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')
	infer_device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')

	model_filepath = args.model_file
	if model_filepath:
		output_dir_path = os.path.dirname(model_filepath)
	else:
		output_dir_prefix = 'simple_training'
		output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		model_filepath = os.path.join(output_dir_path, 'model.pt')

	#--------------------
	runner = MyRunner(batch_size)

	if args.train:
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		history = runner.train(model_filepath, num_epochs, train_device, initial_epoch, is_training_resumed)

		#print('History =', history)
		#swl_ml_util.display_train_history(history)
		#if os.path.exists(output_dir_path):
		#	swl_ml_util.save_train_history(history, output_dir_path)

	if args.test:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return

		runner.test(model_filepath, test_device)

	if args.infer:
		if not model_filepath or not os.path.exists(model_filepath):
			print('[SWL] Error: Model file, {} does not exist.'.format(model_filepath))
			return

		runner.infer(model_filepath, infer_device, batch_size)

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 30 --gpu 0

if '__main__' == __name__:
	main()
