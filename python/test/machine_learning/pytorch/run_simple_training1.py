#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, shutil, argparse, logging, logging.handlers, time, datetime
import numpy as np
import torch
import torchvision
import swl.machine_learning.util as swl_ml_util

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
	def __init__(self, logger):
		self._logger = logger

		# Create datasets.
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5,), (0.5,))
		])

		self._logger.info('Start loading dataset...')
		start_time = time.time()
		self._train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
		self._test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
		self._logger.info('End loading dataset: {} secs.'.format(time.time() - start_time))

	def train(self, model_filepath, model_checkpoint_filepath, batch_size, final_epoch, initial_epoch=0, is_training_resumed=False, device='cpu'):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		if is_training_resumed:
			# Restore a model.
			try:
				self._logger.info('Start restoring a model...')
				start_time = time.time()
				model = torch.load(model_filepath)
				self._logger.info('End restoring a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except:
				self._logger.error('Failed to restore a model from {}.'.format(model_filepath))
				return
		else:
			# Create a model.
			model = MyModel()

		#if torch.cuda.device_count() > 1:
		#	device_ids = [0, 1]
		#	model = torch.nn.DataParallel(model, device_ids=device_ids)
		model = model.to(device)

		# Create a trainer.
		criterion = torch.nn.CrossEntropyLoss().to(device)
		optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

		# Create data loaders.
		train_dataloader = torch.utils.data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
		test_dataloader = torch.utils.data.DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

		#--------------------
		history = {
			'acc': list(),
			'loss': list(),
			'val_acc': list(),
			'val_loss': list()
		}

		#--------------------
		if is_training_resumed:
			self._logger.info('Resume training...')
		else:
			self._logger.info('Start training...')
		best_performance_measure = 0
		best_model_filepath = None
		start_total_time = time.time()
		for epoch in range(initial_epoch, final_epoch):
			self._logger.info('Epoch {}/{}'.format(epoch, final_epoch - 1))

			#--------------------
			start_time = time.time()
			train_loss, train_acc = self._train(train_dataloader, optimizer, model, criterion, device)
			self._logger.info('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

			history['loss'].append(train_loss)
			history['acc'].append(train_acc)

			#--------------------
			start_time = time.time()
			val_loss, val_acc = self._evaluate(test_dataloader, model, criterion, device)
			self._logger.info('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)

			#--------------------
			if val_acc > best_performance_measure:
				self._logger.info('Start saving a model...')
				start_time = time.time()
				best_model_filepath = model_checkpoint_filepath.format(epoch=epoch, val_acc=val_acc)
				torch.save(model, best_model_filepath)  # Saves a model using either a .pt or .pth file extension.
				self._logger.info('End saving a model to {}: {} secs.'.format(best_model_filepath, time.time() - start_time))
				best_performance_measure = val_acc

			sys.stdout.flush()
			time.sleep(0)
		self._logger.info('End training: {} secs.'.format(time.time() - start_total_time))

		if best_model_filepath:
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				self._logger.info('Copied the best model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				self._logger.error('Failed to copy the best model to {}: {}.'.format(model_filepath, ex))
		else:
			torch.save(model, model_filepath)
			self._logger.info('Saved the best model to {}.'.format(model_filepath))

		return history

	def test(self, model, batch_size, device='cpu'):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		# Create a data loader.
		dataloader = torch.utils.data.DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

		#--------------------
		self._logger.info('Start testing...')
		inferences, ground_truths = list(), list()
		start_time = time.time()
		with torch.no_grad():
			for batch_data in dataloader:
				batch_inputs, batch_outputs = batch_data
				#batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
				batch_inputs = batch_inputs.to(device)

				model_outputs = model(batch_inputs)

				model_outputs = torch.argmax(model_outputs, -1)
				inferences.extend(model_outputs.cpu().numpy())
				ground_truths.extend(batch_outputs.numpy())
		self._logger.info('End testing: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.array(inferences), np.array(ground_truths)
		if inferences is not None and ground_truths is not None:
			self._logger.info('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			self._logger.info('Test: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			self._logger.warning('Invalid test results.')

	def infer(self, model, batch_size, shuffle=False, device='cpu'):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		# Create a data loader.
		dataloader = torch.utils.data.DataLoader(self._test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
		inf_images = list(batch_data[0] for batch_data in dataloader)

		#--------------------
		self._logger.info('Start inferring...')
		inferences = list()
		start_time = time.time()
		with torch.no_grad():
			for batch_images in inf_images:
				batch_images = batch_images.to(device)
				model_outputs = model(batch_images)

				model_outputs = torch.argmax(model_outputs, -1)
				inferences.extend(model_outputs.cpu().numpy())
		self._logger.info('End inferring: {} secs.'.format(time.time() - start_time))

		inferences = np.array(inferences)
		if inferences is not None:
			self._logger.info('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			results = {idx: inf for idx, inf in enumerate(inferences) if idx < 100}
			self._logger.info('Inference results (index: inference): {}.'.format(results))
		else:
			self._logger.warning('Invalid inference results.')

	def load_evaluation_model(self, model_filepath, device='cpu'):
		try:
			self._logger.info('Start loading a model...')
			start_time = time.time()
			model = torch.load(model_filepath)
			self._logger.info('End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
		except:
			self._logger.error('Failed to load a model from {}.'.format(model_filepath))
			return None

		#if torch.cuda.device_count() > 1:
		#	device_ids = [0, 1]
		#	model = torch.nn.DataParallel(model, device_ids=device_ids)
		model = model.to(device)
		# Switch to evaluation mode.
		model.eval()

		return model

	def show_data_info(self, batch_size):
		# Create data loaders.
		train_dataloader = torch.utils.data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
		test_dataloader = torch.utils.data.DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

		data_iter = iter(train_dataloader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		self._logger.info('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
		self._logger.info('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

		data_iter = iter(test_dataloader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		self._logger.info('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
		self._logger.info('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	def _train(self, dataloader, optimizer, model, criterion, device):
		# Switch to train mode.
		model.train()

		train_loss, train_acc, num_examples = 0.0, 0.0, 0
		running_loss = 0.0
		start_time = time.time()
		for batch_step, batch_data in enumerate(dataloader):
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

			model_outputs = torch.argmax(model_outputs, -1)
			train_loss += loss.item()
			train_acc += (model_outputs == batch_outputs).sum().item()
			num_examples += batch_outputs.size(0)

			# Print statistics.
			running_loss += loss.item()
			if (batch_step + 1) % 100 == 0:
				self._logger.info('\tStep {}: loss = {:.6f}: {} secs.'.format(batch_step + 1, running_loss / 100, time.time() - start_time))
				running_loss = 0.0
		train_loss /= batch_step + 1
		train_acc /= num_examples
		return train_loss, train_acc

	def _evaluate(self, dataloader, model, criterion, device):
		# Switch to evaluation mode.
		model.eval()

		val_loss, val_acc, num_examples = 0.0, 0.0, 0
		with torch.no_grad():
			for batch_step, batch_data in enumerate(dataloader):
				batch_inputs, batch_outputs = batch_data

				"""
				# One-hot encoding.
				batch_outputs_onehot = torch.LongTensor(batch_outputs.shape[0], self._dataset.num_classes)
				batch_outputs_onehot.zero_()
				batch_outputs_onehot.scatter_(1, batch_outputs.view(batch_outputs.shape[0], -1), 1)
				"""

				batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
				#batch_inputs, batch_outputs, batch_outputs_onehot = batch_inputs.to(device), batch_outputs.to(device), batch_outputs_onehot.to(device)

				model_outputs = model(batch_inputs)
				loss = criterion(model_outputs, batch_outputs)

				model_outputs = torch.argmax(model_outputs, -1)
				val_loss += loss.item()
				val_acc += (model_outputs == batch_outputs).sum().item()
				num_examples += batch_outputs.size(0)
			val_loss /= batch_step + 1
			val_acc /= num_examples
			return val_loss, val_acc

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
		'-o',
		'--out_dir',
		type=str,
		#nargs='?',
		help='The output directory path to save results such as images and log',
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
		help='Final epoch',
		default=30
	)
	parser.add_argument(
		'-b',
		'--batch_size',
		type=int,
		help='Batch size',
		default=32
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
		help='Log level, [0, 50]',  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
		default=None
	)

	return parser.parse_args()

def get_logger(name, log_level=None, log_dir_path=None, is_rotating=True):
	if not log_level: log_level = logging.INFO
	if not log_dir_path: log_dir_path = './log'
	if not os.path.isdir(log_dir_path):
		os.mkdir(log_dir_path)

	log_filepath = os.path.join(log_dir_path, (name if name else 'swl') + '.log')
	if is_rotating:
		file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10000000, backupCount=10)
	else:
		file_handler = logging.FileHandler(log_filepath)
	stream_handler = logging.StreamHandler()

	formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)s][%(asctime)s] [SWL] %(message)s')
	#formatter = logging.Formatter('[%(levelname)s][%(asctime)s] [SWL] %(message)s')
	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def main():
	args = parse_command_line_options()

	logger = get_logger(os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, './log', is_rotating=True)
	logger.info('----------------------------------------------------------------------')
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Python version: {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('Torch version: {}.'.format(torch.__version__))
	logger.info('cuDNN version: {}.'.format(torch.backends.cudnn.version()))

	if not args.train and not args.test and not args.infer:
		logger.error('At least one of command line options "--train", "--test", and "--infer" has to be specified.')
		return

	#if args.gpu:
	#	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')
	logger.info('Device: {}.'.format(device))

	#--------------------
	is_training_resumed = args.resume
	initial_epoch, final_epoch, batch_size = 0, args.epoch, args.batch_size

	model_filepath, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	if model_filepath:
		if not output_dir_path:
			output_dir_path = os.path.dirname(model_filepath)
	else:
		if not output_dir_path:
			output_dir_prefix = 'simple_training'
			output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		model_filepath = os.path.join(output_dir_path, 'model.pt')

	#--------------------
	runner = MyRunner(logger)
	runner.show_data_info(batch_size)

	if args.train:
		model_checkpoint_filepath = os.path.join(output_dir_path, 'model_ckpt.{epoch:04d}-{val_acc:.5f}.pt')
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		# Copy the model file to the output directory.
		new_model_filepath = os.path.join(output_dir_path, os.path.basename(model_filepath))
		if os.path.exists(model_filepath) and not os.path.samefile(model_filepath, new_model_filepath):
			try:
				shutil.copyfile(model_filepath, new_model_filepath)
			except (FileNotFoundError, PermissionError) as ex:
				logger.error('Failed to copy a model, {}: {}.'.format(model_filepath, ex))
				return
		model_filepath = new_model_filepath

		history = runner.train(model_filepath, model_checkpoint_filepath, batch_size, final_epoch, initial_epoch, is_training_resumed, device=device)

		#logger.info('Train history = {}.'.format(history))
		swl_ml_util.display_train_history(history)
		if os.path.exists(output_dir_path):
			swl_ml_util.save_train_history(history, output_dir_path)

	if args.test or args.infer:
		if model_filepath and os.path.exists(model_filepath):
			model = runner.load_evaluation_model(model_filepath, device=device)

			if args.test and model:
				runner.test(model, batch_size, device=device)

			if args.infer and model:
				runner.infer(model, batch_size, device=device)
		else:
			logger.error('Model file, {} does not exist.'.format(model_filepath))

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 20 --gpu 0

if '__main__' == __name__:
	main()
