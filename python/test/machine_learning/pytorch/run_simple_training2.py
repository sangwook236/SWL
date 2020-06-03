#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, shutil, collections, pickle, argparse, logging, logging.handlers, time, datetime
import numpy as np
import torch
import torchvision
#import swl.machine_learning.util as swl_ml_util
import utils

#--------------------------------------------------------------------

class MyModel(torch.nn.Module):
	def __init__(self):
		super().__init__()

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

	def train(self, model, criterion, optimizer, scheduler, train_dataloader, test_dataloader, model_checkpoint_filepath, output_dir_path, initial_epoch=0, final_epoch=10, device='cpu'):
		self._logger.info('Output path: {}.'.format(output_dir_path))
		self._logger.info('Model:\n{}.'.format(model))

		train_log_filepath = os.path.join(output_dir_path, 'train_log.txt')
		train_history_filepath = os.path.join(output_dir_path, 'train_history.pkl')
		train_result_image_filepath = os.path.join(output_dir_path, 'results.png')

		recorder = utils.RecorderMeter(final_epoch)
		history = {
			'acc': list(),
			'loss': list(),
			'val_acc': list(),
			'val_loss': list()
		}
		log_print_freq = 500

		#--------------------
		self._logger.info('Start training...')
		start_train_time = start_epoch_time = time.time()
		epoch_time = utils.AverageMeter()
		best_performance_measure = 0
		best_model_filepath = None
		for epoch in range(initial_epoch, final_epoch):
			self._logger.info('Epoch {}/{}'.format(epoch, final_epoch - 1))

			current_learning_rate = scheduler.get_lr() if scheduler else 0.0
			need_hour, need_mins, need_secs = utils.convert_secs2time(epoch_time.avg * (final_epoch - epoch))
			need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
			self._logger.info('==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(utils.time_string(), epoch, final_epoch, need_time, current_learning_rate) \
				+ ' [Best : Accuracy={:.2f}, Error={:.2f}].'.format(recorder.max_accuracy(False), 100 - recorder.max_accuracy(False)))

			#--------------------
			#start_time = time.time()
			losses, top1, top5 = self._train(model, criterion, optimizer, train_dataloader, epoch, log_print_freq, device)
			train_loss, train_acc = losses.avg, top1.avg
			self._logger.info('\tTrain:      Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}.'.format(top1=top1, top5=top5, error1=100 - top1.avg))
			#self._logger.info('\tTrain:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

			history['loss'].append(train_loss)
			history['acc'].append(train_acc)

			#--------------------
			#start_time = time.time()
			losses, top1, top5 = self._evaluate(model, criterion, test_dataloader, device)
			val_loss, val_acc = losses.avg, top1.avg
			self._logger.info('\tValidation: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f}.'.format(top1=top1, top5=top5, error1=100 - top1.avg, losses=losses))
			#self._logger.info('\tValidation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)

			if scheduler: scheduler.step()

			#--------------------
			dummy = recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

			if val_acc > best_performance_measure:
				best_model_filepath = model_checkpoint_filepath.format(epoch=epoch, val_acc=val_acc)
				self.save_model(best_model_filepath, model)
				best_performance_measure = val_acc

			# Measure elapsed time.
			epoch_time.update(time.time() - start_epoch_time)
			start_epoch_time = time.time()
			recorder.plot_curve(train_result_image_filepath)
		
			#import pdb; pdb.set_trace()
			train_log = collections.OrderedDict()
			train_log['train_loss'] = history['loss']
			train_log['train_acc'] = history['acc']
			train_log['val_loss'] = history['val_loss']
			train_log['val_acc'] = history['val_acc']

			pickle.dump(train_log, open(train_history_filepath, 'wb'))
			utils.plotting(output_dir_path, train_history_filepath)

			sys.stdout.flush()
			time.sleep(0)
		self._logger.info('End training: {} secs.'.format(time.time() - start_train_time))

		return best_model_filepath, history

	def test(self, model, dataloader, device='cpu'):
		# Switch to evaluation mode.
		model.eval()

		self._logger.info('Start testing a model...')
		start_time = time.time()
		inferences, ground_truths = list(), list()
		with torch.no_grad():
			for batch_inputs, batch_outputs in dataloader:
				#batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
				batch_inputs = batch_inputs.to(device)

				model_outputs = model(batch_inputs)

				model_outputs = torch.argmax(model_outputs, -1)
				inferences.extend(model_outputs.cpu().numpy())
				ground_truths.extend(batch_outputs.numpy())
		self._logger.info('End testing a model: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.array(inferences), np.array(ground_truths)
		if inferences is not None and ground_truths is not None:
			self._logger.info('Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			self._logger.info('Test: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			self._logger.warning('Invalid test results.')

	def infer(self, model, inputs, device='cpu'):
		# Switch to evaluation mode.
		model.eval()

		self._logger.info('Start inferring...')
		start_time = time.time()
		with torch.no_grad():
			inputs = inputs.to(device)
			model_outputs = model(inputs)
		self._logger.info('End inferring: {} secs.'.format(time.time() - start_time))
		return torch.argmax(model_outputs, -1)

	def load_model(self, model_filepath, model, device='cpu'):
		try:
			self._logger.info('Start loading a model from {}...'.format(model_filepath))
			start_time = time.time()
			loaded_data = torch.load(model_filepath, map_location=device)
			#model.load_state_dict(loaded_data)
			model.load_state_dict(loaded_data['state_dict'])
			self._logger.info('End loading a model: {} secs.'.format(time.time() - start_time))
			return model
		except Exception as ex:
			self._logger.error('Failed to load a model from {}: {}.'.format(model_filepath, ex))
			#return None
			return model

	def save_model(self, model_filepath, model):
		try:
			self._logger.info('Start saving a model to {}...'.format(model_filepath))
			start_time = time.time()
			# Saves a model using either a .pt or .pth file extension.
			#torch.save(model, model_filepath)
			#torch.save(model.state_dict(), model_filepath)
			torch.save({'state_dict': model.state_dict()}, model_filepath)
			self._logger.info('End saving a model: {} secs.'.format(time.time() - start_time))
		except Exception as ex:
			self._logger.error('Failed to save a model from {}: {}.'.format(model_filepath, ex))

	def load_data(self, batch_size, num_workers=1):
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.5,), (0.5,))
		])

		# Create datasets.
		self._logger.info('Start loading datasets...')
		start_time = time.time()
		train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
		test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
		self._logger.info('End loading datasets: {} secs.'.format(time.time() - start_time))

		# Create data loaders.
		self._logger.info('Start loading data loaders...')
		start_time = time.time()
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		self._logger.info('End loading data loaders: {} secs.'.format(time.time() - start_time))

		return train_dataloader, test_dataloader

	def show_data_info(self, dataloader, mode='Train'):
		data_iter = iter(dataloader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		self._logger.info('{} image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, images.shape, images.dtype, np.min(images), np.max(images)))
		self._logger.info('{} label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mode, labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	def _train(self, model, criterion, optimizer, dataloader, epoch, log_print_freq, device):
		# Switch to train mode.
		model.train()

		batch_time, data_time = utils.AverageMeter(), utils.AverageMeter()
		losses, top1, top5 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
		#running_loss = 0.0
		#start_time = start_batch_time = time.time()
		start_batch_time = time.time()
		for batch_step, (batch_inputs, batch_outputs) in enumerate(dataloader):
			"""
			# One-hot encoding.
			batch_outputs_onehot = torch.LongTensor(batch_outputs.shape[0], self._dataset.num_classes)
			batch_outputs_onehot.zero_()
			batch_outputs_onehot.scatter_(1, batch_outputs.view(batch_outputs.shape[0], -1), 1)
			"""

			batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
			#batch_inputs, batch_outputs, batch_outputs_onehot = batch_inputs.to(device), batch_outputs.to(device), batch_outputs_onehot.to(device)

			data_time.update(time.time() - start_batch_time)

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

			# Measure accuracy and record loss.
			#model_outputs = torch.argmax(model_outputs, -1)
			prec1, prec5 = utils.accuracy(model_outputs, batch_outputs, topk=(1, 5))
			losses.update(loss.item(), batch_inputs.size(0))
			top1.update(prec1.item(), batch_inputs.size(0))
			top5.update(prec5.item(), batch_inputs.size(0))

			"""
			# Print statistics.
			running_loss += loss.item()
			if (batch_step + 1) % 100 == 0:
				self._logger.info('\tStep {}: loss = {:.6f}: {} secs.'.format(batch_step + 1, running_loss / 100, time.time() - start_time))
				running_loss = 0.0
			"""

			# Measure elapsed time.
			batch_time.update(time.time() - start_batch_time)
			start_batch_time = time.time()

			if (batch_step + 1) % log_print_freq == 0:
				self._logger.info('\tEpoch: [{:03d}][{:03d}/{:03d}]   '
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
					'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
					'Loss {loss.val:.4f} ({loss.avg:.4f})   '
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
					'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
					epoch, batch_step + 1, len(dataloader), batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1, top5=top5) + utils.time_string())
		return losses, top1, top5

	def _evaluate(self, model, criterion, dataloader, device):
		# Switch to evaluation mode.
		model.eval()

		losses, top1, top5 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
		with torch.no_grad():
			show = True
			for batch_step, (batch_inputs, batch_outputs) in enumerate(dataloader):
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

				# Measure accuracy and record loss.
				#model_outputs = torch.argmax(model_outputs, -1)
				prec1, prec5 = utils.accuracy(model_outputs.data, batch_outputs, topk=(1, 5))
				losses.update(loss.item(), batch_inputs.size(0))
				top1.update(prec1.item(), batch_inputs.size(0))
				top5.update(prec5.item(), batch_inputs.size(0))

				# Show results.
				if show:
					self._logger.info('\tPrediction: {}.'.format(np.argmax(model_outputs.cpu().numpy(), axis=-1)))
					self._logger.info('\tG/T:        {}.'.format(batch_outputs.cpu().numpy()))
					show = False
		return losses, top1, top5

#--------------------------------------------------------------------

# REF [site] >> https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
class MyLRScheduler(object):
	def __init__(self, optimizer, initial_learning_rate, last_epoch=-1):
		self.optimizer = optimizer
		self.initial_learning_rate = initial_learning_rate

		"""
		# Initialize epoch and base learning rates.
		if last_epoch == -1:
			for group in optimizer.param_groups:
				group.setdefault('initial_lr', group['lr'])
		else:
			for i, group in enumerate(optimizer.param_groups):
				if 'initial_lr' not in group:
					raise KeyError("param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i))
		self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
		"""
		self.last_epoch = last_epoch

		#self.gammas, self.schedule = None, None
		self.gammas = [0.1, 0.1, 0.1, 0.1]  # LR is multiplied by gamma on schedule.
		self.schedule = [20, 30, 40, 50]  # Decrease learning rate at these epochs.

		self.learning_rate = self.initial_learning_rate

	def get_lr(self):
		return self.learning_rate

	def step(self, epoch=None):
		if epoch is None:
			self.last_epoch += 1
		else:
			self.last_epoch = epoch
		if self.gammas and self.schedule:
			self.learning_rate = utils.adjust_learning_rate(self.optimizer, self.last_epoch, self.initial_learning_rate, self.gammas, self.schedule)

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
		help='Number of epochs to train',
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
	initial_epoch, final_epoch, batch_size = 0, args.epoch, args.batch_size
	is_resumed = args.model_file is not None
	num_workers = 4

	initial_learning_rate, momentum, weight_decay = 0.001, 0.9, 0.0001

	model_filepath, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	if model_filepath:
		if not output_dir_path:
			output_dir_path = os.path.dirname(model_filepath)
	else:
		if not output_dir_path:
			output_dir_prefix = 'simple_training2'
			output_dir_suffix = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_{}'.format(output_dir_prefix, output_dir_suffix))
		model_filepath = os.path.join(output_dir_path, 'model.pth')

	#--------------------
	runner = MyRunner(logger)

	# Load datasets.
	train_dataloader, test_dataloader = runner.load_data(batch_size, num_workers=num_workers)
	runner.show_data_info(train_dataloader, 'Train')
	runner.show_data_info(test_dataloader, 'Test')

	if args.train:
		model_checkpoint_filepath = os.path.join(output_dir_path, 'model_ckpt.{epoch:04d}-{val_acc:.5f}.pth')
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		# Create a model.
		model = MyModel()

		if is_resumed:
			# Load a model.
			model = runner.load_model(model_filepath, model, device=device)
		elif False:
			# Initialize model weights.
			for name, param in model.named_parameters():
				#if 'initialized_variable_name' in name:
				#	logger.info(f'Skip {name} as it has already been initialized.')
				#	continue
				try:
					if 'bias' in name:
						torch.nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						torch.nn.init.kaiming_normal_(param)
				except Exception as ex:  # For batch normalization.
					if 'weight' in name:
						param.data.fill_(1)
					continue

		if False:
			# Filter model parameters only that require gradients.
			model_params, num_model_params = list(), list()
			for p in filter(lambda p: p.requires_grad, model.parameters()):
				model_params.append(p)
				num_model_params.append(np.prod(p.size()))
			print('#trainable model parameters = {}.'.format(sum(num_model_params)))
			#print('Trainable model parameters:')
			#[print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]
		else:
			model_params = model.parameters()

		# Create a trainer.
		criterion = torch.nn.CrossEntropyLoss().to(device)
		optimizer = torch.optim.SGD(model_params, lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
		#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
		scheduler = MyLRScheduler(optimizer, initial_learning_rate)

		if model:
			#if torch.cuda.device_count() > 1:
			#	device_ids = [0, 1]
			#	model = torch.nn.DataParallel(model, device_ids=device_ids)
			model = model.to(device)

			# Train a model.
			best_model_filepath, history = runner.train(model, criterion, optimizer, scheduler, train_dataloader, test_dataloader, model_checkpoint_filepath, output_dir_path, initial_epoch, final_epoch, device=device)

			# Save a model.
			model_filepath = os.path.join(output_dir_path, 'best_model.pth')
			if best_model_filepath:
				try:
					shutil.copyfile(best_model_filepath, model_filepath)
					logger.info('Copied the best trained model to {}.'.format(model_filepath))
				except (FileNotFoundError, PermissionError) as ex:
					logger.error('Failed to copy the best trained model to {}: {}.'.format(model_filepath, ex))
					model_filepath = None
			else:
				runner.save_model(model_filepath, model)

			#logger.info('Train history = {}.'.format(history))
			#swl_ml_util.display_train_history(history)
			#if os.path.exists(output_dir_path):
			#	swl_ml_util.save_train_history(history, output_dir_path)

	if args.test or args.infer:
		if model_filepath and os.path.exists(model_filepath):
			# Create a model.
			model = MyModel()
			# Load a model.
			model = runner.load_model(model_filepath, model, device=device)

			if model:
				#if torch.cuda.device_count() > 1:
				#	device_ids = [0, 1]
				#	model = torch.nn.DataParallel(model, device_ids=device_ids)
				model = model.to(device)

			if args.test and model:
				runner.test(model, test_dataloader, device=device)

			if args.infer and model:
				#inputs = torch.cat([batch_data[0] for batch_data in test_dataloader], dim=0)
				data_iter = iter(test_dataloader)
				inputs, _ = data_iter.next()

				inferences = runner.infer(model, inputs, device=device)

				inferences = inferences.cpu().numpy()
				if inferences is not None:
					logger.info('Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

					results = {idx: inf for idx, inf in enumerate(inferences) if idx < 100}
					logger.info('Inference results (index: inference): {}.'.format(results))
				else:
					logger.warning('Invalid inference results.')
		else:
			logger.error('Model file, {} does not exist.'.format(model_filepath))

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training2.py --train --test --infer --epoch 20 --gpu 0

if '__main__' == __name__:
	main()
