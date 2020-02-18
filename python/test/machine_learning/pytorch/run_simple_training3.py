#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os, shutil, collections, pickle, argparse, logging, logging.handlers, time, datetime
import numpy as np
import torch
import torchvision
import cv2
#import swl.machine_learning.util as swl_ml_util
import utils

#--------------------------------------------------------------------

class MyDataset(object):
	def __init__(self):
		#self._image_height, self._image_width, self._image_channel = 28, 28, 1  # 784 = 28 * 28.
		self._num_classes = 10

		# Preprocess.
		self._transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

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
	def __init__(self, batch_size, logger):
		self._logger = logger

		# Create a dataset.
		self._logger.info('[SWL] Start loading dataset...')
		start_time = time.time()
		self._dataset = MyDataset()
		self._train_loader = self._dataset.create_train_data_loader(batch_size, shuffle=True, num_workers=4)
		self._test_loader = self._dataset.create_test_data_loader(batch_size, shuffle=False, num_workers=4)
		self._logger.info('[SWL] End loading dataset: {} secs.'.format(time.time() - start_time))

		data_iter = iter(self._train_loader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		self._logger.info('[SWL] Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
		self._logger.info('[SWL] Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

		data_iter = iter(self._test_loader)
		images, labels = data_iter.next()
		images, labels = images.numpy(), labels.numpy()
		self._logger.info('[SWL] Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
		self._logger.info('[SWL] Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	def train(self, model_filepath, model_checkpoint_filepath, output_dir_path, final_epoch, initial_epoch=0, is_training_resumed=False, log=None, device='cpu'):
		# Create a model.
		model = MyModel()

		#if torch.cuda.device_count() > 1:
		#	device_ids = [0, 1]
		#	model = torch.nn.DataParallel(model, device_ids=device_ids)
		model = model.to(device)

		# Create a trainer.
		criterion = torch.nn.CrossEntropyLoss().to(device)
		initial_learning_rate, momentum, weight_decay = 0.001, 0.9, 0.0001
		optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)

		#--------------------
		train_history_filepath = os.path.join(output_dir_path, 'train_history.pkl')
		train_result_image_filepath = os.path.join(output_dir_path, 'results.png')
		log_print_freq = 100
		if False:
			gammas, schedule = None, None
		else:
			gammas = [0.1, 0.1, 0.1, 0.1]  # LR is multiplied by gamma on schedule.
			schedule = [20, 30, 40, 50]  # Decrease learning rate at these epochs.

		if log:
			utils.print_log('Save path: {}.'.format(output_dir_path), log)
			#state = {k: v for k, v in args._get_kwargs()}
			#utils.print_log(state, log)
			utils.print_log('Python version: {}.'.format(sys.version.replace('\n', ' ')), log)
			utils.print_log('Torch version: {}.'.format(torch.__version__), log)
			utils.print_log('cuDNN version: {}.'.format(torch.backends.cudnn.version()), log)
			utils.print_log('=> Model:\n {}.'.format(model), log)

		history = {
			'acc': list(),
			'loss': list(),
			'val_acc': list(),
			'val_loss': list()
		}

		#--------------------
		if is_training_resumed:
			if os.path.isfile(model_filepath):
				# Restore a model.
				try:
					self._logger.info('[SWL] Start restoring a model...')
					start_time = time.time()
					if log: utils.print_log("=> loading checkpoint '{}'".format(model_filepath), log)
					checkpoint = torch.load(model_filepath)
					initial_epoch = checkpoint['epoch']
					#architecture = checkpoint['arch']
					model.load_state_dict(checkpoint['state_dict'])
					optimizer.load_state_dict(checkpoint['optimizer'])
					recorder = checkpoint['recorder']
					best_acc = recorder.max_accuracy(False)
					if log: utils.print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(model_filepath, best_acc, checkpoint['epoch']), log)
					self._logger.info('[SWL] End restoring a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
				except:
					self._logger.error('[SWL] Failed to restore a model from {}.'.format(model_filepath))
					return
			else:
				self._logger.error('[SWL] Invalid model file, {}.'.format(model_filepath))
				return
			history['acc'] = recorder.epoch_accuracy[:,0].tolist()
			history['loss'] = recorder.epoch_losses[:,0].tolist()
			history['val_acc'] = recorder.epoch_accuracy[:,1].tolist()
			history['val_loss'] = recorder.epoch_losses[:,1].tolist()
			recorder.resize(final_epoch)
		else:
			recorder = utils.RecorderMeter(final_epoch)

		#--------------------
		if is_training_resumed:
			self._logger.info('[SWL] Resume training...')
		else:
			self._logger.info('[SWL] Start training...')
		epoch_time = utils.AverageMeter()
		best_performance_measure = 0
		best_model_filepath = None
		start_total_time = start_epoch_time = time.time()
		for epoch in range(initial_epoch, final_epoch):
			self._logger.info('[SWL] Epoch {}/{}'.format(epoch, final_epoch - 1))

			if not gammas or not schedule:
				current_learning_rate = initial_learning_rate
			else:
				current_learning_rate = utils.adjust_learning_rate(optimizer, epoch, initial_learning_rate, gammas, schedule)
			need_hour, need_mins, need_secs = utils.convert_secs2time(epoch_time.avg * (final_epoch - epoch))
			need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
			if log: utils.print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(utils.time_string(), epoch, final_epoch, need_time, current_learning_rate) \
				+ ' [Best : Accuracy={:.2f}, Error={:.2f}].'.format(recorder.max_accuracy(False), 100 - recorder.max_accuracy(False)), log)

			#--------------------
			losses, top1, top5 = self._train(self._train_loader, optimizer, model, criterion, epoch, log_print_freq, log, device)
			train_loss, train_acc = losses.avg, top1.avg
			if log: utils.print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}.'.format(top1=top1, top5=top5, error1=100 - top1.avg), log)
			#self._logger.info('[SWL]    Train:      loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(train_loss, train_acc, time.time() - start_time))

			history['loss'].append(train_loss)
			history['acc'].append(train_acc)

			#--------------------
			losses, top1, top5 = self._evaluate(self._test_loader, model, criterion, device)
			val_loss, val_acc = losses.avg, top1.avg
			if log: utils.print_log('  **Validation** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f}.'.format(top1=top1, top5=top5, error1=100 - top1.avg, losses=losses), log)
			#self._logger.info('[SWL]    Validation: loss = {:.6f}, accuracy = {:.6f}: {} secs.'.format(val_loss, val_acc, time.time() - start_time))

			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)

			#--------------------
			dummy = recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

			if val_acc > best_performance_measure:
				self._logger.info('[SWL] Start saving a model...')
				start_time = time.time()
				best_model_filepath = model_checkpoint_filepath.format(epoch=epoch, val_acc=val_acc)
				torch.save({
						'epoch': epoch + 1,
						#'arch': architecture,
						'state_dict': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'recorder': recorder,
					},
					best_model_filepath)
				self._logger.info('[SWL] End saving a model to {}: {} secs.'.format(best_model_filepath, time.time() - start_time))
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
		self._logger.info('[SWL] End training: {} secs.'.format(time.time() - start_total_time))

		if best_model_filepath:
			try:
				shutil.copyfile(best_model_filepath, model_filepath)
				self._logger.info('[SWL] Copied the best model to {}.'.format(model_filepath))
			except (FileNotFoundError, PermissionError) as ex:
				self._logger.error('[SWL] Failed to copy the best model to {}: {}.'.format(model_filepath, ex))
		else:
			torch.save({
					'epoch': epoch + 1,
					#'arch': architecture,
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'recorder': recorder,
				},
				model_filepath)
			self._logger.info('[SWL] Saved the best model to {}.'.format(model_filepath))

		return history

	def test(self, model_filepath, log=None, device='cpu'):
		# Create a model.
		model = MyModel()
		model = model.to(device)
		#device_ids = [0, 1]
		#model = torch.nn.DataParallel(model, device_ids=device_ids)

		if os.path.isfile(model_filepath):
			# Load a model.
			try:
				self._logger.info('[SWL] Start loading a model...')
				start_time = time.time()
				if log: utils.print_log("=> loading checkpoint '{}'".format(model_filepath), log)
				checkpoint = torch.load(model_filepath)
				#architecture = checkpoint['arch']
				model.load_state_dict(checkpoint['state_dict'])
				#optimizer.load_state_dict(checkpoint['optimizer'])
				recorder = checkpoint['recorder']
				best_acc = recorder.max_accuracy(False)
				if log: utils.print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(model_filepath, best_acc, checkpoint['epoch']), log)
				self._logger.info('[SWL] End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except:
				self._logger.error('[SWL] Failed to load a model from {}.'.format(model_filepath))
				return
		else:
			self._logger.error('[SWL] Invalid model file, {}.'.format(model_filepath))
			return

		# Switch to evaluation mode.
		model.eval()

		#--------------------
		self._logger.info('[SWL] Start testing...')
		start_time = time.time()
		inferences, ground_truths = list(), list()
		with torch.no_grad():
			for batch_data in self._test_loader:
				batch_inputs, batch_outputs = batch_data
				#batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
				batch_inputs = batch_inputs.to(device)

				model_outputs = model(batch_inputs)

				model_outputs = torch.argmax(model_outputs, -1)
				inferences.extend(model_outputs.cpu().numpy())
				ground_truths.extend(batch_outputs.numpy())
		self._logger.info('[SWL] End testing: {} secs.'.format(time.time() - start_time))

		inferences, ground_truths = np.array(inferences), np.array(ground_truths)
		if inferences is not None and ground_truths is not None:
			self._logger.info('[SWL] Test: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			correct_estimation_count = np.count_nonzero(np.equal(inferences, ground_truths))
			self._logger.info('[SWL] Test: accuracy = {} / {} = {}.'.format(correct_estimation_count, ground_truths.size, correct_estimation_count / ground_truths.size))
		else:
			self._logger.warning('[SWL] Invalid test results.')

	def infer(self, model_filepath, batch_size, shuffle=False, log=None, device='cpu'):
		if batch_size is None or batch_size <= 0:
			raise ValueError('Invalid batch size: {}'.format(batch_size))

		# Create a model.
		model = MyModel()
		model = model.to(device)
		#device_ids = [0, 1]
		#model = torch.nn.DataParallel(model, device_ids=device_ids)

		if os.path.isfile(model_filepath):
			# Load a model.
			try:
				self._logger.info('[SWL] Start loading a model...')
				start_time = time.time()
				if log: utils.print_log("=> loading checkpoint '{}'".format(model_filepath), log)
				checkpoint = torch.load(model_filepath)
				#architecture = checkpoint['arch']
				model.load_state_dict(checkpoint['state_dict'])
				#optimizer.load_state_dict(checkpoint['optimizer'])
				recorder = checkpoint['recorder']
				best_acc = recorder.max_accuracy(False)
				if log: utils.print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(model_filepath, best_acc, checkpoint['epoch']), log)
				self._logger.info('[SWL] End loading a model from {}: {} secs.'.format(model_filepath, time.time() - start_time))
			except:
				self._logger.error('[SWL] Failed to load a model from {}.'.format(model_filepath))
				return
		else:
			self._logger.error('[SWL] Invalid model file, {}.'.format(model_filepath))
			return

		# Switch to evaluation mode.
		model.eval()

		#--------------------
		inf_loader = self._dataset.create_test_data_loader(batch_size, shuffle=shuffle, num_workers=4)
		inf_images = list(batch_data[0] for batch_data in inf_loader)

		#--------------------
		self._logger.info('[SWL] Start inferring...')
		inferences = list()
		start_time = time.time()
		with torch.no_grad():
			for batch_images in inf_images:
				batch_images = batch_images.to(device)
				model_outputs = model(batch_images)

				model_outputs = torch.argmax(model_outputs, -1)
				inferences.extend(model_outputs.cpu().numpy())
		self._logger.info('[SWL] End inferring: {} secs.'.format(time.time() - start_time))

		inferences = np.array(inferences)
		if inferences is not None:
			self._logger.info('[SWL] Inference: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(inferences.shape, inferences.dtype, np.min(inferences), np.max(inferences)))

			results = {idx: inf for idx, inf in enumerate(inferences) if idx < 100}
			self._logger.info('[SWL] Inference results (index: inference): {}.'.format(results))
		else:
			self._logger.warning('[SWL] Invalid inference results.')

	def _train(self, data_loader, optimizer, model, criterion, epoch, log_print_freq, log, device):
		# Switch to train mode.
		model.train()

		batch_time, data_time = utils.AverageMeter(), utils.AverageMeter()
		losses, top1, top5 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
		#running_loss = 0.0
		start_time = start_batch_time = time.time()
		for batch_step, batch_data in enumerate(data_loader):
			batch_inputs, batch_outputs = batch_data

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

			# Print statistics.
			"""
			running_loss += loss.item()
			if (batch_step + 1) % 100 == 0:
				self._logger.info('[SWL]    Step {}: loss = {:.6f}: {} secs.'.format(batch_step + 1, running_loss / 100, time.time() - start_time))
				running_loss = 0.0
			"""

			# Measure elapsed time.
			batch_time.update(time.time() - start_batch_time)
			start_batch_time = time.time()

			if log and batch_step % log_print_freq == 0:
				utils.print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
					'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
					'Loss {loss.val:.4f} ({loss.avg:.4f})   '
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
					'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
					epoch, batch_step, len(self._train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1, top5=top5) + utils.time_string(), log)
		return losses, top1, top5

	def _evaluate(self, data_loader, model, criterion, device):
		# Switch to evaluation mode.
		model.eval()

		losses, top1, top5 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
		start_time = time.time()
		with torch.no_grad():
			for batch_step, batch_data in enumerate(data_loader):
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

				# Measure accuracy and record loss.
				#model_outputs = torch.argmax(model_outputs, -1)
				prec1, prec5 = utils.accuracy(model_outputs.data, batch_outputs, topk=(1, 5))
				losses.update(loss.item(), batch_inputs.size(0))
				top1.update(prec1.item(), batch_inputs.size(0))
				top5.update(prec5.item(), batch_inputs.size(0))
		return losses, top1, top5

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

def get_logger(name, log_level, is_rotating=True):
	if not os.path.isdir('log'):
		os.mkdir('log')

	log_filepath = './log/' + (name if name else 'swl') + '.log'
	if is_rotating:
		file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10000000, backupCount=10)
	else:
		file_handler = logging.FileHandler(log_filepath)
	stream_handler = logging.StreamHandler()

	formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)s][%(asctime)s] %(message)s')
	#formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def main():
	args = parse_command_line_options()

	logger = get_logger(os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, is_rotating=True)
	logger.info('[SWL] ----------------------------------------------------------------------')
	logger.info('[SWL] Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('[SWL] Command-line arguments: {}.'.format(sys.argv))
	logger.info('[SWL] Command-line options: {}.'.format(vars(args)))

	if not args.train and not args.test and not args.infer:
		logger.error('[SWL] At least one of command line options "--train", "--test", and "--infer" has to be specified.')
		return

	#if args.gpu:
	#	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

	if True:
		if output_dir_path and output_dir_path.strip() and not os.path.exists(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)
		train_log_filepath = os.path.join(output_dir_path, 'train_log.txt')
		log = open(train_log_filepath, 'w')
	else:
		log = sys.out

	#--------------------
	runner = MyRunner(batch_size, logger)

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
				logger.error('[SWL] Failed to copy a model, {}: {}.'.format(model_filepath, ex))
				return
		model_filepath = new_model_filepath

		device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')
		history = runner.train(model_filepath, model_checkpoint_filepath, output_dir_path, final_epoch, initial_epoch, is_training_resumed, log=log, device=device)

		#logger.info('[SWL] Train history = {}.'.format(history))
		#swl_ml_util.display_train_history(history)
		#if os.path.exists(output_dir_path):
		#	swl_ml_util.save_train_history(history, output_dir_path)

	if args.test:
		if not model_filepath or not os.path.exists(model_filepath):
			logger.error('[SWL] Model file, {} does not exist.'.format(model_filepath))
			return

		device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')
		runner.test(model_filepath, log=log, device=device)

	if args.infer:
		if not model_filepath or not os.path.exists(model_filepath):
			logger.error('[SWL] Model file, {} does not exist.'.format(model_filepath))
			return

		device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu else 'cpu')
		runner.infer(model_filepath, batch_size, log=log, device=device)

	log.close()

#--------------------------------------------------------------------

# Usage:
#	python run_simple_training.py --train --test --infer --epoch 30 --gpu 0

if '__main__' == __name__:
	main()
