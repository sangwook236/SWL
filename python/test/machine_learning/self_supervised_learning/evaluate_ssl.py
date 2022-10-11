#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, math, logging, pickle, datetime, time
import numpy as np
import torch
import pytorch_lightning as pl
import yaml
import utils

class ClassificationModule(pl.LightningModule):
	def __init__(self, config, num_classes, logger=None):
		super().__init__()
		self.save_hyperparameters()

		self.config = config
		self._logger = logger

		self.is_all_model_params_optimized = True

		# Build a classifier.
		if 'user_defined_model' in config:
			self.model, classifier_output_dim = utils.construct_user_defined_model(config['user_defined_model'])
			assert classifier_output_dim == num_classes, 'The output dimension ({}) of the user-defined model does not match the number of classes ({})'.format(classifier_output_dim, num_classes)
			is_model_initialized = True
		elif 'predefined_model' in config:
			predefined_model_name = config['predefined_model'].get('model_name', 'linear')
			classifier_input_dim = config['predefined_model']['input_dim']
			if predefined_model_name == 'linear':
				# Linear classifier.
				self.model = torch.nn.Linear(classifier_input_dim, num_classes)
			elif predefined_model_name == 'mlp':
				# MLP classifier.
				classifier_hidden_dim = config['predefined_model'].get('hidden_dim', 128)
				self.model = torch.nn.Sequential(
					torch.nn.Linear(classifier_input_dim, classifier_hidden_dim),
					torch.nn.BatchNorm1d(classifier_hidden_dim),
					torch.nn.ReLU(inplace=True),
					torch.nn.Linear(classifier_hidden_dim, num_classes),
				)
			else:
				raise ValueError('Invalid classifier model name, {}'.format(predefined_model_name))
			is_model_initialized = True
		else:
			raise ValueError('No classifier specified')

		# Define a loss.
		self.criterion = torch.nn.NLLLoss(reduction='mean')
		#self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

		#-----
		if is_model_initialized:
			# Initialize model weights.
			for name, param in self.model.named_parameters():
				try:
					if 'bias' in name:
						torch.nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						torch.nn.init.kaiming_normal_(param)
				except Exception as ex:  # For batch normalization.
					if 'weight' in name:
						param.data.fill_(1)
					continue
			'''
			for param in self.model.parameters():
				if param.dim() > 1:
					torch.nn.init.xavier_uniform_(param)  # Initialize parameters with Glorot / fan_avg.
			'''

	def configure_optimizers(self):
		if self.is_all_model_params_optimized:
			model_params = list(self.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, self.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, self.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			if self.trainer and self.trainer.is_global_zero and self._logger:
				self._logger.info('#trainable model parameters = {}.'.format(num_model_params))
				#self._logger.info('Trainable model parameters: {}.'.format([(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, self.named_parameters())]))

		optimizer = utils.construct_optimizer(self.config['optimizer'], model_params)
		scheduler, is_epoch_based = utils.construct_lr_scheduler(self.config.get('lr_scheduler', None), optimizer, self.trainer.max_epochs)
		#scheduler, is_epoch_based = utils.construct_lr_scheduler(self.config.get('lr_scheduler', None), optimizer, self.config['epochs'])

		if scheduler:
			return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch' if is_epoch_based else 'step'}]
		else:
			return optimizer

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		start_time = time.time()
		loss, model_outputs = self._shared_step(batch, batch_idx)
		performances = self._evaluate_performance(model_outputs, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict(
			{'train_loss': loss, 'train_acc': performances['acc'], 'train_time': step_time},
			on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True
		)

		return loss

	def validation_step(self, batch, batch_idx):
		start_time = time.time()
		loss, model_outputs = self._shared_step(batch, batch_idx)
		performances = self._evaluate_performance(model_outputs, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict({'val_loss': loss, 'val_acc': performances['acc'], 'val_time': step_time}, rank_zero_only=True)

	def test_step(self, batch, batch_idx):
		start_time = time.time()
		loss, model_outputs = self._shared_step(batch, batch_idx)
		performances = self._evaluate_performance(model_outputs, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict({'test_loss': loss, 'test_acc': performances['acc'], 'test_time': step_time}, rank_zero_only=True)

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		return self(batch[0])  # Calls forward().

	def _shared_step(self, batch, batch_idx):
		batch_inputs, batch_outputs = batch

		model_outputs = self.model(batch_inputs)
		loss = self.criterion(model_outputs, batch_outputs)

		return loss, model_outputs

	def _evaluate_performance(self, model_outputs, batch, batch_idx):
		_, batch_outputs = batch

		#acc = (batch_outputs == torch.argmax(model_outputs, dim=-1)).sum().item()
		acc = (batch_outputs == torch.argmax(model_outputs, dim=-1)).float().mean().item()

		return {'acc': acc}

def extract_features(model, dataloader, use_projector=False, use_predictor=False, device='cuda'):
	srcs, tgts = list(), list()
	for batch_inputs, batch_outputs in dataloader:
		batch_inputs = batch_inputs.to(device)
		srcs.append(model(batch_inputs, use_projector, use_predictor).cpu())  # [batch size, feature dim].
		tgts.append(batch_outputs)  # [batch size, 1].
		del batch_inputs  # Free memory in CPU or GPU.
	return torch.vstack(srcs), torch.hstack(tgts)

class FeatureDataset(torch.utils.data.Dataset):
	def __init__(self, model, dataloader, use_projector, use_predictor, logger, device):
		super().__init__()

		model = model.to(device)

		if logger: logger.info('Extracting features...')
		start_time = time.time()
		model.eval()
		model.freeze()
		with torch.no_grad():
			self.srcs, self.tgts = extract_features(model, dataloader, use_projector, use_predictor, device)
		if logger: logger.info('Features extracted: {} secs.'.format(time.time() - start_time))
		assert len(self.srcs) == len(self.tgts), 'Unmatched source and target lengths, {} != {}'.format(len(self.srcs), len(self.tgts))

	def __len__(self):
		return len(self.srcs)

	def __getitem__(self, idx):
		return self.srcs[idx], self.tgts[idx]

class FeatureIterableDataset(torch.utils.data.IterableDataset):
	def __init__(self, model, dataloader, use_projector, use_predictor, logger, device):
		super().__init__()

		self.model = model
		self.dataloader = dataloader
		self.use_projector, self.use_predictor = use_projector, use_predictor
		self.logger = logger
		self.device = device

	def __iter__(self):
		self.model = self.model.to(self.device)

		if self.logger: self.logger.info('Extracting features...')
		start_time = time.time()
		self.model.eval()
		self.model.freeze()
		with torch.no_grad():
			srcs, tgts = extract_features(self.model, self.dataloader, self.use_projector, self.use_predictor, self.device)
		if self.logger: self.logger.info('Features extracted: {} secs.'.format(time.time() - start_time))
		assert len(srcs) == len(tgts), 'Unmatched source and target lengths, {} != {}'.format(len(srcs), len(tgts))
		num_examples = len(srcs)

		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:  # Single-process data loading, return the full iterator.
			return iter(zip(srcs, tgts))
		else:  # In a worker process.
			# Split workload.
			worker_id = worker_info.id
			num_examples_per_worker = math.ceil(num_examples / float(worker_info.num_workers))
			iter_start = worker_id * num_examples_per_worker
			iter_end = min(iter_start + num_examples_per_worker, num_examples)
			return iter(zip(srcs[iter_start:iter_end], tgts[iter_start:iter_end]))

def prepare_simple_feature_data(config, model, use_projector=False, use_predictor=False, logger=None, device='cuda'):
	# Create data loaders.
	train_dataloader, test_dataloader, num_classes = utils.prepare_open_data(config, show_info=True, show_data=False, logger=logger)

	# Create feature datasets.
	if logger: logger.info('Creating feature datasets...')
	start_time = time.time()
	train_feature_dataset = FeatureDataset(model, train_dataloader, use_projector, use_predictor, logger, device)
	test_feature_dataset = FeatureDataset(model, test_dataloader, use_projector, use_predictor, logger, device)
	if logger: logger.info('Feature datasets created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train examples = {}, #test examples = {}.'.format(len(train_feature_dataset), len(test_feature_dataset)))

	# Create feature data loaders.
	if logger: logger.info('Creating feature data loaders...')
	start_time = time.time()
	train_feature_dataloader = torch.utils.data.DataLoader(train_feature_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=train_dataloader.num_workers, persistent_workers=False)
	test_feature_dataloader = torch.utils.data.DataLoader(test_feature_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=test_dataloader.num_workers, persistent_workers=False)
	if logger: logger.info('Feature data loaders created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_feature_dataloader), len(test_feature_dataloader)))

	return train_feature_dataloader, test_feature_dataloader, num_classes

def prepare_feature_data(config, model, use_projector=False, use_predictor=False, logger=None, device='cuda'):
	# Create data loaders.
	train_dataloader, test_dataloader, num_classes = utils.prepare_open_data(config, show_info=True, show_data=False, logger=logger)

	# Create feature datasets.
	if logger: logger.info('Creating feature datasets...')
	start_time = time.time()
	train_feature_dataset = FeatureIterableDataset(model, train_dataloader, use_projector, use_predictor, logger, device)
	test_feature_dataset = FeatureIterableDataset(model, test_dataloader, use_projector, use_predictor, logger, device)
	if logger: logger.info('Feature datasets created: {} secs.'.format(time.time() - start_time))
	#if logger: logger.info('#train examples = {}, #test examples = {}.'.format(len(train_feature_dataset), len(test_feature_dataset)))  # NOTE [error] >> TypeError: object of type 'FeatureDataset' has no len().

	# Create feature data loaders.
	if logger: logger.info('Creating feature data loaders...')
	start_time = time.time()
	# TODO [check] >> num_workers = 0?
	#train_feature_dataloader = torch.utils.data.DataLoader(train_feature_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=train_dataloader.num_workers, persistent_workers=False)
	#test_feature_dataloader = torch.utils.data.DataLoader(test_feature_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=test_dataloader.num_workers, persistent_workers=False)
	train_feature_dataloader = torch.utils.data.DataLoader(train_feature_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, persistent_workers=False)
	test_feature_dataloader = torch.utils.data.DataLoader(test_feature_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, persistent_workers=False)
	if logger: logger.info('Feature data loaders created: {} secs.'.format(time.time() - start_time))
	#if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_feature_dataloader), len(test_feature_dataloader)))  # NOTE [error] >> TypeError: object of type 'FeatureDataset' has no len().

	return train_feature_dataloader, test_feature_dataloader, num_classes

def evaluate(config, train_feature_dataloader, test_feature_dataloader, num_classes, output_dir_path, logger=None, device='cuda'):
	from tqdm import tqdm

	# Build a classifier.
	classifier = ClassificationModule(config, num_classes, logger)

	# Create a trainer.
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=os.path.join(output_dir_path, 'checkpoints') if output_dir_path else './checkpoints',
		filename='classifier-{epoch:03d}-{step:05d}-{val_acc:.5f}-{val_loss:.5f}',
		monitor='val_loss',
		mode='min',
		save_top_k=5,
	)
	pl_callbacks = [checkpoint_callback]
	tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=(output_dir_path + '/lightning_logs') if output_dir_path else './lightning_logs', name='', version=None)
	pl_logger = [tensorboard_logger]

	gradient_clip_val = None
	gradient_clip_algorithm = None
	#trainer = pl.Trainer(devices=-1, accelerator='gpu', strategy='dp', auto_select_gpus=True, max_epochs=config['epochs'], callbacks=pl_callbacks, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=output_dir_path)  # When using the default logger.
	trainer = pl.Trainer(devices=-1, accelerator='gpu', strategy='dp', auto_select_gpus=True, max_epochs=config['epochs'], callbacks=pl_callbacks, logger=pl_logger, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=None)

	# Train the classifier.
	if logger: logger.info('Training the classifier...')
	start_time = time.time()
	trainer.fit(classifier, train_dataloaders=train_feature_dataloader, val_dataloaders=test_feature_dataloader, ckpt_path=None)
	if logger: logger.info('The classifier trained: {} secs.'.format(time.time() - start_time))

	#-----
	if False:
		# Validate the classifier.
		if logger: logger.info('Validating the classifier...')
		start_time = time.time()
		val_metrics = trainer.validate(model=classifier, dataloaders=test_feature_dataloader, ckpt_path=None, verbose=True)
		if logger: logger.info('The classifier validated: {} secs.'.format(time.time() - start_time))
		if logger: logger.info('Validation metrics: {}.'.format(val_metrics))

	#--------------------
	# Evaluate.
	classifier = classifier.to(device)

	if logger: logger.info('Evaluating...')
	start_time = time.time()
	classifier.eval()
	classifier.freeze()
	with torch.no_grad():
		gts, predictions = list(), list()
		for batch_inputs, batch_outputs in tqdm(test_feature_dataloader):
			gts.append(batch_outputs.numpy())
			predictions.append(classifier(batch_inputs.to(device)).cpu().numpy())  # [batch size, #classes].
		gts, predictions = np.hstack(gts), np.argmax(np.vstack(predictions), axis=-1)
	if logger: logger.info('Evaluated: {} secs.'.format(time.time() - start_time))
	assert len(gts) == len(predictions)
	num_examples = len(gts)

	results = gts == predictions
	num_correct_examples = results.sum().item()
	acc = results.mean().item()

	if logger: logger.info('Evaluation: accuracy = {} / {} = {}.'.format(num_correct_examples, num_examples, acc))

	#--------------------
	if True:
		# Save evaluation results.
		evaluation_result_filepath = os.path.join(output_dir_path, 'evaluation_results.txt')
		try:
			if logger: logger.info('Saving evaluation results to {}...'.format(evaluation_result_filepath))
			start_time = time.time()
			incorrect_indices = np.argwhere(results == False).squeeze(axis=-1)
			with open(evaluation_result_filepath, 'w+', encoding='utf-8') as fd:
				fd.write('--------------------------------------------------\n')
				for idx in incorrect_indices:
					gt, pred = gts[idx], predictions[idx]
					fd.write('{}:\t{}\t{}\n'.format(idx, gt, pred))
			if logger: logger.info('Evaluation results saved: {} secs.'.format(time.time() - start_time))
		except Exception as ex:
			if logger: logger.warning('Failed to save evaluation results: {}.'.format(ex))

#--------------------------------------------------------------------

def main():
	args = utils.parse_command_line_options(is_training=False)

	try:
		with open(args.config, encoding='utf-8') as fd:
			config = yaml.load(fd, Loader=yaml.Loader)
	except yaml.scanner.ScannerError as ex:
		print('yaml.scanner.ScannerError in {}.'.format(args.config))
		logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		raise
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}.'.format(args.config))
		logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		raise
	except FileNotFoundError as ex:
		print('Config file not found, {}.'.format(args.config))
		logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		raise
	except Exception as ex:
		print('Exception raised in {}.'.format(args.config))
		logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		raise

	#--------------------
	assert ('evaluation' == config['stage']) if isinstance(config['stage'], str) else ('evaluation' in config['stage']), 'Invalid stage(s), {}'.format(config['stage'])
	#config['ssl_type'] = config.get('ssl_type', 'simclr')
	assert config['ssl_type'] in ['byol', 'relic', 'simclr', 'simsiam'], 'Invalid SSL model, {}'.format(config['ssl_type'])
	assert config['data']['dataset'] in ['cifar10', 'imagenet', 'mnist'], 'Invalid dataset, {}'.format(config['data']['dataset'])

	model_filepath_to_load = os.path.normpath(args.model_file) if args.model_file else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)

	output_dir_path = os.path.normpath(args.out_dir) if args.out_dir else None
	#if model_filepath_to_load and not output_dir_path:
	#	output_dir_path = os.path.dirname(model_filepath_to_load)
	if not output_dir_path:
		timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_eval_outputs_{}'.format(config['ssl_type'], timestamp))
	if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
		os.makedirs(output_dir_path, exist_ok=True)

	#--------------------
	logger = utils.get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else output_dir_path, is_rotating=True)
	logger.info('----------------------------------------------------------------------')
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Configuration: {}.'.format(config))
	logger.info('Python: version = {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('Torch: version = {}, distributed = {} & {}.'.format(torch.__version__, 'available' if torch.distributed.is_available() else 'unavailable', 'initialized' if torch.distributed.is_initialized() else 'uninitialized'))
	logger.info('PyTorch Lightning: version = {}, distributed = {}.'.format(pl.__version__, 'available' if pl.utilities.distributed.distributed_available() else 'unavailable'))
	logger.info('CUDA: version = {}, {}.'.format(torch.version.cuda, 'available' if torch.cuda.is_available() else 'unavailable'))
	logger.info('cuDNN: version = {}.'.format(torch.backends.cudnn.version()))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('Device: {}.'.format(device))

	logger.info('Output directory path: {}.'.format(output_dir_path))

	#--------------------
	try:
		use_projector, use_predictor = False, False  # For SSL models.

		# Load a SSL model.
		logger.info('Loading a SSL model from {}...'.format(model_filepath_to_load))
		start_time = time.time()
		ssl_model = utils.load_ssl(config['ssl_type'], model_filepath_to_load)
		logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))

		if True:
			# Evaluate the pretrained SSL model by training a classifier on its features.

			# Prepare feature data.
			if False:
				train_feature_dataloader, test_feature_dataloader, num_classes = prepare_simple_feature_data(config['data'], ssl_model, use_projector, use_predictor, logger, device)
				del ssl_model  # Free memory in CPU or GPU.
			else:
				train_feature_dataloader, test_feature_dataloader, num_classes = prepare_feature_data(config['data'], ssl_model, use_projector, use_predictor, logger, device)

			# Evaluate the pretrained SSL model.
			evaluate(config['evaluation'], train_feature_dataloader, test_feature_dataloader, num_classes, output_dir_path, logger, device)
		else:
			# Extract features from the pretrained SSL model.

			# Prepare data.
			train_dataloader, test_dataloader, num_classes = utils.prepare_open_data(config['data'], show_info=True, show_data=False, logger=logger)

			# Extract features.
			ssl_model = ssl_model.to(device)

			logger.info('Extracting features...')
			start_time = time.time()
			ssl_model.eval()
			ssl_model.freeze()
			with torch.no_grad():
				train_input_features, train_outputs = extract_features(ssl_model, train_dataloader, use_projector, use_predictor, device)
				test_input_features, test_outputs = extract_features(ssl_model, test_dataloader, use_projector, use_predictor, device)
			logger.info('Features extracted: {} secs.'.format(time.time() - start_time))
			logger.info('#train features = {}, #test features = {}.'.format(len(train_input_features), len(test_input_features)))

			if True:
				# Save the extracted features.
				feature_filepath = os.path.join(output_dir_path, '{}_features.pkl'.format(config['ssl_type']))
				try:
					logger.info('Saving features to {}...'.format(feature_filepath))
					start_time = time.time()
					feature_dict = {'train_inputs': train_input_features, 'train_outputs': train_outputs, 'test_inputs': test_input_features, 'test_outputs': test_outputs}
					with open(feature_filepath, 'wb') as fd:
						pickle.dump(feature_dict, fd)
					logger.info('Features saved: {} secs.'.format(time.time() - start_time))
				except Exception as ex:
					logger.warning('Failed to save features: {}.'.format(ex))
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python evaluate_ssl.py --help
#	python evaluate_ssl.py --config ./config/eval_byol.yaml --model_file ./byol_models/model.ckpt
#	python evaluate_ssl.py --config ./config/eval_relic.yaml --model_file ./relic_models/model.ckpt --out_dir ./relic_eval_outputs
#	python evaluate_ssl.py --config ./config/eval_simclr.yaml --model_file ./simclr_models/model.ckpt --log simclr_log --log_dir ./log

if '__main__' == __name__:
	main()
