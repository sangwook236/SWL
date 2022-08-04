#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import numpy as np
import torch
import pytorch_lightning as pl
import yaml
import utils

class ClassificationModule(pl.LightningModule):
	def __init__(self, config, input_dim, num_classes, is_model_initialized=False, is_all_model_params_optimized=True, logger=None):
		super().__init__()
		self.save_hyperparameters()

		self.config = config
		self.model = torch.nn.Linear(input_dim, num_classes)
		'''
		self.model = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.BatchNorm1d(hidden_dim),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(hidden_dim, num_classes),
		)
		'''
		self.is_all_model_params_optimized = is_all_model_params_optimized
		self._logger = logger

		self.criterion = torch.nn.NLLLoss()

		if is_model_initialized:
			# Initialize model weights.
			for name, param in self.model.named_parameters():
				try:
					if 'bias' in name:
						torch.nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						torch.nn.init.kaiming_normal_(param)
					#if param.dim() > 1:
					#	torch.nn.init.xavier_uniform_(param)  # Initialize parameters with Glorot / fan_avg.
				except Exception as ex:  # For batch normalization.
					if 'weight' in name:
						param.data.fill_(1)
					continue

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

def prepare_feature_data(config, ssl_model, train_dataloader, test_dataloader, logger=None, device='cuda'):
	class FeatureDataset(torch.utils.data.Dataset):
		def __init__(self, srcs, tgts):
			super().__init__()
			assert len(srcs) == len(tgts), 'Invalid data length: {} != {}'.format(len(srcs), len(tgts))

			self.srcs, self.tgts = srcs, tgts

		def __len__(self):
			return len(self.srcs)

		def __getitem__(self, idx):
			return self.srcs[idx], self.tgts[idx]

	def create_dataset(model, dataloader, device):
		srcs, tgts = list(), list()
		for batch in dataloader:
			batch_inputs, batch_outputs = batch
			batch_inputs = batch_inputs.to(device)
			srcs.append(model(batch_inputs).cpu())  # [batch size, feature dim].
			tgts.append(batch_outputs)  # [batch size].
			del batch_inputs  # Free memory of CPU or GPU.
		return FeatureDataset(torch.vstack(srcs), torch.hstack(tgts))

	# Create feature datasets.
	if logger: logger.info('Creating feature datasets...')
	start_time = time.time()
	ssl_model = ssl_model.to(device)
	ssl_model.eval()
	ssl_model.freeze()
	with torch.no_grad():
		train_feature_dataset = create_dataset(ssl_model, train_dataloader, device)
		test_feature_dataset = create_dataset(ssl_model, test_dataloader, device)
	if logger: logger.info('Feature datasets created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train examples = {}, #test examples = {}.'.format(len(train_feature_dataset), len(test_feature_dataset)))

	# Create feature data loaders.
	if logger: logger.info('Creating feature data loaders...')
	start_time = time.time()
	train_feature_dataloader = torch.utils.data.DataLoader(train_feature_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=train_dataloader.num_workers, persistent_workers=False)
	test_feature_dataloader = torch.utils.data.DataLoader(test_feature_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=test_dataloader.num_workers, persistent_workers=False)
	if logger: logger.info('Feature data loaders created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_feature_dataloader), len(test_feature_dataloader)))

	return train_feature_dataloader, test_feature_dataloader

def run_linear_evaluation(config, train_feature_dataloader, test_feature_dataloader, input_dim, num_classes, output_dir_path, logger=None):
	is_model_initialized = True
	is_all_model_params_optimized = True

	classifier = ClassificationModule(config, input_dim, num_classes, is_model_initialized, is_all_model_params_optimized, logger)

	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=(output_dir_path + '/checkpoints') if output_dir_path else './checkpoints',
		filename='classifier-{epoch:03d}-{step:05d}-{val_acc:.5f}-{val_loss:.5f}',
		monitor='val_loss',
		mode='min',
		save_top_k=-1,
	)
	pl_callbacks = [checkpoint_callback]
	tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=(output_dir_path + '/lightning_logs') if output_dir_path else './lightning_logs', name='', version=None)
	pl_logger = [tensorboard_logger]

	gradient_clip_val = None
	gradient_clip_algorithm = None
	#trainer = pl.Trainer(devices=-1, accelerator='gpu', strategy='ddp', auto_select_gpus=True, max_epochs=config['epochs'], callbacks=pl_callbacks, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=output_dir_path)  # When using the default logger.
	trainer = pl.Trainer(devices=-1, accelerator='gpu', strategy='ddp', auto_select_gpus=True, max_epochs=config['epochs'], callbacks=pl_callbacks, logger=pl_logger, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=None)

	# Train the classifier.
	if logger: logger.info('Training the classifier...')
	start_time = time.time()
	trainer.fit(classifier, train_dataloaders=train_feature_dataloader, val_dataloaders=test_feature_dataloader, ckpt_path=None)
	if logger: logger.info('The classifier trained: {} secs.'.format(time.time() - start_time))

# REF [function] >> load_ssl() in ./train_ssl.py
def load_ssl(ssl_type, model_filepath):
	if ssl_type == 'simclr':
		import model_simclr
		SslModule = getattr(model_simclr, 'SimclrModule')
		ssl_model = SslModule.load_from_checkpoint(model_filepath, encoder=None, projector=None, augmenter1=None, augmenter2=None)
	elif ssl_type == 'byol':
		import model_byol
		SslModule = getattr(model_byol, 'ByolModule')
		ssl_model = SslModule.load_from_checkpoint(model_filepath, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)
	elif ssl_type == 'relic':
		import model_relic
		SslModule = getattr(model_relic, 'RelicModule')
		ssl_model = SslModule.load_from_checkpoint(model_filepath, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)
	elif ssl_type == 'simsiam':
		import model_simsiam
		SslModule = getattr(model_simsiam, 'SimSiamModule')
		ssl_model = SslModule.load_from_checkpoint(model_filepath, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)

	#ssl_model = SslModule.load_from_checkpoint(model_filepath)
	#ssl_model = SslModule.load_from_checkpoint(model_filepath, map_location={'cuda:1': 'cuda:0'})

	return ssl_model

#--------------------------------------------------------------------

def main():
	args = utils.parse_command_line_options(is_training=False)
	assert os.path.isfile(args.config), 'Config file not found, {}'.format(args.config)

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

	config['out_dir'] = config.get('out_dir', None)
	config['log_name'] = config.get('log_name', os.path.basename(os.path.normpath(__file__)))
	config['log_level'] = config.get('log_level', logging.INFO)
	config['log_dir'] = config.get('log_dir', config['out_dir'])

	#--------------------
	logger = utils.get_logger(config['log_name'], config['log_level'], config['log_dir'], is_rotating=True)
	logger.info('----------------------------------------------------------------------')
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Python: version = {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('Torch: version = {}, distributed = {} & {}.'.format(torch.__version__, 'available' if torch.distributed.is_available() else 'unavailable', 'initialized' if torch.distributed.is_initialized() else 'uninitialized'))
	#logger.info('PyTorch Lightning: version = {}, distributed = {}.'.format(pl.__version__, 'available' if pl.utilities.distributed.distributed_available() else 'unavailable'))
	logger.info('CUDA: version = {}, {}.'.format(torch.version.cuda, 'available' if torch.cuda.is_available() else 'unavailable'))
	logger.info('cuDNN: version = {}.'.format(torch.backends.cudnn.version()))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('Device: {}.'.format(device))

	#--------------------
	#config['ssl_type'] = config.get('ssl_type', 'simclr')
	assert config['ssl_type'] in ['byol', 'relic', 'simclr', 'simsiam'], 'Invalid SSL model, {}'.format(config['ssl_type'])
	assert config['data']['dataset'] in ['cifar10', 'imagenet', 'mnist'], 'Invalid dataset, {}'.format(config['data']['dataset'])

	model_filepath_to_load = os.path.normpath(args.model_file) if args.model_file else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)

	output_dir_path = os.path.normpath(config['out_dir']) if config['out_dir'] else None
	#if model_filepath_to_load and not output_dir_path:
	#	output_dir_path = os.path.dirname(model_filepath_to_load)
	if not output_dir_path:
		timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_eval_outputs_{}'.format(config['ssl_type'], timestamp))
	if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
		os.makedirs(output_dir_path, exist_ok=True)

	logger.info('Output directory path: {}.'.format(output_dir_path))

	#--------------------
	try:
		config_data = config['data']
		config_model = config['model']
		config_linear_eval = config['linear_evaluation']

		# Prepare data.
		train_dataloader, test_dataloader, num_classes = utils.prepare_open_data(config_data, show_info=True, show_data=False, logger=logger)

		# Load a SSL model.
		logger.info('Loading a SSL model from {}...'.format(model_filepath_to_load))
		start_time = time.time()
		ssl_model = load_ssl(config['ssl_type'], model_filepath_to_load)
		logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))

		# Prepare feature datasets.
		train_feature_dataloader, test_feature_dataloader = prepare_feature_data(config_linear_eval, ssl_model, train_dataloader, test_dataloader, logger, device)
		del ssl_model  # Free memory of CPU or GPU.

		# Run a linear evaluation.
		_, feature_dim = utils.construct_encoder(**config_model['encoder'])
		run_linear_evaluation(config_linear_eval, train_feature_dataloader, test_feature_dataloader, feature_dim, num_classes, output_dir_path, logger)
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python evaluate_ssl.py --config ./config/linear_eval_byol.yaml --model_file ./byol_models/model.ckpt
#	python evaluate_ssl.py --config ./config/linear_eval_relic.yaml --model_file ./relic_models/model.ckpt
#	python evaluate_ssl.py --config ./config/linear_eval_simclr.yaml --model_file ./simclr_models/model.ckpt
#	python evaluate_ssl.py --config ./config/linear_eval_simsiam.yaml --model_file ./simsiam_models/model.ckpt

if '__main__' == __name__:
	main()
