#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import torch
import yaml
import utils

def prepare_open_data(config, logger=None):
	import torchvision

	transform = utils.construct_transform(config['transforms'])

	if logger: logger.info('Creating a dataset...')
	start_time = time.time()
	if config['dataset'] == 'imagenet':
		dataset = torchvision.datasets.ImageNet(root=config['data_dir'], split='val', transform=transform, target_transform=None)
	elif config['dataset'] == 'cifar10':
		dataset = torchvision.datasets.CIFAR10(root=config['data_dir'], train=False, download=True, transform=transform, target_transform=None)
	elif config['dataset'] == 'mnist':
		dataset = torchvision.datasets.MNIST(root=config['data_dir'], train=False, download=True, transform=transform, target_transform=None)
	if logger: logger.info('A dataset created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#examples = {}.'.format(len(dataset)))

	return dataset

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
	logger = utils.get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else args.out_dir, is_rotating=True)
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
	assert ('inference' == config['stage']) if isinstance(config['stage'], str) else ('inference' in config['stage']), 'Invalid stage(s), {}'.format(config['stage'])
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
		output_dir_path = os.path.join('.', '{}_results_{}'.format(config['ssl_type'], timestamp))
	if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
		os.makedirs(output_dir_path, exist_ok=True)

	logger.info('Output directory path: {}.'.format(output_dir_path))

	#--------------------
	try:
		config_data = config['data']
		config_model = config['model']

		# Prepare data.
		def create_data_generator(dataset, batch_size, num_workers, shuffle=False):
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=False)
			logger.info('#data batches = {}.'.format(len(dataloader)))
			for batch in dataloader:
				yield batch[0]

		dataset = prepare_open_data(config_data, logger=None)

		# Load a SSL model.
		logger.info('Loading a SSL model from {}...'.format(model_filepath_to_load))
		start_time = time.time()
		ssl_model = load_ssl(config['ssl_type'], model_filepath_to_load)
		logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))

		# Infer by the model.
		predictions = utils.infer(config_model, ssl_model, create_data_generator(dataset, config_data['batch_size'], config_data['num_workers']), logger, device)
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python infer_ssl.py --help
#	python infer_ssl.py --config ./config/infer_byol.yaml --model_file ./byol_models/model.ckpt
#	python infer_ssl.py --config ./config/infer_relic.yaml --model_file ./relic_models/model.ckpt --out_dir ./relic_results
#	python infer_ssl.py --config ./config/infer_simclr.yaml --model_file ./simclr_models/model.ckpt --log simclr_log --log_dir ./log

if '__main__' == __name__:
	main()
