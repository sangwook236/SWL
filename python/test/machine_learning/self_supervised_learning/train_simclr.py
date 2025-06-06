#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import torch
import pytorch_lightning as pl
import yaml
import model_simclr
import utils

def main():
	args = utils.parse_command_line_options(is_training=True)

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
	assert ('training' == config['stage']) if isinstance(config['stage'], str) else ('training' in config['stage']), 'Invalid stage(s), {}'.format(config['stage'])
	#config['ssl_type'] = config.get('ssl_type', 'simclr')
	assert config['ssl_type'] == 'simclr', 'Only SimCLR model is supported, but got {}'.format(config['ssl_type'])
	assert config['data']['dataset'] in ['cifar10', 'imagenet', 'mnist'], 'Invalid dataset, {}'.format(config['data']['dataset'])

	model_filepath_to_load = os.path.normpath(args.model_file) if args.model_file else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)

	#if pl.utilities.distributed.rank_zero_only.rank == 0:
	if True:
		output_dir_path = os.path.normpath(args.out_dir) if args.out_dir else None
		#if model_filepath_to_load and not output_dir_path:
		#	output_dir_path = os.path.dirname(model_filepath_to_load)
		if not output_dir_path:
			timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_train_outputs_{}'.format(config['ssl_type'], timestamp))
		if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)
	else:
		output_dir_path = None

	#--------------------
	logger = utils.get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else output_dir_path, is_rotating=True)
	logger.info('-' * 70)
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Configuration: {}.'.format(config))
	logger.info('Python: version = {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('Torch: version = {}, distributed = {} & {}.'.format(torch.__version__, 'available' if torch.distributed.is_available() else 'unavailable', 'initialized' if torch.distributed.is_initialized() else 'uninitialized'))
	logger.info('PyTorch Lightning: version = {}, distributed = {}.'.format(pl.__version__, 'available' if pl.utilities.distributed.distributed_available() else 'unavailable'))
	logger.info('CUDA: version = {}, {}.'.format(torch.version.cuda, 'available' if torch.cuda.is_available() else 'unavailable'))
	logger.info('cuDNN: version = {}.'.format(torch.backends.cudnn.version()))

	logger.info('Output directory path: {}.'.format(output_dir_path))
	#if model_filepath_to_load: logger.info('Model filepath to load: {}.'.format(model_filepath_to_load))

	#--------------------
	#is_resumed = args.model_file is not None

	try:
		torch.backends.cudnn.backend = True  # Causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
		#torch.backends.cudnn.deterministic = False  # Causes cuDNN to only use deterministic convolution algorithms.

		config_data = config['data']
		config_model = config['model']
		config_training = config['training']

		image_shape = config_data['image_shape']
		#ssl_augmenter = utils.create_simclr_augmenter(*image_shape[:2], *config_data['ssl_transforms']['normalize'])
		ssl_augmenter = utils.construct_transform(config_data['ssl_transforms'])

		# Prepare data.
		train_dataloader, test_dataloader, _ = utils.prepare_open_data(config_data, show_info=True, show_data=False, logger=logger)

		# Build a model.
		if 'pretrained_model' in config_model:
			logger.info('Pretraind model specified.')
			encoder, encoder_feature_dim = utils.construct_pretrained_model(config_model['pretrained_model'], output_dim=None)
			is_model_initialized = False
		else:
			raise ValueError('No encoder specified')
		if True:
			projector = utils.MLP(encoder_feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
		else:
			projector = utils.SimSiamMLP(encoder_feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])

		logger.info('Building a SimCLR model...')
		start_time = time.time()
		ssl_model = model_simclr.SimclrModule(config_training, encoder, projector, ssl_augmenter, ssl_augmenter, is_model_initialized, logger)
		logger.info('A SimCLR model built: {} secs.'.format(time.time() - start_time))

		# Train the model.
		best_model_filepath = utils.train(config_training, ssl_model, train_dataloader, test_dataloader, output_dir_path, model_filepath_to_load, logger)

		if True:
			# Load a model.
			logger.info('Loading a SimCLR model from {}...'.format(best_model_filepath))
			start_time = time.time()
			#ssl_model_loaded = model_simclr.SimclrModule.load_from_checkpoint(best_model_filepath)
			#ssl_model_loaded = model_simclr.SimclrModule.load_from_checkpoint(best_model_filepath, map_location={'cuda:1': 'cuda:0'})
			ssl_model_loaded = model_simclr.SimclrModule.load_from_checkpoint(best_model_filepath, encoder=None, projector=None, augmenter1=None, augmenter2=None)
			logger.info('A SimCLR model loaded: {} secs.'.format(time.time() - start_time))
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python train_simclr.py --help
#	python train_simclr.py --config ./config/train_simclr.yaml
#	python train_simclr.py --config ./config/train_simclr.yaml --model_file ./simclr_models/model.ckpt
#	python train_simclr.py --config ./config/train_simclr.yaml --out_dir ./simclr_train_outputs
#	python train_simclr.py --config ./config/train_simclr.yaml --log simclr_log --log_dir ./log

if '__main__' == __name__:
	main()
