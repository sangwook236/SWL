#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import torch
import pytorch_lightning as pl
import yaml
import model_simsiam
import utils

def main():
	args = utils.parse_command_line_options(is_training=True)
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
	logger.info('PyTorch Lightning: version = {}, distributed = {}.'.format(pl.__version__, 'available' if pl.utilities.distributed.distributed_available() else 'unavailable'))
	logger.info('CUDA: version = {}, {}.'.format(torch.version.cuda, 'available' if torch.cuda.is_available() else 'unavailable'))
	logger.info('cuDNN: version = {}.'.format(torch.backends.cudnn.version()))

	#--------------------
	#config['ssl_type'] = config.get('ssl_type', 'simsiam')
	assert config['ssl_type'] == 'simsiam', 'Only SimSiam model is supported, but got {}'.format(config['ssl_type'])
	assert config['data']['dataset'] in ['cifar10', 'imagenet', 'mnist'], 'Invalid dataset, {}'.format(config['data']['dataset'])

	model_filepath_to_load = os.path.normpath(args.model_file) if args.model_file else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)
	#if model_filepath_to_load: logger.info('Model filepath to load: {}.'.format(model_filepath_to_load))

	#if pl.utilities.distributed.rank_zero_only.rank == 0:
	if True:
		output_dir_path = os.path.normpath(config['out_dir']) if config['out_dir'] else None
		#if model_filepath_to_load and not output_dir_path:
		#	output_dir_path = os.path.dirname(model_filepath_to_load)
		if not output_dir_path:
			timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_train_outputs_{}'.format(config['ssl_type'], timestamp))
		if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		logger.info('Output directory path: {}.'.format(output_dir_path))
	else:
		output_dir_path = None

	#is_resumed = args.model_file is not None

	#--------------------
	try:
		config_data = config['data']
		config_model = config['model']
		config_training = config['training']

		image_shape = config_data['image_shape']
		#ssl_augmenter = utils.create_simclr_augmenter(*image_shape[:2], *config_data['ssl_transforms']['normalize'])
		ssl_augmenter = utils.construct_transform(config_data['ssl_transforms'])

		# Prepare data.
		train_dataloader, test_dataloader, _ = utils.prepare_open_data(config_data, show_info=True, show_data=False, logger=logger)

		# Build a model.
		encoder, feature_dim = utils.construct_encoder(**config_model['encoder'])
		projector = utils.SimSiamMLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
		predictor = utils.MLP(config_model['projector_output_dim'], config_model['predictor_output_dim'], config_model['predictor_hidden_dim'])

		logger.info('Building a SimSiam model...')
		start_time = time.time()
		ssl_model = model_simsiam.SimSiamModule(config_training, encoder, projector, predictor, ssl_augmenter, ssl_augmenter, logger)
		logger.info('A SimSiam model built: {} secs.'.format(time.time() - start_time))

		# Train the model.
		best_model_filepath = utils.train(config_training, ssl_model, train_dataloader, test_dataloader, output_dir_path, model_filepath_to_load, logger)

		if True:
			# Load a model.
			logger.info('Loading a SimSiam model from {}...'.format(best_model_filepath))
			start_time = time.time()
			#ssl_model_loaded = model_simsiam.SimSiamModule.load_from_checkpoint(best_model_filepath)
			#ssl_model_loaded = model_simsiam.SimSiamModule.load_from_checkpoint(best_model_filepath, map_location={'cuda:1': 'cuda:0'})
			ssl_model_loaded = model_simsiam.SimSiamModule.load_from_checkpoint(best_model_filepath, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)
			logger.info('A SimSiam model loaded: {} secs.'.format(time.time() - start_time))
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python train_simsiam.py --config ./config/train_simsiam.yaml
#	python train_simsiam.py --config ./config/train_simsiam.yaml --model_file ./simsiam_models/model.ckpt

if '__main__' == __name__:
	main()
