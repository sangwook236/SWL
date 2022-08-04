#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import torch
import pytorch_lightning as pl
import yaml
import utils

def build_simclr(config, augmenter1, augmenter2, logger=None):
	import model_simclr

	config_model = config['model']
	config_training = config['training']

	encoder, feature_dim = utils.construct_encoder(**config_model['encoder'])
	if True:
		projector = utils.MLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	else:
		projector = utils.SimSiamMLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	ssl_model = model_simclr.SimclrModule(config_training, encoder, projector, augmenter1, augmenter2, logger)

	return ssl_model

def build_byol(config, augmenter1, augmenter2, logger=None):
	import model_byol

	config_model = config['model']
	config_training = config['training']

	encoder, feature_dim = utils.construct_encoder(**config_model['encoder'])
	if config_training.get('is_momentum_encoder_used', True):
		projector = utils.MLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	else:
		projector = utils.SimSiamMLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	predictor = utils.MLP(config_model['projector_output_dim'], config_model['predictor_output_dim'], config_model['predictor_hidden_dim'])
	ssl_model = model_byol.ByolModule(config_training, encoder, projector, predictor, augmenter1, augmenter2, logger)

	return ssl_model

def build_relic(config, augmenter1, augmenter2, logger=None):
	import model_relic

	config_model = config['model']
	config_training = config['training']

	encoder, feature_dim = utils.construct_encoder(**config_model['encoder'])
	if config_training.get('is_momentum_encoder_used', True):
		projector = utils.MLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	else:
		projector = utils.SimSiamMLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	predictor = utils.MLP(config_model['projector_output_dim'], config_model['predictor_output_dim'], config_model['predictor_hidden_dim'])
	ssl_model = model_relic.RelicModule(config_training, encoder, projector, predictor, augmenter1, augmenter2, logger)

	return ssl_model

def build_simsiam(config, augmenter1, augmenter2, logger=None):
	import model_simsiam

	config_model = config['model']
	config_training = config['training']

	encoder, feature_dim = utils.construct_encoder(**config_model['encoder'])
	projector = utils.SimSiamMLP(feature_dim, config_model['projector_output_dim'], config_model['projector_hidden_dim'])
	predictor = utils.MLP(config_model['projector_output_dim'], config_model['predictor_output_dim'], config_model['predictor_hidden_dim'])
	ssl_model = model_simsiam.SimSiamModule(config_training, encoder, projector, predictor, augmenter1, augmenter2, logger)

	return ssl_model

def build_ssl(ssl_type, config, augmenter1, augmenter2, logger=None):
	return globals().get('build_{}'.format(ssl_type))(config, augmenter1, augmenter2, logger)
	'''
	if ssl_type == 'simclr':
		return build_simclr(config, augmenter1, augmenter2, logger)
	elif ssl_type == 'byol':
		return build_byol(config, augmenter1, augmenter2, logger)
	elif ssl_type == 'relic':
		return build_relic(config, augmenter1, augmenter2, logger)
	elif ssl_type == 'simsiam':
		return build_simsiam(config, augmenter1, augmenter2, logger)
	'''

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
	args = utils.parse_config_command_line_options(is_training=True)
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
	#config['ssl_type'] = config.get('ssl_type', 'simclr')
	assert config['ssl_type'] in ['byol', 'relic', 'simclr', 'simsiam'], 'Invalid SSL model, {}'.format(config['ssl_type'])
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
		config_training = config['training']

		image_shape = config_data['image_shape']
		#ssl_augmenter = utils.create_simclr_augmenter(*image_shape[:2], *config_data['ssl_transforms']['normalize'])
		ssl_augmenter = utils.construct_transform(config_data['ssl_transforms'])

		# Prepare data.
		train_dataloader, test_dataloader, _ = utils.prepare_open_data(config_data, show_info=True, show_data=False, logger=logger)

		# Build a SSL model.
		logger.info('Building a SSL model...')
		start_time = time.time()
		ssl_model = build_ssl(config['ssl_type'], config, ssl_augmenter, ssl_augmenter, logger)
		logger.info('A SSL model built: {} secs.'.format(time.time() - start_time))

		# Train the model.
		best_model_filepath = utils.train(config_training, ssl_model, train_dataloader, test_dataloader, output_dir_path, model_filepath_to_load, logger)

		if True:
			# For production.
			# REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html

			# TorchScript.
			try:
				torchscript_filepath = os.path.join(output_dir_path, '{}_ts.pth'.format(config['ssl_type']))
				if True:
					# FIXME [error] >> ReferenceError: weakly-referenced object no longer exists.
					script = ssl_model.to_torchscript(file_path=torchscript_filepath, method='script')
				elif False:
					dummy_inputs = torch.randn((1, image_shape[2], image_shape[0], image_shape[1]))
					script = ssl_model.to_torchscript(file_path=torchscript_filepath, method='trace', example_inputs=dummy_inputs)
				else:
					script = ssl_model.to_torchscript(file_path=None, method='script')
					torch.jit.save(script, torchscript_filepath)
				logger.info('A TorchScript model saved to {}.'.format(torchscript_filepath))
			except Exception as ex:
				logger.error('Failed to save a TorchScript model:')
				logger.exception(ex)

			# ONNX.
			try:
				# FIXME [error] >> ReferenceError: weakly-referenced object no longer exists.
				onnx_filepath = os.path.join(output_dir_path, '{}.onnx'.format(config['ssl_type']))
				dummy_inputs = torch.randn((1, image_shape[2], image_shape[0], image_shape[1]))
				ssl_model.to_onnx(onnx_filepath, dummy_inputs, export_params=True)
				logger.info('An ONNX model saved to {}.'.format(onnx_filepath))
			except Exception as ex:
				logger.error('Failed to save an ONNX model:')
				logger.exception(ex)

		if True:
			# Load a SSL model.
			logger.info('Loading a SSL model from {}...'.format(best_model_filepath))
			start_time = time.time()
			ssl_model_loaded = load_ssl(config['ssl_type'], best_model_filepath)
			logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python train_ssl.py --config ./config/train_byol.yaml
#	python train_ssl.py --config ./config/train_relic.yaml
#	python train_ssl.py --config ./config/train_simclr.yaml
#	python train_ssl.py --config ./config/train_simsiam.yaml
#	python train_ssl.py --config ./config/train_byol.yaml --model_file ./byol_models/model.ckpt
#	python train_ssl.py --config ./config/train_relic.yaml --model_file ./relic_models/model.ckpt
#	python train_ssl.py --config ./config/train_simclr.yaml --model_file ./simclr_models/model.ckpt
#	python train_ssl.py --config ./config/train_simsiam.yaml --model_file ./simsiam_models/model.ckpt

if '__main__' == __name__:
	main()
