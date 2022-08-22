#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import numpy as np
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

	#--------------------
	logger = utils.get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else output_dir_path, is_rotating=True)
	logger.info('----------------------------------------------------------------------')
	logger.info('Logger: name = {}, level = {}.'.format(logger.name, logger.level))
	logger.info('Command-line arguments: {}.'.format(sys.argv))
	logger.info('Command-line options: {}.'.format(vars(args)))
	logger.info('Configuration: {}.'.format(config))
	logger.info('Python: version = {}.'.format(sys.version.replace('\n', ' ')))
	logger.info('Torch: version = {}, distributed = {} & {}.'.format(torch.__version__, 'available' if torch.distributed.is_available() else 'unavailable', 'initialized' if torch.distributed.is_initialized() else 'uninitialized'))
	#logger.info('PyTorch Lightning: version = {}, distributed = {}.'.format(pl.__version__, 'available' if pl.utilities.distributed.distributed_available() else 'unavailable'))
	logger.info('CUDA: version = {}, {}.'.format(torch.version.cuda, 'available' if torch.cuda.is_available() else 'unavailable'))
	logger.info('cuDNN: version = {}.'.format(torch.backends.cudnn.version()))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('Device: {}.'.format(device))

	logger.info('Output directory path: {}.'.format(output_dir_path))

	#--------------------
	try:
		config_data = config['data']

		# Load a SSL model.
		logger.info('Loading a SSL model from {}...'.format(model_filepath_to_load))
		start_time = time.time()
		ssl_model = utils.load_ssl(config['ssl_type'], model_filepath_to_load)
		logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))

		if True:
			# Infer by the trained SSL model.

			use_projector, use_predictor = config['model'].get('use_projector', False), config['model'].get('use_predictor', False)

			# Prepare data.
			def create_data_generator(dataset, batch_size, num_workers, shuffle=False):
				dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=False)
				logger.info('#data batches = {}.'.format(len(dataloader)))
				for batch in dataloader:
					yield batch[0]

			dataset = prepare_open_data(config_data, logger=None)

			if True:
				# Infer by the model.
				logger.info("Inferring...")
				start_time = time.time()
				predictions = utils.infer(ssl_model, create_data_generator(dataset, config_data['batch_size'], config_data['num_workers']), use_projector, use_predictor, device)
				logger.info("Inferred: {} secs.".format(time.time() - start_time))
				logger.info("Prediction: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(predictions.shape, predictions.dtype, np.min(predictions), np.max(predictions)))
			else:
				# Test the model.
				dataloader = torch.utils.data.DataLoader(dataset, batch_size=config_data['batch_size'], shuffle=False, num_workers=config_data['num_workers'], persistent_workers=False)
				utils.test(model, dataloader, logger)
		else:
			# Export the trained SSL model for production.
			# REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html

			# NOTE [info] >>
			#	<error> RuntimeError: Module 'ResNet' has no attribute '_modules'.
			#	<cause> This error occurs when converting a module with an inner module like utils.ModelWrapper using TorchScript scripting.
			#	<solution> Use TorchScript tracing.
			#		Refer to ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_torch_script.py.
			# NOTE [info] >>
			#	<error> ReferenceError: weakly-referenced object no longer exists.
			#		Refer to https://docs.python.org/3/library/weakref.html.
			#	<cause> This error occurs in child processes when training a model using pl.Trainer(strategy='ddp') in a single machine.
			#	<solution> Use pl.Trainer(strategy='dp').
			#		Refer to https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_intermediate.html.

			image_shape = config_data['image_shape']  # (H, W, C).

			# TorchScript.
			try:
				torchscript_filepath = os.path.join(output_dir_path, '{}_ts.pth'.format(config['ssl_type']))

				logger.info('Exporting a TorchScript model to {}.'.format(torchscript_filepath))
				start_time = time.time()
				if False:
					script = ssl_model.to_torchscript(file_path=torchscript_filepath, method='script')
				elif True:
					dummy_inputs = torch.randn((1, image_shape[2], image_shape[0], image_shape[1]))
					script = ssl_model.to_torchscript(file_path=torchscript_filepath, method='trace', example_inputs=dummy_inputs)
				else:
					script = ssl_model.to_torchscript(file_path=None, method='script')
					torch.jit.save(script, torchscript_filepath)
				logger.info('A TorchScript model exported: {} secs.'.format(time.time() - start_time))
			except Exception as ex:
				logger.error('Failed to export a TorchScript model:')
				logger.exception(ex)

			# ONNX.
			try:
				onnx_filepath = os.path.join(output_dir_path, '{}.onnx'.format(config['ssl_type']))

				logger.info('Exporting an ONNX model to {}.'.format(onnx_filepath))
				start_time = time.time()
				dummy_inputs = torch.randn((1, image_shape[2], image_shape[0], image_shape[1]))
				ssl_model.to_onnx(onnx_filepath, dummy_inputs, export_params=True)
				logger.info('An ONNX model exported: {} secs.'.format(time.time() - start_time))
			except Exception as ex:
				logger.error('Failed to export an ONNX model:')
				logger.exception(ex)
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
