#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import torch
import utils

def prepare_open_data(dataset_type, dataset_root_dir_path=None, logger=None):
	if dataset_type == 'imagenet':
		_, test_dataset = utils.create_imagenet_datasets(dataset_root_dir_path, logger)
	elif dataset_type == 'cifar10':
		_, test_dataset, _ = utils.create_cifar10_datasets(logger)
	elif dataset_type == 'mnist':
		_, test_dataset = utils.create_mnist_datasets(logger)
	return test_dataset

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
	args = utils.parse_command_line_options(use_ssl_type=True, use_dataset_type=True)

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
	#assert args.ssl in ['simclr', 'byol', 'relic', 'simsiam'], 'Invalid SSL model, {}'.format(args.ssl)
	#assert args.dataset in ['imagenet', 'cifar10', 'mnist'], 'Invalid dataset, {}'.format(args.dataset)

	use_projector, use_predictor = False, False
	num_workers = 8

	#--------------------
	model_filepath_to_load, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)
	#if model_filepath_to_load and not output_dir_path:
	#	output_dir_path = os.path.dirname(model_filepath_to_load)
	if not output_dir_path:
		timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		output_dir_path = os.path.join('.', '{}_results_{}'.format(args.ssl, timestamp))
	if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
		os.makedirs(output_dir_path, exist_ok=True)

	logger.info('Output directory path: {}.'.format(output_dir_path))

	#--------------------
	try:
		# Prepare data.
		def create_data_generator(dataset):
			dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=num_workers, persistent_workers=False)
			logger.info('#data batches = {}.'.format(len(dataloader)))
			for batch in dataloader:
				yield batch[0]

		if args.dataset == 'imagenet':
			if 'posix' == os.name:
				dataset_root_dir_path = '/home/sangwook/work/dataset/imagenet'
			else:
				dataset_root_dir_path = 'D:/work/dataset/imagenet'
		else:
			dataset_root_dir_path = None
		dataset = prepare_open_data(args.dataset, dataset_root_dir_path, logger=None)

		# Load a SSL model.
		logger.info('Loading a SSL model from {}...'.format(model_filepath_to_load))
		start_time = time.time()
		ssl_model = load_ssl(args.ssl, model_filepath_to_load)
		logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))

		# Infer by the model.
		predictions = utils.infer(ssl_model, create_data_generator(dataset), use_projector, use_predictor, logger, device)
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python infer_ssl.py --ssl simclr --model_file ./ssl_models/model.ckpt --dataset imagenet --out_dir ./ssl_results
#	python infer_ssl.py --ssl byol --model_file ./ssl_models/model.ckpt --dataset cifar10 --batch 64 --out_dir ./ssl_results
#	python infer_ssl.py --ssl relic --model_file ./ssl_models/model.ckpt --dataset mnist --batch 32 --out_dir ./ssl_results --log ssl_log --log_dir ./log

if '__main__' == __name__:
	main()
