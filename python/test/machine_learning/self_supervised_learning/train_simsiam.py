#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')
sys.path.append('./src')

import os, logging, datetime, time
import torch, torchvision
import pytorch_lightning as pl
import model_simsiam
import utils

def main():
	args = utils.parse_train_command_line_options(use_ssl_type=False)

	logger = utils.get_logger(args.log if args.log else os.path.basename(os.path.normpath(__file__)), args.log_level if args.log_level else logging.INFO, args.log_dir if args.log_dir else args.out_dir, is_rotating=True)
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
	#assert args.ssl == 'simsiam', 'Only SimSiam model is supported, but got {}'.format(args.ssl)
	#assert args.dataset in ['imagenet', 'cifar10', 'mnist'], 'Invalid dataset, {}'.format(args.dataset)

	ssl_type = 'simsiam'
	if args.dataset == 'imagenet':
		image_shape = [224, 224, 3]
		normalization_mean, normalization_stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # For ImageNet.
	elif args.dataset == 'cifar10':
		image_shape = [32, 32, 3]
		#normalization_mean, normalization_stddev = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]  # For CIFAR-10.
		normalization_mean, normalization_stddev = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # For RGB images.
	elif args.dataset == 'mnist':
		image_shape = [28, 28, 1]
		#normalization_mean, normalization_stddev = [0.1307], [0.3081]  # For MNIST.
		normalization_mean, normalization_stddev = [0.5], [0.5]  # For grayscale images.
	num_workers = 8
	ssl_augmenter = utils.create_simclr_augmenter(*image_shape[:2], normalization_mean, normalization_stddev)

	feature_dim = 2048  # For ResNet50 or higher.
	projector_output_dim, projector_hidden_dim = 256, 4096  # projector_input_dim = feature_dim.
	predictor_output_dim, predictor_hidden_dim = 256, 4096  # predictor_input_dim = projector_output_dim.
	moving_average_decay = 0.99
	is_momentum_encoder_used = True
	is_model_initialized = False
	is_all_model_params_optimized = True

	#max_gradient_norm = 20.0  # Gradient clipping value.
	max_gradient_norm = None
	swa = False

	#is_resumed = args.model_file is not None

	encoder = utils.ModelWrapper(torchvision.models.resnet50(pretrained=True), layer_name='avgpool')
	if is_momentum_encoder_used:
		projector = utils.MLP(feature_dim, projector_output_dim, projector_hidden_dim)
	else:
		projector = utils.SimSiamMLP(feature_dim, projector_output_dim, projector_hidden_dim)
	predictor = utils.MLP(projector_output_dim, predictor_output_dim, predictor_hidden_dim)

	#--------------------
	model_filepath_to_load, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)
	#if pl.utilities.distributed.rank_zero_only.rank == 0:
	if True:
		#if model_filepath_to_load and not output_dir_path:
		#	output_dir_path = os.path.dirname(model_filepath_to_load)
		if not output_dir_path:
			timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_train_outputs_{}'.format(ssl_type, timestamp))
		if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		logger.info('Output directory path: {}.'.format(output_dir_path))
		#if model_filepath_to_load: logger.info('Model filepath to load: {}.'.format(model_filepath_to_load))
	else:
		output_dir_path = None

	#--------------------
	try:
		# Prepare data.
		if args.dataset == 'imagenet':
			if 'posix' == os.name:
				dataset_root_dir_path = '/home/sangwook/work/dataset/imagenet'
			else:
				dataset_root_dir_path = 'D:/work/dataset/imagenet'
		else:
			dataset_root_dir_path = None
		train_dataloader, test_dataloader, _ = utils.prepare_open_data(args.dataset, args.batch, num_workers, dataset_root_dir_path, show_info=True, show_data=False, logger=logger)

		# Build a model.
		logger.info('Building a SimSiam model...')
		start_time = time.time()
		ssl_model = model_simsiam.SimSiamModule(encoder, projector, predictor, moving_average_decay, is_momentum_encoder_used, ssl_augmenter, ssl_augmenter, is_model_initialized, is_all_model_params_optimized, logger)
		logger.info('A SimSiam model built: {} secs.'.format(time.time() - start_time))

		# Train the model.
		best_model_filepath = utils.train(ssl_model, train_dataloader, test_dataloader, max_gradient_norm, args.epoch, output_dir_path, model_filepath_to_load, swa, logger)

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
#	python train_simsiam.py --dataset imagenet --epoch 40 --batch 64 --out_dir ./simsiam_train_outputs
#	python train_simsiam.py --dataset cifar10 --epoch 20 --batch 32 --out_dir ./simsiam_train_outputs --log simsiam_log --log_dir ./log
#	python train_simsiam.py --dataset mnist --epoch 10 --batch 48 --model_file ./simsiam_models/model.ckpt --out_dir ./simsiam_train_outputs

if '__main__' == __name__:
	main()
