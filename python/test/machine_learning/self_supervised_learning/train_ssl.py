#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, logging, datetime, time
import torch, torchvision
import pytorch_lightning as pl
import utils

def build_simclr(feature_dim, projector_hidden_dim, projector_output_dim, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger=None):
	import model_simclr

	encoder = utils.ModelWrapper(torchvision.models.resnet50(pretrained=True), layer_name='avgpool')
	#feature_dim = encoder(torch.randn(1, *input_shape).permute(0, 3, 1, 2)).shape[-1]  # FIXME [check] >>
	if True:
		projector = utils.MLP(feature_dim, projector_output_dim, projector_hidden_dim)
	else:
		projector = utils.SimSiamMLP(feature_dim, projector_output_dim, projector_hidden_dim)
	ssl_model = model_simclr.SimclrModule(encoder, projector, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)

	return ssl_model

def build_byol(feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger=None):
	import model_byol

	encoder = utils.ModelWrapper(torchvision.models.resnet50(pretrained=True), layer_name='avgpool')
	#feature_dim = encoder(torch.randn(1, *input_shape).permute(0, 3, 1, 2)).shape[-1]  # FIXME [check] >>
	if is_momentum_encoder_used:
		projector = utils.MLP(feature_dim, projector_output_dim, projector_hidden_dim)
	else:
		projector = utils.SimSiamMLP(feature_dim, projector_output_dim, projector_hidden_dim)
	predictor = utils.MLP(projector_output_dim, predictor_output_dim, predictor_hidden_dim)
	ssl_model = model_byol.ByolModule(encoder, projector, predictor, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)

	return ssl_model

def build_relic(feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger=None):
	import model_relic

	encoder = utils.ModelWrapper(torchvision.models.resnet50(pretrained=True), layer_name='avgpool')
	#feature_dim = encoder(torch.randn(1, *input_shape).permute(0, 3, 1, 2)).shape[-1]  # FIXME [check] >>
	if is_momentum_encoder_used:
		projector = utils.MLP(feature_dim, projector_output_dim, projector_hidden_dim)
	else:
		projector = utils.SimSiamMLP(feature_dim, projector_output_dim, projector_hidden_dim)
	predictor = utils.MLP(projector_output_dim, predictor_output_dim, predictor_hidden_dim)
	ssl_model = model_relic.RelicModule(encoder, projector, predictor, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)

	return ssl_model

def build_ssl(ssl_type, feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger=None):
	if ssl_type == 'simclr':
		return build_simclr(feature_dim, projector_hidden_dim, projector_output_dim, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)
	elif ssl_type == 'byol':
		return build_byol(feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)
	elif ssl_type == 'relic':
		return build_relic(feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)

def load_ssl(ssl_type, model_filepath):
	if ssl_type == 'simclr':
		import model_simclr
		SslModule = getattr(model_simclr, 'SimclrModule')
	elif ssl_type == 'byol':
		import model_byol
		SslModule = getattr(model_byol, 'ByolModule')
	elif ssl_type == 'relic':
		import model_relic
		SslModule = getattr(model_relic, 'RelicModule')

	ssl_model = SslModule.load_from_checkpoint(model_filepath)
	#ssl_model = SslModule.load_from_checkpoint(model_filepath, map_location={'cuda:1': 'cuda:0'})

	return ssl_model

#--------------------------------------------------------------------

def main():
	args = utils.parse_train_command_line_options()

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
	#assert args.ssl in ['simclr', 'byol', 'relic'], 'Invalid SSL model, {}'.format(args.ssl)
	#assert args.dataset in ['imagenet', 'cifar10', 'mnist'], 'Invalid dataset type, {}'.format(args.dataset)

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
	augmenter = utils.create_simclr_augmenter(*image_shape[:2], normalization_mean, normalization_stddev)

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
	num_workers = 8

	#is_resumed = args.model_file is not None

	#--------------------
	model_filepath_to_load, output_dir_path = os.path.normpath(args.model_file) if args.model_file else None, os.path.normpath(args.out_dir) if args.out_dir else None
	assert model_filepath_to_load is None or os.path.isfile(model_filepath_to_load), 'Model file not found, {}'.format(model_filepath_to_load)
	#if pl.utilities.distributed.rank_zero_only.rank == 0:
	if True:
		#if model_filepath_to_load and not output_dir_path:
		#	output_dir_path = os.path.dirname(model_filepath_to_load)
		if not output_dir_path:
			timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
			output_dir_path = os.path.join('.', '{}_train_outputs_{}'.format(args.ssl, timestamp))
		if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		logger.info('Output directory path: {}.'.format(output_dir_path))
		#if model_filepath_to_load: logger.info('Model filepath to load: {}.'.format(model_filepath_to_load))
	else:
		output_dir_path = None

	#--------------------
	try:
		# Prepare data.
		train_dataloader, test_dataloader = utils.prepare_open_data(args.dataset, args.batch, num_workers, show_info=True, show_data=False, logger=logger)

		# Build a SSL model.
		logger.info('Building a SSL model...')
		start_time = time.time()
		ssl_model = build_ssl(args.ssl, feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter, augmenter, is_model_initialized, is_all_model_params_optimized, logger)
		logger.info('A SSL model built: {} secs.'.format(time.time() - start_time))

		# Train the model.
		best_model_filepath = utils.train(ssl_model, train_dataloader, test_dataloader, max_gradient_norm, args.epoch, output_dir_path, model_filepath_to_load, swa, logger)

		if True:
			# Load a SSL model.
			logger.info('Loading a SSL model from {}...'.format(best_model_filepath))
			start_time = time.time()
			ssl_model_loaded = load_ssl(args.ssl, best_model_filepath)
			logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python train_ssl.py --ssl simclr --dataset imagenet --epoch 40 --batch 64 --out_dir ./ssl_train_outputs
#	python train_ssl.py --ssl byol --dataset cifar10 --epoch 20 --batch 32 --model_file ./ssl_models/model.ckpt --out_dir ./ssl_train_outputs
#	python train_ssl.py --ssl relic --dataset mnist --epoch 10 --batch 64 --out_dir ./ssl_train_outputs --log ssl_log --log_dir ./log

if '__main__' == __name__:
	main()
