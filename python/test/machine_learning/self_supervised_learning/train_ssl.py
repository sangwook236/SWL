#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, logging, datetime, time
import numpy as np
import torch, torchvision
import pytorch_lightning as pl
import utils

# REF [function] >> train_text_recognizer() in ${SWLP_HOME}/app/text/run_text_recognition_pl.py
def train(model, train_dataloader, test_dataloader, max_gradient_norm, num_epochs, output_dir_path, model_filepath_to_load, swa=False, logger=None):
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=(output_dir_path + '/checkpoints') if output_dir_path else './checkpoints',
		filename='model-{epoch:03d}-{step:05d}-{val_acc:.5f}-{val_loss:.5f}',
		monitor='val_loss',
		mode='min',
		save_top_k=-1,
	)
	if swa:
		swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=2, annealing_strategy='cos', avg_fn=None)
		pl_callbacks = [checkpoint_callback, swa_callback]
	else:
		pl_callbacks = [checkpoint_callback]
	tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=(output_dir_path + '/lightning_logs') if output_dir_path else './lightning_logs', name='', version=None)
	pl_logger = [tensorboard_logger]

	if max_gradient_norm:
		gradient_clip_val = max_gradient_norm
		gradient_clip_algorithm = 'norm'  # {'norm', 'value'}.
	else:
		gradient_clip_val = None
		gradient_clip_algorithm = None
	#trainer = pl.Trainer(devices=-1, accelerator='gpu', strategy='ddp', auto_select_gpus=True, max_epochs=num_epochs, callbacks=pl_callbacks, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=output_dir_path)  # When using the default logger.
	trainer = pl.Trainer(devices=-1, accelerator='gpu', strategy='ddp', auto_select_gpus=True, max_epochs=num_epochs, callbacks=pl_callbacks, logger=pl_logger, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=None)

	# Train a model.
	if logger: logger.info('Training the model...')
	start_time = time.time()
	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=model_filepath_to_load if model_filepath_to_load else None)
	if logger: logger.info('The model trained: {} secs.'.format(time.time() - start_time))

	#--------------------
	if False:
		# Validate the trained model.
		if logger: logger.info('Validating the model...')
		start_time = time.time()
		#model.eval()
		trainer.validate(model, dataloaders=test_dataloader, ckpt_path=None, verbose=True)
		if logger: logger.info('The model validated: {} secs.'.format(time.time() - start_time))

	#--------------------
	#best_model_filepath = trainer.checkpoint_callback.best_model_path
	best_model_filepath = checkpoint_callback.best_model_path
	if logger: logger.info('The best trained model saved to {}.'.format(best_model_filepath))
	if False:
		# Save the final model.
		final_model_filepath = (output_dir_path + '/final_model.ckpt') if output_dir_path else './final_model.ckpt'
		trainer.save_checkpoint(final_model_filepath, weights_only=False)
		if logger: logger.info('The final trained model saved to {}.'.format(final_model_filepath))

	return best_model_filepath

def prepare_data(dataset_type, batch_size, num_workers, logger=None):
	# Create datasets.
	if dataset_type == 'imagenet':
		if 'posix' == os.name:
			imagenet_dir_path = '/home/sangwook/work/dataset/imagenet'
		else:
			imagenet_dir_path = 'D:/work/dataset/imagenet'

		train_dataset, test_dataset = utils.create_imagenet_datasets(imagenet_dir_path, logger)
		class_names = None
	elif dataset_type == 'cifar10':
		train_dataset, test_dataset, class_names = utils.create_cifar10_datasets(logger)
	elif dataset_type == 'mnist':
		train_dataset, test_dataset = utils.create_mnist_datasets(logger)
		class_names = None

	# Create data loaders.
	if logger: logger.info('Creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)
	if logger: logger.info('Data loaders created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	if True:
		# Show data info.
		data_iter = iter(train_dataloader)
		srcs, tgts = data_iter.next()
		srcs, tgts = srcs.numpy(), tgts.numpy()
		if logger: logger.info('Train source (batch): Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(srcs.shape, srcs.dtype, np.min(srcs), np.max(srcs)))
		if logger: logger.info('Train target (batch): Shape = {}, dtype = {}, classes = {}.'.format(tgts.shape, tgts.dtype, np.unique(tgts)))

		data_iter = iter(test_dataloader)
		srcs, tgts = data_iter.next()
		srcs, tgts = srcs.numpy(), tgts.numpy()
		if logger: logger.info('Test source (batch): Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(srcs.shape, srcs.dtype, np.min(srcs), np.max(srcs)))
		if logger: logger.info('Test target (batch): Shape = {}, dtype = {}, classes = {}.'.format(tgts.shape, tgts.dtype, np.unique(tgts)))

	if False:
		# Visualize data.
		print('Visualizing training data...')
		utils.visualize_data(train_dataloader, num_data=10, class_names=class_names)
		print('Visualizing test data...')
		utils.visualize_data(test_dataloader, num_data=10, class_names=class_names)

	return train_dataloader, test_dataloader

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
	if ssl_type == 'byol':
		return build_byol(feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)
	elif ssl_type == 'relic':
		return build_relic(feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter1, augmenter2, is_model_initialized, is_all_model_params_optimized, logger)

def load_ssl(ssl_type, model_filepath):
	if ssl_type == 'byol':
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
	#assert args.ssl_type in ['byol', 'relic'], 'Invalid SSL model, {}'.format(args.ssl_type)
	#assert args.dataset_type in ['imagenet', 'cifar10', 'mnist'], 'Invalid dataset type, {}'.format(args.dataset_type)

	if args.dataset_type == 'imagenet':
		image_shape = [224, 224, 3]
		normalization_mean, normalization_stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # For ImageNet.
	elif args.dataset_type == 'cifar10':
		image_shape = [32, 32, 3]
		#normalization_mean, normalization_stddev = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]  # For CIFAR-10.
		normalization_mean, normalization_stddev = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # For RGB images.
	elif args.dataset_type == 'mnist':
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
			output_dir_path = os.path.join('.', '{}_train_outputs_{}'.format(args.ssl_type, timestamp))
		if output_dir_path and output_dir_path.strip() and not os.path.isdir(output_dir_path):
			os.makedirs(output_dir_path, exist_ok=True)

		logger.info('Output directory path: {}.'.format(output_dir_path))
		#if model_filepath_to_load: logger.info('Model filepath to load: {}.'.format(model_filepath_to_load))
	else:
		output_dir_path = None

	#--------------------
	try:
		# Prepare data.
		train_dataloader, test_dataloader = prepare_data(args.dataset_type, args.batch, num_workers, logger)

		# Build a SSL model.
		logger.info('Building a SSL model...')
		start_time = time.time()
		ssl_model = build_ssl(args.ssl_type, feature_dim, projector_hidden_dim, projector_output_dim, predictor_hidden_dim, predictor_output_dim, moving_average_decay, is_momentum_encoder_used, augmenter, augmenter, is_model_initialized, is_all_model_params_optimized, logger)
		logger.info('A SSL model built: {} secs.'.format(time.time() - start_time))

		# Train the model.
		best_model_filepath = train(ssl_model, train_dataloader, test_dataloader, max_gradient_norm, args.epoch, output_dir_path, model_filepath_to_load, swa, logger)

		if True:
			# Load a SSL model.
			logger.info('Loading a SSL model from {}...'.format(best_model_filepath))
			start_time = time.time()
			ssl_model_loaded = load_ssl(args.ssl_type, best_model_filepath)
			logger.info('A SSL model loaded: {} secs.'.format(time.time() - start_time))
	except Exception as ex:
		#logging.exception(ex)  # Logs a message with level 'ERROR' on the root logger.
		logger.exception(ex)
		raise

#--------------------------------------------------------------------

# Usage:
#	python train_ssl.py --epoch 20 --batch 64 --out_dir ./ssl_train_outputs
#	python train_ssl.py --epoch 10 --batch 32 --model_file ./ssl_model/model.ckpt --out_dir ./ssl_train_outputs
#	python train_ssl.py --epoch 40 --batch 64 --out_dir ./ssl_train_outputs --log ssl_log --log_dir ./log

if '__main__' == __name__:
	main()
