import os, random, argparse, logging, logging.handlers, time
import numpy as np
import torch, torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt

def get_logger(name, log_level=None, log_dir_path=None, is_rotating=True):
	if not log_level: log_level = logging.INFO
	if not log_dir_path: log_dir_path = "./log"
	if not os.path.exists(log_dir_path):
		os.makedirs(log_dir_path, exist_ok=True)

	log_filepath = os.path.join(log_dir_path, (name if name else "inisys") + ".log")
	if is_rotating:
		file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10000000, backupCount=10)
	else:
		file_handler = logging.FileHandler(log_filepath)
	stream_handler = logging.StreamHandler()

	#formatter = logging.Formatter("[%(levelname)s][%(process)d][%(filename)s:%(lineno)s][%(asctime)s] [SWL] %(message)s")
	formatter = logging.Formatter("[%(levelname)s][%(process)d][%(filename)s:%(lineno)s][%(asctime)s] [SWL] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
	#formatter = logging.Formatter("[%(levelname)s][%(process)d][%(asctime)s] [SWL] %(message)s")
	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	logger = logging.getLogger(name if name else __name__)
	logger.setLevel(log_level)  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
	logger.addHandler(file_handler) 
	logger.addHandler(stream_handler) 

	return logger

def parse_train_command_line_options(use_ssl_type=True):
	parser = argparse.ArgumentParser(description="Training options for self-supervised learning.")

	if use_ssl_type:
		parser.add_argument(
			"-s",
			"--ssl",
			choices={"simclr", "byol", "relic", "simsiam"},
			help="A SSL model to train",
			#required=True,
			default="simclr"
		)
	parser.add_argument(
		"-d",
		"--dataset",
		choices={"imagenet", "cifar10", "mnist"},
		help="A dataset for training",
		#required=True,
		default="cifar10"
	)
	parser.add_argument(
		"-mf",
		"--model_file",
		type=str,
		#nargs="?",
		help="A model file path to resume training",
		#required=True,
		default=None
	)
	parser.add_argument(
		"-o",
		"--out_dir",
		type=str,
		#nargs="?",
		help="An output directory path to save results such as images and log",
		#required=True,
		default=None
	)
	parser.add_argument(
		"-e",
		"--epoch",
		type=int,
		help="Number of epochs to train",
		default=20
	)
	parser.add_argument(
		"-b",
		"--batch",
		type=int,
		help="Batch size",
		default=64
	)
	parser.add_argument(
		"-l",
		"--log",
		type=str,
		help="The name of logger and log files",
		default=None
	)
	parser.add_argument(
		"-ll",
		"--log_level",
		type=int,
		help="Log level, [0, 50]",  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
		default=None
	)
	parser.add_argument(
		"-ld",
		"--log_dir",
		type=str,
		help="A directory path to log",
		default=None
	)

	return parser.parse_args()

def parse_command_line_options(use_ssl_type=True, use_dataset_type=True):
	parser = argparse.ArgumentParser(description="Options for self-supervised learning.")

	if use_ssl_type:
		parser.add_argument(
			"-s",
			"--ssl",
			choices={"simclr", "byol", "relic", "simsiam"},
			help="A SSL model to train",
			#required=True,
			default="simclr"
		)
	parser.add_argument(
		"-mf",
		"--model_file",
		type=str,
		#nargs="?",
		help="A file path to load a pretrained model",
		required=True,
	)
	if use_dataset_type:
		parser.add_argument(
			"-d",
			"--dataset",
			choices={"imagenet", "cifar10", "mnist"},
			help="A dataset for training",
			#required=True,
			default="cifar10"
		)
	else:
		parser.add_argument(
			"-d",
			"--data_dir",
			type=str,
			#nargs="?",
			help="A directory path to load data",
			required=True,
		)
	parser.add_argument(
		"-o",
		"--out_dir",
		type=str,
		#nargs="?",
		help="An output directory path to save results such as images and log",
		#required=True,
		default=None
	)
	parser.add_argument(
		"-b",
		"--batch",
		type=int,
		help="Batch size",
		default=64
	)
	parser.add_argument(
		"-l",
		"--log",
		type=str,
		help="The name of logger and log files",
		default=None
	)
	parser.add_argument(
		"-ll",
		"--log_level",
		type=int,
		help="Log level, [0, 50]",  # {NOTSET=0, DEBUG=10, INFO=20, WARNING=WARN=30, ERROR=40, CRITICAL=FATAL=50}.
		default=None
	)
	parser.add_argument(
		"-ld",
		"--log_dir",
		type=str,
		help="A directory path to log",
		default=None
	)

	return parser.parse_args()

def visualize_data(dataloader, num_data=10, class_names=None):
	data_iter = iter(dataloader)
	srcs, tgts = data_iter.next()  # torch.Tensor & torch.Tensor.
	srcs, tgts = srcs.numpy(), tgts.numpy()
	srcs = srcs.transpose(0, 2, 3, 1).squeeze(axis=-1)

	num_data = min(num_data, len(srcs), len(tgts)) if num_data else min(len(srcs), len(tgts))
	for src, tgt in random.sample(list(zip(srcs, tgts)), num_data):
		print('Label = {}.'.format(class_names[tgt] if class_names else tgt))
		plt.imshow(src)
		#plt.title('Image')
		plt.axis('off')
		plt.tight_layout()
		plt.show()

def create_imagenet_datasets(imagenet_dir_path, logger=None):
	train_transform = torchvision.transforms.Compose([
		#torchvision.transforms.Resize(256),
		#torchvision.transforms.CenterCrop(224),
		torchvision.transforms.RandomResizedCrop(224),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(256),
		torchvision.transforms.CenterCrop(224),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	if logger: logger.info('Creating ImageNet datasets...')
	start_time = time.time()
	train_dataset = torchvision.datasets.ImageNet(imagenet_dir_path, split='train', transform=train_transform, target_transform=None)
	test_dataset = torchvision.datasets.ImageNet(imagenet_dir_path, split='val', transform=test_transform, target_transform=None)
	if logger: logger.info('ImageNet datasets created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	return train_dataset, test_dataset

def create_cifar10_datasets(logger=None):
	if True:
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])
	else:
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(32, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
		])
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	if logger: logger.info('Creating CIFAR-10 datasets...')
	start_time = time.time()
	train_dataset = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=train_transform, target_transform=None)
	test_dataset = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=test_transform, target_transform=None)
	if logger: logger.info('CIFAR-10 datasets created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	return train_dataset, test_dataset, classes

def create_mnist_datasets(logger=None):
	if True:
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
		])
	else:
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(28, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
		])

	if logger: logger.info('Creating MNIST datasets...')
	start_time = time.time()
	train_dataset = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=train_transform, target_transform=None)
	test_dataset = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=test_transform, target_transform=None)
	if logger: logger.info('MNIST datasets created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train examples = {}, #test examples = {}.'.format(len(train_dataset), len(test_dataset)))

	return train_dataset, test_dataset

def prepare_open_data(dataset_type, batch_size, num_workers, dataset_root_dir_path=None, show_info=True, show_data=False, logger=None):
	# Create datasets.
	if dataset_type == 'imagenet':
		train_dataset, test_dataset = create_imagenet_datasets(dataset_root_dir_path, logger)
		class_names = None
	elif dataset_type == 'cifar10':
		train_dataset, test_dataset, class_names = create_cifar10_datasets(logger)
	elif dataset_type == 'mnist':
		train_dataset, test_dataset = create_mnist_datasets(logger)
		class_names = None

	# Create data loaders.
	if logger: logger.info('Creating data loaders...')
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)
	if logger: logger.info('Data loaders created: {} secs.'.format(time.time() - start_time))
	if logger: logger.info('#train steps per epoch = {}, #test steps per epoch = {}.'.format(len(train_dataloader), len(test_dataloader)))

	if show_info:
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

	if show_data:
		# Visualize data.
		print('Visualizing training data...')
		visualize_data(train_dataloader, num_data=10, class_names=class_names)
		print('Visualizing test data...')
		visualize_data(test_dataloader, num_data=10, class_names=class_names)

	return train_dataloader, test_dataloader

def create_simclr_augmenter(image_height, image_width, normalization_mean, normalization_stddev):
	class RandomApply(torch.nn.Module):
		def __init__(self, fn, p):
			super().__init__()

			self.fn = fn
			self.p = p

		def forward(self, x):
			if random.random() > self.p:
				return x
			return self.fn(x)

	s = 1.0  # The strength of color distortion.
	return torch.nn.Sequential(
		torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
		#torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
		torchvision.transforms.RandomHorizontalFlip(p=0.5),
		RandomApply(
			torchvision.transforms.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s),
			#torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
			p=0.8
		),
		torchvision.transforms.RandomGrayscale(p=0.2),
		RandomApply(
			torchvision.transforms.GaussianBlur(kernel_size=(max(round(image_height * 0.1), 3), max(round(image_width * 0.1), 3)), sigma=(0.1, 2.0)),
			p=0.5
		),
		torchvision.transforms.Normalize(mean=torch.tensor(normalization_mean), std=torch.tensor(normalization_stddev)),
	)

class ModelWrapper(torch.nn.Module):
	def __init__(self, module, layer_name):
		super().__init__()

		assert layer_name in module._modules.keys(), 'Layer name, {} not found in module'.format(layer_name)
		self.submodule = module
		self.name = layer_name
		#self.linear = torch.nn.Linear(feature_dim, output_dim)

	def forward(self, x):
		for name, module in self.submodule._modules.items():
			x = module(x)
			if name is self.name:
				return x.view(x.size(0), -1)
				#return self.linear(x.view(x.size(0), -1))
		return None

# REF [site] >> https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
# MLP class for projector and predictor.
class MLP(torch.nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim=4096):
		super().__init__()

		self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
		self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim)
		self.relu1 = torch.nn.ReLU(inplace=True)
		self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = self.linear1(x)
		x = self.batchnorm1(x)
		x = self.relu1(x)
		x = self.linear2(x)
		return x

# REF [site] >> https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
class SimSiamMLP(torch.nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim=4096):
		super().__init__()

		self.linear1 = torch.nn.Linear(input_dim, hidden_dim, bias=False)
		self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim)
		self.relu1 = torch.nn.ReLU(inplace=True)
		self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim)
		self.relu2 = torch.nn.ReLU(inplace=True)
		self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
		self.batchnorm3 = torch.nn.BatchNorm1d(output_dim, affine=False)

	def forward(self, x):
		x = self.linear1(x)
		x = self.batchnorm1(x)
		x = self.relu1(x)
		x = self.linear2(x)
		x = self.batchnorm2(x)
		x = self.relu2(x)
		x = self.linear3(x)
		x = self.batchnorm3(x)
		return x

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

	# Train the model.
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
