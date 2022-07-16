import os, random, argparse, logging, logging.handlers, time
import numpy as np
import torch, torchvision
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

def parse_train_command_line_options():
	parser = argparse.ArgumentParser(description="Training options for self-supervised learning.")

	parser.add_argument(
		"-st",
		"--ssl_type",
		choices={"byol", "relic"},
		help="The SSL type to train a model",
		#required=True,
		default="byol"
	)
	parser.add_argument(
		"-dt",
		"--dataset_type",
		choices={"imagenet", "cifar10", "mnist"},
		help="The dataset type for training",
		#required=True,
		default="cifar10"
	)
	parser.add_argument(
		"-mf",
		"--model_file",
		type=str,
		#nargs="?",
		help="The file path to load a pretrained model",
		#required=True,
		default=None
	)
	parser.add_argument(
		"-o",
		"--out_dir",
		type=str,
		#nargs="?",
		help="The output directory path to save results such as images and log",
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
		help="The directory path to log",
		default=None
	)

	return parser.parse_args()

def parse_command_line_options():
	parser = argparse.ArgumentParser(description="Options for self-supervised learning.")

	parser.add_argument(
		"-mf",
		"--model_file",
		type=str,
		#nargs="?",
		help="The file path to load a pretrained model",
		required=True,
		default=None
	)
	parser.add_argument(
		"-d",
		"--data_dir",
		type=str,
		#nargs="?",
		help="The directory path to load data",
		required=True,
		default=None
	)
	parser.add_argument(
		"-o",
		"--out_dir",
		type=str,
		#nargs="?",
		help="The output directory path to save results such as images and log",
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
		help="The directory path to log",
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

	return torch.nn.Sequential(
		RandomApply(
			torchvision.transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
			p=0.3
		),
		torchvision.transforms.RandomGrayscale(p=0.2),
		torchvision.transforms.RandomHorizontalFlip(p=0.5),
		RandomApply(
			torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0)),
			p=0.2
		),
		torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.08, 1.0), ratio=(0.75, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
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
