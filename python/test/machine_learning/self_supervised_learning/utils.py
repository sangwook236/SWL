import os, random, math, argparse, logging, logging.handlers, time
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

def parse_evaluation_command_line_options(use_ssl_type=True, use_dataset_type=True):
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

def parse_config_command_line_options(is_training=True):
	parser = argparse.ArgumentParser(description="Options for self-supervised learning.")

	parser.add_argument(
		"-c",
		"--config",
		type=str,
		#nargs="?",
		help="A path to configuration file",
		required=True,
		default="config.yaml"
	)
	parser.add_argument(
		"-mf",
		"--model_file",
		type=str,
		#nargs="?",
		help="A model file path to resume training" if is_training else "A file path to load a pretrained model",
		#required=True,
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
		print("Label = {}.".format(class_names[tgt] if class_names else tgt))
		plt.imshow(src)
		#plt.title("Image")
		plt.axis("off")
		plt.tight_layout()
		plt.show()

def create_imagenet_datasets(config, logger=None):
	if False:
		# No additional augmentation is required for unsupervised pretraining.
		#	REF [function] >> create_simclr_augmenter().

		# NOTE [info] >> All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. (torchvision.transforms)
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			#torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			#torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	elif False:
		# NOTE [info] >> All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. (torchvision.transforms)
		train_transform = torchvision.transforms.Compose([
			#torchvision.transforms.RandomCrop(224, padding=16),
			torchvision.transforms.RandomResizedCrop(224),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		test_transform = torchvision.transforms.Compose([
			#torchvision.transforms.Resize(256),
			#torchvision.transforms.CenterCrop(224),
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	else:
		train_transform = construct_transform(config["train_transforms"])
		test_transform = construct_transform(config["test_transforms"])

	if logger: logger.info("Creating ImageNet datasets...")
	start_time = time.time()
	train_dataset = torchvision.datasets.ImageNet(root=config["data_dir"], split="train", transform=train_transform, target_transform=None)
	test_dataset = torchvision.datasets.ImageNet(root=config["data_dir"], split="val", transform=test_transform, target_transform=None)
	if logger: logger.info("ImageNet datasets created: {} secs.".format(time.time() - start_time))
	if logger: logger.info("#train examples = {}, #test examples = {}.".format(len(train_dataset), len(test_dataset)))

	return train_dataset, test_dataset

def create_cifar10_datasets(config, logger=None):
	if False:
		# No additional augmentation is required for unsupervised pretraining.
		#	REF [function] >> create_simclr_augmenter().

		# NOTE [info] >> All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. (torchvision.transforms)
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			#torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0, 1] -> [-1, 1].
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			#torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0, 1] -> [-1, 1].
		])
	elif False:
		# NOTE [info] >> All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. (torchvision.transforms)
		train_transform = torchvision.transforms.Compose([
			#torchvision.transforms.RandomCrop(32, padding=4),
			torchvision.transforms.RandomResizedCrop(32),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
			#torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])
		test_transform = torchvision.transforms.Compose([
			#torchvision.transforms.Resize(40),
			#torchvision.transforms.CenterCrop(32),
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
			#torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])
	else:
		train_transform = construct_transform(config["train_transforms"])
		test_transform = construct_transform(config["test_transforms"])
	classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

	if logger: logger.info("Creating CIFAR-10 datasets...")
	start_time = time.time()
	train_dataset = torchvision.datasets.CIFAR10(root=config["data_dir"], train=True, download=True, transform=train_transform, target_transform=None)
	test_dataset = torchvision.datasets.CIFAR10(root=config["data_dir"], train=False, download=True, transform=test_transform, target_transform=None)
	if logger: logger.info("CIFAR-10 datasets created: {} secs.".format(time.time() - start_time))
	if logger: logger.info("#train examples = {}, #test examples = {}.".format(len(train_dataset), len(test_dataset)))

	return train_dataset, test_dataset, classes

def create_mnist_datasets(config, logger=None):
	if False:
		# No additional augmentation is required for unsupervised pretraining.
		#	REF [function] >> create_simclr_augmenter().

		# NOTE [info] >> All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. (torchvision.transforms)
		train_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			#torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),  # [0, 1] -> [-1, 1].
		])
		test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			#torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),  # [0, 1] -> [-1, 1].
		])
	elif False:
		# NOTE [info] >> All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input. (torchvision.transforms)
		train_transform = torchvision.transforms.Compose([
			#torchvision.transforms.RandomCrop(28, padding=4),
			torchvision.transforms.RandomResizedCrop(28),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081]),
		])
		test_transform = torchvision.transforms.Compose([
			#torchvision.transforms.Resize(36),
			#torchvision.transforms.CenterCrop(28),
			torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
			torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081]),
		])
	else:
		train_transform = construct_transform(config["train_transforms"])
		test_transform = construct_transform(config["test_transforms"])

	if logger: logger.info("Creating MNIST datasets...")
	start_time = time.time()
	train_dataset = torchvision.datasets.MNIST(root=config["data_dir"], train=True, download=True, transform=train_transform, target_transform=None)
	test_dataset = torchvision.datasets.MNIST(root=config["data_dir"], train=False, download=True, transform=test_transform, target_transform=None)
	if logger: logger.info("MNIST datasets created: {} secs.".format(time.time() - start_time))
	if logger: logger.info("#train examples = {}, #test examples = {}.".format(len(train_dataset), len(test_dataset)))

	return train_dataset, test_dataset

def prepare_open_data(config, show_info=True, show_data=False, logger=None):
	# Create datasets.
	if config["dataset"] == "imagenet":
		train_dataset, test_dataset = create_imagenet_datasets(config, logger)
		class_names = None
		num_classes = 1000
	elif config["dataset"] == "cifar10":
		train_dataset, test_dataset, class_names = create_cifar10_datasets(config, logger)
		num_classes = 10
	elif config["dataset"] == "mnist":
		train_dataset, test_dataset = create_mnist_datasets(config, logger)
		class_names = None
		num_classes = 10

	# Create data loaders.
	if logger: logger.info("Creating data loaders...")
	start_time = time.time()
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], persistent_workers=False)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], persistent_workers=False)
	if logger: logger.info("Data loaders created: {} secs.".format(time.time() - start_time))
	if logger: logger.info("#train steps per epoch = {}, #test steps per epoch = {}.".format(len(train_dataloader), len(test_dataloader)))

	if show_info:
		# Show data info.
		data_iter = iter(train_dataloader)
		srcs, tgts = data_iter.next()
		srcs, tgts = srcs.numpy(), tgts.numpy()
		if logger: logger.info("Train source (batch): Shape = {}, dtype = {}, (min, max) = ({}, {}).".format(srcs.shape, srcs.dtype, np.min(srcs), np.max(srcs)))
		if logger: logger.info("Train target (batch): Shape = {}, dtype = {}, classes = {}.".format(tgts.shape, tgts.dtype, np.unique(tgts)))

		data_iter = iter(test_dataloader)
		srcs, tgts = data_iter.next()
		srcs, tgts = srcs.numpy(), tgts.numpy()
		if logger: logger.info("Test source (batch): Shape = {}, dtype = {}, (min, max) = ({}, {}).".format(srcs.shape, srcs.dtype, np.min(srcs), np.max(srcs)))
		if logger: logger.info("Test target (batch): Shape = {}, dtype = {}, classes = {}.".format(tgts.shape, tgts.dtype, np.unique(tgts)))

	if show_data:
		# Visualize data.
		print("Visualizing training data...")
		visualize_data(train_dataloader, num_data=10, class_names=class_names)
		print("Visualizing test data...")
		visualize_data(test_dataloader, num_data=10, class_names=class_names)

	return train_dataloader, test_dataloader, num_classes

# REF [site] >> https://github.com/NightShade99/Self-Supervised-Vision/blob/main/utils/augmentations.py
def construct_transform(config, *args, **kwargs):
	if not config: return None

	# REF [site] >> https://github.com/NightShade99/Self-Supervised-Vision/blob/main/utils/augmentations.py
	class GaussianBlur:
		def __init__(self, sigma=[0.1, 2.0]):
			self.sigma = sigma

		def __call__(self, img):
			from PIL import ImageFilter

			sigma = random.uniform(self.sigma[0], self.sigma[1])
			img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
			return img

	# REF [site] >> https://github.com/NightShade99/Self-Supervised-Vision/blob/main/utils/augmentations.py
	class Cutout:
		def __init__(self, n_cuts=0, max_len=1):
			self.n_cuts = n_cuts
			self.max_len = max_len

		def __call__(self, img):
			h, w = img.shape[1:3]
			cut_len = random.randint(1, self.max_len)
			mask = np.ones((h, w), np.float32)

			for _ in range(self.n_cuts):
				x, y = random.randint(0, w), random.randint(0, h)
				x1 = np.clip(x - cut_len // 2, 0, w)
				x2 = np.clip(x + cut_len // 2, 0, w)
				y1 = np.clip(y - cut_len // 2, 0, h)
				y2 = np.clip(y + cut_len // 2, 0, h)
				mask[y1:y2, x1:x2] = 0

			mask = torch.from_numpy(mask)
			mask = mask.expand_as(img)
			return img * mask

	# REF [site] >> https://github.com/NightShade99/Self-Supervised-Vision/blob/main/utils/augmentations.py
	class RandomAugment:
		def __init__(self, n_aug=4):
			self.n_aug = n_aug
			self.aug_list = [
				("identity", 1, 1),
				("autocontrast", 1, 1),
				("equalize", 1, 1),
				("rotate", -30, 30),
				("solarize", 1, 1),
				("color", 1, 1),
				("contrast", 1, 1),
				("brightness", 1, 1),
				("sharpness", 1, 1),
				("shear_x", -0.1, 0.1),
				("shear_y", -0.1, 0.1),
				("translate_x", -0.1, 0.1),
				("translate_y", -0.1, 0.1),
				("posterize", 1, 1),
			]

		def __call__(self, img):
			from PIL import Image, ImageOps, ImageEnhance

			aug_choices = random.choices(self.aug_list, k=self.n_aug)
			for aug, min_value, max_value in aug_choices:
				v = random.uniform(min_value, max_value)
				if aug == "identity":
					pass
				elif aug == "autocontrast":
					img = ImageOps.autocontrast(img)
				elif aug == "equalize":
					img = ImageOps.equalize(img)
				elif aug == "rotate":
					if random.random() > 0.5:
						v = -v
					img = img.rotate(v)
				elif aug == "solarize":
					img = ImageOps.solarize(img, v)
				elif aug == "color":
					img = ImageEnhance.Color(img).enhance(v)
				elif aug == "contrast":
					img = ImageEnhance.Contrast(img).enhance(v)
				elif aug == "brightness":
					img = ImageEnhance.Brightness(img).enhance(v)
				elif aug == "sharpness":
					img = ImageEnhance.Sharpness(img).enhance(v)
				elif aug == "shear_x":
					if random.random() > 0.5:
						v = -v
					img = img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))
				elif aug == "shear_y":
					if random.random() > 0.5:
						v = -v
					img = img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))
				elif aug == "translate_x":
					if random.random() > 0.5:
						v = -v
					v = v * img.size[0]
					img = img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
				elif aug == "translate_y":
					if random.random() > 0.5:
						v = -v
					v = v * img.size[1]
					img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
				elif aug == "posterize":
					img = ImageOps.posterize(img, int(v))
				else:
					raise NotImplementedError(f"{aug} not implemented")
			return img

	TRANSFORM_HELPER = {
		"color_jitter": torchvision.transforms.ColorJitter,
		"random_gray": torchvision.transforms.RandomGrayscale,
		"random_crop": torchvision.transforms.RandomCrop,
		"random_resized_crop": torchvision.transforms.RandomResizedCrop,
		"center_crop": torchvision.transforms.CenterCrop,
		"resize": torchvision.transforms.Resize,
		"random_horizontal_flip": torchvision.transforms.RandomHorizontalFlip,
		"to_tensor": torchvision.transforms.ToTensor,
		"normalize": torchvision.transforms.Normalize,
		"gaussian_blur": torchvision.transforms.GaussianBlur,
		#"gaussian_blur": GaussianBlur,
		"rand_aug": RandomAugment,
		"cutout": Cutout,
	}

	transforms = list()
	for key, value in config.items():
		if value is not None:
			random_apply = value.pop("random_apply", None)
			tr = TRANSFORM_HELPER[key](**value)
			if random_apply is not None:
				tr = torchvision.transforms.RandomApply([tr], p=random_apply["p"])
		else:
			tr = TRANSFORM_HELPER[key]()
		transforms.append(tr)
	return torchvision.transforms.Compose(transforms)

def create_simclr_augmenter(image_height, image_width, normalization_mean, normalization_stddev):
	s = 1.0  # The strength of color distortion.
	return torchvision.transforms.Compose([
		torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
		#torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
		torchvision.transforms.RandomHorizontalFlip(p=0.5),
		torchvision.transforms.RandomApply(
			torchvision.transforms.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s),
			#torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
			p=0.8
		),
		torchvision.transforms.RandomGrayscale(p=0.2),
		torchvision.transforms.RandomApply(
			#torchvision.transforms.GaussianBlur(kernel_size=(max(math.floor(image_height * 0.1 * 0.5) * 2 + 1, 3), max(math.floor(image_width * 0.1 * 0.5) * 2 + 1, 3)), sigma=(0.1, 2.0)),
			torchvision.transforms.GaussianBlur(kernel_size=(max(math.floor(image_height * 0.05) * 2 + 1, 3), max(math.floor(image_width * 0.05) * 2 + 1, 3)), sigma=(0.1, 2.0)),
			p=0.5
		),
		torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
		torchvision.transforms.Normalize(mean=normalization_mean, std=normalization_stddev),
	])

def construct_encoder(model_type, pretrained=True, *args, **kwargs):
	ENCODERS = {
		"resnet18": {"model": torchvision.models.resnet18, "feature_dim": 512},
		"resnet50": {"model": torchvision.models.resnet50, "feature_dim": 2048},
		"resnext50": {"model": torchvision.models.resnext50_32x4d, "feature_dim": 2048},
		"resnext101": {"model": torchvision.models.resnext101_32x8d, "feature_dim": 2048},
		"wide_resnet50": {"model": torchvision.models.wide_resnet50_2, "feature_dim": 2048},
		"wide_resnet101": {"model": torchvision.models.wide_resnet101_2, "feature_dim": 2048},
	}

	return ModelWrapper(ENCODERS[model_type]["model"](pretrained=pretrained), layer_name="avgpool"), ENCODERS[model_type]["feature_dim"]

class ModelWrapper(torch.nn.Module):
	def __init__(self, model, layer_name):
		super().__init__()

		assert layer_name in model._modules.keys(), "Layer name, {} not found in model".format(layer_name)
		self.model = model
		self.layer_name = layer_name
		#self.linear = torch.nn.Linear(feature_dim, output_dim)

	def forward(self, x):
		for name, module in self.model._modules.items():
			x = module(x)
			if name is self.layer_name:
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

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, T_max, T_warmup, last_epoch=-1, verbose=False):
		self.T_max = T_max
		self.T_warmup = T_warmup
		super().__init__(optimizer, last_epoch, verbose)

	def get_lr(self):
		lr_factor = self.get_lr_factor(epoch=self.last_epoch)
		return [base_lr * lr_factor for base_lr in self.base_lrs]

	def get_lr_factor(self, epoch):
		"""
		lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.T_max))
		if epoch <= self.T_warmup:
			lr_factor *= epoch / self.T_warmup
		return lr_factor
		"""
		if epoch <= self.T_warmup:
			return epoch / self.T_warmup
		else:
			return 0.5 * (1 + math.cos(math.pi * (epoch - self.T_warmup) / (self.T_max - self.T_warmup)))

# REF [function] >> train_text_recognizer() in ${SWLP_HOME}/app/text/run_text_recognition_pl.py
def train(config, model, train_dataloader, test_dataloader, output_dir_path, model_filepath_to_load, logger=None):
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=(output_dir_path + "/checkpoints") if output_dir_path else "./checkpoints",
		filename="model-{epoch:03d}-{step:05d}-{val_acc:.5f}-{val_loss:.5f}",
		monitor="val_loss",
		mode="min",
		save_top_k=-1,
	)
	if config.get("swa", False):
		swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=2, annealing_strategy="cos", avg_fn=None)
		pl_callbacks = [checkpoint_callback, swa_callback]
	else:
		pl_callbacks = [checkpoint_callback]
	tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=(output_dir_path + "/lightning_logs") if output_dir_path else "./lightning_logs", name="", version=None)
	pl_logger = [tensorboard_logger]

	if config.get("max_gradient_norm", None):
		gradient_clip_val = config["max_gradient_norm"]
		gradient_clip_algorithm = "norm"  # {"norm", "value"}.
	else:
		gradient_clip_val = None
		gradient_clip_algorithm = None
	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="ddp", auto_select_gpus=True, max_epochs=config["epochs"], callbacks=pl_callbacks, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=output_dir_path)  # When using the default logger.
	trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="ddp", auto_select_gpus=True, max_epochs=config["epochs"], callbacks=pl_callbacks, logger=pl_logger, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=None)

	# Train the model.
	if logger: logger.info("Training the model...")
	start_time = time.time()
	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=model_filepath_to_load if model_filepath_to_load else None)
	if logger: logger.info("The model trained: {} secs.".format(time.time() - start_time))

	#--------------------
	if False:
		# Validate the trained model.
		if logger: logger.info("Validating the model...")
		start_time = time.time()
		#model.eval()
		trainer.validate(model, dataloaders=test_dataloader, ckpt_path=None, verbose=True)
		if logger: logger.info("The model validated: {} secs.".format(time.time() - start_time))

	#--------------------
	#best_model_filepath = trainer.checkpoint_callback.best_model_path
	best_model_filepath = checkpoint_callback.best_model_path
	if logger: logger.info("The best trained model saved to {}.".format(best_model_filepath))
	if False:
		# Save the final model.
		final_model_filepath = (output_dir_path + "/final_model.ckpt") if output_dir_path else "./final_model.ckpt"
		trainer.save_checkpoint(final_model_filepath, weights_only=False)
		if logger: logger.info("The final trained model saved to {}.".format(final_model_filepath))

	return best_model_filepath

def infer(config, model, data_iter, logger=None, device="cuda"):
	model = model.to(device)

	use_projector, use_predictor = config.get('use_projector', False), config.get('use_predictor', False)
	if logger: logger.info("Inferring...")
	start_time = time.time()
	model.eval()
	model.freeze()
	with torch.no_grad():
		predictions = list()
		for inputs in data_iter:
			predictions.append(model(inputs.to(device), use_projector, use_predictor).cpu().numpy())
		predictions = np.vstack(predictions)
	if logger: logger.info("Inferred: {} secs.".format(time.time() - start_time))
	#if logger: logger.info("Prediction: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(predictions.shape, predictions.dtype, np.min(predictions), np.max(predictions)))
	if logger: logger.info("Prediction: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(predictions.shape, predictions.dtype, np.min(np.abs(predictions)), np.max(np.abs(predictions))))

	return predictions
