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

def parse_command_line_options(is_training=True):
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
		required=not is_training,
		default=None
	)
	parser.add_argument(
		"-o",
		"--out_dir",
		type=str,
		help="The output directory path to save results such as images and log",
		default=None
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

# REF [function] >> construct_transform() in ${SWL_PYTHON_HOME}/test/machine_learning/config_test.py.
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

	TRANSFORMS = {
		"color_jitter": torchvision.transforms.ColorJitter,
		"random_grayscale": torchvision.transforms.RandomGrayscale,
		"random_crop": torchvision.transforms.RandomCrop,
		"random_resized_crop": torchvision.transforms.RandomResizedCrop,
		"center_crop": torchvision.transforms.CenterCrop,
		"resize": torchvision.transforms.Resize,
		"random_horizontal_flip": torchvision.transforms.RandomHorizontalFlip,
		"to_tensor": torchvision.transforms.ToTensor,
		"normalize": torchvision.transforms.Normalize,
		"gaussian_blur": torchvision.transforms.GaussianBlur,
		#"gaussian_blur": GaussianBlur,
		"random_augument": RandomAugment,
		"cutout": Cutout,
	}

	transforms = list()
	for key, value in config.items():
		if value is not None:
			random_apply = value.pop("random_apply", None)
			tr = TRANSFORMS[key](**value)
			if random_apply is not None:
				tr = torchvision.transforms.RandomApply([tr], p=random_apply["p"])
		else:
			tr = TRANSFORMS[key]()
		transforms.append(tr)
	return torchvision.transforms.Compose(transforms)

def create_simclr_augmenter(image_height, image_width, normalization_mean, normalization_stddev):
	s = 1.0  # The strength of color distortion.
	return torchvision.transforms.Compose([
		torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
		#torchvision.transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
		torchvision.transforms.RandomHorizontalFlip(p=0.5),
		torchvision.transforms.RandomApply(
			[torchvision.transforms.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s)],
			#[torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
			p=0.8
		),
		torchvision.transforms.RandomGrayscale(p=0.2),
		torchvision.transforms.RandomApply(
			#[torchvision.transforms.GaussianBlur(kernel_size=(max(math.floor(image_height * 0.1 * 0.5) * 2 + 1, 3), max(math.floor(image_width * 0.1 * 0.5) * 2 + 1, 3)), sigma=(0.1, 2.0))],
			[torchvision.transforms.GaussianBlur(kernel_size=(max(math.floor(image_height * 0.05) * 2 + 1, 3), max(math.floor(image_width * 0.05) * 2 + 1, 3)), sigma=(0.1, 2.0))],
			p=0.5
		),
		torchvision.transforms.ToTensor(),  # (H, W) or (H, W, C) -> (C, H, W).
		torchvision.transforms.Normalize(mean=normalization_mean, std=normalization_stddev),
	])

# REF [class] >> ModelWrapper class in ${SWL_PYTHON_HOME}/test/machine_learning/config_test.py.
class ModelWrapper(torch.nn.Module):
	def __init__(self, model, layer_name, feature_dim=None, output_dim=None):
		super().__init__()

		assert layer_name in model._modules.keys(), "Layer name, {} not found in model".format(layer_name)
		self.model = model
		self.layer_name = layer_name
		self.generator = torch.nn.Linear(feature_dim, output_dim) if feature_dim is not None and output_dim is not None else None

	def forward(self, x):
		for name, module in self.model._modules.items():
			x = module(x)
			if name == self.layer_name:
				return x.view(x.size(0), -1) if self.generator is None else self.generator(x.view(x.size(0), -1))
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

def load_ssl(ssl_type, model_filepath):
	if ssl_type == "simclr":
		import model_simclr
		SslModule = getattr(model_simclr, "SimclrModule")
		ssl_model = SslModule.load_from_checkpoint(model_filepath, config=None, encoder=None, projector=None, augmenter1=None, augmenter2=None)
	elif ssl_type == "byol":
		import model_byol
		SslModule = getattr(model_byol, "ByolModule")
		ssl_model = SslModule.load_from_checkpoint(model_filepath, config=None, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)
	elif ssl_type == "relic":
		import model_relic
		SslModule = getattr(model_relic, "RelicModule")
		ssl_model = SslModule.load_from_checkpoint(model_filepath, config=None, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)
	elif ssl_type == "simsiam":
		import model_simsiam
		SslModule = getattr(model_simsiam, "SimSiamModule")
		ssl_model = SslModule.load_from_checkpoint(model_filepath, config=None, encoder=None, projector=None, predictor=None, augmenter1=None, augmenter2=None)

	#ssl_model = SslModule.load_from_checkpoint(model_filepath)
	#ssl_model = SslModule.load_from_checkpoint(model_filepath, map_location={"cuda:1": "cuda:0"})

	return ssl_model

# REF [function] >> construct_pretrained_model() in ${SWL_PYTHON_HOME}/test/machine_learning/config_test.py.
def construct_pretrained_model(config, output_dim=None, *args, **kwargs):
	PRETRAINED_MODELS = {
		"resnet18": torchvision.models.resnet18,
		"resnet34": torchvision.models.resnet34,
		"resnet50": torchvision.models.resnet50,
		"resnet101": torchvision.models.resnet101,
		"resnet152": torchvision.models.resnet152,
		"resnext50": torchvision.models.resnext50_32x4d,
		"resnext101": torchvision.models.resnext101_32x8d,
		#"resnext101": torchvision.models.resnext101_64x4d,
		"wide_resnet50": torchvision.models.wide_resnet50_2,
		"wide_resnet101": torchvision.models.wide_resnet101_2,
	}

	for name in config:
		if name in PRETRAINED_MODELS:
			feature_layer = config[name].pop("feature_layer", "avgpool")
			feature_dim = config[name].pop("feature_dim", 0)
			return ModelWrapper(PRETRAINED_MODELS[name](**config[name]), layer_name=feature_layer, feature_dim=feature_dim, output_dim=output_dim), feature_dim
		elif hasattr(torchvision.models, name):
			pretrained_model = getattr(torchvision.models, name)
			feature_layer = config[name].pop("feature_layer", "avgpool")
			feature_dim = config[name].pop("feature_dim", 0)
			return ModelWrapper(pretrained_model(**config[name]), layer_name=feature_layer, feature_dim=feature_dim, output_dim=output_dim), feature_dim
		else:
			raise ValueError("Unsupported pretrained model, {}".format(name))
	raise ValueError("Invalid pretrained model, {}".format(config))

# REF [function] >> construct_user_defined_model() in ${SWL_PYTHON_HOME}/test/machine_learning/config_test.py.
def construct_user_defined_model(config, *args, **kwargs):
	MODULES = {
		"conv_1d": torch.nn.Conv1d,
		"conv_2d": torch.nn.Conv2d,
		"linear": torch.nn.Linear,
		"max_pool_1d": torch.nn.MaxPool1d,
		"max_pool_2d": torch.nn.MaxPool2d,
		"batch_norm_1d": torch.nn.BatchNorm1d,
		"batch_norm_2d": torch.nn.BatchNorm2d,
		"dropout": torch.nn.Dropout,
		"flatten": torch.nn.Flatten,
		"softmax": torch.nn.Softmax,
		"sigmoid": torch.nn.Sigmoid,
		"tanh": torch.nn.Tanh,
		"relu": torch.nn.ReLU,
	}

	modules = list()
	for module_config in config["architecture"]:
		module_type = module_config.pop("module_type")
		if module_type in MODULES:
			modules.append(MODULES[module_type](**module_config))
		elif hasattr(torch.nn, module_type):
			module = getattr(torch.nn, module_type)
			modules.append(module(**module_config))
		else:
			raise ValueError("Unsupported module, {}".format(module_type))
	output_dim = config.pop("output_dim", 0)

	return torch.nn.Sequential(*modules), output_dim

# REF [function] >> construct_optimizer() in ${SWL_PYTHON_HOME}/test/machine_learning/config_test.py.
def construct_optimizer(config, model_params, *args, **kwargs):
	OPTIMIZERS = {
		"sgd": torch.optim.SGD,
		"adam": torch.optim.Adam,
		"adadelta": torch.optim.SGD,
		"adagrad": torch.optim.Adagrad,
		"rmsprop": torch.optim.RMSprop,
	}

	for name in config:
		if name in OPTIMIZERS:
			return OPTIMIZERS[name](model_params, **config[name])
		elif hasattr(torch.optim, name):
			optimizer = getattr(torch.optim, name)
			return optimizer(model_params, **config[name])
		else:
			raise ValueError("Unsupported optimizer, {}".format(name))
	raise ValueError("Invalid optimizer, {}".format(config))

# REF [class] >> CosineAnnealingWarmupLR class in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_optimization.py
class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
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

# REF [class] >> CosineAnnealingWarmUpRestartsLR class in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_optimization.py
class CosineAnnealingWarmUpRestartsLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, T_0, T_mult=1, T_up=0, eta_max=0.1, gamma=1.0, last_epoch=-1):
		if T_0 <= 0 or not isinstance(T_0, int):
			raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
		if T_mult < 1 or not isinstance(T_mult, int):
			raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
		if T_up < 0 or not isinstance(T_up, int):
			raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
		self.T_0 = T_0
		self.T_mult = T_mult
		self.base_eta_max = eta_max
		self.eta_max = eta_max
		self.T_up = T_up
		self.T_i = T_0
		self.gamma = gamma
		self.cycle = 0
		self.T_cur = last_epoch
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.T_cur == -1:
			return self.base_lrs
		elif self.T_cur < self.T_up:
			return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
		else:
			return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2 for base_lr in self.base_lrs]

	def step(self, epoch=None):
		if epoch is None:
			epoch = self.last_epoch + 1
			self.T_cur = self.T_cur + 1
			if self.T_cur >= self.T_i:
				self.cycle += 1
				self.T_cur = self.T_cur - self.T_i
				self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
		else:
			if epoch >= self.T_0:
				if self.T_mult == 1:
					self.T_cur = epoch % self.T_0
					self.cycle = epoch // self.T_0
				else:
					n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
					self.cycle = n
					self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
					self.T_i = self.T_0 * self.T_mult ** (n)
			else:
				self.T_i = self.T_0
				self.T_cur = epoch

		self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
		self.last_epoch = math.floor(epoch)
		for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
			param_group["lr"] = lr

		self._last_lr = [param_group["lr"] for param_group in self.optimizer.param_groups]

# REF [function] >> construct_lr_scheduler() in ${SWL_PYTHON_HOME}/test/machine_learning/config_test.py.
def construct_lr_scheduler(config, optimizer, num_epochs, *args, **kwargs):
	if not config:
		return None, True

	LR_SCHEDULERS = {
		"step": torch.optim.lr_scheduler.StepLR,
		"multi_step": torch.optim.lr_scheduler.MultiStepLR,
		"cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
		"cosine_warmup": CosineAnnealingWarmupLR,
		"cosine_restart": CosineAnnealingWarmUpRestartsLR,
		#"noam": NoamLR,  # For transformer. Step-based LR scheduler.
	}

	for name in config:
		if name in LR_SCHEDULERS:
			epoch_based = config[name].pop("epoch_based", True)
			if "T_max" in config[name]:
				T_max = config[name].pop("T_max")
				return LR_SCHEDULERS[name](optimizer, T_max=T_max if T_max is not None else num_epochs, **config[name]), epoch_based
			else:
				return LR_SCHEDULERS[name](optimizer, **config[name]), epoch_based
		elif hasattr(torch.optim.lr_scheduler, name):
			lr_scheduler = getattr(torch.optim.lr_scheduler, name)
			epoch_based = config[name].pop("epoch_based", True)
			if "T_max" in config[name]:
				T_max = config[name].pop("T_max")
				return lr_scheduler(optimizer, T_max=T_max if T_max is not None else num_epochs, **config[name]), epoch_based
			else:
				return lr_scheduler(optimizer, **config[name]), epoch_based
		else:
			raise ValueError("Unsupported LR scheduler, {}".format(name))
	return None, True

# REF [function] >> train_text_recognizer() in ${SWLP_HOME}/app/text/run_text_recognition_pl.py
def train(config, model, train_dataloader, test_dataloader, output_dir_path, model_filepath_to_load, logger=None):
	# Create a trainer.
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=os.path.join(output_dir_path, "checkpoints") if output_dir_path else None,
		filename="model-{epoch:03d}-{step:05d}-{val_acc:.5f}-{val_loss:.5f}",
		#monitor="val_acc", mode="max",
		monitor="val_loss", mode="min",
		save_top_k=5,
	)
	lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
	if config.get("swa", False):
		swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=0.1, swa_epoch_start=0.8, annealing_epochs=2, annealing_strategy="cos", avg_fn=None)
		pl_callbacks = [checkpoint_callback, lr_monitor_callback, swa_callback]
	else:
		pl_callbacks = [checkpoint_callback, lr_monitor_callback]
	tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=(output_dir_path + "/lightning_logs") if output_dir_path else "./lightning_logs", name="", version=None)
	pl_logger = [tensorboard_logger]

	if config.get("max_gradient_norm", None):
		gradient_clip_val = config["max_gradient_norm"]
		gradient_clip_algorithm = "norm"  # {"norm", "value"}.
	else:
		gradient_clip_val = None
		gradient_clip_algorithm = None
	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=config["epochs"], callbacks=pl_callbacks, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=output_dir_path)  # When using the default logger.
	trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=config["epochs"], callbacks=pl_callbacks, logger=pl_logger, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, default_root_dir=None)

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
		trainer.validate(model=model, dataloaders=test_dataloader, ckpt_path=None, verbose=True)
		if logger: logger.info("The model validated: {} secs.".format(time.time() - start_time))

	#--------------------
	#best_model_filepath = trainer.checkpoint_callback.best_model_path
	best_model_filepath = checkpoint_callback.best_model_path
	if logger: logger.info("The best trained model saved to {}.".format(best_model_filepath))
	if False:
		# Save the final model.
		final_model_filepath = os.path.join(output_dir_path, "final_model.ckpt") if output_dir_path else "./final_model.ckpt"
		trainer.save_checkpoint(final_model_filepath, weights_only=False)
		if logger: logger.info("The final trained model saved to {}.".format(final_model_filepath))

	return best_model_filepath

def validate(model, dataloader, logger=None):
	# Create a trainer.
	trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=-1, default_root_dir=None)

	# Validate the model.
	if logger: logger.info("Validating the model...")
	start_time = time.time()
	val_metrics = trainer.validate(model=model, dataloaders=dataloader, ckpt_path=None, verbose=True)
	if logger: logger.info("The model validated: {} secs.".format(time.time() - start_time))
	if logger: logger.info("Validation metrics: {}.".format(val_metrics))

def test(model, dataloader, logger=None):
	# Create a trainer.
	trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=-1, default_root_dir=None)

	# Test the model.
	if logger: logger.info("Testing the model...")
	start_time = time.time()
	test_metrics = trainer.test(model=model, dataloaders=dataloader, ckpt_path=None, verbose=True)
	if logger: logger.info("The model tested: {} secs.".format(time.time() - start_time))
	if logger: logger.info("Test metrics: {}.".format(test_metrics))

	# Predict.
	if logger: logger.info("Predicting...")
	start_time = time.time()
	predictions = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None, return_predictions=None)  # A list of [batch size, feature dim or #classes]'s.
	predictions = torch.vstack(predictions)
	if logger: logger.info("Predicted: {} secs.".format(time.time() - start_time))
	if logger: logger.info("Prediction: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(predictions.shape, predictions.dtype, torch.min(predictions), torch.max(predictions)))

def infer(model, data_iter, use_projector=False, use_predictor=False, device="cuda"):
	from tqdm import tqdm

	model = model.to(device)

	model.eval()
	model.freeze()
	with torch.no_grad():
		predictions = list()
		for inputs in tqdm(data_iter):
			predictions.append(model(inputs.to(device), use_projector, use_predictor).cpu().numpy())  # [batch size, feature dim].
		predictions = np.vstack(predictions)

	return predictions
