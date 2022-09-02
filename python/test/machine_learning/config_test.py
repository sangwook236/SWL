#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../../src")

import random, math, time
import numpy as np
import yaml

def pytorch_ml_config_test():
	import torch, torchvision
	import PIL.Image, PIL.ImageOps, PIL.ImageEnhance

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
				aug_choices = random.choices(self.aug_list, k=self.n_aug)
				for aug, min_value, max_value in aug_choices:
					v = random.uniform(min_value, max_value)
					if aug == "identity":
						pass
					elif aug == "autocontrast":
						img = PIL.ImageOps.autocontrast(img)
					elif aug == "equalize":
						img = PIL.ImageOps.equalize(img)
					elif aug == "rotate":
						if random.random() > 0.5:
							v = -v
						img = img.rotate(v)
					elif aug == "solarize":
						img = PIL.ImageOps.solarize(img, v)
					elif aug == "color":
						img = PIL.ImageEnhance.Color(img).enhance(v)
					elif aug == "contrast":
						img = PIL.ImageEnhance.Contrast(img).enhance(v)
					elif aug == "brightness":
						img = PIL.ImageEnhance.Brightness(img).enhance(v)
					elif aug == "sharpness":
						img = PIL.ImageEnhance.Sharpness(img).enhance(v)
					elif aug == "shear_x":
						if random.random() > 0.5:
							v = -v
						img = img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
					elif aug == "shear_y":
						if random.random() > 0.5:
							v = -v
						img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
					elif aug == "translate_x":
						if random.random() > 0.5:
							v = -v
						v = v * img.size[0]
						img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
					elif aug == "translate_y":
						if random.random() > 0.5:
							v = -v
						v = v * img.size[1]
						img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
					elif aug == "posterize":
						img = PIL.ImageOps.posterize(img, int(v))
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

	# REF [class] >> NoamLR class in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_optimization.py
	class NoamLR(torch.optim.lr_scheduler._LRScheduler):
		def __init__(self, optimizer, dim_feature, warmup_steps, factor=1, last_step=-1):
			#self.optimizer = optimizer
			self.dim_feature = dim_feature
			self.warmup_steps = warmup_steps
			self.factor = factor
			self.last_step = last_step
			super().__init__(optimizer, last_step)

			"""
			# Initialize step and base learning rates.
			if last_step == -1:
				for group in optimizer.param_groups:
					group.setdefault("initial_lr", group["lr"])
			else:
				for i, group in enumerate(optimizer.param_groups):
					if "initial_lr" not in group:
						raise KeyError("param "initial_lr" is not specified in param_groups[{}] when resuming an optimizer".format(i))
			self.base_lrs = list(map(lambda group: group["initial_lr"], optimizer.param_groups))
			"""

		def get_lr(self):
			if self.last_step == -1 or self.last_step == 0:
				return self.base_lrs
			else:
				lr = self.factor * (self.dim_feature**(-0.5) * min(self.last_step**(-0.5), self.last_step * self.warmup_steps**(-1.5)))
				#return [base_lr + lr for base_lr in self.base_lrs]
				#return [base_lr * lr for base_lr in self.base_lrs]
				return [lr for _ in self.base_lrs]

		def step(self, step=None):
			if step is None:
				self.last_step += 1
			else:
				self.last_step = step

			for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
				param_group["lr"] = lr

			self._last_lr = [param_group["lr"] for param_group in self.optimizer.param_groups]

	def construct_lr_scheduler(config, optimizer, num_epochs, *args, **kwargs):
		if not config:
			return None, True

		LR_SCHEDULERS = {
			"step": torch.optim.lr_scheduler.StepLR,
			"multi_step": torch.optim.lr_scheduler.MultiStepLR,
			"cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
			"cosine_warmup": CosineAnnealingWarmupLR,
			"cosine_restart": CosineAnnealingWarmUpRestartsLR,
			"noam": NoamLR,  # For transformer. Step-based LR scheduler.
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

	#--------------------
	config_filepath = "./config/pytorch_ml_config.yaml"

	print("Loading a PyTorch ML config from {}...".format(config_filepath))
	start_time = time.time()
	try:
		with open(config_filepath, encoding="utf-8") as fd:
			config = yaml.load(fd, Loader=yaml.Loader)
	except yaml.scanner.ScannerError as ex:
		print("yaml.scanner.ScannerError in {}: {}.".format(config_filepath, ex))
		return
	except UnicodeDecodeError as ex:
		print("Unicode decode error in {}: {}.".format(config_filepath, ex))
		return
	except FileNotFoundError as ex:
		print("File not found, {}: {}.".format(config_filepath, ex))
		return
	except Exception as ex:
		print("Exception raised in {}: {}.".format(config_filepath, ex))
		return
	print("A PyTorch ML config loaded: {} secs.".format(time.time() - start_time))

	#--------------------
	assert config["library"] == "pytorch", "The ML configuration for PyTorch library: Invalid library = {}".format(config["library"])

	if "transforms" in config:
		print("Transform --------------------------------------------------")
		print("Processing transforms...")
		start_time = time.time()
		transforms = construct_transform(config["transforms"])
		print("Transforms processed: {} secs.".format(time.time() - start_time))

		print("-----")
		print(transforms)

		print("-----")
		inputs = PIL.Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))  # (channel, height, width).
		outputs = transforms(inputs)
		print("Outputs: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(outputs.shape, outputs.dtype, torch.min(outputs), torch.max(outputs)))

	if "pretrained_model" in config:
		print("Pretrained model --------------------------------------------------")
		print("Processing Pretrained models...")
		start_time = time.time()
		pretrained_model, feature_dim = construct_pretrained_model(config["pretrained_model"], output_dim=None)
		print("Pretrained models processed: {} secs.".format(time.time() - start_time))

		print("-----")
		print(pretrained_model)
		print("Feature dimension = {}.".format(feature_dim))

		print("-----")
		inputs = torch.randn((5, 3, 244, 244), dtype=torch.float32)  # (batch size, image channel, image height, image width).
		outputs = pretrained_model(inputs)
		print("Inputs:  shape = {}, dtype = {}, (min, max) = ({}, {}).".format(inputs.shape, inputs.dtype, torch.min(inputs), torch.max(inputs)))
		print("Outputs: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(outputs.shape, outputs.dtype, torch.min(outputs), torch.max(outputs)))
		if isinstance(feature_dim, list) or isinstance(feature_dim, tuple):
			assert list(feature_dim) == list(outputs.shape[1:]), "The feature dimension of the pretrained model is not matched, {} != {}".format(feature_dim, list(outputs.shape[1:]))
		elif isinstance(feature_dim, int):
			assert [feature_dim] == list(outputs.shape[1:]), "The feature dimension of the pretrained model is not matched, {} != {}".format(feature_dim, list(outputs.shape[1:]))
		else:
			raise ValueError("Invalid feature dimension type, {}".format(type(feature_dim)))

	if "user_defined_model" in config:
		print("User-defined model --------------------------------------------------")
		print("Processing User-defined models...")
		start_time = time.time()
		user_defined_model, output_dim = construct_user_defined_model(config["user_defined_model"])
		print("User-defined models processed: {} secs.".format(time.time() - start_time))

		print("-----")
		print(user_defined_model)
		print("Output dimension = {}.".format(output_dim))

		print("-----")
		# For simple test.
		#inputs = torch.randn((5, 2048), dtype=torch.float32)  # (batch size, feature dim).
		# For LeNet5 + MNIST.
		inputs = torch.randn((5, 1, 28, 28), dtype=torch.float32)  # (batch size, image channel, image height, image width).
		outputs = user_defined_model(inputs)
		print("Inputs:  shape = {}, dtype = {}, (min, max) = ({}, {}).".format(inputs.shape, inputs.dtype, torch.min(inputs), torch.max(inputs)))
		print("Outputs: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(outputs.shape, outputs.dtype, torch.min(outputs), torch.max(outputs)))
		if isinstance(output_dim, list) or isinstance(output_dim, tuple):
			assert list(output_dim) == list(outputs.shape[1:]), "The output dimension of the user-defined model is not matched, {} != {}".format(output_dim, list(outputs.shape[1:]))
		elif isinstance(output_dim, int):
			assert [output_dim] == list(outputs.shape[1:]), "The output dimension of the user-defined model is not matched, {} != {}".format(output_dim, list(outputs.shape[1:]))
		else:
			raise ValueError("Invalid output dimension type, {}".format(type(output_dim)))

	if "optimizer" in config:
		model = torchvision.models.resnet18()

		print("Optimizer --------------------------------------------------")
		print("Processing Optimizers...")
		start_time = time.time()
		optimizer = construct_optimizer(config["optimizer"], model.parameters())
		print("Optimizers processed: {} secs.".format(time.time() - start_time))

		print("-----")
		print(optimizer)

		print("-----")
		num_steps = 100
		for _ in range(num_steps):
			optimizer.zero_grad()
			optimizer.step()
		print("Optimization done.")

	if "lr_scheduler" in config:
		model = torchvision.models.resnet18()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False)
		num_epochs = 100

		print("LR scheduler --------------------------------------------------")
		print("Processing LR schedulers...")
		start_time = time.time()
		lr_scheduler, is_epoch_based = construct_lr_scheduler(config["lr_scheduler"], optimizer, num_epochs)
		print("LR schedulers processed: {} secs.".format(time.time() - start_time))

		print("-----")
		print(lr_scheduler)
		print("Is epoch based = {}.".format(is_epoch_based))

		print("-----")
		num_steps = 10
		lrs = list()
		for _ in range(num_steps):
			lr_scheduler.optimizer.step()
			lrs.append(lr_scheduler.get_last_lr())
			lr_scheduler.step()
		print("LRs: {}.".format(lrs))

def main():
	pytorch_ml_config_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
