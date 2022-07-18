import typing, math, collections, copy, time
import numpy as np
import torch
import pytorch_lightning as pl

# REF [site] >> https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
class ByolModule(pl.LightningModule):
	def __init__(self, encoder, projector, predictor, augmenter1, augmenter2, moving_average_decay, is_momentum_encoder_used, is_model_initialized=False, is_all_model_params_optimized=True, logger=None):
		super().__init__()
		#self.save_hyperparameters()  # UserWarning: Attribute 'encoder' is an instance of 'nn.Module' and is already saved during checkpointing.
		self.save_hyperparameters(ignore=['encoder', 'projector', 'predictor' , 'augmenter1', 'augmenter2'])

		self.online_model = torch.nn.Sequential(encoder, projector)
		self.online_predictor = predictor
		self.target_model = None
		self.moving_average_decay = moving_average_decay
		self.is_momentum_encoder_used = is_momentum_encoder_used

		self.augmenter1 = augmenter1
		self.augmenter2 = augmenter2

		self.is_all_model_params_optimized = is_all_model_params_optimized
		self._logger = logger

		if is_model_initialized:
			# Initialize model weights.
			for name, param in self.online_model.named_parameters():
				try:
					if 'bias' in name:
						torch.nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						torch.nn.init.kaiming_normal_(param)
				except Exception as ex:  # For batch normalization.
					if 'weight' in name:
						param.data.fill_(1)
					continue

	"""
	def load_model(self, model_filepath):
		model_dict = torch.load(model_filepath)

		self.online_model.load_state_dict(model_dict['model_state_dict'])
		self.online_predictor.load_state_dict(model_dict['predictor_state_dict'])
		self.target_model.load_state_dict(model_dict['target_model_state_dict'])
		#self.augmenter1 = model_dict['augmenter1']
		#self.augmenter2 = model_dict['augmenter2']

	def save_model(self, model_filepath):
		torch.save({
			'model_state_dict': self.online_model.state_dict(),
			'predictor_state_dict': self.online_predictor.state_dict(),
			'target_model_state_dict': self.target_model.state_dict(),
			#'augmenter1': self.augmenter1,
			#'augmenter2': self.augmenter2,
		}, model_filepath)
	"""

	def on_load_checkpoint(self, checkpoint: typing.Dict[str, typing.Any]) -> None:
		self.online_model = checkpoint['model']
		self.online_predictor = checkpoint['predictor']
		self.target_model = checkpoint['target_model']
		#self.augmenter1 = checkpoint['augmenter1']
		#self.augmenter2 = checkpoint['augmenter2']

	def on_save_checkpoint(self, checkpoint: typing.Dict[str, typing.Any]) -> None:
		checkpoint['model'] = self.online_model
		checkpoint['predictor'] = self.online_predictor
		checkpoint['target_model'] = self.target_model
		#checkpoint['augmenter1'] = self.augmenter1
		#checkpoint['augmenter2'] = self.augmenter2

	def configure_optimizers(self):
		if self.is_all_model_params_optimized:
			model_params = list(self.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, self.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, self.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			if self.trainer and self.trainer.is_global_zero and self._logger: self._logger.info('#trainable model parameters = {}.'.format(num_model_params))
			#if self.trainer and self.trainer.is_global_zero and self._logger: self._logger.info('Trainable model parameters:')
			#if self.trainer and self.trainer.is_global_zero and self._logger: [self._logger.info(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, self.named_parameters())]

		#optimizer = torch.optim.SGD(model_params, lr=3e-4, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
		optimizer = torch.optim.Adam(model_params, lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

		#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0, last_epoch=-1)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0, last_epoch=-1)
		#scheduler = None

		if scheduler:
			return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
		else:
			return optimizer

	def forward(self, x, use_projector=False, use_predictor=False, *args, **kwargs):
		x = self.online_model(x) if use_projector else self.online_model[0](x)
		if use_predictor:
			x = self.online_predictor(x)
		return x

	def training_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		step_time = time.time() - start_time

		if self.is_momentum_encoder_used and self.target_model is not None:
			#tau = self._update_tau(step, max_steps, tau_lower=self.moving_average_decay, tau_upper=1.0)  # TODO [check] >>
			self._update_target_model(tau=self.moving_average_decay)

		self.log_dict(
			{'train_loss': loss, 'train_time': step_time},
			on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True
		)

		return loss

	def validation_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict({'val_loss': loss, 'val_time': step_time}, rank_zero_only=True)

	def test_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict({'test_loss': loss, 'test_time': step_time}, rank_zero_only=True)

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		return self(batch[0])  # Calls forward().

	def on_train_epoch_start(self):
		if self.trainer and self.trainer.is_global_zero and self._logger:
			if self.lr_schedulers():
				learning_rate = [scheduler.get_last_lr() for scheduler in self.lr_schedulers() if scheduler is not None] if isinstance(self.lr_schedulers(), collections.abc.Iterable) else self.lr_schedulers().get_last_lr()
				self._logger.info('Epoch {}/{}: Learning rate = {}.'.format(self.current_epoch, self.trainer.max_epochs, learning_rate))
			else:
				self._logger.info('Epoch {}/{}.'.format(self.current_epoch, self.trainer.max_epochs))

	def on_train_epoch_end(self):
		if self.trainer and self.trainer.is_global_zero and self._logger: self._logger.info('Epoch {}/{} done.'.format(self.current_epoch, self.trainer.max_epochs))

	def _shared_step(self, batch, batch_idx):
		x, _ = batch

		x1, x2 = self.augmenter1(x), self.augmenter2(x)

		z1_online = self.online_model(x1)
		z2_online = self.online_model(x2)
		z1_online = self.online_predictor(z1_online)
		z2_online = self.online_predictor(z2_online)

		with torch.no_grad():
			target_model = self._get_target_model() if self.is_momentum_encoder_used else self.online_model
			z1_target = target_model(x1)
			z2_target = target_model(x2)
			z1_target.detach_()
			z2_target.detach_()

		# TODO [check] >> Are z1_target.detach() & z2_target.detach() required?
		loss1 = self._compute_loss(z1_online, z2_target.detach())  # Stop gradient.
		loss2 = self._compute_loss(z2_online, z1_target.detach())  # Stop gradient.

		loss = loss1 + loss2

		return loss.mean()

	@torch.no_grad()
	def _update_target_model(self, tau):
		for current_params, ma_params in zip(self.online_model.parameters(), self.target_model.parameters()):
			# Exponential moving average.
			ma_params.data = current_params.data if ma_params.data is None else (tau * ma_params.data + (1 - tau) * current_params.data)

	def _get_target_model(self):
		if self.target_model is None:
			self.target_model = copy.deepcopy(self.online_model)
			self._set_requires_grad(self.target_model, False)
		return self.target_model

	def _reset_moving_average(self):
		del self.target_model
		self.target_model = None

	@staticmethod
	def _compute_loss(x1, x2):
		x1 = torch.nn.functional.normalize(x1, dim=-1, p=2)
		x2 = torch.nn.functional.normalize(x2, dim=-1, p=2)
		return 2 - 2 * (x1 * x2).sum(dim=-1)

	@staticmethod
	def _set_requires_grad(model, val):
		for p in model.parameters():
			p.requires_grad = val

	@staticmethod
	def _update_tau(step, max_steps, tau_lower=0.996, tau_upper=1.0):
		return tau_upper - (tau_upper - tau_lower) * (math.cos(math.pi * step / max_steps) + 1) / 2
