import math, collections, copy, time
import numpy as np
import torch
import pytorch_lightning as pl


# REF [site] >> https://github.com/NightShade99/Self-Supervised-Vision/blob/main/utils/losses.py
class RelicLoss(torch.nn.Module):
	def __init__(self, normalize=True, temperature=1.0, alpha=0.5):
		super().__init__()

		self.normalize = normalize
		self.temperature = temperature
		self.alpha = alpha

	def forward(self, zi, zj, z_orig):
		bs = zi.shape[0]
		labels = torch.zeros((2 * bs,)).long()
		mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

		if self.normalize:
			zi_norm = torch.nn.functional.normalize(zi, p=2, dim=-1)
			zj_norm = torch.nn.functional.normalize(zj, p=2, dim=-1)
			zo_norm = torch.nn.functional.normalize(z_orig, p=2, dim=-1)
		else:
			zi_norm = zi
			zj_norm = zj
			zo_norm = z_orig

		logits_ii = torch.mm(zi_norm, zi_norm.t()) / self.temperature
		logits_ij = torch.mm(zi_norm, zj_norm.t()) / self.temperature
		logits_ji = torch.mm(zj_norm, zi_norm.t()) / self.temperature
		logits_jj = torch.mm(zj_norm, zj_norm.t()) / self.temperature

		logits_ij_pos = logits_ij[torch.logical_not(mask)]  # Shape (N,).
		logits_ji_pos = logits_ji[torch.logical_not(mask)]  # Shape (N,).
		logits_ii_neg = logits_ii[mask].reshape(bs, -1)  # Shape (N, N - 1).
		logits_ij_neg = logits_ij[mask].reshape(bs, -1)  # Shape (N, N - 1).
		logits_ji_neg = logits_ji[mask].reshape(bs, -1)  # Shape (N, N - 1).
		logits_jj_neg = logits_jj[mask].reshape(bs, -1)  # Shape (N, N - 1).

		pos = torch.cat((logits_ij_pos, logits_ji_pos), dim=0).unsqueeze(1)  # Shape (2N, 1).
		neg_i = torch.cat((logits_ii_neg, logits_ij_neg), dim=1)  # Shape (N, 2N - 2).
		neg_j = torch.cat((logits_ji_neg, logits_jj_neg), dim=1)  # Shape (N, 2N - 2).
		neg = torch.cat((neg_i, neg_j), dim=0)  # Shape (2N, 2N - 2).

		logits = torch.cat((pos, neg), dim=1)  # Shape (2N, 2N - 1).
		contrastive_loss = torch.nn.functional.cross_entropy(logits, labels)

		logits_io = torch.mm(zi_norm, zo_norm.t()) / self.temperature
		logits_jo = torch.mm(zj_norm, zo_norm.t()) / self.temperature
		probs_io = torch.nn.functional.softmax(logits_io[torch.logical_not(mask)], -1)
		probs_jo = torch.nn.functional.log_softmax(logits_jo[torch.logical_not(mask)], -1)
		kl_div_loss = torch.nn.functional.kl_div(probs_io, probs_jo, log_target=True, reduction='sum')
		return contrastive_loss + self.alpha * kl_div_loss

class RelicModule(pl.LightningModule):
	def __init__(self, model, projector, predictor, moving_average_decay, use_momentum, augmenter1, augmenter2, is_model_initialized=False, is_all_model_params_optimized=True, logger=None):
		super().__init__()
		#self.save_hyperparameters()  # UserWarning: Attribute 'model' is an instance of 'nn.Module' and is already saved during checkpointing.
		self.save_hyperparameters(ignore=['model', 'projector', 'predictor' , 'augmenter1', 'augmenter2'])

		self.online_encoder = torch.nn.Sequential(model, projector)
		self.online_predictor = predictor
		self.target_encoder = None
		self.moving_average_decay = moving_average_decay
		self.use_momentum = use_momentum

		self.augmenter1 = augmenter1
		self.augmenter2 = augmenter2

		self.is_all_model_params_optimized = is_all_model_params_optimized
		self._logger = logger

		self._loss_fn = RelicLoss(normalize=True, temperature=1.0, alpha=0.5)

		if is_model_initialized:
			# Initialize model weights.
			for name, param in self.online_encoder.named_parameters():
				try:
					if 'bias' in name:
						torch.nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						torch.nn.init.kaiming_normal_(param)
				except Exception as ex:  # For batch normalization.
					if 'weight' in name:
						param.data.fill_(1)
					continue

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

		optimizer = torch.optim.SGD(model_params, lr=0.2, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
		#optimizer = torch.optim.Adam(model_params, lr=0.2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

		#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0, last_epoch=-1)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0, last_epoch=-1)

		if scheduler:
			return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
		else:
			return optimizer

	def forward(self, x, use_projector=False, use_predictor=False):
		x = self.online_encoder(x) if use_projector else self.online_encoder[0](x)
		if use_predictor:
			x = self.online_predictor(x)
		return x

	def training_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		step_time = time.time() - start_time

		if self.use_momentum and self.target_encoder is not None:
			#tau = self._update_tau(step, max_steps, tau_lower=self.moving_average_decay, tau_upper=1.0)  # TODO [check] >>
			self._update_target_encoder(tau=self.moving_average_decay)

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

		z = self.online_encoder(x)
		z1_online = self.online_encoder(x1)
		z2_online = self.online_encoder(x2)
		z1_online = self.online_predictor(z1_online)
		z2_online = self.online_predictor(z2_online)

		with torch.no_grad():
			target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
			z1_target = target_encoder(x1)
			z2_target = target_encoder(x2)
			z1_target.detach_()
			z2_target.detach_()

		# TODO [check] >> Are z1_target.detach() & z2_target.detach() required?
		loss1 = self._loss_fn(z1_online, z2_target.detach(), z)
		loss2 = self._loss_fn(z2_online, z1_target.detach(), z)

		loss = loss1 + loss2

		return loss.mean()

	@torch.no_grad()
	def _update_target_encoder(self, tau):
		for current_params, ma_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
			# Exponential moving average.
			ma_params.data = current_params.data if ma_params.data is None else (tau * ma_params.data + (1 - tau) * current_params.data)

	def _get_target_encoder(self):
		if self.target_encoder is None:
			self.target_encoder = copy.deepcopy(self.online_encoder)
			self._set_requires_grad(self.target_encoder, False)
		return self.target_encoder

	def _reset_moving_average(self):
		del self.target_encoder
		self.target_encoder = None

	@staticmethod
	def _set_requires_grad(model, val):
		for p in model.parameters():
			p.requires_grad = val

	@staticmethod
	def _update_tau(step, max_steps, tau_lower=0.996, tau_upper=1.0):
		return tau_upper - (tau_upper - tau_lower) * (math.cos(math.pi * step / max_steps) + 1) / 2
