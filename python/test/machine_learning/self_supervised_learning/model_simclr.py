import typing, collections, time
import numpy as np
import torch
import pytorch_lightning as pl
import utils

# REF [site] >> https://github.com/NightShade99/Self-Supervised-Vision/blob/88f221d2bcba846f99f47d7b7b407af5c3bbbb7c/utils/losses.py
class SimclrLoss(torch.nn.Module):
	def __init__(self, normalize=False, temperature=1.0):
		super().__init__()

		self.normalize = normalize
		self.temperature = temperature

	def forward(self, zi, zj, device):
		# NOTE [caution] >> This loss is different from the one in the original paper, "A Simple Framework for Contrastive Learning of Visual Representations"

		bs = zi.shape[0]
		labels = torch.zeros((2 * bs,), dtype=torch.long, device=device)
		mask = torch.ones((bs, bs), dtype=torch.bool).fill_diagonal_(0)

		if self.normalize:
			zi_norm = torch.nn.functional.normalize(zi, p=2, dim=-1)
			zj_norm = torch.nn.functional.normalize(zj, p=2, dim=-1)
		else:
			zi_norm = zi
			zj_norm = zj

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
		loss = torch.nn.functional.cross_entropy(logits, labels)
		return loss

class SimclrModule(pl.LightningModule):
	def __init__(self, config, encoder, projector, augmenter1, augmenter2, is_model_initialized=True, logger=None):
		super().__init__()
		#self.save_hyperparameters()  # UserWarning: Attribute 'encoder' is an instance of 'nn.Module' and is already saved during checkpointing.
		self.save_hyperparameters(ignore=['encoder', 'projector', 'augmenter1', 'augmenter2', 'logger'])

		self.config = config
		self.model = torch.nn.Sequential(encoder, projector)
		self.augmenter1 = augmenter1
		self.augmenter2 = augmenter2
		self._logger = logger

		self.criterion = SimclrLoss(**config['loss']) if config else None

		#-----
		if is_model_initialized:
			# Initialize model weights.
			for name, param in self.model.named_parameters():
				try:
					if 'bias' in name:
						torch.nn.init.constant_(param, 0.0)
					elif 'weight' in name:
						torch.nn.init.kaiming_normal_(param)
				except Exception as ex:  # For batch normalization.
					if 'weight' in name:
						param.data.fill_(1)
					continue
			'''
			for param in self.model.parameters():
				if param.dim() > 1:
					torch.nn.init.xavier_uniform_(param)  # Initialize parameters with Glorot / fan_avg.
			'''

	"""
	def load_model(self, model_filepath):
		model_dict = torch.load(model_filepath)

		self.model.load_state_dict(model_dict['model_state_dict'])
		#self.augmenter1 = model_dict['augmenter1']
		#self.augmenter2 = model_dict['augmenter2']

	def save_model(self, model_filepath):
		torch.save({
			'model_state_dict': self.model.state_dict(),
			#'augmenter1': self.augmenter1,
			#'augmenter2': self.augmenter2,
		}, model_filepath)
	"""

	def on_load_checkpoint(self, checkpoint: typing.Dict[str, typing.Any]) -> None:
		self.model = checkpoint['model']
		#self.augmenter1 = checkpoint['augmenter1']
		#self.augmenter2 = checkpoint['augmenter2']

	def on_save_checkpoint(self, checkpoint: typing.Dict[str, typing.Any]) -> None:
		checkpoint['model'] = self.model
		#checkpoint['augmenter1'] = self.augmenter1
		#checkpoint['augmenter2'] = self.augmenter2

	def configure_optimizers(self):
		if self.config.get('is_all_model_params_optimized', True):
			model_params = list(self.parameters())
		else:
			# Filter model parameters only that require gradients.
			#model_params = filter(lambda p: p.requires_grad, self.parameters())
			model_params, num_model_params = list(), 0
			for p in filter(lambda p: p.requires_grad, self.parameters()):
				model_params.append(p)
				num_model_params += np.prod(p.size())
			if self.trainer and self.trainer.is_global_zero and self._logger:
				self._logger.info('#trainable model parameters = {}.'.format(num_model_params))
				#self._logger.info('Trainable model parameters: {}.'.format([(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, self.named_parameters())]))

		optimizer = utils.construct_optimizer(self.config['optimizer'], model_params)
		scheduler, is_epoch_based = utils.construct_lr_scheduler(self.config.get('lr_scheduler', None), optimizer, self.trainer.max_epochs)
		#scheduler, is_epoch_based = utils.construct_lr_scheduler(self.config.get('lr_scheduler', None), optimizer, self.config['epochs'])

		if scheduler:
			return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch' if is_epoch_based else 'step'}]
		else:
			return optimizer

	#def forward(self, x, use_projector=False):
	def forward(self, x, use_projector=False, use_predictor=False):
		x = self.model(x) if use_projector else self.model[0](x)
		return x

	def training_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		step_time = time.time() - start_time

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

		z1 = self.model(x1)
		z2 = self.model(x2)

		loss = self.criterion(z1, z2, self.device)

		return loss.mean()
