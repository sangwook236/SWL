import os, sys, time, random, pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# REF [site] >>
# 	https://github.com/fastai/imagenet-fast/blob/master/cifar10/models/cifar10/utils.py
#	https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class RecorderMeter(object):
	"""Computes and stores the minimum loss value and its epoch index"""
	def __init__(self, total_epoch):
		self.reset(total_epoch)

	def reset(self, total_epoch):
		assert total_epoch > 0
		self.total_epoch = total_epoch
		self.current_epoch = 0
		self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
		self.epoch_losses = self.epoch_losses

		self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
		self.epoch_accuracy = self.epoch_accuracy

	def resize(self, total_epoch):
		assert total_epoch > self.total_epoch
		epoch_losses = np.zeros((total_epoch, 2), dtype=np.float32) # [epoch, train/val]
		epoch_losses[:self.total_epoch] = self.epoch_losses
		self.epoch_losses = epoch_losses

		epoch_accuracy = np.zeros((total_epoch, 2), dtype=np.float32) # [epoch, train/val]
		epoch_accuracy[:self.total_epoch] = self.epoch_accuracy
		self.epoch_accuracy = epoch_accuracy

		self.total_epoch = total_epoch
		#self.current_epoch = 0

	def update(self, idx, train_loss, train_acc, val_loss, val_acc):
		assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
		self.epoch_losses[idx, 0] = train_loss
		self.epoch_losses[idx, 1] = val_loss
		self.epoch_accuracy[idx, 0] = train_acc
		self.epoch_accuracy[idx, 1] = val_acc
		self.current_epoch = idx + 1
		return self.max_accuracy(False) == val_acc

	def max_accuracy(self, istrain):
		if self.current_epoch <= 0: return 0
		if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
		else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

	def plot_curve(self, save_path):
		title = 'The accuracy/loss curve of train/val'
		dpi = 80  
		width, height = 1200, 800
		legend_fontsize = 10
		scale_distance = 48.8
		figsize = width / float(dpi), height / float(dpi)

		#fig = plt.figure(figsize=figsize)
		fig, ax1 = plt.subplots(figsize=figsize)
		x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
		y_axis = np.zeros(self.total_epoch)

		plt.xlim(0, self.total_epoch)
		plt.ylim(0, 100)
		interval_y = 5
		interval_x = 5
		plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
		plt.yticks(np.arange(0, 100 + interval_y, interval_y))
		plt.grid()
		plt.title(title, fontsize=20)

		plt.xlabel('Training epoch', fontsize=16)
		ax1.set_ylabel('Accuracy', fontsize=16)

		y_axis[:] = self.epoch_accuracy[:, 0]
		ax1.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
		ax1.legend(loc=3, fontsize=legend_fontsize)

		y_axis[:] = self.epoch_accuracy[:, 1]
		ax1.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
		ax1.legend(loc=3, fontsize=legend_fontsize)

		ax2 = ax1.twinx()
		ax2.set_ylabel('Loss', fontsize=16)

		y_axis[:] = self.epoch_losses[:, 0]
		ax2.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss', lw=2)
		ax2.legend(loc=4, fontsize=legend_fontsize)

		y_axis[:] = self.epoch_losses[:, 1]
		ax2.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss', lw=2)
		ax2.legend(loc=4, fontsize=legend_fontsize)

		if save_path is not None:
			fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
			print('---- save figure {} into {}'.format(title, save_path))
		plt.close(fig)

def time_string():
	ISOTIMEFORMAT='%Y-%m-%d %X'
	string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
	return string

def convert_secs2time(epoch_time):
	need_hour = int(epoch_time / 3600)
	need_mins = int((epoch_time - 3600*need_hour) / 60)
	need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
	return need_hour, need_mins, need_secs

def time_file_str():
	ISOTIMEFORMAT='%Y-%m-%d'
	string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
	return string + '-{}'.format(random.randint(1, 10000))

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def print_log(print_string, log):
	print('{}'.format(print_string))
	log.write('{}\n'.format(print_string))
	log.flush()

def adjust_learning_rate(optimizer, epoch, initial_learning_rate, gammas, schedule):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = initial_learning_rate
	assert len(gammas) == len(schedule), 'Length of gammas and schedule should be equal.'
	for (gamma, step) in zip(gammas, schedule):
		if (epoch >= step):
			lr = lr * gamma
		else:
			break
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def plotting(exp_dir, log_file):
	# Load the training log dictionary:
	train_dict = pickle.load(open(log_file, 'rb'))

	###########################################################
	### Make the vanilla train and test loss per epoch plot ###
	###########################################################

	plt.plot(np.asarray(train_dict['train_loss']), label='train_loss')
	plt.plot(np.asarray(train_dict['val_loss']), label='val_loss')
		
	#plt.ylim(0, 2000)
	plt.xlabel('Evaluation step')
	plt.ylabel('Loss')
	plt.tight_layout()
	plt.legend(loc='upper right')
	plt.savefig(os.path.join(exp_dir, 'loss.png'))
	plt.clf()

	## accuracy###
	plt.plot(np.asarray(train_dict['train_acc']), label='train_acc')
	plt.plot(np.asarray(train_dict['val_acc']), label='val_acc')
		
	#plt.ylim(0, 2000)
	plt.xlabel('Evaluation step')
	plt.ylabel('Accuracy')
	plt.tight_layout()
	plt.legend(loc='lower right')
	plt.savefig(os.path.join(exp_dir, 'acc.png'))
	plt.clf()
