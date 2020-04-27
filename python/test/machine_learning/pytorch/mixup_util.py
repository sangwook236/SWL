import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py

def to_one_hot(inp,num_classes):
	y_onehot = torch.FloatTensor(inp.size(0), num_classes)
	y_onehot.zero_()

	y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

	return Variable(y_onehot.cuda(),requires_grad=False)


def mixup_process(out, target_reweighted, lam):
	indices = np.random.permutation(out.size(0))
	out = out*lam + out[indices]*(1-lam)
	target_shuffled_onehot = target_reweighted[indices]
	target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

	#t1 = target.data.cpu().numpy()
	#t2 = target[indices].data.cpu().numpy()
	#print (np.sum(t1==t2))
	return out, target_reweighted


def mixup_data(x, y, alpha):

	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	batch_size = x.size()[0]
	index = torch.randperm(batch_size).cuda()
	mixed_x = lam * x + (1 - lam) * x[index,:]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def get_lambda(alpha=1.0):
	'''Return lambda'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	return lam

class Cutout(object):
	"""Randomly mask out one or more patches from an image.
	Args:
		n_holes (int): Number of patches to cut out of each image.
		length (int): The length (in pixels) of each square patch.
	"""
	def __init__(self, n_holes, length):
		self.n_holes = n_holes
		self.length = length

	def apply(self, img):
		"""
		Args:
			img (Tensor): Tensor image of size (C, H, W).
		Returns:
			Tensor: Image with n_holes of dimension length x length cut out of it.
		"""
		h = img.size(2)
		w = img.size(3)

		mask = np.ones((h, w), np.float32)

		for n in range(self.n_holes):
			y = np.random.randint(h)
			x = np.random.randint(w)

			y1 = int(np.clip(y - self.length / 2, 0, h))
			y2 = int(np.clip(y + self.length / 2, 0, h))
			x1 = int(np.clip(x - self.length / 2, 0, w))
			x2 = int(np.clip(x + self.length / 2, 0, w))

			mask[y1: y2, x1: x2] = 0.

		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img).cuda()
		img = img * mask

		return img


def create_val_folder(data_set_path):
	"""
	Used for Tiny-imagenet dataset
	Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
	This method is responsible for separating validation images into separate sub folders,
	so that test and val data can be read by the pytorch dataloaders
	"""
	path = os.path.join(data_set_path, 'val/images')  # path where validation data is present now
	filename = os.path.join(data_set_path, 'val/val_annotations.txt')  # file where image2class mapping is present
	fp = open(filename, "r")  # open file in read mode
	data = fp.readlines()  # read line by line

	# Create a dictionary with image names as key and corresponding classes as values
	val_img_dict = {}
	for line in data:
		words = line.split("\t")
		val_img_dict[words[0]] = words[1]
	fp.close()

	# Create folder if not present, and move image into proper folder
	for img, folder in val_img_dict.items():
		newpath = (os.path.join(path, folder))
		if not os.path.exists(newpath):  # check if folder exists
			os.makedirs(newpath)

		if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
			os.rename(os.path.join(path, img), os.path.join(newpath, img))

if __name__ == "__main__":
	create_val_folder('data/tiny-imagenet-200')  # Call method to create validation image folders

