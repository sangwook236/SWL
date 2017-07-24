import numpy as np
#from scipy import ndimage, misc
from PIL import Image
import os, re
from .dataset import Dataset

class DataLoader(object):
	def __init__(self, width=0, height=0):
		self.width = width
		self.height = height

	# REF [file] >> load_images_by_pil(), load_labels_by_pil(), load_images_by_scipy(), and load_labels_by_scipy() in ${SWL_PYTHON_HOME}/src/swl/image_processing/util.py
	def load(self, data_dir_path, label_dir_path=None, data_suffix='', data_extension='png', label_suffix='', label_extension='png'):
		data = []
		labels = []
		if None == label_dir_path:
			for root, dirnames, filenames in os.walk(data_dir_path):
				for filename in filenames:
					if re.search(label_suffix + "\." + label_extension + "$", filename):
						filepath = os.path.join(root, filename)
						# Use scipy.
						#label = np.uint16(ndimage.imread(filepath, flatten=True))  # Gray image.
						##label = np.uint16(ndimage.imread(filepath))  # Do not correctly work.
						#if (self.height > 0 and label.shape[0] != self.height) or (self.width > 0 and label.shape[1] != self.width):
						#	#labels.append(misc.imresize(label, (height, width)))  # Do not work.
						#	labels.append(misc.imresize(label, (height, width), interp='nearest'))  # Do not correctly work.
						#else:
						#	labels.append(label)
						# Use PIL.
						label = Image.open(filepath)
						if (self.height > 0 and label.size[1] != self.height) or (self.width > 0 and label.size[0] != self.width):
							labels.append(np.asarray(label.resize((self.width, self.height), resample=Image.NEAREST)))
						else:
							labels.append(np.asarray(label))
					elif re.search(data_suffix + "\." + data_extension + "$", filename):
						filepath = os.path.join(root, filename)
						# Use scipy.
						#image = ndimage.imread(filepath, mode="RGB")  # RGB image.
						##image = ndimage.imread(filepath)
						#if (self.height > 0 and image.shape[0] != self.height) or (self.width > 0 and image.shape[1] != self.width):
						#	data.append(misc.imresize(image, (self.height, self.width)))
						#else:
						#	data.append(image)
						# Use PIL.
						image = Image.open(filepath)
						if (self.height > 0 and image.size[1] != self.height) or (self.width > 0 and image.size[0] != self.width):
							data.append(np.asarray(image.resize((self.width, self.height), resample=Image.NEAREST)))
						else:
							data.append(np.asarray(image))
		else:
			for root, dirnames, filenames in os.walk(data_dir_path):
				for filename in filenames:
					if re.search(data_suffix + "\." + data_extension + "$", filename):
						filepath = os.path.join(root, filename)
						# Use scipy.
						#image = ndimage.imread(filepath, mode="RGB")  # RGB image.
						##image = ndimage.imread(filepath)
						#if (self.height > 0 and image.shape[0] != self.height) or (self.width > 0 and image.shape[1] != self.width):
						#	data.append(misc.imresize(image, (self.height, self.width)))
						#else:
						#	data.append(image)
						# Use PIL.
						image = Image.open(filepath)
						if (self.height > 0 and image.size[1] != self.height) or (self.width > 0 and image.size[0] != self.width):
							data.append(np.asarray(image.resize((self.width, self.height), resample=Image.NEAREST)))
						else:
							data.append(np.asarray(image))
			for root, dirnames, filenames in os.walk(label_dir_path):
				for filename in filenames:
					if re.search(label_suffix + "\." + label_extension + "$", filename):
						filepath = os.path.join(root, filename)
						# Use scipy.
						#label = np.uint16(ndimage.imread(filepath, flatten=True))  # Gray image.
						##label = np.uint16(ndimage.imread(filepath))  # Do not correctly work.
						#if (self.height > 0 and label.shape[0] != self.height) or (self.width > 0 and label.shape[1] != self.width):
						#	#labels.append(misc.imresize(label, (self.height, self.width)))  # Do not work.
						#	labels.append(misc.imresize(label, (self.height, self.width), interp='nearest'))  # Do not correctly work.
						#else:
						#	labels.append(label)
						# Use PIL.
						label = Image.open(filepath)
						if (self.height > 0 and label.size[1] != self.height) or (self.width > 0 and label.size[0] != self.width):
							labels.append(np.asarray(label.resize((self.width, self.height), resample=Image.NEAREST)))
						else:
							labels.append(np.asarray(label))
		return Dataset(data = np.array(data), labels = np.array(labels))
