import numpy as np
from scipy import ndimage, misc
import os, re

class ImageLoader():
	def __init__(self, width=0, height=0):
		self.width = width
		self.height = height

	def load(self, dir_path, img_suffix, img_extension, label_suffix, label_extension):
		images = []
		labels = []
		for root, dirnames, filenames in os.walk(dir_path):
			for filename in filenames:
				if re.search(img_suffix + "\." + img_extension + "$", filename):
					filepath = os.path.join(root, filename)
					image = ndimage.imread(filepath, mode="RGB")
					#image = ndimage.imread(filepath)
					if (self.height > 0 and image.shape[0] != self.height) or (self.width > 0 and image.shape[1] != self.width):
						images.append(misc.imresize(image, (self.height, self.width)))
					else:
						images.append(image)
				if re.search(label_suffix + "\." + label_extension + "$", filename):
					filepath = os.path.join(root, filename)
					label = np.uint16(ndimage.imread(filepath, flatten=True))
					#label = np.uint16(ndimage.imread(filepath))  # Do not correctly work.
					if (self.height > 0 and label.shape[0] != self.height) or (self.width > 0 and label.shape[1] != self.width):
						labels.append(misc.imresize(label, (self.height, self.width)))
					else:
						labels.append(label)
		return np.array(images), np.array(labels)
