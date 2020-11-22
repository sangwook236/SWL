import random
import numpy as np
import torch
import torchvision

# REF [site] >> https://github.com/pytorch/vision/blob/master/references/detection/transforms.py

def _flip_coco_person_keypoints(kps, width):
	flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
	flipped_data = kps[:, flip_inds]
	flipped_data[..., 0] = width - flipped_data[..., 0]
	# Maintain COCO convention that if visibility == 0, then x, y = 0
	inds = flipped_data[..., 2] == 0
	flipped_data[inds] = 0
	return flipped_data

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, image, target):
		for t in self.transforms:
			image, target = t(image, target)
		return image, target

class ToTensor(object):
	def __call__(self, image, target):
		image = torchvision.transforms.functional.to_tensor(image)
		return image, target

class Normalize(object):
	def __init__(self, mean, std, inplace=False):
		self.normalize = torchvision.transforms.Normalize(mean, std, inplace)

	def __call__(self, image, target):
		image = self.normalize(image)
		return image, target

class ConvertPILMode(object):
	def __init__(self, mode='RGB'):
		self.mode = mode

	def __call__(self, image, target):
		return image.convert(self.mode), target

class RandomHorizontalFlip(object):
	def __init__(self, prob):
		self.prob = prob

	def __call__(self, image, target):
		if random.random() < self.prob:
			height, width = image.shape[-2:]
			image = image.flip(-1)
			bbox = target['boxes']
			bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
			target['boxes'] = bbox
			if 'masks' in target:
				target['masks'] = target['masks'].flip(-1)
			if 'keypoints' in target:
				keypoints = target['keypoints']
				keypoints = _flip_coco_person_keypoints(keypoints, width)
				target['keypoints'] = keypoints
		return image, target

class AugmentByImgaug(object):
	def __init__(self, augmenter, is_pil=True):
		self.augmenter = augmenter
		self.is_pil = is_pil

	def __call__(self, image, target):
		if self.is_pil: image = np.array(image)
		if 'boxes' in target:
			boxes = target['boxes']
			boxes = np.expand_dims(boxes, axis=0)
		else: boxes = None
		if 'keypoints' in target:
			#keypoints0 = target['keypoints'].numpy()
			keypoints0 = target['keypoints']
			keypoints = keypoints0.reshape(-1, keypoints0.shape[-1])
			keypoints = np.expand_dims(keypoints[...,:2], axis=0)
		else: keypoints = None

		image, boxes, keypoints = self.augmenter(image=image, bounding_boxes=boxes, keypoints=keypoints, return_batch=False)
		#image, boxes, keypoints = self.augmenter(images=images, bounding_boxes=boxes, keypoints=keypoints, return_batch=True)

		if self.is_pil:
			import PIL.Image
			image = PIL.Image.fromarray(image)
		if 'boxes' in target:
			target['boxes'] = np.squeeze(boxes, axis=0)
		if 'keypoints' in target:
			keypoints0[...,:2] = np.squeeze(keypoints, axis=0).reshape(-1, 4, keypoints.shape[-1])
			target['keypoints'] = keypoints0
		return image, target
