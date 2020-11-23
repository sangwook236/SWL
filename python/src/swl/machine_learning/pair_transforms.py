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

	def __call__(self, input, target):
		for t in self.transforms:
			input, target = t(input, target)
		return input, target

class ToTensor(object):
	def __call__(self, input, target):
		input = torchvision.transforms.functional.to_tensor(input)
		return input, target

class Normalize(object):
	def __init__(self, mean, std, inplace=False):
		self.normalize = torchvision.transforms.Normalize(mean, std, inplace)

	def __call__(self, input, target):
		input = self.normalize(input)
		return input, target

class ConvertPILMode(object):
	def __init__(self, mode='RGB'):
		self.mode = mode

	def __call__(self, input, target):
		return input.convert(self.mode), target

#--------------------------------------------------------------------
# For object detection.

class ResizeToFixedSize(object):
	def __init__(self, height, width, warn_about_small_image=False, is_pil=True, logger=None):
		self.height, self.width = height, width
		self.resize_functor = self._resize_by_pil if is_pil else self._resize_by_opencv
		self.logger = logger

		self.min_height_threshold, self.min_width_threshold = 30, 30
		self.warn = self._warn_about_small_image if warn_about_small_image else lambda *args, **kwargs: None

	def __call__(self, input, target):
		return self.resize_functor(input, target, self.height, self.width)

	@staticmethod
	def _compute_scale_factor(canvas_height, canvas_width, image_height, image_width, max_scale_factor=3, re_scale_factor=0.5):
		h_scale_factor, w_scale_factor = canvas_height / image_height, canvas_width / image_width
		#scale_factor = min(h_scale_factor, w_scale_factor)
		scale_factor = min(h_scale_factor, w_scale_factor, max_scale_factor)
		#return scale_factor, scale_factor
		return max(scale_factor, min(h_scale_factor, re_scale_factor)), max(scale_factor, min(w_scale_factor, re_scale_factor))

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_opencv() in ${SWL_PYTHON_HOME}/test/language_processing/text_line_data.py.
	def _resize_by_opencv(self, image, target, canvas_height, canvas_width, *args, **kwargs):
		min_height, min_width = canvas_height // 2, canvas_width // 2

		image_height, image_width = image.shape[:2]
		self.warn(image_height, image_width)
		image_height, image_width = max(image_height, 1), max(image_width, 1)

		h_scale_factor, w_scale_factor = self._compute_scale_factor(canvas_height, canvas_width, image_height, image_width)

		#tgt_height, tgt_width = image_height, canvas_width
		tgt_height, tgt_width = int(image_height * h_scale_factor), int(image_width * w_scale_factor)
		#tgt_height, tgt_width = max(int(image_height * h_scale_factor), min_height), max(int(image_width * w_scale_factor), min_width)
		assert tgt_height > 0 and tgt_width > 0

		h_scale_factor, w_scale_factor = tgt_height / image_height, tgt_width / image_width
		if 'boxes' in target:
			target['boxes'] = target['boxes'] * [w_scale_factor, h_scale_factor, w_scale_factor, h_scale_factor]
		if 'area' in target:
			# TODO [check] >>
			target['area'] = target['area'] * h_scale_factor * w_scale_factor
		if 'keypoints' in target:
			target['keypoints'] = target['keypoints'] * [w_scale_factor, h_scale_factor, 1]

		import cv2
		zeropadded = np.zeros((canvas_height, canvas_width) + image.shape[2:], dtype=image.dtype)
		zeropadded[:tgt_height,:tgt_width] = cv2.resize(image, (tgt_width, tgt_height), interpolation=cv2.INTER_AREA)
		return zeropadded, target

	# REF [function] >> RunTimeTextLineDatasetBase._resize_by_pil() in ${SWL_PYTHON_HOME}/test/language_processing/text_line_data.py.
	def _resize_by_pil(self, image, target, canvas_height, canvas_width, *args, **kwargs):
		min_height, min_width = canvas_height // 2, canvas_width // 2

		image_width, image_height = image.size
		self.warn(image_height, image_width)
		image_height, image_width = max(image_height, 1), max(image_width, 1)

		h_scale_factor, w_scale_factor = self._compute_scale_factor(canvas_height, canvas_width, image_height, image_width)

		#tgt_height, tgt_width = image_height, canvas_width
		tgt_height, tgt_width = int(image_height * h_scale_factor), int(image_width * w_scale_factor)
		#tgt_height, tgt_width = max(int(image_height * h_scale_factor), min_height), max(int(image_width * w_scale_factor), min_width)
		assert tgt_height > 0 and tgt_width > 0

		h_scale_factor, w_scale_factor = tgt_height / image_height, tgt_width / image_width
		if 'boxes' in target:
			target['boxes'] = target['boxes'] * [w_scale_factor, h_scale_factor, w_scale_factor, h_scale_factor]
		if 'area' in target:
			# TODO [check] >>
			target['area'] = target['area'] * h_scale_factor * w_scale_factor
		if 'keypoints' in target:
			target['keypoints'] = target['keypoints'] * [w_scale_factor, h_scale_factor, 1]

		import PIL.Image
		zeropadded = PIL.Image.new(image.mode, (canvas_width, canvas_height), color=0)
		zeropadded.paste(image.resize((tgt_width, tgt_height), resample=PIL.Image.BICUBIC), (0, 0, tgt_width, tgt_height))
		return zeropadded, target

	def _warn_about_small_image(self, height, width):
		if height < self.min_height_threshold:
			if self.logger: self.logger.warning('Too small image: The image height {} should be larger than or equal to {}.'.format(height, self.min_height_threshold))
		#if width < self.min_width_threshold:
		#	if self.logger: self.logger.warning('Too small image: The image width {} should be larger than or equal to {}.'.format(width, self.min_width_threshold))

class RandomHorizontalFlip(object):
	def __init__(self, prob):
		self.prob = prob

	def __call__(self, input, target):
		if random.random() < self.prob:
			height, width = input.shape[-2:]
			input = input.flip(-1)
			bbox = target['boxes']
			bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
			target['boxes'] = bbox
			if 'masks' in target:
				target['masks'] = target['masks'].flip(-1)
			if 'keypoints' in target:
				keypoints = target['keypoints']
				keypoints = _flip_coco_person_keypoints(keypoints, width)
				target['keypoints'] = keypoints
		return input, target

class AugmentByImgaug(object):
	def __init__(self, augmenter, is_pil=True):
		self.augmenter = augmenter
		self.is_pil = is_pil

	def __call__(self, input, target):
		if self.is_pil: input = np.array(input)
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

		input, boxes, keypoints = self.augmenter(image=input, bounding_boxes=boxes, keypoints=keypoints, return_batch=False)
		#input, boxes, keypoints = self.augmenter(images=images, bounding_boxes=boxes, keypoints=keypoints, return_batch=True)

		if self.is_pil:
			import PIL.Image
			input = PIL.Image.fromarray(input)
		if 'boxes' in target:
			target['boxes'] = np.squeeze(boxes, axis=0)
		if 'keypoints' in target:
			# FIXME [modify] >> 4 means 4 keypoints. Actually keypoints can be a arbitrary number.
			keypoints0[...,:2] = np.squeeze(keypoints, axis=0).reshape(-1, 4, keypoints.shape[-1])
			target['keypoints'] = keypoints0
		return input, target
