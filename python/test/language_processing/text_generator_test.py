#!/usr/bin/env python

import sys
sys.path.append('../../src')

import os, math, random, csv, time, copy
import numpy as np
import cv2
#import imgaug as ia
#from imgaug import augmenters as iaa
import swl.language_processing.util as swl_langproc_util
from swl.language_processing.text_generator import Transformer, HangeulJamoGenerator, HangeulLetterGenerator, TextGenerator
import swl.language_processing.phd08_dataset as phd08_dataset

#class IdentityTransformer(Transformer):
class IdentityTransformer(object):
	"""Transforms an object of numpy.array.
	"""

	def __call__(self, input, mask, *args, **kwargs):
		"""Transforms an object of numpy.array.

		Inputs:
			input (numpy.array): a 2D or 3D numpy.array to transform.
			mask (numpy.array): a mask of input to transform. It can be None.
		Outputs:
			transformed input (numpy.array): a transformed 2D or 3D numpy.array.
			transformed mask (numpy.array): a transformed mask.
		"""

		return input, mask

#class RotationTransformer(Transformer):
class RotationTransformer(object):
	"""Rotates an object of numpy.array.
	"""

	def __call__(self, input, mask, *args, **kwargs):
		"""Rotates an object of numpy.array.

		Inputs:
			input (numpy.array): a 2D or 3D numpy.array to rotate.
			mask (numpy.array): a mask of input to rotate. It can be None.
		Outputs:
			transformed input (numpy.array): a rotated 2D or 3D numpy.array.
			transformed mask (numpy.array): a rotated mask.
		"""

		rot_angle = random.uniform(-30, 30)

		rows, cols = input.shape[:2]
		M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), rot_angle, 1)
		input = cv2.warpAffine(input, M, (cols, rows))
		#input = cv2.warpAffine(input, M, (cols, rows), cv2.INTER_CUBIC, cv2.BORDER_DEFAULT + cv2.BORDER_TRANSPARENT)
		if mask is not None:
			mask = cv2.warpAffine(mask, M, (cols, rows))

		return input, mask

#class ImgaugTransformer(Transformer):
class ImgaugTransformer(object):
	"""Transforms an object of numpy.array.
	"""

	def __init__(self):
		self._seq = iaa.Sequential([
			iaa.Sometimes(0.5, iaa.Affine(
				scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent (per axis).
				rotate=(-10, 10),  # Rotate by -10 to +10 degrees.
				shear=(-5, 5),  # Shear by -5 to +5 degrees.
				order=[0, 1],  # Use nearest neighbour or bilinear interpolation (fast).
				cval=(0, 255),  # If mode is constant, use a cval between 0 and 255.
				mode=ia.ALL  # Use any of scikit-image's warping modes (see 2nd image from the top for examples).
			)),
		])

	def __call__(self, input, mask, *args, **kwargs):
		"""Transforms an object of numpy.array.

		Inputs:
			input (numpy.array): a 2D or 3D numpy.array to transform.
			mask (numpy.array): a mask of input to transform. It can be None.
		Outputs:
			transformed input (numpy.array): a transformed 2D or 3D numpy.array.
			transformed mask (numpy.array): a transformed mask.
		"""

		seq_det = self._seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		return seq_det.augment_images(input), seq_det.augment_images(mask)

#class MyCharacterGenerator(CharacterGenerator):
class MyCharacterGenerator(object):
	"""Generates a character and its mask of numpy.array.

	Make sure if only a mask of a character may be needed.
	A character can be RGB, grayscale, or binary (black and white).
	A mask is binary (black(bg) and white(fg)).
	"""

	def __init__(self, font_size):
		"""Constructor.

		Inputs:
			font_size (int): a font size.
		"""

		#self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

		# Font.
		if 'posix' == os.name:
			font_info_list = [
				('/usr/share/fonts/truetype/gulim.ttf', 4),  # 굴림, 굴림체, 돋움, 돋움체.
				('/usr/share/fonts/truetype/batang.ttf', 4),  # 바탕, 바탕체, 궁서, 궁서체.
			]
		else:
			font_info_list = [
				('C:/Windows/Fonts/gulim.ttc', 4),  # 굴림, 굴림체, 돋움, 돋움체.
				('C:/Windows/Fonts/batang.ttc', 4),  # 바탕, 바탕체, 궁서, 궁서체.
			]
		self._font_list = list()
		for font_filepath, font_count in font_info_list:
			for font_idx in range(font_count):
				self._font_list.append((font_filepath, font_idx))
		self._font_size = font_size

		self._text_offset = (0, 0)
		self._image_size = (math.ceil(self._font_size * 1.1), math.ceil(self._font_size * 1.1))

		self._crop_text_area = True
		self._draw_text_border = False
		self._inverted_image = True

		if False:
			# Loads PHD08 image dataset.
			phd08_image_dataset_info_filepath = './phd08_png_dataset.csv'
			print('Start loading PHD08 image dataset...')
			start_time = time.time()
			self._letter_dict = phd08_dataset.load_phd08_image(phd08_image_dataset_info_filepath, self._inverted_image)
			print('\tElapsed time = {}'.format(time.time() - start_time))
			print('End loading PHD08 image dataset.')
		else:
			# Loads PHD08 npy dataset.
			# REF [function] >> generate_npy_dataset_from_phd08_conversion_result() in ${DataAnalysis_HOME}/test/language_processing_test/phd08_dataset_test.py
			phd08_npy_dataset_info_filepath = './phd08_npy_dataset.csv'
			print('Start loading PHD08 npy dataset...')
			start_time = time.time()
			self._letter_dict = phd08_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath, self._inverted_image)
			print('\tElapsed time = {}'.format(time.time() - start_time))
			print('End loading PHD08 npy dataset.')

	def __call__(self, char, *args, **kwargs):
		"""Generates a character and its mask of numpy.array.

		Inputs:
			char (str): a single character.
		Outputs:
			char (numpy.array): a numpy.array generated from an input character.
			mask (numpy.array): a mask of an input character, char. A mask is binary (black(bg) and white(fg)). It can be None.
		"""

		if char in self._letter_dict:
			use_printed_letter = 0 == random.randrange(2)
		else:
			use_printed_letter = True

		if use_printed_letter:
			print('Generate a printed Hangeul letter.')
			font_id = random.randrange(len(self._font_list))
			font_type, font_index = self._font_list[font_id]

			if self._inverted_image:
				#font_color = (random.randint(224, 255),) * 3
				#bg_color = (random.randint(0, 127),) * 3
				font_color = (255, 255, 255)
				bg_color = (0, 0, 0)
			else:
				#font_color = (random.randint(0, 31),) * 3
				#bg_color = (random.randint(128, 255),) * 3
				font_color = (0, 0, 0)
				bg_color = (255, 255, 255)

			arr = swl_langproc_util.generate_text_image(char, font_type, font_index, self._font_size, font_color, bg_color, self._image_size, self._text_offset, self._crop_text_area, self._draw_text_border)
			arr = np.asarray(arr, dtype=np.uint8)
		else:
			print('Generate a handwritten Hangeul letter.')
			letters = self._letter_dict[char]
			letter_id = random.randrange(len(letters))
			arr = letters[letter_id]

		if 2 == arr.ndim:
			mask = copy.deepcopy(arr)
			arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
		elif 3 == arr.ndim:
			mask = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
		elif 4 == arr.ndim:
			mask = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
			arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
		else:
			raise ValueError('Invalid image dimension: {}'.format(arr.ndim))

		#return arr, self._generate_mask_by_thresholding(mask)
		return arr, self._generate_mask_by_simple_binarization(mask)
		#return arr, mask
		#return arr, None

	def _generate_mask_by_thresholding(self, arr, *args, **kwargs):
		"""Generates a mask of an input object of numpy.array.

		Inputs:
			arr (numpy.array): an input array.
		Outputs:
			Mask (numpy.array): a mask of an input array.
		"""

		mask = arr
		#mask = self._clahe.apply(mask)
		_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		#mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
		#mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

		return mask

	def _generate_mask_by_simple_binarization(self, arr, *args, **kwargs):
		"""Generates a mask of an input object of numpy.array.

		Inputs:
			arr (numpy.array): an input array.
		Outputs:
			Mask (numpy.array): a mask of an input array.
		"""

		#mask = copy.deepcopy(arr)
		#mask[arr >= 64] = 255
		#mask[arr < 64] = 0
		mask = np.zeros_like(arr)
		mask[arr > 64] = 255

		return mask

#class MyCharacterPositioner(CharacterPositioner):
class MyCharacterPositioner(object):
	"""Place characters to construct a text line.
	"""

	def __init__(self, char_space):
		"""Constructor.

		Inputs:
			char_space (int): a space between characters.
		"""

		self._char_space = char_space
		self._inverted_image = True

	def __call__(self, char_list, mask_list, *args, **kwargs):
		"""Places characters to construct a single text line.

		Inputs:
			char_list (a list of numpy.array): a list of characters of type numpy.array to compose a text line.
			mask_list (a list of numpy.array): a list of masks of characters in char_list.
		Outputs:
			A text line (numpy.array): a text line of type 2D numpy.array made up of char_list.
			Masks (a list of numpy.array): a list of characters' masks.
		"""

		max_height = 0
		for ch in char_list:
			if ch.shape[0] > max_height:
				max_height = ch.shape[0]
		max_width = math.ceil(char_list[0].shape[1] / 2) + (len(char_list) - 1) * self._char_space + math.ceil(char_list[-1].shape[1] / 2)
		max_channels = 3

		if self._inverted_image:
			font_color = (random.randint(224, 255),) * 3
			bg_color = (random.randint(0, 127),) * 3
		else:
			font_color = (random.randint(0, 31),) * 3
			bg_color = (random.randint(128, 255),) * 3

		#text = np.zeros((max_height, max_width, max_channels), dtype=np.uint8)
		text = np.full((max_height, max_width, max_channels), bg_color)
		#text_mask = np.zeros((max_height, max_width), dtype=np.uint8)
		text_mask = list()

		for idx, (ch, mask) in enumerate(zip(char_list, mask_list)):
			sy = (max_height - ch.shape[0]) // 2
			sx = self._char_space * idx

			pixels = np.where(mask > 0)
			#pixels = np.where(ch > 0)

			#text[sy:sy+ch.shape[0],sx:sx+ch.shape[1]][pixels] = ch[pixels]
			text[sy:sy+ch.shape[0],sx:sx+ch.shape[1]][pixels] = font_color
			#if mask is not None:
			#	text_mask[sy:sy+mask.shape[0],sx:sx+mask.shape[1]][pixels] = mask[pixels]
			text_mask.append((mask, (sy, sx)))

		return text, text_mask

class MyTextGenerator(TextGenerator):
	"""Generates a single text line and its mask made up of characters.
	"""

	def __init__(self, font_size, char_space):
		"""Constructor.

		Inputs:
			font_size (int): a font size.
			char_space (int): a space between characters.
		"""

		#super().__init__(MyCharacterGenerator(font_size=font_size), IdentityTransformer(), MyCharacterPositioner(char_space=char_space))
		super().__init__(MyCharacterGenerator(font_size=font_size), RotationTransformer(), MyCharacterPositioner(char_space=char_space))
		#super().__init__(MyCharacterGenerator(font_size=font_size), ImgaugTransformer(), MyCharacterPositioner(char_space=char_space))

def hangeul_jamo_generator_test():
	raise NotImplementedError

def hangeul_letter_generator_test():
	raise NotImplementedError

def text_generator_test():
	textGenerator = MyTextGenerator(font_size=32, char_space=30)
	text, text_mask = textGenerator('가나다라마바사아자차카타파하')

	#mask = np.zeros_like(text, dtype=np.uint8)
	mask = np.zeros(text.shape[:2], dtype=np.uint8)
	for mk, (sy, sx) in text_mask:
		pixels = np.where(mk > 0)
		mask[sy:sy+mk.shape[0],sx:sx+mk.shape[1]][pixels] = mk[pixels]

	if 'posix' == os.name:
		cv2.imwrite('./text.png', text)
		cv2.imwrite('./mask.png', mask)
	else:
		cv2.imshow('Text', text)
		cv2.imshow('Mask', mask)
		cv2.waitKey(0)

		cv2.destroyAllWindows()

def main():
	#hangeul_jamo_generator_test()
	#hangeul_letter_generator_test()

	text_generator_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
