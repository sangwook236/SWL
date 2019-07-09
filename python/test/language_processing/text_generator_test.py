#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math, random, csv, time, copy, glob, json
from functools import reduce
import numpy as np
import cv2
#import imgaug as ia
#from imgaug import augmenters as iaa
import swl.language_processing.util as swl_langproc_util
from swl.language_processing.text_generator import Transformer, HangeulJamoGenerator, HangeulLetterGenerator, TextGenerator, SceneProvider, SceneTextGenerator
import swl.language_processing.phd08_dataset as phd08_dataset
import swl.machine_vision.util as swl_cv_util
from swl.util.util import make_dir

#class IdentityTransformer(Transformer):
class IdentityTransformer(object):
	"""Transforms an object of numpy.array.
	"""

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Transforms an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to transform.
			mask (numpy.array): A mask of input to transform. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A transformed 2D or 3D numpy.array.
			A transformed mask (numpy.array): A transformed mask.
			A transformed bounding rectangle (numpy.array): A transformed bounding rectangle. 4 x 2.
		"""

		height, width = input.shape[:2]
		dst_rect = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

		return input, mask, dst_rect

#class RotationTransformer(Transformer):
class RotationTransformer(object):
	"""Rotates an object of numpy.array.
	"""

	def __init__(self, min_angle, max_angle):
		"""Constructor.

		Inputs:
			min_angle (float): A min. rotation angle in degrees.
			max_angle (float): A max. rotation angle in degrees.
		"""

		self._min_angle, self._max_angle = min_angle, max_angle

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Rotates an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to rotate.
			mask (numpy.array): A mask of input to rotate. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A rotated 2D or 3D numpy.array.
			A transformed mask (numpy.array): A rotated mask.
			A transformed bounding rectangle (numpy.array): A transformed bounding rectangle. 4 x 2.
		"""

		height, width = input.shape[:2]
		if canvas_size is None:
			canvas_height, canvas_width = height, width
		else:
			canvas_height, canvas_width = canvas_size

		rot_angle = random.uniform(self._min_angle, self._max_angle)
		T = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), rot_angle, 1)

		input = cv2.warpAffine(input, T, (canvas_width, canvas_height))
		#input = cv2.warpAffine(input, T, (canvas_width, canvas_height), cv2.INTER_CUBIC, cv2.BORDER_DEFAULT + cv2.BORDER_TRANSPARENT)
		if mask is not None:
			mask = cv2.warpAffine(mask, T, (canvas_width, canvas_height))

		dst_rect = np.matmul(T, np.array([(0, 0, 1), (width, 0, 1), (width, height, 1), (0, height, 1)], dtype=np.float32).T)

		return input, mask, dst_rect.T

#class AffineTransformer(Transformer):
class AffineTransformer(object):
	"""Affinely transforms an object of numpy.array.
	"""

	def __init__(self, paper_size, min_angle, max_angle, min_scale, max_scale):
		"""Constructor.

		Inputs:
			paper_size (tuple of ints): The size of a paper (height, width).
			min_angle (float): A min. rotation angle in degrees.
			max_angle (float): A max. rotation angle in degrees.
			min_scale (float): A min. scale.
			max_scale (float): A max. scale.
		"""

		self._paper_height, self._paper_width = paper_size
		self._min_angle, self._max_angle = min_angle, max_angle
		self._min_scale, self._max_scale = min_scale, max_scale

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Affinely transforms an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to rotate.
			mask (numpy.array): A mask of input to rotate. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A rotated 2D or 3D numpy.array.
			A transformed mask (numpy.array): A rotated mask.
			A transformed bounding rectangle (numpy.array): A transformed bounding rectangle. 4 x 2.
		"""

		height, width = input.shape[:2]
		if canvas_size is None:
			canvas_height, canvas_width = height, width
		else:
			canvas_height, canvas_width = canvas_size

		dx, dy = random.uniform(0, self._paper_width - width), random.uniform(0, self._paper_height - height)
		rot_angle = random.uniform(self._min_angle, self._max_angle)
		scale = random.uniform(self._min_scale, self._max_scale)
		T = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rot_angle, scale)
		T[:,2] += (dx, dy)

		input = cv2.warpAffine(input, T, (canvas_width, canvas_height))
		#input = cv2.warpAffine(input, T, (canvas_width, canvas_height), cv2.INTER_CUBIC, cv2.BORDER_DEFAULT + cv2.BORDER_TRANSPARENT)
		if mask is not None:
			mask = cv2.warpAffine(mask, T, (canvas_width, canvas_height))

		dst_rect = np.matmul(T, np.array([(0, 0, 1), (width, 0, 1), (width, height, 1), (0, height, 1)], dtype=np.float32).T)

		return input, mask, dst_rect.T

#class PerspectiveTransformer(Transformer):
class PerspectiveTransformer(object):
	"""Perspectively transforms an object of numpy.array.
	"""

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Perspectively transforms an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to transform.
			mask (numpy.array): A mask of input to transform. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A transformed 2D or 3D numpy.array.
			A transformed mask (numpy.array): A transformed mask.
			A transformed bounding rectangle (numpy.array): A transformed bounding rectangle. 4 x 2.
		"""

		height, width = input.shape[:2]
		if canvas_size is None:
			canvas_height, canvas_width = height, width
		else:
			canvas_height, canvas_width = canvas_size

		delta_x_range, delta_y_range = (-10, 10), (-10, 10)
		#rot_angle = random.uniform(-math.pi, math.pi)
		rot_angle = random.uniform(-math.pi / 4, math.pi / 4)

		src_rect = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)
		dst_rect = src_rect + np.array([(random.uniform(*delta_x_range), random.uniform(*delta_y_range)) for _ in range(4)], dtype=np.float32)

		cos_angle, sin_angle = math.cos(rot_angle), math.sin(rot_angle)
		R = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
		dst_rect = np.matmul(R, dst_rect.T).T

		(x_min, y_min), (x_max, y_max) = np.min(dst_rect, axis=0), np.max(dst_rect, axis=0)
		dx, dy = random.uniform(min(-x_min, canvas_width - x_max), max(-x_min, canvas_width - x_max)), random.uniform(min(-y_min, canvas_height - y_max), max(-y_min, canvas_height - y_max))
		dst_rect += np.array([dx, dy], dtype=np.float32)

		T = cv2.getPerspectiveTransform(src_rect, dst_rect)

		#--------------------
		input = cv2.warpPerspective(input, T, (canvas_width, canvas_height))
		#input = cv2.warpPerspective(input, T, (canvas_width, canvas_height), cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS, cv2.BORDER_DEFAULT + cv2.BORDER_TRANSPARENT)
		if mask is not None:
			mask = cv2.warpPerspective(mask, T, (canvas_width, canvas_height))

		return input, mask, dst_rect

#class ProjectiveTransformer(Transformer):
class ProjectiveTransformer(object):
	"""Projectively transforms an object of numpy.array.
	"""

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Projectively transforms an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to transform.
			mask (numpy.array): A mask of input to transform. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A transformed 2D or 3D numpy.array.
			A transformed mask (numpy.array): A transformed mask.
			A transformed bounding rectangle (numpy.array): A transformed bounding rectangle. 4 x 2.
		"""

		height, width = input.shape[:2]
		if canvas_size is None:
			canvas_height, canvas_width = height, width
		else:
			canvas_height, canvas_width = canvas_size

		#rot_angle = random.uniform(-math.pi, math.pi)
		rot_angle = random.uniform(-math.pi / 4, math.pi / 4)
		scale = random.uniform(0.5, 2)
		while True:
			K11, K12 = random.uniform(-2, 2), random.uniform(-2, 2)
			if abs(K11) > -1.0e-3:
				break
		#v1, v2 = random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005)
		v1, v2 = random.uniform(0, 0.005), random.uniform(0, 0.005)

		cos_angle, sin_angle = math.cos(rot_angle), math.sin(rot_angle)
		Hs = scale * np.array([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]], dtype=np.float32)
		Ha = np.array([[K11, K12, 0], [0, 1 / K11, 0], [0, 0, 1]], dtype=np.float32)
		Hp = np.array([[1, 0, 0], [0, 1, 0], [v1, v2, 1]], dtype=np.float32)
		H = np.matmul(np.matmul(Hs, Ha), Hp)

		src_rect = np.array([(0, 0, 1), (width, 0, 1), (width, height, 1), (0, height, 1)], dtype=np.float32)
		dst_rect = np.matmul(H, src_rect.T).T
		#src_rect, dst_rect = src_rect[:,:2], (dst_rect[:,:2] / dst_rect[:,2].reshape(dst_rect[:,2].shape + (-1,)))  # Error: Error: Assertion failed (src.checkVector(2, 5) == 4 && dst.checkVector(2, 5) == 4) in cv::getPerspectiveTransform.
		src_rect, dst_rect = src_rect[:,:2].astype(np.float32), (dst_rect[:,:2] / dst_rect[:,2].reshape(dst_rect[:,2].shape + (-1,))).astype(np.float32)

		(x_min, y_min), (x_max, y_max) = np.min(dst_rect, axis=0), np.max(dst_rect, axis=0)
		dx, dy = random.uniform(min(-x_min, canvas_width - x_max), max(-x_min, canvas_width - x_max)), random.uniform(min(-y_min, canvas_height - y_max), max(-y_min, canvas_height - y_max))
		dst_rect += np.array([dx, dy], dtype=np.float32)

		T = cv2.getPerspectiveTransform(src_rect, dst_rect)

		#--------------------
		input = cv2.warpPerspective(input, T, (canvas_width, canvas_height))
		#input = cv2.warpPerspective(input, T, (canvas_width, canvas_height), cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS, cv2.BORDER_DEFAULT + cv2.BORDER_TRANSPARENT)
		if mask is not None:
			mask = cv2.warpPerspective(mask, T, (canvas_width, canvas_height))

		return input, mask, dst_rect

#class ImgaugAffineTransformer(Transformer):
class ImgaugAffineTransformer(object):
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

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Transforms an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to transform.
			mask (numpy.array): A mask of input to transform. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A transformed 2D or 3D numpy.array.
			A transformed mask (numpy.array): A transformed mask.
		"""

		seq_det = self._seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		return seq_det.augment_images(input), seq_det.augment_images(mask)

#class ImgaugPerspectiveTransformer(Transformer):
class ImgaugPerspectiveTransformer(object):
	"""Transforms an object of numpy.array.
	"""

	def __init__(self):
		self._seq = iaa.Sequential([
			iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
		])

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Transforms an object of numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to transform.
			mask (numpy.array): A mask of input to transform. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A transformed 2D or 3D numpy.array.
			A transformed mask (numpy.array): A transformed mask.
		"""

		seq_det = self._seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		return seq_det.augment_images(input), seq_det.augment_images(mask)

#class MyHangeulCharacterAlphaMatteGenerator(CharacterAlphaMatteGenerator):
class MyHangeulCharacterAlphaMatteGenerator(object):
	"""Generates an alpha-matte [0, 1] for a character which reflects the proportion of foreground (when alpha=1) and background (when alpha=0).
	"""

	def __init__(self):
		"""Constructor.
		"""

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

		#self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

		# Font.
		if 'posix' == os.name:
			system_font_dir = '/usr/share/fonts'
			font_dir = '/home/sangwook/work/font'
		else:
			system_font_dir = 'C:/Windows/Fonts'
			font_dir = 'D:/work/font'

		"""
		if 'posix' == os.name:
			font_info_list = [
				(system_font_dir + '/truetype/gulim.ttf', 4),  # 굴림, 굴림체, 돋움, 돋움체.
				(system_font_dir + '/truetype/batang.ttf', 4),  # 바탕, 바탕체, 궁서, 궁서체.
			]
		else:
			font_info_list = [
				(system_font_dir + '/gulim.ttc', 4),  # 굴림, 굴림체, 돋움, 돋움체.
				(system_font_dir + '/batang.ttc', 4),  # 바탕, 바탕체, 궁서, 궁서체.
			]
		font_info_list += [
		"""
		font_info_list = [
			(font_dir + '/gulim.ttf', 4),  # 굴림, 굴림체, 돋움, 돋움체.
			(font_dir + '/batang.ttf', 4),  # 바탕, 바탕체, 궁서, 궁서체.
			(font_dir + '/gabia_bombaram.ttf', 1),
			(font_dir + '/gabia_napjakBlock.ttf', 1),
			(font_dir + '/gabia_solmee.ttf', 1),
			(font_dir + '/godoMaum.ttf', 1),
			(font_dir + '/godoRounded R.ttf', 1),
			#(font_dir + '/HS1.ttf', 1),  # HS가을생각체1.0 Regular.ttf
			(font_dir + '/HS2.ttf', 1),  # HS가을생각체2.0.ttf
			(font_dir + '/HS3.ttf', 1),  # HS겨울눈꽃체.ttf
			(font_dir + '/HS4.ttf', 1),  # HS두꺼비체.ttf
			#(font_dir + '/HS5.ttf', 1),  # HS봄바람체1.0.ttf
			(font_dir + '/HS6.ttf', 1),  # HS봄바람체2.0.ttf
			(font_dir + '/HS7.ttf', 1),  # HS여름물빛체.ttf
			(font_dir + '/NanumBarunGothic.ttf', 1),
			(font_dir + '/NanumBarunpenR.ttf', 1),
			(font_dir + '/NanumBrush.ttf', 1),
			(font_dir + '/NanumGothic.ttf', 1),
			(font_dir + '/NanumMyeongjo.ttf', 1),
			(font_dir + '/NanumPen.ttf', 1),
			(font_dir + '/NanumSquareR.ttf', 1),
			#(font_dir + '/NanumSquareRoundR.ttf', 1),
			(font_dir + '/SDMiSaeng.ttf', 1),
		]

		self._font_list = list()
		for font_filepath, font_count in font_info_list:
			for font_idx in range(font_count):
				self._font_list.append((font_filepath, font_idx))

		if True:
			# Loads PHD08 image dataset.
			phd08_image_dataset_info_filepath = './phd08_png_dataset.csv'
			print('Start loading PHD08 image dataset...')
			start_time = time.time()
			self._handwriting_dict = phd08_dataset.load_phd08_image(phd08_image_dataset_info_filepath, is_dark_background=False)
			for key, values in self._handwriting_dict.items():
				self._handwriting_dict[key] = list()
				for val in values:
					val = cv2.cvtColor(cv2.bitwise_not(val), cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255
					self._handwriting_dict[key].append(val)
			print('End loading PHD08 image dataset: {} secs.'.format(time.time() - start_time))
		elif False:
			# Loads PHD08 npy dataset.
			# Generate an info file for npy files generated from the PHD08 dataset.
			#	Refer to generate_npy_dataset_from_phd08_conversion_result() in ${SWL_PYTHON_HOME}/test/language_processing/phd08_datset_test.py.
			phd08_npy_dataset_info_filepath = './phd08_npy_dataset.csv'
			print('Start loading PHD08 npy dataset...')
			start_time = time.time()
			self._handwriting_dict = phd08_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath, is_dark_background=False)
			for key, values in self._handwriting_dict.items():
				self._handwriting_dict[key] = list()
				for val in values:
					val = cv2.cvtColor(cv2.bitwise_not(val), cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255
					self._handwriting_dict[key].append(val)
			print('End loading PHD08 npy dataset: {} secs.'.format(time.time() - start_time))
		else:
			self._handwriting_dict = dict()

	def __call__(self, char, font_size, *args, **kwargs):
		"""Generates a character and its mask of numpy.array.

		Inputs:
			char (str): A single character.
			font_size (int): A font size for the character.
		Outputs:
			A character (numpy.array): An alpha matte for an input character.
		"""

		image_size = (math.ceil(font_size * 1.1), math.ceil(font_size * 1.1))

		if char in self._handwriting_dict:
			use_printed_letter = 0 == random.randrange(2)
		else:
			use_printed_letter = True

		if use_printed_letter:
			#print('Generate a printed Hangeul letter.')
			font_id = random.randrange(len(self._font_list))
			font_type, font_index = self._font_list[font_id]

			font_color = (255, 255, 255)
			bg_color = (0, 0, 0)

			alpha = swl_langproc_util.generate_text_image(char, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border)
			alpha = cv2.cvtColor(np.array(alpha), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
		else:
			#print('Generate a handwritten Hangeul letter.')
			letters = self._handwriting_dict[char]
			letter_id = random.randrange(len(letters))
			alpha = letters[letter_id]

		return alpha

#class MyCharacterAlphaMattePositioner(CharacterAlphaMattePositioner):
class MyCharacterAlphaMattePositioner(object):
	"""Place characters to construct a text line.
	"""

	def __call__(self, char_alpha_list, char_space, *args, **kwargs):
		"""Places characters to construct a single text line.

		Inputs:
			char_alpha_list (a list of numpy.array): A list of character alpha mattes of type numpy.array to compose a text line.
			char_space (int): A space between characters.
				If char_space <= 0, default widths of characters are used.
		Outputs:
			A list of coordinates of character alpha mattes (a list of (int, int)): A list of (y position, x position) of the character alpha mattes.
				A text line can be constructed by MyCharacterAlphaMattePositioner.constructTextLine().
		"""

		if char_space <= 0:
			max_height, max_width = 0, 0
			for alpha in char_alpha_list:
				if alpha.shape[0] > max_height:
					max_height = alpha.shape[0]
				max_width += alpha.shape[1]
		else:
			max_height = 0
			for alpha in char_alpha_list:
				if alpha.shape[0] > max_height:
					max_height = alpha.shape[0]
			#max_width = math.ceil(char_alpha_list[0].shape[1] / 2) + (len(char_alpha_list) - 1) * char_space + math.ceil(char_alpha_list[-1].shape[1] / 2)
			max_width = (len(char_alpha_list) - 1) * char_space + math.ceil(char_alpha_list[-1].shape[1])

		char_alpha_coordinate_list = list()
		sx = 0
		for idx, alpha in enumerate(char_alpha_list):
			sy = (max_height - alpha.shape[0]) // 2  # Vertical center.

			char_alpha_coordinate_list.append((sy, sx))

			sx += alpha.shape[1] if char_space <= 0 else char_space

		return char_alpha_coordinate_list

#class MyTextGenerator(TextGenerator):
class MyTextGenerator(object):
	"""Generates a single text line and masks for individual characters.
	"""

	def __init__(self, characterGenerator, characterTransformer, characterPositioner):
		"""Constructor.

		Inputs:
			characterGenerator (CharacterGenerator): An object to generate each character.
			characterTransformer (Transformer): An object to tranform each character.
			characterPositioner (CharacterPositioner): An object to place characters.
		"""

		self._characterGenerator = characterGenerator
		self._characterTransformer = characterTransformer
		self._characterPositioner = characterPositioner

	def __call__(self, text, char_space, font_size, *args, **kwargs):
		"""Generates a single text line and masks for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
			char_space (int): A space between characters.
				If char_space <= 0, widths of characters are used.
			font_size (int): A font size for the characters.
			font_color (tuple): A font color for the characters. If None, random colors are used.
			is_single_mask_generated (bool): Specifies whether a list of masks or a single mask is generated.
		Outputs:
			A text line (numpy.array): A text line of type 2D numpy.array made up of char_list.
			Masks (a list of (numpy.array, int, int)) or a mask (numpy.array): A list of masks and (y position, x position) of characters or a mask of the text line.
				Which mask is generated depends on the input parameter is_single_mask_generated.
		"""

		char_alpha_list = list()
		for ch in text:
			alpha = self._characterGenerator(ch, font_size, *args, **kwargs)
			alpha, _, _ = self._characterTransformer(alpha, None, *args, **kwargs)
			char_alpha_list.append(alpha)

		char_alpha_coordinate_list = self._characterPositioner(char_alpha_list, char_space, *args, **kwargs)
		return char_alpha_list, char_alpha_coordinate_list

	@staticmethod
	def constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=None, text_line_size=None):
		"""Constructs a text line from character alpha mattes.

		Inputs:
			char_alpha_list (a list of numpy.array): A list of character alpha mattes.
			char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
			font_color (tuple): A font color for the characters. If None, random colors are used.
			text_line_size (tuple): The size of the text line.
		Outputs:
			A text line (numpy.array): A single text line constructed from character alpha mattes.
		"""

		if text_line_size is None:
			text_line_size = reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])), map(lambda alpha, coord: (coord[0] + alpha.shape[0], coord[1] + alpha.shape[1]), char_alpha_list, char_alpha_coordinate_list))

		text_line = np.zeros(text_line_size + (3,), dtype=np.float32)
		text_line_alpha = np.zeros(text_line_size, dtype=np.float32)
		if font_color is None:
			for alpha, (sy, sx) in zip(char_alpha_list, char_alpha_coordinate_list):
				#pixels = np.where(alpha > 0)
				#text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]][pixels] = alpha[pixels]	
				font_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
				#font_color = (random.randrange(256), random.randrange(256), random.randrange(256))
				text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full(alpha.shape + (3,), font_color, dtype=np.float32), text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
				text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full_like(alpha, 1.0, dtype=np.float32), text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
		else:
			for alpha, (sy, sx) in zip(char_alpha_list, char_alpha_coordinate_list):
				#pixels = np.where(alpha > 0)
				#text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]][pixels] = alpha[pixels]	
				text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full(alpha.shape + (3,), font_color, dtype=np.float32), text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
				text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full_like(alpha, 1.0, dtype=np.float32), text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
		#return text_line, text_line_alpha
		return np.round(text_line * 255).astype(np.uint8), text_line_alpha

#class MySceneTextGenerator(SceneTextGenerator):
class MySceneTextGenerator(object):
	"""Generates a scene containing multiple transformed text lines in a background.
	"""

	def __init__(self, textTransformer):
		"""Constructor.

		Inputs:
			textTransformer (Transformer): An object to transform a single text line.
		"""

		self._textTransformer = textTransformer

	def __call__(self, scene, texts, text_alphas, *args, **kwargs):
		"""Generates a scene containing multiple transformed text lines in a background.

		Inputs:
			scene (numpy.array): An object to be used as a scene or a background.
			texts (list of numpy.arrays): A list with multiple text lines.
			text_alphas (list of numpy.arrays): A list of alpha mattes of the text lines.
		Outputs:
			A scene (numpy.array): A scene containing transformed text lines.
			A scene text mask (numpy.array) or a list of text masks (list of numpy.array's): A scene mask containing masks of transformed text lines in a scene.
			A list of transformed bounding rectangles (list of numpy.array's): A list of transformed bounding rectangles (4 x 2) in a scene.
		"""

		scene_size = scene.shape[:2]

		scene_mask = np.zeros(scene_size, dtype=np.uint16)
		#scene_text_masks = list()
		bboxes = list()
		for idx, (text, alpha) in enumerate(zip(texts, text_alphas)):
			text, alpha, bbox = self._textTransformer(text, alpha, scene_size, *args, **kwargs)

			#--------------------
			"""
			pixels = np.where(alpha > 0)
			#pixels = np.where(text > 0)

			if 2 == text.ndim:
				scene[:,:][pixels] = text[pixels]
			elif 3 == text.ndim:
				scene[:,:,:text.shape[-1]][pixels] = text[pixels]
			else:
				print('[SWL] Invalid number {} of channels in the {}-th text, {}.'.format(text.shape[-1], idx, text))
				continue
			scene_mask[pixels] = idx + 1
			#scene_text_masks.append(alpha)
			bboxes.append(bbox)
			"""
			scene = swl_cv_util.blend_image(text.astype(np.float32), scene.astype(np.float32), alpha)
			scene_mask[alpha > 0] = idx + 1
			#scene_text_masks.append(alpha)
			bboxes.append(bbox)

		print('***********', scene_mask.shape, scene_mask.dtype)
		return np.round(scene).astype(np.uint8), scene_mask, np.array(bboxes)
		#return np.round(scene).astype(np.uint8), scene_text_masks, np.array(bboxes)

#class MyGrayscaleBackgroundProvider(SceneProvider):
class MyGrayscaleBackgroundProvider(object):
	def __init__(self, shape):
		"""Constructor.

		Inputs:
			shape (int or tuple of ints): Shape of a new scene.
		"""

		self._shape = shape

	def __call__(self, shape=None, *args, **kwargs):
		"""Generates and provides a scene.

		Inputs:
			shape (int or tuple of ints): Shape of a new scene. If shape = None, a scene of a prespecified or a random shape is generated.
		Outputs:
			A scene (numpy.array): A scene generated.
		"""

		if shape is None:
			shape = self._shape

		#return np.full(shape, random.randrange(256), dtype=np.uint8)
		return np.full(shape, random.randrange(1, 255), dtype=np.uint8)

#class MySceneProvider(SceneProvider):
class MySceneProvider(object):
	def __init__(self):
		self._scene_filepaths = list()
		self._scene_filepaths.append('./background_image/image1.jpg')
		self._scene_filepaths.append('./background_image/image2.jpg')
		self._scene_filepaths.append('./background_image/image3.jpg')
		self._scene_filepaths.append('./background_image/image4.jpg')
		self._scene_filepaths.append('./background_image/image5.jpg')

	def __call__(self, shape=None, *args, **kwargs):
		"""Generates and provides a scene.

		Inputs:
			shape (int or tuple of ints): Shape of a new scene. If shape = None, a scene of a prespecified or a random shape is generated.
		Outputs:
			A scene (numpy.array): A scene generated.
		"""

		idx = random.randrange(len(self._scene_filepaths))
		scene = cv2.imread(self._scene_filepaths[idx], cv2.IMREAD_COLOR)

		if shape is None:
			return scene
		else:
			return cv2.resize(scene, shape[:2], interpolation=cv2.INTER_AREA)

class MySimpleSceneProvider(MySceneProvider):
	def __init__(self):
		self._scene_filepaths = glob.glob('./background_image/*.jpg', recursive=True)

def hangeul_jamo_generator_test():
	raise NotImplementedError

def hangeul_letter_generator_test():
	raise NotImplementedError

def text_generator_test():
	font_size = 32
	font_color = None  # Uses random character colors.
	#font_color = (random.randint(0, 255),) * 3  # Uses a random text color.
	char_space = 30  # If char_space <= 0, widths of characters are used.

	characterAlphaMatteGenerator = MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = IdentityTransformer()
	characterTransformer = RotationTransformer(-30, 30)
	#characterTransformer = ImgaugAffineTransformer()
	characterAlphaMattePositioner = MyCharacterAlphaMattePositioner()
	textGenerator = MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	char_alpha_list, char_alpha_coordinate_list = textGenerator('가나다라마바사아자차카타파하', char_space, font_size)
	text_line, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=None)

	#--------------------
	# No background.
	if 'posix' == os.name:
		cv2.imwrite('./text_line.png', text_line)
	else:
		cv2.imshow('Text line', text_line)

		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	#--------------------
	sceneTextGenerator = MySceneTextGenerator(IdentityTransformer())

	# Grayscale background.
	bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
	scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])

	if 'posix' == os.name:
		cv2.imwrite('./scene.png', scene)
		cv2.imwrite('./scene_text_mask.png', scene_text_mask)
	else:
		cv2.imshow('Scene', scene)
		#scene_text_mask[scene_text_mask > 0] = 255
		#scene_text_mask = scene_text_mask.astype(np.uint8)
		minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
		scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)
		cv2.imshow('Scene Mask', scene_text_mask)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def scene_text_generator_test():
	font_color = None  # Uses random character colors.
	#font_color = (random.randint(0, 255),) * 3  # Uses a random text color.

	characterAlphaMatteGenerator = MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = IdentityTransformer()
	characterTransformer = RotationTransformer(-30, 30)
	#characterTransformer = ImgaugAffineTransformer()
	characterAlphaMattePositioner = MyCharacterAlphaMattePositioner()
	textGenerator = MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	#--------------------
	texts, text_alphas = list(), list()

	char_alpha_list, char_alpha_coordinate_list = textGenerator('가나다라마바사아자차카타파하', char_space=30, font_size=32)
	text_line, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#char_alpha_list, char_alpha_coordinate_list = textGenerator('ABCDEFGHIJKLMNOPQRSTUVWXYZ', char_space=40, font_size=24)
	char_alpha_list, char_alpha_coordinate_list = textGenerator('ABCDEFGHIJKLM', char_space=40, font_size=24)
	text_line, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#char_alpha_list, char_alpha_coordinate_list = textGenerator('abcdefghijklmnopqrstuvwxyz', char_space=14, font_size=16)
	char_alpha_list, char_alpha_coordinate_list = textGenerator('abcdefghijklm', char_space=14, font_size=16)
	text_line, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)
	texts.append(text_line)
	text_alphas.append(text_line_alpha)

	#--------------------
	if True:
		textTransformer = PerspectiveTransformer()
	else:
		textTransformer = ProjectiveTransformer()
	sceneTextGenerator = MySceneTextGenerator(textTransformer)

	if True:
		sceneProvider = MySceneProvider()
	else:
		# Grayscale background.
		scene_shape = (800, 1000, 3)  # Some handwritten characters have 3 channels.
		sceneProvider = MyGrayscaleBackgroundProvider(scene_shape)

	#--------------------
	scene = sceneProvider()
	if 3 == scene.ndim and 3 != scene.shape[-1]:
		raise ValueError('Invalid image shape')

	scene, scene_text_mask, _ = sceneTextGenerator(scene, texts, text_alphas)
	#scene, scene_text_mask, _ = sceneTextGenerator(scene, texts, text_alphas)

	#--------------------
	if 'posix' == os.name:
		cv2.imwrite('./scene.png', scene)
		cv2.imwrite('./scene_text_mask.png', scene_text_mask)
	else:
		cv2.imshow('Scene', scene)
		#scene_text_mask[scene_text_mask > 0] = 255
		#scene_text_mask = scene_text_mask.astype(np.uint8)
		minval, maxval = np.min(scene_text_mask), np.max(scene_text_mask)
		scene_text_mask = (scene_text_mask.astype(np.float32) - minval) / (maxval - minval)

		cv2.imshow('Scene Mask', scene_text_mask)
		cv2.waitKey(0)

		cv2.destroyAllWindows()

def generate_scene_text_dataset(dir_path, json_filename, sceneTextGenerator, sceneProvider, textGenerator, num_images):
	scene_subdir_name = 'scene'
	mask_subdir_name = 'mask'
	scene_dir_path = os.path.join(dir_path, scene_subdir_name)
	mask_dir_path = os.path.join(dir_path, mask_subdir_name)

	make_dir(dir_path)
	make_dir(scene_dir_path)
	make_dir(mask_dir_path)

	hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
	#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
	with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
		#data = fd.readlines()  # A string.
		#data = fd.read().strip('\n')  # A list of strings.
		#data = fd.read().splitlines()  # A list of strings.
		data = fd.read().replace(' ', '').replace('\n', '')  # A string.
	count = 80
	hangeul_charset = str()
	for idx in range(0, len(data), count):
		txt = ''.join(data[idx:idx+count])
		#hangeul_charset += ('' if 0 == idx else '\n') + txt
		hangeul_charset += txt
	alphabet_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	digit_charset = '0123456789'
	symbol_charset = ' `~!@#$%^&*()-_=+[]{}\\|;:\'\",.<>/?'

	#print('Hangeul charset =', len(hangeul_charset), hangeul_charset)
	#print('Alphabet charset =', len(alphabet_charset), alphabet_charset)
	#print('Digit charset =', len(digit_charset), digit_charset)
	#print('Symbol charset =', len(symbol_charset), symbol_charset)

	#charset_list = [ hangeul_charset, alphabet_charset, digit_charset, symbol_charset ]
	#charset_selection_ratios = [ 0.25, 0.5, 0.75, 1.0 ]
	charset_list = [ hangeul_charset, alphabet_charset, digit_charset, hangeul_charset + alphabet_charset, hangeul_charset + alphabet_charset + digit_charset ]
	charset_selection_ratios = [ 0.6, 0.8, 0.9, 0.95, 1.0 ]

	min_char_count_per_text, max_char_count_per_text = 1, 10
	min_text_count_per_image, max_text_count_per_image = 2, 10
	min_font_size, max_font_size = 15, 30
	min_char_space, max_char_space = 10, 60
	min_char_space_ratio, max_char_space_ratio = 0.8, 3

	#--------------------
	data_list = list()
	for idx in range(num_images):
		num_texts_per_image = random.randint(min_text_count_per_image, max_text_count_per_image)

		texts, text_images, text_alphas = list(), list(), list()
		for ii in range(num_texts_per_image):
			font_size = random.randint(min_font_size, max_font_size)
			char_space = random.randint(min_char_space, max_char_space)
			char_space = max(max(char_space, int(font_size * min_char_space_ratio)), min(char_space, int(font_size * max_char_space_ratio)))

			font_color = None  # Uses random character colors.
			#font_color = (random.randint(0, 255),) * 3  # Uses a random text color.

			num_chars_per_text = random.randint(min_char_count_per_text, max_char_count_per_text)

			charset_selection_ratio = random.uniform(0.0, 1.0)
			for charset_idx, ratio in enumerate(charset_selection_ratios):
				if charset_selection_ratio < ratio:
					break

			charset = charset_list[charset_idx]
			charset_len = len(charset)
			text = ''.join(list(charset[random.randrange(charset_len)] for _ in range(num_chars_per_text)))

			char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space=char_space, font_size=font_size)
			text_line_image, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color=font_color)

			texts.append(text)
			text_images.append(text_line_image)
			text_alphas.append(text_line_alpha)

		#--------------------
		scene = sceneProvider()
		if 3 == scene.ndim and 3 != scene.shape[-1]:
			#raise ValueError('Invalid image shape')
			print('Error: Invalid image shape.')
			continue

		scene, scene_text_mask, bboxes = sceneTextGenerator(scene, text_images, text_alphas)
		#scene, scene_text_mask, bboxes = sceneTextGenerator(scene, text_images, text_alphas)
		#scene, scene_text_mask, bboxes = sceneTextGenerator(scene, text_images, text_alphas)

		#--------------------
		if True:
			text_image_filepath = os.path.join(scene_subdir_name, 'scene_{:07}.png'.format(idx))
			mask_image_filepath = os.path.join(mask_subdir_name, 'mask_{:07}.png'.format(idx))
		elif False:
			# For MS-D-Net.
			#scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255.0
			scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR)
			text_image_filepath = os.path.join(scene_subdir_name, 'scene_{:07}.tiff'.format(idx))
			mask_image_filepath = os.path.join(mask_subdir_name, 'mask_{:07}.tiff'.format(idx))
		elif False:
			# For MS-D_Net_PyTorch.
			#scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255.0
			scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR)
			scene_text_mask[scene_text_mask > 0] = 1
			#scene_text_mask = scene_text_mask.astype(np.uint8)
			text_image_filepath = os.path.join(scene_subdir_name, 'img_{:07}.tif'.format(idx))
			mask_image_filepath = os.path.join(mask_subdir_name, 'img_{:07}.tif'.format(idx))
		"""
		# Draw bounding rectangles.
		for box in bboxes:
			scene = cv2.line(scene, (box[0,0], box[0,1]), (box[1,0], box[1,1]), (0, 0, 255, 255), 2, cv2.LINE_8)
			scene = cv2.line(scene, (box[1,0], box[1,1]), (box[2,0], box[2,1]), (0, 255, 0, 255), 2, cv2.LINE_8)
			scene = cv2.line(scene, (box[2,0], box[2,1]), (box[3,0], box[3,1]), (255, 0, 0, 255), 2, cv2.LINE_8)
			scene = cv2.line(scene, (box[3,0], box[3,1]), (box[0,0], box[0,1]), (255, 0, 255, 255), 2, cv2.LINE_8)
		"""
		cv2.imwrite(os.path.join(dir_path, text_image_filepath), scene)
		cv2.imwrite(os.path.join(dir_path, mask_image_filepath), scene_text_mask)

		datum = {
			'image': text_image_filepath,
			#'image': os.path.abspath(text_image_filepath),
			'mask': mask_image_filepath,
			#'mask': os.path.abspath(mask_image_filepath),
			'texts': texts,
			'bboxes': bboxes.tolist(),
		}
		data_list.append(datum)

	json_filepath = os.path.join(dir_path, json_filename)
	with open(json_filepath, 'w', encoding='UTF-8') as json_file:
		#json.dump(data_list, json_file)
		json.dump(data_list, json_file, ensure_ascii=False, indent='  ')

def load_scene_text_dataset(dir_path, json_filename):
	json_filepath = os.path.join(dir_path, json_filename)
	with open(json_filepath, 'r', encoding='UTF-8') as json_file:
		json_data = json.load(json_file)

	image_filepaths, mask_filepaths, gt_texts, gt_boxes = list(), list(), list(), list()
	for dat in json_data:
		image_filepaths.append(dat['image'])
		mask_filepaths.append(dat['mask'])
		gt_texts.append(dat['texts'])
		#gt_boxes.append(dat['bboxes'])
		gt_boxes.append(np.array(dat['bboxes']))

	return image_filepaths, mask_filepaths, gt_texts, gt_boxes

def generate_hangeul_synthetic_scene_text_dataset():
	characterAlphaMatteGenerator = MyHangeulCharacterAlphaMatteGenerator()
	#characterTransformer = IdentityTransformer()
	characterTransformer = RotationTransformer(-30, 30)
	#characterTransformer = ImgaugAffineTransformer()
	characterAlphaMattePositioner = MyCharacterAlphaMattePositioner()
	textGenerator = MyTextGenerator(characterAlphaMatteGenerator, characterTransformer, characterAlphaMattePositioner)

	#--------------------
	if True:
		textTransformer = PerspectiveTransformer()
	else:
		textTransformer = ProjectiveTransformer()
	sceneTextGenerator = MySceneTextGenerator(textTransformer)

	if True:
		sceneProvider = MySimpleSceneProvider()
	else:
		# Grayscale background.
		scene_shape = (800, 1000, 3)  # Some handwritten characters have 4 channels.
		sceneProvider = MyGrayscaleBackgroundProvider(scene_shape)

	# Generate a scene dataset.
	scene_text_dataset_dir_path = './scene_text_dataset'
	#scene_text_dataset_dir_path = './scene_text_dataset_for_ms_d_net'
	#scene_text_dataset_dir_path = './scene_text_dataset_for_ms_d_net_pytorch'
	scene_text_dataset_json_filename = 'scene_text_dataset.json'
	num_images = 50000
	generate_scene_text_dataset(scene_text_dataset_dir_path, scene_text_dataset_json_filename, sceneTextGenerator, sceneProvider, textGenerator, num_images)

	# Load a scene dataset.
	image_filepaths, mask_filepaths, gt_texts, gt_boxes = load_scene_text_dataset(scene_text_dataset_dir_path, scene_text_dataset_json_filename)
	print('Generated scene dataset: #images = {}, #masks = {}, #texts = {}, #boxes = {}.'.format(len(image_filepaths), len(mask_filepaths), len(gt_texts), len(gt_boxes)))

def main():
	#hangeul_jamo_generator_test()  # Not yet implemented.
	#hangeul_letter_generator_test()  # Not yet implemented.

	#text_generator_test()
	#scene_text_generator_test()

	# Application.
	generate_hangeul_synthetic_scene_text_dataset()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
