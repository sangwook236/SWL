import os, math, random, csv, time, copy, glob, json
from functools import reduce
import numpy as np
import cv2
#import imgaug as ia
#from imgaug import augmenters as iaa
import swl.language_processing.util as swl_langproc_util
from swl.language_processing.text_generator import Transformer, HangeulJamoGenerator, HangeulLetterGenerator, TextGenerator, SceneProvider, SceneTextGenerator
import swl.language_processing.hangeul_handwriting_dataset as hg_hw_dataset
import swl.machine_vision.util as swl_cv_util

def generate_random_word_set(num_words, charset, min_char_count, max_char_count):
	charset_len = len(charset)
	word_set = set()
	for idx in range(num_words):
		num_chars = random.randint(min_char_count, max_char_count)
		text = ''.join(list(charset[random.randrange(charset_len)] for _ in range(num_chars)))
		word_set.add(text)

	return word_set

def generate_repetitive_word_set(num_char_repetitions, charset, min_char_count, max_char_count):
	indices = list(range(len(charset)))
	multi_indices = indices * num_char_repetitions
	random.shuffle(multi_indices)
	#print(multi_indices[0:100], len(multi_indices))

	num_chars = len(multi_indices)

	word_set = set()
	start_idx = 0
	while start_idx < num_chars:
		#end_idx = min(start_idx + random.randint(min_char_count, max_char_count), num_chars)
		end_idx = start_idx + random.randint(min_char_count, max_char_count)
		char_indices = multi_indices[start_idx:end_idx]
		text = ''.join(map(lambda idx: charset[idx], char_indices))
		word_set.add(text)

		start_idx = end_idx

	return word_set

#--------------------------------------------------------------------

def generate_font_list(font_filepaths):
	num_fonts = 1
	font_list = list()
	for fpath in font_filepaths:
		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	return font_list

def generate_hangeul_font_list(font_filepaths):
	# NOTE [caution] >>
	#	Font가 깨져 (한글) 문자가 물음표로 표시되는 경우 발생.
	#	생성된 (한글) 문자의 하단부가 일부 짤리는 경우 발생.
	#	Image resizing에 의해 얇은 획이 사라지는 경우 발생.

	font_list = list()
	for fpath in font_filepaths:
		num_fonts = 4 if os.path.basename(fpath).lower() in ['gulim.ttf', 'batang.ttf'] else 1

		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	return font_list

def generate_phd08_dict(from_npy=True):
	if from_npy:
		# Loads PHD08 npy dataset.
		# Generate an info file for npy files generated from the PHD08 dataset.
		#	Refer to generate_npy_dataset_from_phd08_conversion_result() in ${SWL_PYTHON_HOME}/test/language_processing/phd08_datset_test.py.
		phd08_npy_dataset_info_filepath = './phd08_npy_dataset.csv'
		print('Start loading PHD08 npy dataset...')
		start_time = time.time()
		handwriting_dict = hg_hw_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath, is_dark_background=False)
		for key, values in handwriting_dict.items():
			handwriting_dict[key] = list()
			for val in values:
				val = cv2.cvtColor(cv2.bitwise_not(val), cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255
				handwriting_dict[key].append(val)
		print('End loading PHD08 npy dataset: {} secs.'.format(time.time() - start_time))
	else:
		# Loads PHD08 image dataset.
		phd08_image_dataset_info_filepath = './phd08_png_dataset.csv'
		print('Start loading PHD08 image dataset...')
		start_time = time.time()
		handwriting_dict = phd08_dataset.load_phd08_image(phd08_image_dataset_info_filepath, is_dark_background=False)
		for key, values in handwriting_dict.items():
			handwriting_dict[key] = list()
			for val in values:
				val = cv2.cvtColor(cv2.bitwise_not(val), cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255
				handwriting_dict[key].append(val)
		print('End loading PHD08 image dataset: {} secs.'.format(time.time() - start_time))

	return handwriting_dict

#--------------------------------------------------------------------

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
		font_color = list(map(lambda x: x / 255, font_color))
		for alpha, (sy, sx) in zip(char_alpha_list, char_alpha_coordinate_list):
			#pixels = np.where(alpha > 0)
			#text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]][pixels] = alpha[pixels]	
			text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full(alpha.shape + (3,), font_color, dtype=np.float32), text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
			text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full_like(alpha, 1.0, dtype=np.float32), text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)

	#return text_line, text_line_alpha
	return np.round(text_line * 255).astype(np.uint8), text_line_alpha

#--------------------------------------------------------------------

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

#class MyCharacterAlphaMatteGenerator(CharacterAlphaMatteGenerator):
class MyCharacterAlphaMatteGenerator(object):
	"""Generates an alpha-matte [0, 1] for a character which reflects the proportion of foreground (when alpha=1) and background (when alpha=0).
	"""

	def __init__(self, font_list, handwriting_dict=None, mode='1'):
		"""Constructor.

		Inputs:
			font_list (a list of (font file path, font index) pairs): A list of the file paths and the font indices of fonts.
			handwriting_dict (a dict of (character, a list of images)): A dictionary of characters and their corresponding list of images.
			mode (str): A color mode: 'L' or '1'.
		"""

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

		self._font_list = font_list
		self._handwriting_dict = handwriting_dict
		self._mode = mode

	def __call__(self, char, font_size, *args, **kwargs):
		"""Generates a character and its mask of numpy.array.

		Inputs:
			char (str): A single character.
			font_size (int): A font size for the character.
		Outputs:
			A character (numpy.array): An alpha matte for an input character.
		"""

		image_size = (math.ceil(font_size * 1.1), math.ceil(font_size * 1.1))

		if self._handwriting_dict is not None and char in self._handwriting_dict:
			use_printed_letter = 0 == random.randrange(2)
		else:
			use_printed_letter = True

		if use_printed_letter:
			#print('Generate a printed letter.')
			font_type, font_index = random.choice(self._font_list)
			font_color, bg_color = 255, 0

			alpha = swl_langproc_util.generate_text_image(char, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, mode=self._mode)
			if '1' != self._mode:
				return np.array(alpha, dtype=np.float32) / 255
			else:
				return np.array(alpha, dtype=np.float32)
		else:
			#print('Generate a handwritten Hangeul letter.')
			return random.choice(self._handwriting_dict[char])

#class MyCharacterPositioner(CharacterPositioner):
class MyCharacterPositioner(object):
	"""Place characters to construct a text line.
	"""

	def __call__(self, char_image_list, char_space_ratio, *args, **kwargs):
		"""Places characters to construct a single text line.

		Inputs:
			char_image_list (a list of numpy.array): A list of character images of type numpy.array to compose a text line.
			char_space_ratio (float): A ratio of space between characters.
		Outputs:
			A list of coordinates of character images (a list of (int, int)): A list of (y position, x position) of the character images.
		"""

		max_height = reduce(lambda x, y: max(x, y.shape[0]), char_image_list, 0)

		char_image_coordinate_list = list()
		sx = 0
		for idx, img in enumerate(char_image_list):
			sy = (max_height - img.shape[0]) // 2  # Vertical center.

			char_image_coordinate_list.append((sy, sx))

			sx += math.ceil(img.shape[1] * char_space_ratio)

		return char_image_coordinate_list

class MyBasicPrintedTextGenerator(object):
	"""Generates a basic printed text line for individual characters.
	"""

	def __init__(self, font_list, font_size_interval, char_space_ratio_interval=None, mode='RGB', mask_mode='1'):
		"""Constructor.

		Inputs:
			font_list (a list of (font file path, font index) pairs): A list of the file paths and the font indices of fonts.
			font_size_interval (a tuple of two ints): A font size interval for the characters.
			char_space_ratio_interval (a tuple of two floats): A space ratio interval between two characters.
			mode (str): RGB mode ('RGB') or grayscale mode ('L') of image.
			mask_mode (str): Black-white mode ('1') or grayscale mode ('L') of image mask.
		"""

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

		self._font_list = font_list
		self._font_size_interval = font_size_interval
		self._char_space_ratio_interval = char_space_ratio_interval
		self._mode = mode
		self._mask_mode = mask_mode
		self._image_channel = 1 if 'L' == mode or '1' == mode else 3

	def __call__(self, text, *args, **kwargs):
		"""Generates a single text line for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
		Outputs:
			text_image (numpy.array): A generated text image.
			text_mask (numpy.array): A generated text mask.
		"""

		#image_size = (math.ceil(len(text) * font_size * 2), math.ceil(font_size * 2))
		image_size = None

		font_type, font_index = random.choice(self._font_list)

		font_size = random.randint(*self._font_size_interval)
		char_space_ratio = None if self._char_space_ratio_interval is None else random.uniform(*self._char_space_ratio_interval)

		#font_color = (255,) * self._image_channel
		#font_color = tuple(random.randrange(256) for _ in range(self._image_channel))  # Uses a specific RGB font color.
		#font_color = (random.randrange(256),) * self._image_channel  # Uses a specific grayscale font color.
		font_color = (random.randrange(0, 128),) * self._image_channel  # Uses a specific black font color.
		#font_color = (random.randrange(128, 256),) * self._image_channel  # Uses a specific white font color.
		#font_color = None  # Uses a random font color.
		#bg_color = (0,) * self._image_channel
		#bg_color = tuple(random.randrange(256) for _ in range(self._image_channel))  # Uses a specific RGB background color.
		#bg_color = (random.randrange(256),) * self._image_channel  # Uses a specific grayscale background color.
		#bg_color = (random.randrange(0, 128),) * self._image_channel  # Uses a specific black background color.
		bg_color = (random.randrange(128, 256),) * self._image_channel  # Uses a specific white background color.
		#bg_color = None  # Uses a random background color.

		text_image, text_mask = swl_langproc_util.generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, char_space_ratio, mode=self._mode, mask=True, mask_mode=self._mask_mode)

		#return np.array(text_image), np.array(text_mask)  # text_mask: np.bool.
		return np.array(text_image), np.array(text_mask, dtype=np.uint8)

	def create_generator(self, word_set, batch_size):
		if batch_size <= 0 or batch_size > len(word_set):
			raise ValueError('Invalid batch size: 0 < batch_size <= len(word_set)')

		while True:
			texts = random.sample(word_set, k=batch_size)
			#texts = random.choices(word_set, k=batch_size)

			text_list, image_list, mask_list = list(), list(), list()
			for text in texts:
				text_line, mask = self.__call__(text)

				text_list.append(text)
				image_list.append(text_line)
				mask_list.append(mask)

			yield text_list, image_list, mask_list

class MySimpleTextAlphaMatteGenerator(object):
	"""Generates a simple text line and masks for individual characters.
	"""

	def __init__(self, characterTransformer, characterPositioner, font_list, font_size_interval, char_space_ratio_interval=None, handwriting_dict=None, mode='1'):
		"""Constructor.

		Inputs:
			characterTransformer (Transformer): An object to tranform each character.
			characterPositioner (CharacterPositioner): An object to place characters.
			font_list (a list of (font file path, font index) pairs): A list of the file paths and the font indices of fonts.
			font_size_interval (a tuple of two ints): A font size interval for the characters.
			char_space_ratio_interval (a tuple of two floats): A space ratio interval between two characters.
			handwriting_dict (a dict of (character, a list of images)): A dictionary of characters and their corresponding list of images.
			mode (str): A color mode: 'L' or '1'.
		"""

		self._characterTransformer = characterTransformer
		self._characterPositioner = characterPositioner
		self._mode = mode

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

		self._font_list = font_list
		self._handwriting_dict = handwriting_dict  # FIXME [fix] >> Currently not used.
		self._font_size_interval = font_size_interval
		self._char_space_ratio_interval = char_space_ratio_interval

	def __call__(self, text, *args, **kwargs):
		"""Generates a single text line and masks for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
		Outputs:
			char_alpha_list (a list of numpy.array): A list of character images.
			char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
				A text line can be constructed by constructTextLine().
		"""

		font_type, font_index = random.choice(self._font_list)

		font_size = random.randint(*self._font_size_interval)
		char_space_ratio = None if self._char_space_ratio_interval is None else random.uniform(*self._char_space_ratio_interval)

		image_size = (math.ceil(font_size * 1.1), math.ceil(font_size * 1.1))

		font_color, bg_color = 255, 0

		char_alpha_list = list()
		for ch in text:
			alpha = swl_langproc_util.generate_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, mode=self._mode)
			if '1' != self._mode:
				alpha = np.array(alpha, dtype=np.float32) / 255
			else:
				alpha = np.array(alpha, dtype=np.float32)
			alpha, _, _ = self._characterTransformer(alpha, None, *args, **kwargs)
			char_alpha_list.append(alpha)

		char_alpha_coordinate_list = self._characterPositioner(char_alpha_list, char_space_ratio, *args, **kwargs)
		return char_alpha_list, char_alpha_coordinate_list

	def create_generator(self, word_set, batch_size):
		if batch_size <= 0 or batch_size > len(word_set):
			raise ValueError('Invalid batch size: 0 < batch_size <= len(word_set)')

		sceneTextGenerator = MyAlphaMatteSceneTextGenerator(IdentityTransformer())

		font_color = None  # Uses a random font color.
		while True:
			texts = random.sample(word_set, k=batch_size)
			#texts = random.choices(word_set, k=batch_size)

			text_list, scene_list, scene_text_mask_list = list(), list(), list()
			for text in texts:
				#font_color = (255, 255, 255)
				#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
				#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
				#font_color = None  # Uses a random font color.
				#bg_color = (0, 0, 0)
				#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
				bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
				#bg_color = None  # Uses a random background color.

				char_alpha_list, char_alpha_coordinate_list = self.__call__(text)
				text_line, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

				scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
				text_list.append(text)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)

			yield text_list, scene_list, scene_text_mask_list

#class MyTextAlphaMatteGenerator(TextAlphaMatteGenerator):
class MyTextAlphaMatteGenerator(object):
	"""Generates a single text line and masks for individual characters.
	"""

	def __init__(self, characterGenerator, characterTransformer, characterPositioner, font_size_interval, char_space_ratio_interval=None):
		"""Constructor.

		Inputs:
			characterGenerator (CharacterGenerator): An object to generate each character.
			characterTransformer (Transformer): An object to tranform each character.
			characterPositioner (CharacterPositioner): An object to place characters.
			font_size_interval (a tuple of two ints): A font size interval for the characters.
			char_space_ratio_interval (a tuple of two floats): A space ratio interval between two characters.
		"""

		self._characterGenerator = characterGenerator
		self._characterTransformer = characterTransformer
		self._characterPositioner = characterPositioner

		self._font_size_interval = font_size_interval
		self._char_space_ratio_interval = char_space_ratio_interval

	def __call__(self, text, *args, **kwargs):
		"""Generates a single text line and masks for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
			char_space_ratio (float): A ratio of space between characters.
			font_size (int): A font size for the characters.
		Outputs:
			char_alpha_list (a list of numpy.array): A list of character alpha mattes.
			char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
				A text line can be constructed by constructTextLine().
		"""

		font_size = random.randint(*self._font_size_interval)
		char_space_ratio = None if self._char_space_ratio_interval is None else random.uniform(*self._char_space_ratio_interval)

		char_alpha_list = list()
		for ch in text:
			alpha = self._characterGenerator(ch, font_size, *args, **kwargs)
			alpha, _, _ = self._characterTransformer(alpha, None, *args, **kwargs)
			char_alpha_list.append(alpha)

		char_alpha_coordinate_list = self._characterPositioner(char_alpha_list, char_space_ratio, *args, **kwargs)
		return char_alpha_list, char_alpha_coordinate_list

	def create_generator(self, word_set, batch_size):
		if batch_size <= 0 or batch_size > len(word_set):
			raise ValueError('Invalid batch size: 0 < batch_size <= len(word_set)')

		sceneTextGenerator = MyAlphaMatteSceneTextGenerator(IdentityTransformer())

		font_color = None  # Uses a random font color.
		while True:
			texts = random.sample(word_set, k=batch_size)
			#texts = random.choices(word_set, k=batch_size)

			text_list, scene_list, scene_text_mask_list = list(), list(), list()
			for text in texts:
				#font_color = (255, 255, 255)
				#font_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB font color.
				#font_color = (random.randrange(256),) * 3  # Uses a specific grayscale font color.
				#font_color = None  # Uses a random font color.
				#bg_color = (0, 0, 0)
				#bg_color = tuple(random.randrange(256) for _ in range(3))  # Uses a specific RGB background color.
				bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
				#bg_color = None  # Uses a random background color.

				char_alpha_list, char_alpha_coordinate_list = self.__call__(text)
				text_line, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

				scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
				text_list.append(text)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)

			yield text_list, scene_list, scene_text_mask_list

#class MyAlphaMatteSceneTextGenerator(AlphaMatteSceneTextGenerator):
class MyAlphaMatteSceneTextGenerator(object):
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
			# TODO [revise] >> Need to enhance alpha > 0.
			scene_mask[alpha > 0] = idx + 1
			#scene_text_masks.append(alpha)
			bboxes.append(bbox)

		return np.round(scene).astype(np.uint8), scene_mask, np.array(bboxes)
		#return np.round(scene).astype(np.uint8), scene_text_masks, np.array(bboxes)

	def create_generator(self, textGenerator, sceneProvider, word_set, batch_size, text_count_interval, font_color=None):
		while True:
			texts_list, scene_list, scene_text_mask_list, bboxes_list = list(), list(), list(), list()
			for _ in range(batch_size):
				num_texts_per_image = random.randint(*text_count_interval)
				texts = random.choices(word_set, k=num_texts_per_image)

				text_images, text_alphas = list(), list()
				for text in texts:
					char_alpha_list, char_alpha_coordinate_list = textGenerator(text)
					text_line_image, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

					text_images.append(text_line_image)
					text_alphas.append(text_line_alpha)

				#--------------------
				scene = sceneProvider()
				if 3 == scene.ndim and 3 != scene.shape[-1]:
					#raise ValueError('Invalid image shape')
					print('Error: Invalid image shape.')
					continue

				scene, scene_text_mask, bboxes = self.__call__(scene, text_images, text_alphas)
				texts_list.append(texts)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)
				bboxes_list.append(bboxes)

			yield texts_list, scene_list, scene_text_mask_list, bboxes_list

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
		if 'posix' == os.name:
			base_dir_path = '/home/sangwook/work/dataset'
		else:
			base_dir_path = 'D:/work/dataset'
		self._scene_filepaths = glob.glob(base_dir_path + '/background_image/*.jpg', recursive=True)

	def __call__(self, shape=None, *args, **kwargs):
		"""Generates and provides a scene.

		Inputs:
			shape (int or tuple of ints): Shape of a new scene. If shape = None, a scene of a prespecified or a random shape is generated.
		Outputs:
			A scene (numpy.array): A scene generated.
		"""

		scene_filepath = random.choice(self._scene_filepaths)
		scene = cv2.imread(scene_filepath, cv2.IMREAD_COLOR)

		if shape is None:
			return scene
		else:
			return cv2.resize(scene, shape[:2], interpolation=cv2.INTER_AREA)

class MySimpleSceneProvider(MySceneProvider):
	def __init__(self):
		if 'posix' == os.name:
			base_dir_path = '/home/sangwook/work/dataset'
		else:
			base_dir_path = 'D:/work/dataset'
		self._scene_filepaths = glob.glob(base_dir_path + '/background_image/*.jpg', recursive=True)
