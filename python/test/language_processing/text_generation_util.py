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
from swl.util.util import make_dir

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

def generate_hangeul_font_list():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_dir_path = 'D:/work/font'

	"""
	if 'posix' == os.name:
		font_info_list = [
			(system_font_dir_path + '/truetype/gulim.ttf', 4),  # 굴림, 굴림체, 돋움, 돋움체.
			(system_font_dir_path + '/truetype/batang.ttf', 4),  # 바탕, 바탕체, 궁서, 궁서체.
		]
	else:
		font_info_list = [
			(system_font_dir_path + '/gulim.ttc', 4),  # 굴림, 굴림체, 돋움, 돋움체.
			(system_font_dir_path + '/batang.ttc', 4),  # 바탕, 바탕체, 궁서, 궁서체.
		]
	font_info_list += [
	"""
	font_info_list = [
		(font_dir_path + '/gulim.ttf', 4),  # 굴림, 굴림체, 돋움, 돋움체.
		(font_dir_path + '/batang.ttf', 4),  # 바탕, 바탕체, 궁서, 궁서체.
		(font_dir_path + '/gabia_bombaram.ttf', 1),
		#(font_dir_path + '/gabia_napjakBlock.ttf', 1),  # 한글 하단부 잘림.
		(font_dir_path + '/gabia_solmee.ttf', 1),
		(font_dir_path + '/godoMaum.ttf', 1),
		#(font_dir_path + '/godoRounded R.ttf', 1),  # 한글 깨짐.
		#(font_dir_path + '/HS1.ttf', 1),  # HS가을생각체1.0 Regular.ttf
		(font_dir_path + '/HS2.ttf', 1),  # HS가을생각체2.0.ttf
		(font_dir_path + '/HS3.ttf', 1),  # HS겨울눈꽃체.ttf
		#(font_dir_path + '/HS4.ttf', 1),  # HS두꺼비체.ttf  # 한글/영문/숫자/기호 하단부 잘림.
		#(font_dir_path + '/HS5.ttf', 1),  # HS봄바람체1.0.ttf
		(font_dir_path + '/HS6.ttf', 1),  # HS봄바람체2.0.ttf
		(font_dir_path + '/HS7.ttf', 1),  # HS여름물빛체.ttf
		(font_dir_path + '/NanumBarunGothic.ttf', 1),
		(font_dir_path + '/NanumBarunpenR.ttf', 1),
		(font_dir_path + '/NanumBrush.ttf', 1),
		(font_dir_path + '/NanumGothic.ttf', 1),
		(font_dir_path + '/NanumMyeongjo.ttf', 1),
		(font_dir_path + '/NanumPen.ttf', 1),
		(font_dir_path + '/NanumSquareR.ttf', 1),
		#(font_dir_path + '/NanumSquareRoundR.ttf', 1),
		(font_dir_path + '/SDMiSaeng.ttf', 1),
	]

	font_list = list()
	for font_filepath, font_count in font_info_list:
		for font_idx in range(font_count):
			font_list.append((font_filepath, font_idx))

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

		self._font_list = generate_hangeul_font_list()
		#self._handwriting_dict = generate_phd08_dict(from_npy=True)
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

	def __call__(self, char_alpha_list, char_space_ratio, *args, **kwargs):
		"""Places characters to construct a single text line.

		Inputs:
			char_alpha_list (a list of numpy.array): A list of character alpha mattes of type numpy.array to compose a text line.
			char_space_ratio (float): A ratio of space between characters.
		Outputs:
			A list of coordinates of character alpha mattes (a list of (int, int)): A list of (y position, x position) of the character alpha mattes.
		"""

		max_height = reduce(lambda x, y: max(x, y.shape[0]), char_alpha_list, 0)

		char_alpha_coordinate_list = list()
		sx = 0
		for idx, alpha in enumerate(char_alpha_list):
			sy = (max_height - alpha.shape[0]) // 2  # Vertical center.

			char_alpha_coordinate_list.append((sy, sx))

			sx += math.ceil(alpha.shape[1] * char_space_ratio)

		return char_alpha_coordinate_list

class MySimplePrintedHangeulTextGenerator(object):
	"""Generates a simple printed Hangeul text line and masks for individual characters.
	"""

	def __init__(self, characterTransformer, characterPositioner):
		"""Constructor.

		Inputs:
			characterTransformer (Transformer): An object to tranform each character.
			characterPositioner (CharacterPositioner): An object to place characters.
		"""

		self._characterTransformer = characterTransformer
		self._characterPositioner = characterPositioner

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

		#self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

		self._font_list = generate_hangeul_font_list()
		#self._handwriting_dict = generate_phd08_dict(from_npy=True)

	def __call__(self, text, char_space_ratio, font_size, *args, **kwargs):
		"""Generates a single text line and masks for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
			char_space_ratio (float): A ratio of space between characters.
			font_size (int): A font size for the characters.
		Outputs:
			char_alpha_list (a list of numpy.array): A list of character alpha mattes.
			char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
				A text line can be constructed by MyTextGenerator.constructTextLine().
		"""

		image_size = (math.ceil(font_size * 1.1), math.ceil(font_size * 1.1))

		#print('Generate a printed Hangeul letter.')
		font_id = random.randrange(len(self._font_list))
		font_type, font_index = self._font_list[font_id]

		font_color = (255, 255, 255)
		bg_color = (0, 0, 0)

		char_alpha_list = list()
		for ch in text:
			alpha = swl_langproc_util.generate_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border)
			alpha = cv2.cvtColor(np.array(alpha), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
			alpha, _, _ = self._characterTransformer(alpha, None, *args, **kwargs)
			char_alpha_list.append(alpha)

		char_alpha_coordinate_list = self._characterPositioner(char_alpha_list, char_space_ratio, *args, **kwargs)
		return char_alpha_list, char_alpha_coordinate_list

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

	def __call__(self, text, char_space_ratio, font_size, *args, **kwargs):
		"""Generates a single text line and masks for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
			char_space_ratio (float): A ratio of space between characters.
			font_size (int): A font size for the characters.
		Outputs:
			char_alpha_list (a list of numpy.array): A list of character alpha mattes.
			char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
				A text line can be constructed by MyTextGenerator.constructTextLine().
		"""

		char_alpha_list = list()
		for ch in text:
			alpha = self._characterGenerator(ch, font_size, *args, **kwargs)
			alpha, _, _ = self._characterTransformer(alpha, None, *args, **kwargs)
			char_alpha_list.append(alpha)

		char_alpha_coordinate_list = self._characterPositioner(char_alpha_list, char_space_ratio, *args, **kwargs)
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
			font_color = list(map(lambda x: x / 255, font_color))
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

#--------------------------------------------------------------------

def generate_text_lines(word_set, textGenerator, font_size_interval, char_space_ratio_interval, batch_size, font_color=None, bg_color=None):
	sceneTextGenerator = MySceneTextGenerator(IdentityTransformer())

	scene_list, scene_text_mask_list = list(), list()
	step = 0
	while True:
		font_size = random.randint(*font_size_interval)
		char_space_ratio = random.uniform(*char_space_ratio_interval)

		text = random.sample(word_set, 1)[0]

		char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space_ratio, font_size)
		text_line, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

		if bg_color is None:
			# Grayscale background.
			bg = np.full_like(text_line, random.randrange(256), dtype=np.uint8)
		else:
			bg = np.full_like(text_line, bg_color, dtype=np.uint8)

		scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
		scene_list.append(scene)
		scene_text_mask_list.append(scene_text_mask)

		step += 1
		if 0 == step % batch_size:
			yield scene_list, scene_text_mask_list
			scene_list, scene_text_mask_list = list(), list()
			step = 0

def generate_scene_texts(word_set, sceneTextGenerator, sceneProvider, textGenerator, text_count_interval, font_size_interval, char_space_ratio_interval,  batch_size, font_color=None):
	scene_list, scene_text_mask_list, bboxes_list, text_list = list(), list(), list(), list()
	step = 0
	while True:
		num_texts_per_image = random.randint(*text_count_interval)

		texts, text_images, text_alphas = list(), list(), list()
		for ii in range(num_texts_per_image):
			font_size = random.randint(*font_size_interval)
			char_space_ratio = random.uniform(*char_space_ratio_interval)

			text = random.sample(word_set, 1)[0]

			char_alpha_list, char_alpha_coordinate_list = textGenerator(text, char_space_ratio, font_size)
			text_line_image, text_line_alpha = MyTextGenerator.constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

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
		scene_list.append(scene)
		scene_text_mask_list.append(scene_text_mask)
		bboxes_list.append(bboxes)

		step += 1
		if 0 == step % batch_size:
			yield scene_list, scene_text_mask_list, bboxes_list
			scene_list, scene_text_mask_list, bboxes_list = list(), list(), list()
			step = 0
