import random
import numpy as np
import cv2

#--------------------------------------------------------------------

class Transformer(object):
	"""Transforms a numpy.array.
	"""

	def __call__(self, input, mask, canvas_size=None, *args, **kwargs):
		"""Transforms a numpy.array.

		Inputs:
			input (numpy.array): A 2D or 3D numpy.array to transform.
			mask (numpy.array): A mask of input to transform. It can be None.
			canvas_size (tuple of ints): The size of a canvas (height, width). If canvas_size = None, the size of input is used.
		Outputs:
			A transformed input (numpy.array): A transformed 2D or 3D numpy.array.
			A transformed mask (numpy.array): A transformed mask.
			A transformed bounding rectangle (numpy.array): A transformed bounding rectangle. 4 x 2.
		"""

		raise NotImplementedError

#--------------------------------------------------------------------

class HangeulJamoGenerator(object):
	"""Generates a Hangeul jamo and its mask of numpy.array.

	Make sure if only a mask of a jamo may be needed.
	A jamo can be RGB, grayscale, or binary (black and white).
	A mask is binary (black(bg) and white(fg)).
	"""

	def __call__(self, jamo, font_color=None, *args, **kwargs):
		"""Generates a Hangeul jamo and its mask of numpy.array.

		Inputs:
			jamo (str): A single Hangeul jamo.
			font_color (tuple): A font color for the jamo. If None, random colors are used.
		Outputs:
			A jamo (numpy.array): A numpy.array generated from an input Hangeul jamo.
		"""

		raise NotImplementedError

class HangeulJamoPositioner(object):
	"""Places jamos to construct a Hangeul letter.
	"""

	def __call__(self, jamo_list, mask_list, shape_type, *args, **kwargs):
		"""Places jamos to construct a Hangeul letter.

		Inputs:
			jamo_list (a list of numpy.array): A list of jamos of type numpy.array to compose a letter.
			mask_list (a list of numpy.array): A list of masks of jamos in jamo_list.
			shape_type (int): The shape type of a letter. [1, 6].
		Outputs:
			A Hangeul letter (numpy.array): A Hangeul letter made up of jamos.
			Masks (a list of numpy.array): A list of jamos' masks. A mask is binary (black(bg) and white(fg)). It can be None.
		"""

		raise NotImplementedError

		"""
		# FIXME [change] >> Center positions should be changed.
		if 1 == shape_type:
			position_list = [(0.5, 0.25), (0.5, 0.75)]
		elif 2 == shape_type:
			position_list = [(0.25, 0.5), (0.75, 0.5)]
		elif 3 == shape_type:
			position_list = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
		elif 4 == shape_type:
			position_list = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.5)]
		elif 5 == shape_type:
			position_list = [(0.16, 0.5), (0.5, 0.5), (0.83, 0.5)]
		elif 6 == shape_type:
			position_list = [(0.16, 0.25), (0.5, 0.25), (0.33, 0.75), (0.83, 0.5)]
		else:
			raise ValueError('Invalid shape type: it must be [1, 6]')
		"""

class HangeulLetterGenerator(object):
	"""Generates a Hangeul letter made up of jamos.
	"""

	def __init__(self, jamoGenerator, jamoTransformer, jamoPositioner):
		"""Constructor.

		Inputs:
			jamoGenerator (HangeulJamoGenerator): An object to generate each jamo.
			jamoTransformer (Transformer): An object to tranform each jamo.
			jamoPositioner (HangeulJamoPositioner): An object to place jamos.
		"""

		self._jamoGenerator = jamoGenerator
		self._jamoTransformer = jamoTransformer
		self._jamoPositioner = jamoPositioner

	def __call__(self, jamos, shape_type, *args, **kwargs):
		"""Generates a Hangeul letter made up of jamos.

		Inputs:
			jamos (str): Jamos to compose a letter.
			shape_type (int): The shape type of a letter. [1, 6].
		Outputs:
			A Hangeul letter (numpy.array): A Hangeul letter made up of jamos.
			Masks (a list of numpy.array): A list of jamos' masks.
		"""

		jamo_list, mask_list = list(), list()
		for jamo in jamos:
			jamo, mask = self._jamoGenerator(jamo)
			jamo, mask = self._jamoTransformer(jamo, mask)
			jamo_list.append(jamo)
			mask_list.append(mask)

		return self._jamoPositioner(jamo_list, mask_list, shape_type, *args, **kwargs)

#--------------------------------------------------------------------

class CharacterGenerator(object):
	"""Generates a character and its mask of numpy.array.

	Make sure if only a mask of a character may be needed.
	A character can be RGB, grayscale, or binary (black and white).
	A mask is binary (black(bg) and white(fg)).
	"""

	def __call__(self, char, font_size, font_color=None, *args, **kwargs):
		"""Generates a character and its mask of numpy.array.

		Inputs:
			char (str): A single character.
			font_size (int): A font size for the character.
			font_color (tuple): A font color for the character. If None, random colors are used.
		Outputs:
			A character (numpy.array): A numpy.array generated from an input character.
			A mask (numpy.array): A mask of an input character, char. A mask is binary (black(bg) and white(fg)). It can be None.
		"""

		raise NotImplementedError

class CharacterPositioner(object):
	"""Places characters to construct a text line.
	"""

	def __call__(self, char_list, mask_list, char_space, is_single_mask_generated=True, *args, **kwargs):
		"""Places characters to construct a single text line.

		Inputs:
			char_list (a list of numpy.array): A list of characters of type numpy.array to compose a text line.
			mask_list (a list of numpy.array): A list of masks of characters in char_list.
			char_space (int): A space between characters.
				If char_space <= 0, widths of characters are used.
			is_single_mask_generated (bool): Specifies whether a list of masks or a single mask is generated.
		Outputs:
			A text line (numpy.array): A text line of type 2D numpy.array made up of char_list.
			Masks (a list of (numpy.array, int, int)) or a mask (numpy.array): A list of masks and (y position, x position) of characters or a mask of the text line.
				Which mask is generated depends on the input parameter is_single_mask_generated.
		"""

		raise NotImplementedError

#--------------------------------------------------------------------

class TextGenerator(object):
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

	def __call__(self, text, char_space, font_size, font_color=None, is_single_mask_generated=True, *args, **kwargs):
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

		char_list, mask_list = list(), list()
		for ch in text:
			ch, mask = self._characterGenerator(ch, font_size, font_color, *args, **kwargs)
			ch, mask, _ = self._characterTransformer(ch, mask, *args, **kwargs)
			char_list.append(ch)
			mask_list.append(mask)

		return self._characterPositioner(char_list, mask_list, char_space, is_single_mask_generated, *args, **kwargs)

#--------------------------------------------------------------------

class SceneProvider(object):
	"""Generates and provides a scene.
	"""

	def __call__(self, shape=None, *args, **kwargs):
		"""Generates and provides a scene.

		Inputs:
			shape (int or tuple of ints): Shape of a new scene. If shape = None, a scene of a prespecified or a random shape is generated.
		Outputs:
			A scene (numpy.array): A scene generated.
		"""

		raise NotImplementedError

class SceneTextGenerator(object):
	"""Generates a scene containing multiple transformed text lines in a background.
	"""

	def __init__(self, textTransformer):
		"""Constructor.

		Inputs:
			textTransformer (Transformer): An object to transform a single text line.
		"""

		self._textTransformer = textTransformer

	def __call__(self, scene, texts, text_masks, blend_ratio_interval=None, *args, **kwargs):
		"""Generates a scene containing multiple transformed text lines in a background.

		Inputs:
			scene (numpy.array): An object to be used as a scene or a background.
			texts (list of numpy.arrays): A list object with multiple text lines.
			text_masks (list of numpy.arrays): A list object of masks of the text lines.
			blend_ratio_interval (tuple of two floats): Specifies min and max ratios to blend texts and a scene. 0.0 <= min ratio <= max ratio <= 1.0. If None, texts and a scene are not blended.
		Outputs:
			A scene (numpy.array): A scene containing transformed text lines.
			A scene text mask (numpy.array) or a list of text masks (list of numpy.array's): A scene mask containing masks of transformed text lines in a scene.
			A list of transformed bounding rectangles (list of numpy.array's): A list of transformed bounding rectangles (4 x 2) in a scene.
		"""

		scene_size = scene.shape[:2]

		scene_mask = np.zeros(scene_size, dtype=np.uint16)
		#scene_text_masks = list()
		bboxes = list()
		if blend_ratio_interval is None:
			for idx, (text, mask) in enumerate(zip(texts, text_masks)):
				text, mask, bbox = self._textTransformer(text, mask, scene_size, *args, **kwargs)

				#--------------------
				pixels = np.where(mask > 0)
				#pixels = np.where(text > 0)

				if 2 == text.ndim:
					scene[:,:][pixels] = text[pixels]
				elif 3 == text.ndim:
					scene[:,:,:text.shape[-1]][pixels] = text[pixels]
				else:
					print('[SWL] Invalid number {} of channels in the {}-th text, {}.'.format(text.shape[-1], idx, text))
					continue
				scene_mask[pixels] = idx + 1
				#scene_text_masks.append(mask)
				bboxes.append(bbox)
		else:
			scene_text = np.zeros_like(scene)
			for idx, (text, mask) in enumerate(zip(texts, text_masks)):
				text, mask, bbox = self._textTransformer(text, mask, scene_size, *args, **kwargs)

				#--------------------
				pixels = np.where(mask > 0)
				#pixels = np.where(text > 0)

				if 2 == text.ndim:
					scene_text[:,:][pixels] = text[pixels]
				elif 3 == text.ndim:
					scene_text[:,:,:text.shape[-1]][pixels] = text[pixels]
				else:
					print('[SWL] Invalid number {} of channels in the {}-th text, {}.'.format(text.shape[-1], idx, text))
					continue
				scene_mask[pixels] = idx + 1
				#scene_text_masks.append(mask)
				bboxes.append(bbox)

			pixels = np.where(scene_mask > 0)
			alpha = random.uniform(blend_ratio_interval[0], blend_ratio_interval[1])
			scene[pixels] = (1.0 - alpha) * scene[pixels] + alpha * scene_text[pixels]

		return scene, scene_mask, np.array(bboxes)
		#return scene, scene_text_masks, np.array(bboxes)
