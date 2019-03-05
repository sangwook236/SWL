import abc
import cv2

#%%------------------------------------------------------------------

class Transformer(object):
	"""Transforms a numpy.array.
	"""

	def __call__(self, input, mask, *args, **kwargs):
		"""Transforms a numpy.array.

		Inputs:
			input (numpy.array): a 2D or 3D numpy.array to transform.
			mask (numpy.array): a mask of input to transform. It can be None.
		Outputs:
			transformed input (numpy.array): a transformed 2D or 3D numpy.array.
			transformed mask (numpy.array): a transformed mask.
		"""

		raise NotImplementedError

#%%------------------------------------------------------------------

class HangeulJamoGenerator(object):
	"""Generates a Hangeul jamo and its mask of numpy.array.

	Make sure if only a mask of a jamo may be needed.
	A jamo can be RGB, grayscale, or binary (black and white).
	A mask is binary (black(bg) and white(fg)).
	"""

	def __call__(self, jamo, *args, **kwargs):
		"""Generates a Hangeul jamo and its mask of numpy.array.

		Inputs:
			jamo (str): a single Hangeul jamo.
		Outputs:
			jamo (numpy.array): a numpy.array generated from an input Hangeul jamo.
		"""

		raise NotImplementedError

class HangeulJamoPositioner(object):
	"""Places jamos to construct a Hangeul letter.
	"""

	def __call__(self, jamo_list, mask_list, shape_type, *args, **kwargs):
		"""Places jamos to construct a Hangeul letter.

		Inputs:
			jamo_list (a list of numpy.array): a list of jamos of type numpy.array to compose a letter.
			mask_list (a list of numpy.array): a list of masks of jamos in jamo_list.
			shape_type (int): the shape type of a letter. [1, 6].
		Outputs:
			A Hangeul letter (numpy.array): a Hangeul letter made up of jamos.
			Masks (a list of numpy.array): a list of jamos' masks. A mask is binary (black(bg) and white(fg)). It can be None.
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

class HangeulLetterGenerator(abc.ABC):
	"""Generates a Hangeul letter made up of jamos.
	"""

	def __init__(self, jamoGenerator, jamoTransformer, jamoPositioner):
		"""Constructor.

		Inputs:
			jamoGenerator (HangeulJamoGenerator): an object to generate each jamo.
			jamoTransformer (Transformer): an object to tranform each jamo.
			jamoPositioner (HangeulJamoPositioner): an object to place jamos.
		"""

		self._jamoGenerator = jamoGenerator
		self._jamoTransformer = jamoTransformer
		self._jamoPositioner = jamoPositioner

	def __call__(self, jamos, shape_type, *args, **kwargs):
		"""Generates a Hangeul letter made up of jamos.

		Inputs:
			jamos (str): jamos to compose a letter.
			shape_type (int): the shape type of a letter. [1, 6].
		Outputs:
			A Hangeul letter (numpy.array): a Hangeul letter made up of jamos.
			Masks (a list of numpy.array): a list of jamos' masks.
		"""

		jamo_list, mask_list = list(), list()
		for jamo in jamos:
			jamo, mask = self._jamoGenerator(jamo)
			jamo, mask = self._jamoTransformer(jamo, mask)
			jamo_list.append(jamo)
			mask_list.append(mask)

		return self._jamoPositioner(jamo_list, mask_list, shape_type, *args, **kwargs)

#%%------------------------------------------------------------------

class CharacterGenerator(object):
	"""Generates a character and its mask of numpy.array.

	Make sure if only a mask of a character may be needed.
	A character can be RGB, grayscale, or binary (black and white).
	A mask is binary (black(bg) and white(fg)).
	"""

	def __call__(self, char, *args, **kwargs):
		"""Generates a character and its mask of numpy.array.

		Inputs:
			char (str): a single character.
		Outputs:
			char (numpy.array): a numpy.array generated from an input character.
			mask (numpy.array): a mask of an input character, char. A mask is binary (black(bg) and white(fg)). It can be None.
		"""

		raise NotImplementedError

class CharacterPositioner(object):
	"""Place characters to construct a text line.
	"""

	def __call__(self, char_list, mask_list, *args, **kwargs):
		"""Places characters to construct a single text line.

		Inputs:
			char_list (a list of numpy.array): a list of characters of type numpy.array to compose a text line.
			mask_list (a list of numpy.array): a list of masks of characters in char_list.
		Outputs:
			A text line (numpy.array): a text line of type 2D numpy.array made up of char_list.
			Masks (a list of numpy.array): a list of characters' masks. It can be None.
		"""

		raise NotImplementedError

class TextGenerator(abc.ABC):
	"""Generates a single text line and its mask made up of characters.
	"""

	def __init__(self, characterGenerator, characterTransformer, characterPositioner):
		"""Constructor.

		Inputs:
			characterGenerator (CharacterGenerator): an object to generate each character.
			characterTransformer (Transformer): an object to tranform each character.
			characterPositioner (CharacterPositioner): an object to place characters.
		"""

		self._characterGenerator = characterGenerator
		self._characterTransformer = characterTransformer
		self._characterPositioner = characterPositioner

	def __call__(self, text, *args, **kwargs):
		"""Generates a single text line and its mask made up of characters.

		Inputs:
			text (str): characters to compose a text line.
		Outputs:
			A text line (numpy.array): a text line made up of characters.
			Masks (a list of numpy.array): a list of characters' masks.
		"""

		char_list, mask_list = list(), list()
		for ch in text:
			ch, mask = self._characterGenerator(ch)
			ch, mask = self._characterTransformer(ch, mask)
			char_list.append(ch)
			mask_list.append(mask)

		return self._characterPositioner(char_list, mask_list, *args, **kwargs)
