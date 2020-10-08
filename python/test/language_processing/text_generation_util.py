import os, math, random, glob, csv, time, copy
from functools import reduce
import numpy as np
import cv2
import swl.language_processing.util as swl_langproc_util
from swl.language_processing.text_generator import Transformer, HangeulJamoGenerator, HangeulLetterGenerator, TextGenerator, SceneProvider, SceneTextGenerator

def construct_charset(digit=True, alphabet_uppercase=True, alphabet_lowercase=True, punctuation=True, space=True, whitespace=False, hangeul=False, hangeul_jamo=False, latin=False, greek_uppercase=False, greek_lowercase=False, chinese=False, hiragana=False, katakana=False, unit=False, currency=False, symbol=False, math_symbol=False, hangeul_letter_filepath=None):
	charset = ''

	# Latin.
	# Unicode: Basic Latin (U+0020 ~ U+007F).
	import string
	if digit:
		charset += string.digits
	if alphabet_uppercase:
		charset += string.ascii_uppercase
	if alphabet_lowercase:
		charset += string.ascii_lowercase
	if punctuation:
		charset += string.punctuation
	if space:
		charset += ' '
	if whitespace:
		#charset += '\n\t\v\b\r\f\a'
		charset += '\n\t\v\r\f'

	if hangeul:
		# Unicode: Hangul Syllables (U+AC00 ~ U+D7AF).
		#charset += ''.join([chr(ch) for ch in range(0xAC00, 0xD7A3 + 1)])

		if hangeul_letter_filepath is None:
			hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
			#hangeul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
			#hangeul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
		with open(hangeul_letter_filepath, 'r', encoding='UTF-8') as fd:
			#charset += fd.read().strip('\n')  # A string.
			charset += fd.read().replace(' ', '').replace('\n', '')  # A string.
			#charset += fd.readlines()  # A list of strings.
			#charset += fd.read().splitlines()  # A list of strings.
	if hangeul_jamo:
		# Unicode: Hangul Jamo (U+1100 ~ U+11FF), Hangul Compatibility Jamo (U+3130 ~ U+318F), Hangul Jamo Extended-A (U+A960 ~ U+A97F), & Hangul Jamo Extended-B (U+D7B0 ~ U+D7FF).
		##unicodes = list(range(0x1100, 0x11FF + 1)) + list(range(0x3131, 0x318E + 1))
		#unicodes = range(0x3131, 0x318E + 1)
		#charset += ''.join([chr(ch) for ch in unicodes])

		#charset += 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		charset += 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		#charset += 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	if latin:
		# Unicode: Latin-1 Supplement (U+0080 ~ U+00FF), Latin Extended-A (U+0100 ~ U+017F), Latin Extended-B (U+0180 ~ U+024F).
		charset += ''.join([chr(ch) for ch in range(0x00C0, 0x024F + 1)])

	# Unicode: Greek and Coptic (U+0370 ~ U+03FF) & Greek Extended (U+1F00 ~ U+1FFF).
	if greek_uppercase:
		unicodes = list(range(0x0391, 0x03A1 + 1)) + list(range(0x03A3, 0x03A9 + 1))
		charset += ''.join([chr(ch) for ch in unicodes])
	if greek_lowercase:
		unicodes = list(range(0x03B1, 0x03C1 + 1)) + list(range(0x03C3, 0x03C9 + 1))
		charset += ''.join([chr(ch) for ch in unicodes])

	if chinese:
		# Unicode: CJK Unified Ideographs (U+4E00 ~ U+9FFF) & CJK Unified Ideographs Extension A (U+3400 ~ U+4DBF).
		unicodes = list(range(0x4E00, 0x9FD5 + 1)) + list(range(0x3400, 0x4DB5 + 1))
		charset += ''.join([chr(ch) for ch in unicodes])

	if hiragana:
		# Unicode: Hiragana (U+3040 ~ U+309F).
		charset += ''.join([chr(ch) for ch in range(0x3041, 0x3096 + 1)])
	if katakana:
		# Unicode: Katakana (U+30A0 ~ U+30FF).
		charset += ''.join([chr(ch) for ch in range(0x30A1, 0x30FA + 1)])

	if unit:
		# REF [site] >> http://xahlee.info/comp/unicode_units.html
		unicodes = list(range(0x3371, 0x337A + 1)) + list(range(0x3380, 0x33DF + 1)) + [0x33FF]
		charset += ''.join([chr(ch) for ch in unicodes])

	if currency:
		# Unicode: Currency Symbols (U+20A0 ~ U+20CF).
		charset += ''.join([chr(ch) for ch in range(0x20A0, 0x20BF + 1)])

	if symbol:
		# Unicode: Letterlike Symbols (U+2100 ~ U+214F).
		charset += ''.join([chr(ch) for ch in range(0x2100, 0x214F + 1)])
		# Unicode: Number Forms (U+2150 ~ U+218F).
		charset += ''.join([chr(ch) for ch in range(0x2150, 0x218F + 1)])
		# Unicode: Arrows (U+2190 ~ U+21FF).
		charset += ''.join([chr(ch) for ch in range(0x2190, 0x21FF + 1)])
		# Unicode: Enclosed Alphanumerics (U+2460 ~ U+24FF).
		#charset += ''.join([chr(ch) for ch in range(0x2460, 0x24FF + 1)])
		# Unicode: Geometric Shapes (U+25A0 ~ U+25FF).
		#charset += ''.join([chr(ch) for ch in range(0x25A0, 0x25FF + 1)])
		# Unicode: Miscellaneous Symbols (U+2600 ~ U+26FF).
		charset += ''.join([chr(ch) for ch in range(0x2600, 0x26FF + 1)])
		# Unicode: Dingbats (U+2700 ~ U+27BF).
		#charset += ''.join([chr(ch) for ch in range(0x2700, 0x27BF + 1)])
		# Unicode: Supplemental Arrows-A (U+27F0 ~ U+27FF).
		charset += ''.join([chr(ch) for ch in range(0x27F0, 0x27FF + 1)])
		# Unicode: Supplemental Arrows-B (U+2900 ~ U+297F).
		charset += ''.join([chr(ch) for ch in range(0x2900, 0x297F + 1)])
		# Unicode: Miscellaneous Symbols and Arrows (U+2B00 ~ U+2BFF).
		charset += ''.join([chr(ch) for ch in range(0x2B00, 0x2BFF + 1)])

	if math_symbol:
		# Unicode: Mathematical Operators (U+2200 ~ U+22FF).
		charset += ''.join([chr(ch) for ch in range(0x2200, 0x22FF + 1)])
		# Unicode: Miscellaneous Mathematical Symbols-A (U+27C0 ~ U+27EF).
		charset += ''.join([chr(ch) for ch in range(0x27C0, 0x27EF + 1)])
		# Unicode: Miscellaneous Mathematical Symbols-B (U+2980 ~ U+29FF).
		charset += ''.join([chr(ch) for ch in range(0x2980, 0x29FF + 1)])
		# Unicode: Supplemental Mathematical Operators (U+2A00 ~ U+2AFF).
		charset += ''.join([chr(ch) for ch in range(0x2A00, 0x2AFF + 1)])

	return charset

def construct_word_set(korean=True, english=True, korean_dictionary_filepath=None, english_dictionary_filepath=None):
	words = list()
	if korean:
		if korean_dictionary_filepath is None:
			korean_dictionary_filepath = '../../data/language_processing/dictionary/korean_wordslistUnique.txt'

		print('Start loading a Korean dictionary...')
		start_time = time.time()
		with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
			#korean_words = fd.readlines()
			#korean_words = fd.read().strip('\n')
			korean_words = fd.read().splitlines()
		print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))
		words += korean_words
	if english:
		if english_dictionary_filepath is None:
			#english_dictionary_filepath = '../../data/language_processing/dictionary/english_words.txt'
			english_dictionary_filepath = '../../data/language_processing/wordlist_mono_clean.txt'
			#english_dictionary_filepath = '../../data/language_processing/wordlist_bi_clean.txt'

		print('Start loading an English dictionary...')
		start_time = time.time()
		with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
			#english_words = fd.readlines()
			#english_words = fd.read().strip('\n')
			english_words = fd.read().splitlines()
		print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))
		words += english_words

	return set(words)

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

	num_fonts = 1
	font_list = list()
	for fpath in font_filepaths:
		#num_fonts = 4 if os.path.basename(fpath).lower() in ['gulim.ttf', 'batang.ttf'] else 1
		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	return font_list

def construct_font(font_dir_paths):
	font_list = list()
	for dir_path in font_dir_paths:
		font_filepaths = glob.glob(os.path.join(dir_path, '*.ttf'))
		#font_list = generate_hangeul_font_list(font_filepaths)
		font_list.extend(generate_font_list(font_filepaths))
	return font_list

#--------------------------------------------------------------------

def generate_random_words(chars, min_char_len=1, max_char_len=10):
	chars = list(chars)
	random.shuffle(chars)
	chars = ''.join(chars)
	num_chars = len(chars)

	words = list()
	start_idx = 0
	while True:
		end_idx = start_idx + random.randint(min_char_len, max_char_len)
		words.append(chars[start_idx:end_idx])
		if end_idx >= num_chars:
			break
		start_idx = end_idx

	return words

def generate_random_text_lines(words, min_word_len=1, max_word_len=5):
	random.shuffle(words)
	num_words = len(words)

	texts = list()
	start_idx = 0
	while True:
		end_idx = start_idx + random.randint(min_word_len, max_word_len)
		texts.append(' '.join(words[start_idx:end_idx]))
		if end_idx >= num_words:
			break
		start_idx = end_idx

	return texts

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

def generate_phd08_dict(from_npy=True):
	if from_npy:
		import swl.language_processing.hangeul_handwriting_dataset as hg_hw_dataset

		# Loads PHD08 npy dataset.
		# Generate an info file for npy files generated from the PHD08 dataset.
		#	Refer to generate_npy_dataset_from_phd08_conversion_result() in ${SWL_PYTHON_HOME}/test/language_processing/phd08_datset_test.py.
		phd08_npy_dataset_info_filepath = './phd08_npy_dataset.csv'
		print('Start loading PHD08 npy dataset...')
		start_time = time.time()
		char_images_dict = hg_hw_dataset.load_phd08_npy(phd08_npy_dataset_info_filepath, is_dark_background=False)
		for key, values in char_images_dict.items():
			char_images_dict[key] = list()
			for val in values:
				val = cv2.cvtColor(cv2.bitwise_not(val), cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255
				char_images_dict[key].append(val)
		print('End loading PHD08 npy dataset: {} secs.'.format(time.time() - start_time))
	else:
		# Loads PHD08 image dataset.
		phd08_image_dataset_info_filepath = './phd08_png_dataset.csv'
		print('Start loading PHD08 image dataset...')
		start_time = time.time()
		char_images_dict = phd08_dataset.load_phd08_image(phd08_image_dataset_info_filepath, is_dark_background=False)
		for key, values in char_images_dict.items():
			char_images_dict[key] = list()
			for val in values:
				val = cv2.cvtColor(cv2.bitwise_not(val), cv2.COLOR_BGRA2GRAY).astype(np.float32) / 255
				char_images_dict[key].append(val)
		print('End loading PHD08 image dataset: {} secs.'.format(time.time() - start_time))

	return char_images_dict

#--------------------------------------------------------------------

def constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color, text_line_size=None):
	"""Constructs a text line from character alpha mattes.

	Inputs:
		char_alpha_list (a list of numpy.array): A list of character alpha mattes.
		char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
		font_color (tuple): A font color for the characters.
		text_line_size (tuple): The size of the text line.
	Outputs:
		A text line (numpy.array): A single text line constructed from character alpha mattes.
	"""

	import swl.machine_vision.util as swl_cv_util

	if text_line_size is None:
		text_line_size = reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])), map(lambda alpha, coord: (coord[0] + alpha.shape[0], coord[1] + alpha.shape[1]), char_alpha_list, char_alpha_coordinate_list))

	image_channel = len(font_color)
	text_line_alpha = np.zeros(text_line_size, dtype=np.float32)
	font_color = list(map(lambda x: x / 255, font_color))
	if 1 == image_channel:
		text_line = np.zeros(text_line_size, dtype=np.float32)
		def apply_blending(alpha, alpha_coords):
			sy, sx = alpha_coords
			#pixels = np.where(alpha > 0)
			#text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]][pixels] = alpha[pixels]	
			text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full(alpha.shape, font_color, dtype=np.float32), text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
			text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full_like(alpha, 1.0, dtype=np.float32), text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)		
	else:
		text_line = np.zeros(text_line_size + (image_channel,), dtype=np.float32)
		def apply_blending(alpha, alpha_coords):
			sy, sx = alpha_coords
			#pixels = np.where(alpha > 0)
			#text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]][pixels] = alpha[pixels]	
			text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full(alpha.shape + (image_channel,), font_color, dtype=np.float32), text_line[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)
			text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]] = swl_cv_util.blend_image(np.full_like(alpha, 1.0, dtype=np.float32), text_line_alpha[sy:sy+alpha.shape[0],sx:sx+alpha.shape[1]], alpha)		
	for alpha, alpha_coords in zip(char_alpha_list, char_alpha_coordinate_list):
		apply_blending(alpha, alpha_coords)

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
		#import imgaug as ia
		from imgaug import augmenters as iaa

		self._seq = iaa.Sequential([
			iaa.Sometimes(0.5, iaa.Affine(
				scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  # Scale images to 80-120% of their size, individually per axis.
				translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translate by -10 to +10 percent along x-axis and -10 to +10 percent along y-axis.
				rotate=(-2, 2),  # Rotate by -2 to +2 degrees.
				shear=(-2, 2),  # Shear by -2 to +2 degrees.
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
		#import imgaug as ia
		from imgaug import augmenters as iaa

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

#class SimpleCharacterAlphaMatteGenerator(CharacterAlphaMatteGenerator):
class SimpleCharacterAlphaMatteGenerator(object):
	"""Generates an alpha-matte [0, 1] for a character which reflects the proportion of foreground (when alpha=1) and background (when alpha=0).
	"""

	def __init__(self, font_list, char_images_dict=None, mode='1'):
		"""Constructor.

		Inputs:
			font_list (a list of (font file path, font index) pairs): A list of the file paths and the font indices of fonts.
			char_images_dict (a dict of (character, a list of images)): A dictionary of characters and their corresponding list of images.
			mode (str): The color mode for alpha matte. Black-white mode ('1') or grayscale mode ('L').
		"""

		self._font_list = font_list
		self._char_images_dict = char_images_dict
		self._alpha_matte_mode = mode

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

	def __call__(self, char, font_size, *args, **kwargs):
		"""Generates a character and its mask of numpy.array.

		Inputs:
			char (str): A single character.
			font_size (int): A font size for the character.
		Outputs:
			A character (numpy.array): An alpha matte for an input character.
		"""

		#image_size = (math.ceil(font_size * 1.1), math.ceil(font_size * 1.1))
		image_size = (font_size * 2, font_size * 2)
		#image_size = None

		if self._char_images_dict is not None and char in self._char_images_dict:
			use_printed_letter = 0 == random.randrange(2)
		else:
			use_printed_letter = True

		if use_printed_letter:
			#print('Generate a printed letter.')
			font_color, bg_color = 255, 0

			while True:
				font_type, font_index = random.choice(self._font_list)
				try:
					alpha = swl_langproc_util.generate_text_image(char, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, mode=self._alpha_matte_mode)
					break
				except OSError as ex:
					print('Warning: Font = {}, Index = {}: {}.'.format(font_type, font_index, ex))

			if '1' == self._alpha_matte_mode:
				return np.array(alpha, dtype=np.float32)
			else:
				return np.array(alpha, dtype=np.float32) / 255
		else:
			#print('Generate a handwritten Hangeul letter.')
			return random.choice(self._char_images_dict[char])

#class SimpleCharacterPositioner(CharacterPositioner):
class SimpleCharacterPositioner(object):
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

class BasicPrintedTextGenerator(object):
	"""Generates a basic printed text line for individual characters.
	"""

	def __init__(self, font_list, font_size_interval, char_space_ratio_interval=None, mode='RGB', mask_mode='1'):
		"""Constructor.

		Inputs:
			font_list (a list of (font file path, font index) pairs): A list of the file paths and the font indices of fonts.
			font_size_interval (a tuple of two ints): A font size interval for the characters.
			char_space_ratio_interval (a tuple of two floats): A space ratio interval between two characters.
			mode (str): The color mode of image. RGB mode ('RGB') or grayscale mode ('L').
			mask_mode (str): The color mode of image mask. Black-white mode ('1') or grayscale mode ('L').
		"""

		self._font_list = font_list
		self._font_size_interval = font_size_interval
		self._char_space_ratio_interval = char_space_ratio_interval
		self._mode = mode
		self._mask_mode = mask_mode

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

	def __call__(self, text, font_color, bg_color, *args, **kwargs):
		"""Generates a single text line for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
			font_color (int or list of int): A font color.
			bg_color (int or list of int): A background color.
		Outputs:
			text_image (numpy.array): A generated text image.
			text_mask (numpy.array): A generated text mask.
		"""

		#image_size = (math.ceil(len(text) * font_size * 2), math.ceil(font_size * 2))
		image_size = None

		font_size = random.randint(*self._font_size_interval)
		char_space_ratio = None if self._char_space_ratio_interval is None else random.uniform(*self._char_space_ratio_interval)

		while True:
			font_type, font_index = random.choice(self._font_list)
			try:
				text_image, text_mask = swl_langproc_util.generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, char_space_ratio, mode=self._mode, mask=True, mask_mode=self._mask_mode)
				break
			except OSError as ex:
				print('Warning: Font = {}, Index = {}: {}.'.format(font_type, font_index, ex))

		#return np.array(text_image), np.array(text_mask)  # text_mask: np.bool.
		return np.array(text_image), np.array(text_mask, dtype=np.uint8)

	def create_subset_generator(self, texts, batch_size, color_functor=None):
		if batch_size <= 0 or batch_size > len(texts):
			raise ValueError('Invalid batch size: 0 < batch_size <= len(texts)')

		if color_functor is None:
			color_functor = lambda: (None, None)

		while True:
			sub_texts = random.sample(texts, k=batch_size)
			#sub_texts = random.choices(texts, k=batch_size)

			text_list, image_list, mask_list = list(), list(), list()
			for txt in sub_texts:
				font_color, bg_color = color_functor()
				text_line, mask = self.__call__(txt, font_color, bg_color)

				text_list.append(txt)
				image_list.append(text_line)
				mask_list.append(mask)

			yield text_list, image_list, mask_list

	def create_whole_generator(self, texts, batch_size, color_functor=None, shuffle=False):
		num_words = len(texts)
		if batch_size <= 0 or batch_size > num_words:
			raise ValueError('Invalid batch size: 0 < batch_size <= len(texts)')

		if color_functor is None:
			color_functor = lambda: (None, None)

		if shuffle:
			random.shuffle(texts)
		start_idx = 0
		while True:
			end_idx = start_idx + batch_size
			sub_texts = texts[start_idx:end_idx]

			text_list, image_list, mask_list = list(), list(), list()
			for txt in sub_texts:
				font_color, bg_color = color_functor()
				text_line, mask = self.__call__(txt, font_color, bg_color)

				text_list.append(txt)
				image_list.append(text_line)
				mask_list.append(mask)

			yield text_list, image_list, mask_list

			if end_idx >= num_words:
				break
			start_idx = end_idx

class BasicTextAlphaMatteGenerator(object):
	"""Generates a simple text line and masks for individual characters.
	"""

	def __init__(self, characterTransformer, characterPositioner, font_list, font_size_interval, char_space_ratio_interval=None, char_images_dict=None, alpha_matte_mode='1'):
		"""Constructor.

		Inputs:
			characterTransformer (Transformer): An object to tranform each character.
			characterPositioner (CharacterPositioner): An object to place characters.
			font_list (a list of (font file path, font index) pairs): A list of the file paths and the font indices of fonts.
			font_size_interval (a tuple of two ints): A font size interval for the characters.
			char_space_ratio_interval (a tuple of two floats): A space ratio interval between two characters.
			char_images_dict (a dict of (character, a list of images)): A dictionary of characters and their corresponding list of images.
			alpha_matte_mode (str): The color mode for alpha matte. Black-white mode ('1') or grayscale mode ('L').
		"""

		self._characterTransformer = characterTransformer
		self._characterPositioner = characterPositioner

		self._font_list = font_list
		self._char_images_dict = char_images_dict  # FIXME [fix] >> Currently not used.
		self._font_size_interval = font_size_interval
		self._char_space_ratio_interval = char_space_ratio_interval
		self._alpha_matte_mode = alpha_matte_mode

		self._text_offset = (0, 0)
		self._crop_text_area = True
		self._draw_text_border = False

	def __call__(self, text, *args, **kwargs):
		"""Generates a single text line and masks for individual characters.

		Inputs:
			text (str): Characters to compose a text line.
		Outputs:
			char_alpha_list (a list of numpy.array): A list of character images.
			char_alpha_coordinate_list (a list of (int, int)): A list of (y position, x position) of character alpha mattes.
				A text line can be constructed by constructTextLine().
		"""

		font_size = random.randint(*self._font_size_interval)
		char_space_ratio = None if self._char_space_ratio_interval is None else random.uniform(*self._char_space_ratio_interval)

		#image_size = (math.ceil(font_size * 1.1), math.ceil(font_size * 1.1))
		image_size = (font_size * 2, font_size * 2)
		#image_size = None

		font_color, bg_color = 255, 0

		while True:
			font_type, font_index = random.choice(self._font_list)
			try:
				# For testing fonts.
				swl_langproc_util.generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, mode=self._alpha_matte_mode)
				break
			except OSError as ex:
				print('Warning: Font = {}, Index = {}: {}.'.format(font_type, font_index, ex))

		char_alpha_list = list()
		for ch in text:
			alpha = swl_langproc_util.generate_text_image(ch, font_type, font_index, font_size, font_color, bg_color, image_size, self._text_offset, self._crop_text_area, self._draw_text_border, mode=self._alpha_matte_mode)
			if '1' == self._alpha_matte_mode:
				alpha = np.array(alpha, dtype=np.float32)
			else:
				alpha = np.array(alpha, dtype=np.float32) / 255
			alpha, _, _ = self._characterTransformer(alpha, None, *args, **kwargs)
			char_alpha_list.append(alpha)

		char_alpha_coordinate_list = self._characterPositioner(char_alpha_list, char_space_ratio, *args, **kwargs)
		return char_alpha_list, char_alpha_coordinate_list

	def create_subset_generator(self, texts, batch_size, color_functor):
		if batch_size <= 0 or batch_size > len(texts):
			raise ValueError('Invalid batch size: 0 < batch_size <= len(texts)')

		sceneTextGenerator = SimpleAlphaMatteSceneTextGenerator(IdentityTransformer())

		while True:
			sub_texts = random.sample(texts, k=batch_size)
			#sub_texts = random.choices(texts, k=batch_size)

			text_list, scene_list, scene_text_mask_list = list(), list(), list()
			for txt in sub_texts:
				font_color, bg_color = color_functor()
				#if font_color is None:
				#	font_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
				#if bg_color is None:
				#	bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.

				char_alpha_list, char_alpha_coordinate_list = self.__call__(txt)
				text_line, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

				scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
				text_list.append(txt)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)

			yield text_list, scene_list, scene_text_mask_list

	def create_whole_generator(self, texts, batch_size, color_functor, shuffle=False):
		num_words = len(texts)
		if batch_size <= 0 or batch_size > num_words:
			raise ValueError('Invalid batch size: 0 < batch_size <= len(texts)')

		sceneTextGenerator = SimpleAlphaMatteSceneTextGenerator(IdentityTransformer())

		if shuffle:
			random.shuffle(texts)
		start_idx = 0
		while True:
			end_idx = start_idx + batch_size
			sub_texts = texts[start_idx:end_idx]

			text_list, scene_list, scene_text_mask_list = list(), list(), list()
			for txt in sub_texts:
				font_color, bg_color = color_functor()
				#if font_color is None:
				#	font_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
				#if bg_color is None:
				#	bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.

				char_alpha_list, char_alpha_coordinate_list = self.__call__(txt)
				text_line, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

				scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
				text_list.append(txt)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)

			yield text_list, scene_list, scene_text_mask_list

			if end_idx >= num_words:
				break
			start_idx = end_idx

#class SimpleTextAlphaMatteGenerator(TextAlphaMatteGenerator):
class SimpleTextAlphaMatteGenerator(object):
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

	def create_subset_generator(self, texts, batch_size, color_functor):
		if batch_size <= 0 or batch_size > len(texts):
			raise ValueError('Invalid batch size: 0 < batch_size <= len(texts)')

		sceneTextGenerator = SimpleAlphaMatteSceneTextGenerator(IdentityTransformer())

		font_color = None  # Uses a random font color.
		while True:
			sub_texts = random.sample(texts, k=batch_size)
			#sub_texts = random.choices(texts, k=batch_size)

			text_list, scene_list, scene_text_mask_list = list(), list(), list()
			for txt in sub_texts:
				font_color, bg_color = color_functor()
				#if font_color is None:
				#	font_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
				#if bg_color is None:
				#	bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.

				char_alpha_list, char_alpha_coordinate_list = self.__call__(txt)
				text_line, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

				scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
				text_list.append(txt)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)

			yield text_list, scene_list, scene_text_mask_list

	def create_whole_generator(self, texts, batch_size, color_functor=None, shuffle=False):
		num_words = len(texts)
		if batch_size <= 0 or batch_size > num_words:
			raise ValueError('Invalid batch size: 0 < batch_size <= len(texts)')

		sceneTextGenerator = SimpleAlphaMatteSceneTextGenerator(IdentityTransformer())

		if shuffle:
			random.shuffle(texts)
		start_idx = 0
		while True:
			end_idx = start_idx + batch_size
			sub_texts = texts[start_idx:end_idx]

			text_list, scene_list, scene_text_mask_list = list(), list(), list()
			for txt in sub_texts:
				font_color, bg_color = color_functor()
				#if font_color is None:
				#	font_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
				#if bg_color is None:
				#	bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.

				char_alpha_list, char_alpha_coordinate_list = self.__call__(txt)
				text_line, text_line_alpha = constructTextLine(char_alpha_list, char_alpha_coordinate_list, font_color)

				bg = np.full_like(text_line, bg_color, dtype=np.uint8)

				scene, scene_text_mask, _ = sceneTextGenerator(bg, [text_line], [text_line_alpha])
				text_list.append(txt)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)

			yield text_list, scene_list, scene_text_mask_list

			if end_idx >= num_words:
				break
			start_idx = end_idx

#class SimpleAlphaMatteSceneTextGenerator(AlphaMatteSceneTextGenerator):
class SimpleAlphaMatteSceneTextGenerator(object):
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

		import swl.machine_vision.util as swl_cv_util

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

	def create_generator(self, textGenerator, sceneProvider, texts, batch_size, text_count_interval, color_functor):
		while True:
			texts_list, scene_list, scene_text_mask_list, bboxes_list = list(), list(), list(), list()
			for _ in range(batch_size):
				num_texts_per_image = random.randint(*text_count_interval)
				sub_texts = random.choices(texts, k=num_texts_per_image)

				text_images, text_alphas = list(), list()
				for txt in sub_texts:
					font_color, _ = color_functor()
					#if font_color is None:
					#	font_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.
					#if bg_color is None:
					#	bg_color = (random.randrange(256),) * 3  # Uses a specific grayscale background color.

					char_alpha_list, char_alpha_coordinate_list = textGenerator(txt)
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
				texts_list.append(sub_texts)
				scene_list.append(scene)
				scene_text_mask_list.append(scene_text_mask)
				bboxes_list.append(bboxes)

			yield texts_list, scene_list, scene_text_mask_list, bboxes_list

#class GrayscaleBackgroundProvider(SceneProvider):
class GrayscaleBackgroundProvider(object):
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

#class SimpleSceneProvider(SceneProvider):
class SimpleSceneProvider(object):
	def __init__(self, scene_filepaths):
		self._scene_filepaths = scene_filepaths

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
