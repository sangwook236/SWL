import hgtk
import functools, operator

def hangeul2jamo(text, eoc_str, compose_code='\u1d25', use_separate_consonants=False, use_separate_vowels=False):
	"""Convert a string to a list of jamos and letters.

	Inputs:
		text (str): Input string.
		eoc_str (str): End-of-Character string.
		compose_code (str): A code to split sets of jamos which consists of each letter.
		use_separate_consonants (bool): Decides whether consonants which consist of two consonants are used or not. e.g.) ㄼ -> ㄹㅂ if True.
		use_separate_vowels (bool): Decides whether vowels which consist of two vowels are used or not. e.g.) ㅘ -> ㅗㅏ if True.
	Outputs:
		jamo_text (a list of str): A list of jamos decomposed from the input string.
	"""

	consonants = {'ㄲ': 'ㄱㄱ', 'ㄳ': 'ㄱㅅ', 'ㄵ': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄸ': 'ㄷㄷ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ', 'ㅃ': 'ㅂㅂ', 'ㅄ': 'ㅂㅅ', 'ㅆ': 'ㅅㅅ', 'ㅉ': 'ㅈㅈ'}
	vowels = {'ㅘ': 'ㅗㅏ', 'ㅙ': 'ㅗㅐ', 'ㅚ': 'ㅗㅣ', 'ㅝ': 'ㅜㅓ', 'ㅞ': 'ㅜㅔ', 'ㅟ': 'ㅜㅣ', 'ㅢ': 'ㅡㅣ'}

	jamo_text = list(hgtk.text.decompose(text, latin_filter=True, compose_code=compose_code))
	if use_separate_consonants:
		jamo_text = list(map(lambda jm: list(consonants[jm]) if jm in consonants else jm, jamo_text))
		jamo_text = functools.reduce(operator.iconcat, jamo_text, [])
	if use_separate_vowels:
		jamo_text = list(map(lambda jm: list(vowels[jm]) if jm in vowels else jm, jamo_text))
		jamo_text = functools.reduce(operator.iconcat, jamo_text, [])
	return list(map(lambda x: eoc_str if x == compose_code else x, jamo_text))
	"""
	jamo_text = list(map(lambda ch: hgtk.letter.decompose(ch) + (eoc_str,) if hgtk.checker.is_hangul(ch) else ch, text))
	jamo_text = functools.reduce(operator.iconcat, jamo_text, [])
	return list(filter(lambda x: x, jamo_text))  # Removes empty strings.
	"""

def jamo2hangeul(jamo_text, eoc_str, compose_code='\u1d25', use_separate_consonants=False, use_separate_vowels=False):
	"""Convert a list of jamos and letters to a string.

	Inputs:
		jamo_text (a list of str): A list of jamos decomposed from the input string.
		eoc_str (str): End-of-Character string.
		compose_code (str): A code to split sets of jamos which consists of each letter.
		use_separate_consonants (bool): Decides whether consonants which consist of two consonants are used or not. e.g.) ㄹㅂ -> ㄼ if True.
		use_separate_vowels (bool): Decides whether vowels which consist of two vowels are used or not. e.g.) ㅗㅏ -> ㅘ if True.
	Outputs:
		text (str): Input string.
	"""

	consonants = {'ㄱㄱ': 'ㄲ', 'ㄱㅅ': 'ㄳ', 'ㄴㅈ': 'ㄵ', 'ㄴㅎ': 'ㄶ', 'ㄷㄷ': 'ㄸ', 'ㄹㄱ': 'ㄺ', 'ㄹㅁ': 'ㄻ', 'ㄹㅂ': 'ㄼ', 'ㄹㅅ': 'ㄽ', 'ㄹㅌ': 'ㄾ', 'ㄹㅍ': 'ㄿ', 'ㄹㅎ': 'ㅀ', 'ㅂㅂ': 'ㅃ', 'ㅂㅅ': 'ㅄ', 'ㅅㅅ': 'ㅆ', 'ㅈㅈ': 'ㅉ'}
	vowels = {'ㅗㅏ': 'ㅘ', 'ㅗㅐ': 'ㅙ', 'ㅗㅣ': 'ㅚ', 'ㅜㅓ': 'ㅝ', 'ㅜㅔ': 'ㅞ', 'ㅜㅣ': 'ㅟ', 'ㅡㅣ': 'ㅢ'}

	"""
	if use_separate_consonants:
		jamo_text = list(map(lambda jm: consonants[jm] if jm in consonants else jm, jamo_text))
	if use_separate_vowels:
		jamo_text = list(map(lambda jm: vowels[jm] if jm in vowels else jm, jamo_text))

	text = ''.join(list(map(lambda x: compose_code if x == eoc_str else x, jamo_text)))
	return hgtk.text.compose(text, compose_code=compose_code)
	"""
	text = ''.join(jamo_text)
	if use_separate_consonants:
		for jm in consonants:
			text = text.replace(jm, consonants[jm])
	if use_separate_vowels:
		for jm in vowels:
			text = text.replace(jm, vowels[jm])
	#text = list(map(lambda x: compose_code if x == eoc_str else x, text))
	text = text.replace(eoc_str, compose_code)
	return hgtk.text.compose(text, compose_code=compose_code)
