#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import hangeul_util as hg_util
import hgtk

def hgtk_error_case_test():
	# NOTE [info] >> Some special Hangeul jamos (e.g. 'ㆍ', 'ㆅ', 'ㆆ') are ignored in the hgtk library.
	print("hgtk.text.decompose('ㆍ') =", hgtk.text.decompose('ㆍ'))
	print("hgtk.text.decompose('가ㆍ나') =", hgtk.text.decompose('가ㆍ나'))
	print("hgtk.text.decompose('ㆅ') =", hgtk.text.decompose('ㆅ'))
	print("hgtk.text.decompose('ㆆ') =", hgtk.text.decompose('ㆆ'))

def hangeul_jamo_set_test():
	jamo1 = 'ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ'.replace(' ', '')
	jamo2 = 'ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ'.replace(' ', '')
	jamo3 = 'ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅄ ㅅ ㅆ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ'.replace(' ', '')

	jamo_set1 = set(jamo1)
	jamo_set1 = jamo_set1.union(jamo2)
	jamo_set1 = jamo_set1.union(jamo3)
	jamo_set1 = sorted(jamo_set1)

	print('#jamo_set1 =', len(jamo_set1))
	print('jamo_set1 =', jamo_set1)
	#print('jamo_set1 =', ''.join(jamo_set1))

	#--------------------
	jamo1 = 'ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄸ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅃ ㅄ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ'.replace(' ', '')
	jamo2 = 'ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ'.replace(' ', '')

	jamo_set2 = set(jamo1)
	jamo_set2 = jamo_set2.union(jamo2)
	jamo_set2 = sorted(jamo_set2)

	print('#jamo_set2 =', len(jamo_set2))
	print('jamo_set2 =', jamo_set2)
	#print('jamo_set2 =', ''.join(jamo_set2))

	#--------------------
	jamo1 = 'ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄸ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅃ ㅄ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ'.replace(' ', '')
	jamo2 = 'ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ'.replace(' ', '')

	jamo_set3 = set(jamo1)
	jamo_set3 = jamo_set3.union(jamo2)
	jamo_set3 = sorted(jamo_set3)

	print('#jamo_set3 =', len(jamo_set3))
	print('jamo_set3 =', jamo_set3)
	#print('jamo_set3 =', ''.join(jamo_set3))

	#--------------------
	jamo1 = 'ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ'.replace(' ', '')
	jamo2 = 'ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ'.replace(' ', '')

	jamo_set4 = set(jamo1)
	jamo_set4 = jamo_set4.union(jamo2)
	jamo_set4 = sorted(jamo_set4)

	print('#jamo_set4 =', len(jamo_set4))
	print('jamo_set4 =', jamo_set4)
	#print('jamo_set4 =', ''.join(jamo_set4))

	#--------------------
	hangeul_jamo_charset = jamo_set4

	import string
	charset = \
		hangeul_jamo_charset + \
		string.ascii_uppercase + \
		string.ascii_lowercase + \
		string.digits + \
		string.punctuation + \
		' '

	charset = sorted(charset)

	print('#charset =', len(charset))
	print('charset =', charset)
	#print('charset =', ''.join(charset))

def hangeul_jamo_conversion_test():
	text = 'ㄱ나ㄷ 대한 민국이 123 ABC abcDEF def 확률 회의 훠귕궤 딸꾹 밟다 쑥갓'
	#text = '대한 민국이 123'

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOJC>', use_separate_consonants=True, use_separate_vowels=True)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOJC>', use_separate_consonants=True, use_separate_vowels=True)
	print('Text =', text2)

	print('Comparison =', text == text2)

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOJC>', use_separate_consonants=True, use_separate_vowels=False)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOJC>', use_separate_consonants=True, use_separate_vowels=False)
	print('Text =', text2)

	print('Comparison =', text == text2)

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOJC>', use_separate_consonants=False, use_separate_vowels=True)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOJC>', use_separate_consonants=False, use_separate_vowels=True)
	print('Text =', text2)

	print('Comparison =', text == text2)

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOJC>', use_separate_consonants=False, use_separate_vowels=False)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOJC>', use_separate_consonants=False, use_separate_vowels=False)
	print('Text =', text2)

	print('Comparison =', text == text2)

def main():
	hgtk_error_case_test()

	#hangeul_jamo_set_test()

	hangeul_jamo_conversion_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
