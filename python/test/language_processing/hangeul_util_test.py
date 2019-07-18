#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import hangeul_util as hg_util

def hangeul_jamo_test():
	text = 'ㄱ나ㄷ 대한 민국이 123 ABC abcDEF def 확률 회의 훠귕궤 딸꾹 밟다 쑥갓'
	#text = '대한 민국이 123'

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOC>', use_separate_consonants=True, use_separate_vowels=True)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOC>', use_separate_consonants=True, use_separate_vowels=True)
	print('Text =', text2)

	print('Comparison =', text == text2)

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOC>', use_separate_consonants=True, use_separate_vowels=False)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOC>', use_separate_consonants=True, use_separate_vowels=False)
	print('Text =', text2)

	print('Comparison =', text == text2)

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOC>', use_separate_consonants=False, use_separate_vowels=True)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOC>', use_separate_consonants=False, use_separate_vowels=True)
	print('Text =', text2)

	print('Comparison =', text == text2)

	#--------------------
	jamo_text = hg_util.hangeul2jamo(text, '<EOC>', use_separate_consonants=False, use_separate_vowels=False)
	print('Jamo text =', jamo_text)

	text2 = hg_util.jamo2hangeul(jamo_text, '<EOC>', use_separate_consonants=False, use_separate_vowels=False)
	print('Text =', text2)

	print('Comparison =', text == text2)

def main():
	hangeul_jamo_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
