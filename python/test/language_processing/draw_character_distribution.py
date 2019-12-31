#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')

import os, math
import numpy as np
import scipy.stats
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

def draw_ellipse_on_character():
	if 'posix' == os.name:
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		font_base_dir_path = '/work/font'
	font_dir_path = font_base_dir_path + '/kor'

	font_type = font_dir_path + '/gulim.ttf'
	font_index = 0
	font_size = 32
	text_offset = (0, 0)
	draw_text_border, crop_text_area = False, False
	font_color, bg_color = 255, 0

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	import string
	#text = string.ascii_letters
	text = '가나다라마바사자차카타파하'

	image_size = font.getsize(text)
	#image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))

	img = Image.new(mode='L', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	# Draws text.
	draw.text(xy=text_offset, text=text, font=font, fill=font_color)

	if draw_text_border or crop_text_area:
		#text_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
		text_size = draw.textsize(text, font=font)  # (width, height).
		font_offset = font.getoffset(text)  # (x, y).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)

	rgb = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
	offset = np.array(text_offset)
	for ch in text:
		#ch_size = font.getsize(ch)  # (width, height). This is erroneous for multiline text.
		ch_size = draw.textsize(ch, font=font)  # (width, height).
		font_offset = font.getoffset(ch)  # (x, y).
		text_rect = (offset[0], offset[1], offset[0] + ch_size[0] + font_offset[0], offset[1] + ch_size[1] + font_offset[1])

		if False:
			center = (text_rect[0] + text_rect[2]) / 2, (text_rect[1] + text_rect[3]) / 2
			axis = (text_rect[2] - text_rect[0], text_rect[3] - text_rect[1])
			cv2.ellipse(rgb, (center, axis, 0), (0, 0, 255), 1, cv2.LINE_AA)
		elif False:
			pts = cv2.findNonZero(np.array(img)[text_rect[1]:text_rect[3],text_rect[0]:text_rect[2]]) + offset
			obb = cv2.minAreaRect(pts)
			cv2.ellipse(rgb, obb, (0, 0, 255), 1, cv2.LINE_AA)
		elif True:
			try:
				pts = cv2.findNonZero(np.array(img)[text_rect[1]:text_rect[3],text_rect[0]:text_rect[2]])
				pts = np.squeeze(pts, axis=1)
				center = np.mean(pts, axis=0)
				size = np.max(pts, axis=0) - np.min(pts, axis=0)
				pts = pts - center  # Centering.
				u, s, vh = np.linalg.svd(pts, full_matrices=True)
				angle = math.degrees(math.atan2(vh[0,1], vh[0,0]))
				#obb = (center + offset, s * max(size) / max(s), angle)
				obb = (center + offset, s * math.sqrt((size[0] * size[0] + size[1] * size[1]) / (s[0] * s[0] + s[1] * s[1])), angle)
				cv2.ellipse(rgb, obb, (0, 255, 0), 1, cv2.LINE_AA)
			except np.linalg.LinAlgError:
				print('np.linalg.LinAlgError raised.')
				raise

		offset[0] = text_rect[2]

		cv2.imshow('Ellipse', rgb)

	cv2.imshow('Text', np.array(img))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def draw_normal_distribution_on_character():
	if 'posix' == os.name:
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		font_base_dir_path = '/work/font'
	font_dir_path = font_base_dir_path + '/kor'

	font_type = font_dir_path + '/gulim.ttf'
	font_index = 0
	font_size = 32
	font_color, bg_color = 255, 0
	text_offset = (0, 0)
	draw_text_border, crop_text_area = False, False

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	import string
	text = string.ascii_letters
	#text = '가나다라마바사자차카타파하'

	image_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
	#image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))

	img = Image.new(mode='L', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	# Draws text.
	draw.text(xy=text_offset, text=text, font=font, fill=font_color)

	if draw_text_border or crop_text_area:
		#text_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
		text_size = draw.textsize(text, font=font)  # (width, height).
		font_offset = font.getoffset(text)  # (x, y).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)

	#x, y = np.mgrid[0:img.size[0], 0:img.size[1]]
	x, y = np.mgrid[0:img.size[0]:0.5, 0:img.size[1]:0.5]
	pos = np.dstack((x, y))
	text_pdf_unnormalized = np.zeros(x.shape, dtype=np.float32)
	offset = np.array(text_offset)
	for ch in text:
		#char_size = font.getsize(ch)  # (width, height). This is erroneous for multiline text.
		char_size = draw.textsize(ch, font=font)  # (width, height).
		font_offset = font.getoffset(ch)  # (x, y).
		text_rect = (offset[0], offset[1], offset[0] + char_size[0] + font_offset[0], offset[1] + char_size[1] + font_offset[1])

		if True:
			pts = cv2.findNonZero(np.array(img)[text_rect[1]:text_rect[3],text_rect[0]:text_rect[2]]) + offset
			center, axis, angle = cv2.minAreaRect(pts)
			angle = math.radians(angle)
		elif False:
			try:
				pts = cv2.findNonZero(np.array(img)[text_rect[1]:text_rect[3],text_rect[0]:text_rect[2]])
				pts = np.squeeze(pts, axis=1)
				center = np.mean(pts, axis=0)
				size = np.max(pts, axis=0) - np.min(pts, axis=0)
				pts = pts - center  # Centering.

				u, s, vh = np.linalg.svd(pts, full_matrices=True)
				center = center + offset
				#axis = s * max(size) / max(s)
				axis = s * math.sqrt((size[0] * size[0] + size[1] * size[1]) / (s[0] * s[0] + s[1] * s[1]))
				angle = math.atan2(vh[0,1], vh[0,0])
			except np.linalg.LinAlgError:
				print('np.linalg.LinAlgError raised.')
				raise

		cos_theta, sin_theta = math.cos(angle), math.sin(angle)
		R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
		# TODO [decide] >> Which one is better?
		if True:
			cov = np.diag(np.array(axis))  # 1 * sigma.
		else:
			cov = np.diag(np.array(axis) * 2)  # 2 * sigma.
		cov = np.matmul(R, np.matmul(cov, R.T))

		rv = scipy.stats.multivariate_normal(center, cov)
		# TODO [decide] >> Which one is better?
		if False:
			text_pdf_unnormalized += rv.pdf(pos)
		else:
			char_pdf = rv.pdf(pos)
			text_pdf_unnormalized += char_pdf / np.max(char_pdf)

		offset[0] = text_rect[2]

	rotation_angle = 22.5
	text_pdf_unnormalized = np.array(Image.fromarray(text_pdf_unnormalized.T).rotate(rotation_angle, expand=1)).T
	img = img.rotate(rotation_angle, expand=1)
	img = img.resize(text_pdf_unnormalized.shape, resample=Image.BICUBIC)
	#x, y = np.mgrid[0:img.size[0], 0:img.size[1]]
	#pos = np.dstack((x, y))

	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	ax1.imshow(img, cmap='gray', aspect='equal')
	#ax2.contourf(x, y, text_pdf_unnormalized, cmap='Reds')
	#ax2.set_aspect('equal')
	ax2.imshow(text_pdf_unnormalized.T, cmap='Reds', aspect='equal')

	#text_pdf_blended = 0.5 * text_pdf_unnormalized + 0.5 * np.array(img).T / 255
	text_pdf_blended = 0.5 * text_pdf_unnormalized / np.max(text_pdf_unnormalized) + 0.5 * np.array(img).T / 255
	#ax3.contourf(x, y, text_pdf_blended, cmap='gray')
	#ax3.set_aspect('equal')
	ax3.imshow(text_pdf_blended.T, cmap='gray', aspect='equal')

	plt.show()

def transform_ellipse_projectively():
	raise NotImplementedError

def main():
	#draw_ellipse_on_character()
	draw_normal_distribution_on_character()

	#transform_ellipse_projectively()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
