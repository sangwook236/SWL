import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def compute_text_size(text, font_type, font_index, font_size):
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_size = font.getsize(text)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).

	return text_size[0] + font_offset[0], text_size[1] + font_offset[1]

def generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False):
	if image_size is None:
		image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_size = font.getsize(text)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).

	img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	#text_size = draw.textsize(text, font=font)  # (x, y).
	text_area = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

	# Draws text.
	draw.text(xy=text_offset, text=text, font=font, fill=font_color)

	# Draws rectangle surrounding text.
	if draw_text_border:
		draw.rectangle(text_area, outline='red', width=5)

	# Crops text area.
	if crop_text_area:
		img = img.crop(text_area)

	return img

def draw_text_on_image(img, text, font_type, font_index, font_size, font_color, text_offset=(0, 0), rotation_angle=None):
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).
	text_area = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

	bg_img = Image.fromarray(img)

	# Draws text.
	if rotation_angle is None:
		bg_draw = ImageDraw.Draw(bg_img)
		bg_draw.text(xy=text_offset, text=text, font=font, fill=font_color)

		text_mask = Image.new('L', bg_img.size, (0,))
		mask_draw = ImageDraw.Draw(text_mask)
		mask_draw.text(xy=text_offset, text=text, font=font, fill=(255,))

		x1, y1, x2, y2 = text_area
		text_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
	else:
		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))
		sx0, sy0 = text_img.size

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.

		sx, sy = text_img.size
		bg_img.paste(text_img, (text_offset[0], text_offset[1], text_offset[0] + sx, text_offset[1] + sy), text_img)

		text_mask = Image.new('L', bg_img.size, (0,))
		text_mask.paste(text_img, (text_offset[0], text_offset[1], text_offset[0] + sx, text_offset[1] + sy), text_img)

		dx, dy = (sx0 - sx) / 2, (sy0 - sy) / 2
		x1, y1, x2, y2 = text_area
		rect = (((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), -rotation_angle)
		text_bbox = cv2.boxPoints(rect)
		text_bbox = list(map(lambda xy: [xy[0] - dx, xy[1] - dy], text_bbox))

	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)
	return img, text_mask, text_bbox

def transform_text(text, tx, ty, rotation_angle, font, text_offset=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offset is None:
		text_offset = (0, 0)  # The coordinates (x, y) before transformation.
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).
	
	# z = -y.
	#	xy: left-handed, xz: right-handed.
	x1, z1 = transform(text_offset[0], -text_offset[1])
	x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
	x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
	x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
	xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])

	##x0, y0 = xmin, -zmax
	#text_bbox = np.array([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	dy = -zmax - ty
	#x0, y0 = xmin, -zmax - dy
	text_bbox = np.array([[x1, -z1 - dy], [x2, -z2 - dy], [x3, -z3 - dy], [x4, -z4 - dy]])

	return text_bbox

def transform_text_on_image(text, tx, ty, rotation_angle, img, font, font_color, bg_color, text_offset=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offset is None:
		text_offset = (0, 0)  # The coordinates (x, y) before transformation.
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).

	# z = -y.
	#	xy: left-handed, xz: right-handed.
	x1, z1 = transform(text_offset[0], -text_offset[1])
	x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
	x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
	x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
	xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])

	#x0, y0 = xmin, -zmax
	#text_bbox = np.array([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	dy = -zmax - ty
	x0, y0 = xmin, -zmax - dy
	text_bbox = np.array([[x1, -z1 - dy], [x2, -z2 - dy], [x3, -z3 - dy], [x4, -z4 - dy]])

	#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
	text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

	text_draw = ImageDraw.Draw(text_img)
	text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

	text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
	text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

	bg_img = Image.fromarray(img)
	bg_img.paste(text_img, text_rect, text_img)

	text_mask = Image.new('L', bg_img.size, (0,))
	text_mask.paste(text_img, text_rect, text_img)

	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)

	return text_bbox, img, text_mask

def transform_texts(texts, tx, ty, rotation_angle, font, text_offsets=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offsets is None:
		text_offsets, text_sizes = list(), list()
		max_text_height = 0
		for idx, text in enumerate(texts):
			if 0 == idx:
				text_offset = (0, 0)  # The coordinates (x, y) before transformation.
			else:
				prev_texts = ' '.join(texts[:idx]) + ' '
				text_size = font.getsize(prev_texts)  # (width, height).
				text_offset = (text_size[0], 0)  # (x, y).
			text_offsets.append(text_offset)

			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			font_offset = font.getoffset(text)  # (x, y).
			sx, sy = text_size[0] + font_offset[0], text_size[1] + font_offset[1]
			text_sizes.append((sx, sy))
			if sy > max_text_height:
				max_text_height = sy
		tmp_text_offsets = list()
		for offset, sz in zip(text_offsets, text_sizes):
			dy = int(round((max_text_height - sz[1]) / 2))
			tmp_text_offsets.append((offset[0], offset[1] + dy))
		text_offsets = tmp_text_offsets
	else:
		if len(texts) != len(text_offsets):
			print('[SWL] Error: Unmatched lengths of texts and text offsets {} != {}.'.format(len(texts), len(text_offsets)))
			return None

		text_sizes = list()
		for text in texts:
			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			font_offset = font.getoffset(text)  # (x, y).
			text_sizes.append((text_size[0] + font_offset[0], text_size[1] + font_offset[1]))

	text_bboxes = list()
	"""
	for text, text_offset, text_size in zip(texts, text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])
		#x0, y0 = xmin, -zmax

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	"""
	dy = None
	for text, text_offset, text_size in zip(reversed(texts), reversed(text_offsets), reversed(text_sizes)):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])

		if dy is None:
			dy = -zmax - ty
		x0, y0 = xmin, -zmax - dy
		text_bboxes.append([[x1, -z1 - dy], [x2, -z2 - dy], [x3, -z3 - dy], [x4, -z4 - dy]])
	text_bboxes.reverse()

	return np.array(text_bboxes)

def transform_texts_on_image(texts, tx, ty, rotation_angle, img, font, font_color, bg_color, text_offsets=None):
	cos_angle, sin_angle = math.cos(math.radians(rotation_angle)), math.sin(math.radians(rotation_angle))
	def transform(x, z):
		return int(round(x * cos_angle - z * sin_angle)) + tx, int(round(x * sin_angle + z * cos_angle)) - ty

	if text_offsets is None:
		text_offsets, text_sizes = list(), list()
		max_text_height = 0
		for idx, text in enumerate(texts):
			if 0 == idx:
				text_offset = (0, 0)  # The coordinates (x, y) before transformation.
			else:
				prev_texts = ' '.join(texts[:idx]) + ' '
				text_size = font.getsize(prev_texts)  # (width, height).
				text_offset = (text_size[0], 0)  # (x, y).
			text_offsets.append(text_offset)

			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			font_offset = font.getoffset(text)  # (x, y).
			sx, sy = text_size[0] + font_offset[0], text_size[1] + font_offset[1]
			text_sizes.append((sx, sy))
			if sy > max_text_height:
				max_text_height = sy
		tmp_text_offsets = list()
		for offset, sz in zip(text_offsets, text_sizes):
			dy = int(round((max_text_height - sz[1]) / 2))
			tmp_text_offsets.append((offset[0], offset[1] + dy))
		text_offsets = tmp_text_offsets
	else:
		if len(texts) != len(text_offsets):
			print('[SWL] Error: Unmatched lengths of texts and text offsets {} != {}.'.format(len(texts), len(text_offsets)))
			return None, None, None

		text_sizes = list()
		for text in texts:
			text_size = font.getsize(text)  # (width, height).
			#text_size = draw.textsize(text, font=font)  # (width, height).
			font_offset = font.getoffset(text)  # (x, y).
			text_sizes.append((text_size[0] + font_offset[0], text_size[1] + font_offset[1]))

	bg_img = Image.fromarray(img)
	text_mask = Image.new('L', bg_img.size, (0,))
	text_bboxes = list()
	"""
	for text, text_offset, text_size in zip(texts, text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])
		x0, y0 = xmin, -zmax

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])

		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
		text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

		bg_img.paste(text_img, text_rect, text_img)
		text_mask.paste(text_img, text_rect, text_img)
	"""
	dy = None
	for text, text_offset, text_size in zip(reversed(texts), reversed(text_offsets), reversed(text_sizes)):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xmin, zmax = min([x1, x2, x3, x4]), max([z1, z2, z3, z4])

		if dy is None:
			dy = -zmax - ty
		x0, y0 = xmin, -zmax - dy
		text_bboxes.append([[x1, -z1 - dy], [x2, -z2 - dy], [x3, -z3 - dy], [x4, -z4 - dy]])

		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
		text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

		bg_img.paste(text_img, text_rect, text_img)
		text_mask.paste(text_img, text_rect, text_img)
	text_bboxes.reverse()

	text_bboxes = np.array(text_bboxes)
	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)

	return text_bboxes, img, text_mask
