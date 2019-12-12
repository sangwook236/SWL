import math, random, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def compute_simple_text_recognition_accuracy(text_pairs, case_sensitive=True):
	total_text_count = len(text_pairs)
	correct_word_count, total_word_count, correct_char_count, total_char_count = 0, 0, 0, 0
	if case_sensitive:
		correct_text_count = len(list(filter(lambda x: x[0] == x[1], text_pairs)))
		for inf_text, gt_text in text_pairs:
			inf_words, gt_words = inf_text.split(' '), gt_text.split(' ')
			total_word_count += max(len(inf_words), len(gt_words))
			correct_word_count += len(list(filter(lambda x: x[0] == x[1], zip(inf_words, gt_words))))

			total_char_count += max(len(inf_text), len(gt_text))
			correct_char_count += len(list(filter(lambda x: x[0] == x[1], zip(inf_text, gt_text))))
	else:
		correct_text_count = len(list(filter(lambda x: x[0].lower() == x[1].lower(), text_pairs)))
		for inf_text, gt_text in text_pairs:
			inf_words, gt_words = inf_text.split(' '), gt_text.split(' ')
			total_word_count += max(len(inf_words), len(gt_words))
			correct_word_count += len(list(filter(lambda x: x[0].lower() == x[1].lower(), zip(inf_words, gt_words))))

			total_char_count += max(len(inf_text), len(gt_text))
			correct_char_count += len(list(filter(lambda x: x[0].lower() == x[1].lower(), zip(inf_text, gt_text))))

	return correct_text_count, total_text_count, correct_word_count, total_word_count, correct_char_count, total_char_count

def compute_string_distance(text_pairs, case_sensitive=True):
	import jellyfish

	#string_distance_functor = jellyfish.hamming_distance
	string_distance_functor = jellyfish.levenshtein_distance
	#string_distance_functor = jellyfish.damerau_levenshtein_distance
	#string_distance_functor = jellyfish.jaro_distance
	#string_distance_functor = functools.partial(jellyfish.jaro_winkler, long_tolerance=False)
	#string_distance_functor = jellyfish.match_rating_comparison

	total_text_count = len(text_pairs)
	word_distance, total_word_count, char_distance, total_char_count = 0, 0, 0, 0
	if case_sensitive:
		text_distance = functools.reduce(lambda ss, x: ss + string_distance_functor(x[0], x[1]), text_pairs, 0)
		for inf_text, gt_text in text_pairs:
			inf_words, gt_words = inf_text.split(' '), gt_text.split(' ')
			total_word_count += max(len(inf_words), len(gt_words))
			word_distance += functools.reduce(lambda ss, x: ss + string_distance_functor(x[0], x[1]), zip(inf_words, gt_words), 0)

			total_char_count += max(len(inf_text), len(gt_text))
			char_distance += functools.reduce(lambda ss, x: ss + string_distance_functor(x[0], x[1]), zip(inf_text, gt_text), 0)
	else:
		text_distance = functools.reduce(lambda ss, x: ss + string_distance_functor(x[0].lower(), x[1].lower()), text_pairs, 0)
		for inf_text, gt_text in text_pairs:
			inf_words, gt_words = inf_text.split(' '), gt_text.split(' ')
			total_word_count += max(len(inf_words), len(gt_words))
			word_distance += functools.reduce(lambda ss, x: ss + string_distance_functor(x[0].lower(), x[1].lower()), zip(inf_words, gt_words), 0)

			total_char_count += max(len(inf_text), len(gt_text))
			char_distance += functools.reduce(lambda ss, x: ss + string_distance_functor(x[0].lower(), x[1].lower()), zip(inf_text, gt_text), 0)

	return text_distance, word_distance, char_distance, total_text_count, total_word_count, total_char_count

#--------------------------------------------------------------------

def compute_text_size(text, font_type, font_index, font_size):
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_size = font.getsize(text)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).

	return text_size[0] + font_offset[0], text_size[1] + font_offset[1]

def generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=None, mode='RGB', mask=False, mask_mode='1'):
	if char_space_ratio is None or 1 == char_space_ratio:
		if mask:
			return generate_simple_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, mode, mask_mode)
		else:
			return generate_simple_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, mode)
	else:
		if mask:
			return generate_per_character_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, char_space_ratio, mode, mask_mode)
		else:
			return generate_per_character_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size, text_offset, crop_text_area, draw_text_border, char_space_ratio, mode)

def generate_simple_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode='RGB'):
	if image_size is None:
		image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if font_color is None:
		#font_color = (random.randrange(256),) * image_depth  # Uses a random grayscale font color.
		font_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB font color.
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
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

	return img

def generate_simple_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, mode='RGB', mask_mode='1'):
	if image_size is None:
		image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if font_color is None:
		#font_color = (random.randrange(256),) * image_depth  # Uses a random grayscale font color.
		font_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB font color.
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random RGB background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw_img = ImageDraw.Draw(img)

	msk = Image.new(mode=mask_mode, size=image_size, color=0)
	#msk = Image.new(mode='1', size=image_size, color=0)  # {0, 1}, bool.
	#msk = Image.new(mode='L', size=image_size, color=0)  # [0, 255], uint8.
	draw_msk = ImageDraw.Draw(msk)

	# Draws text.
	draw_img.text(xy=text_offset, text=text, font=font, fill=font_color)
	draw_msk.text(xy=text_offset, text=text, font=font, fill=255)

	if draw_text_border or crop_text_area:
		#text_size = font.getsize(text)  # (width, height). This is erroneous for multiline text.
		text_size = draw_img.textsize(text, font=font)  # (width, height).
		font_offset = font.getoffset(text)  # (x, y).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw_img.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)
			msk = msk.crop(text_rect)

	return img, msk

def generate_per_character_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=None, mode='RGB'):
	num_chars, num_newlines = len(text), text.count('\n')
	if image_size is None:
		image_size = (math.ceil(num_chars * font_size * char_space_ratio * 1.1), math.ceil((num_newlines + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	# Draws text.
	char_offset = list(text_offset)
	char_space = math.ceil(font_size * char_space_ratio)
	if font_color is None:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw.text(xy=char_offset, text=ch, font=font, fill=tuple(random.randrange(256) for _ in range(image_depth)))  # Random font color.
			char_offset[0] += char_space
	#elif len(font_colors) == num_chars:
	#	for idx, (ch, fcolor) in enumerate(zip(text, font_colors)):
	#		char_offset[0] = text_offset[0] + char_space * idx
	#		draw.text(xy=char_offset, text=ch, font=font, fill=fcolor)
	else:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw.text(xy=char_offset, text=ch, font=font, fill=font_color)
			char_offset[0] += char_space

	if draw_text_border or crop_text_area:
		#text_size = list(font.getsize(text))  # (width, height). This is erroneous for multiline text.
		text_size = list(draw.textsize(text, font=font))  # (width, height).
		if num_chars > 1:
			max_chars_in_line = functools.reduce(lambda ll, line: max(ll, len(line)), text.splitlines(), 0)
			#text_size[0] = char_space * (max_chars_in_line - 1) + font_size
			text_size[0] = char_space * (max_chars_in_line - 1) + font.getsize(text[-1])[0]
			text_size[1] = (num_newlines + 1) * font_size
		font_offset = font.getoffset(text)  # (x, y).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)

	return img

def generate_per_character_text_image_and_mask(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False, char_space_ratio=None, mode='RGB', mask_mode='1'):
	num_chars, num_newlines = len(text), text.count('\n')
	if image_size is None:
		image_size = (math.ceil(num_chars * font_size * char_space_ratio * 1.1), math.ceil((num_newlines + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)
	# TODO [improve] >> Other color modes have to be supported.
	if 'L' == mode or '1' == mode:
		image_depth = 1
	elif 'RGBA' == mode:
		image_depth = 4
	else:
		image_depth = 3
	if bg_color is None:
		#bg_color = (random.randrange(256),) * image_depth  # Uses a random grayscale background color.
		bg_color = tuple(random.randrange(256) for _ in range(image_depth))  # Uses a random background color.

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)

	img = Image.new(mode=mode, size=image_size, color=bg_color)
	#img = Image.new(mode='RGB', size=image_size, color=bg_color)
	#img = Image.new(mode='RGBA', size=image_size, color=bg_color)
	#img = Image.new(mode='L', size=image_size, color=bg_color)
	#img = Image.new(mode='1', size=image_size, color=bg_color)
	draw_img = ImageDraw.Draw(img)

	msk = Image.new(mode=mask_mode, size=image_size, color=0)
	#msk = Image.new(mode='1', size=image_size, color=0)  # {0, 1}, bool.
	#msk = Image.new(mode='L', size=image_size, color=0)  # [0, 255], uint8.
	draw_msk = ImageDraw.Draw(msk)

	# Draws text.
	char_offset = list(text_offset)
	char_space = math.ceil(font_size * char_space_ratio)
	if font_color is None:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw_img.text(xy=char_offset, text=ch, font=font, fill=tuple(random.randrange(256) for _ in range(image_depth)))  # Random font color.
			draw_msk.text(xy=char_offset, text=ch, font=font, fill=255)
			char_offset[0] += char_space
	#elif len(font_colors) == num_chars:
	#	for idx, (ch, fcolor) in enumerate(zip(text, font_colors)):
	#		char_offset[0] = text_offset[0] + char_space * idx
	#		draw_img.text(xy=char_offset, text=ch, font=font, fill=fcolor)
	#		draw_msk.text(xy=char_offset, text=ch, font=font, fill=255)
	else:
		for ch in text:
			if '\n' == ch:
				char_offset[0] = text_offset[0]
				char_offset[1] += font_size
				continue
			draw_img.text(xy=char_offset, text=ch, font=font, fill=font_color)
			draw_msk.text(xy=char_offset, text=ch, font=font, fill=255)
			char_offset[0] += char_space

	if draw_text_border or crop_text_area:
		#text_size = list(font.getsize(text))  # (width, height). This is erroneous for multiline text.
		text_size = list(draw_img.textsize(text, font=font))  # (width, height).
		if num_chars > 1:
			max_chars_in_line = functools.reduce(lambda ll, line: max(ll, len(line)), text.splitlines(), 0)
			#text_size[0] = char_space * (max_chars_in_line - 1) + font_size
			text_size[0] = char_space * (max_chars_in_line - 1) + font.getsize(text[-1])[0]
			text_size[1] = (num_newlines + 1) * font_size
		font_offset = font.getoffset(text)  # (x, y).
		text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

		# Draws a rectangle surrounding text.
		if draw_text_border:
			draw_img.rectangle(text_rect, outline='red', width=5)

		# Crops text area.
		if crop_text_area:
			img = img.crop(text_rect)
			msk = msk.crop(text_rect)

	return img, msk

#--------------------------------------------------------------------

def draw_text_on_image(img, text, font_type, font_index, font_size, font_color, text_offset=(0, 0), rotation_angle=None):
	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	text_size = font.getsize(text)  # (width, height).
	#text_size = draw.textsize(text, font=font)  # (width, height).
	font_offset = font.getoffset(text)  # (x, y).
	text_rect = (text_offset[0], text_offset[1], text_offset[0] + text_size[0] + font_offset[0], text_offset[1] + text_size[1] + font_offset[1])

	bg_img = Image.fromarray(img)

	# Draws text.
	if rotation_angle is None:
		bg_draw = ImageDraw.Draw(bg_img)
		bg_draw.text(xy=text_offset, text=text, font=font, fill=font_color)

		text_mask = Image.new('L', bg_img.size, (0,))
		mask_draw = ImageDraw.Draw(text_mask)
		mask_draw.text(xy=text_offset, text=text, font=font, fill=(255,))

		x1, y1, x2, y2 = text_rect
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
		x1, y1, x2, y2 = text_rect
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
	dx, dy = xmin - tx, -zmax - ty
	#x0, y0 = xmin - dx, -zmax - dy
	text_bbox = np.array([[x1 - dx, -z1 - dy], [x2 - dx, -z2 - dy], [x3 - dx, -z3 - dy], [x4 - dx, -z4 - dy]])

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
	dx, dy = xmin - tx, -zmax - ty
	x0, y0 = xmin - dx, -zmax - dy
	text_bbox = np.array([[x1 - dx, -z1 - dy], [x2 - dx, -z2 - dy], [x3 - dx, -z3 - dy], [x4 - dx, -z4 - dy]])

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

	return np.array(text_bboxes)
	"""
	xy0_list = list()
	for text_offset, text_size in zip(text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xy0_list.append((min([x1, x2, x3, x4]), -max([z1, z2, z3, z4])))

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	text_bboxes = np.array(text_bboxes)

	dxy = functools.reduce(lambda xym, xy0: (min(xym[0], xy0[0] - tx), min(xym[1], xy0[1] - ty)), xy0_list, (0, 0))
	text_bboxes[:,:] -= dxy

	return text_bboxes

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
	xy0_list = list()
	for text_offset, text_size in zip(text_offsets, text_sizes):
		# z = -y.
		#	xy: left-handed, xz: right-handed.
		x1, z1 = transform(text_offset[0], -text_offset[1])
		x2, z2 = transform(text_offset[0] + text_size[0], -text_offset[1])
		x3, z3 = transform(text_offset[0] + text_size[0], -(text_offset[1] + text_size[1]))
		x4, z4 = transform(text_offset[0], -(text_offset[1] + text_size[1]))
		xy0_list.append((min([x1, x2, x3, x4]), -max([z1, z2, z3, z4])))

		text_bboxes.append([[x1, -z1], [x2, -z2], [x3, -z3], [x4, -z4]])
	text_bboxes = np.array(text_bboxes)

	dxy = functools.reduce(lambda xym, xy0: (min(xym[0], xy0[0] - tx), min(xym[1], xy0[1] - ty)), xy0_list, (0, 0))
	text_bboxes[:,:] -= dxy

	for text, text_size, xy0 in zip(texts, text_sizes, xy0_list):
		x0, y0 = xy0[0] - dxy[0], xy0[1] - dxy[1]

		#text_img = Image.new('RGBA', text_size, (0, 0, 0, 0))
		text_img = Image.new('RGBA', text_size, (255, 255, 255, 0))

		text_draw = ImageDraw.Draw(text_img)
		text_draw.text(xy=(0, 0), text=text, font=font, fill=font_color)

		text_img = text_img.rotate(rotation_angle, expand=1)  # Rotates the image around the top-left corner point.
		text_rect = (x0, y0, x0 + text_img.size[0], y0 + text_img.size[1])

		bg_img.paste(text_img, text_rect, text_img)
		text_mask.paste(text_img, text_rect, text_img)

	img = np.asarray(bg_img, dtype=img.dtype)
	text_mask = np.asarray(text_mask, dtype=np.uint8)

	return text_bboxes, img, text_mask

#--------------------------------------------------------------------

def draw_character_histogram(texts, charset=None):
	if charset is None:
		import string
		if True:
			charset = \
				string.ascii_uppercase + \
				string.ascii_lowercase + \
				string.digits + \
				string.punctuation + \
				' '
		else:
			hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001.txt'
			#hangul_letter_filepath = '../../data/language_processing/hangul_ksx1001_1.txt'
			#hangul_letter_filepath = '../../data/language_processing/hangul_unicode.txt'
			with open(hangul_letter_filepath, 'r', encoding='UTF-8') as fd:
				#hangeul_charset = fd.read().strip('\n')  # A strings.
				hangeul_charset = fd.read().replace(' ', '').replace('\n', '')  # A string.
				#hangeul_charset = fd.readlines()  # A list of string.
				#hangeul_charset = fd.read().splitlines()  # A list of strings.
			#hangeul_jamo_charset = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
			#hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
			hangeul_jamo_charset = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

			charset = \
				hangeul_charset + \
				hangeul_jamo_charset + \
				string.ascii_uppercase + \
				string.ascii_lowercase + \
				string.digits + \
				string.punctuation + \
				' '

	charset = sorted(charset)
	#charset = ''.join(sorted(charset))

	#--------------------
	char_dict = dict()
	for ch in charset:
		char_dict[ch] = 0

	for txt in texts:
		if not txt:
			continue

		for ch in txt:
			try:
				char_dict[ch] += 1
			except KeyError:
				print('[SWL] Warning: Invalid character, {} in {}.'.format(ch, txt))

	#--------------------
	import numpy as np
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(10, 6))
	x_label = np.arange(len(char_dict.keys()))
	plt.bar(x_label, char_dict.values(), align='center', alpha=0.5)
	plt.xticks(x_label, char_dict.keys())
	plt.show()

	fig.savefig('./character_frequency.png')
	plt.close(fig)
