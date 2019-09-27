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
