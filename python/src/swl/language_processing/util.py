import math
from PIL import Image, ImageDraw, ImageFont

def generate_text_image(text, font_type, font_index, font_size, font_color, bg_color, image_size=None, text_offset=None, crop_text_area=True, draw_text_border=False):
	if image_size is None:
		image_size = (math.ceil(len(text) * font_size * 1.1), math.ceil((text.count('\n') + 1) * font_size * 1.1))
	if text_offset is None:
		text_offset = (0, 0)

	font = ImageFont.truetype(font=font_type, size=font_size, index=font_index)
	img = Image.new(mode='RGB', size=image_size, color=bg_color)
	draw = ImageDraw.Draw(img)

	text_size = draw.textsize(text, font=font)
	font_offset = font.getoffset(text)
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
