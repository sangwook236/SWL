import numpy as np
from PIL import Image

# REF [site] >> http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
def to_rgb(gray):
    return np.dstack([gray.astype(np.uint8)] * 3)

# REF [size] >> https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def stack_images_horzontally(images):
	'''
	images: PIL image list.
	'''

	widths, heights = zip(*(img.size for img in images))

	total_width = sum(widths)
	max_height = max(heights)

	stacked_img = Image.new('RGB', (total_width, max_height))

	x_offset = 0
	for img in images:
		stacked_img.paste(img, (x_offset, 0))
		x_offset += img.size[0]

	return stacked_img

# REF [size] >> https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
def stack_images_vertically(images):
	'''
	images: PIL image list.
	'''

	widths, heights = zip(*(img.size for img in images))

	max_width = max(widths)
	total_height = sum(heights)

	stacked_img = Image.new('RGB', (max_width, total_height))

	y_offset = 0
	for img in images:
		stacked_img.paste(img, (0, y_offset))
		y_offset += img.size[1]

	return stacked_img
