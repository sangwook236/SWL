#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math, glob
import numpy as np
import skimage, skimage.io, skimage.filters
import matplotlib.pyplot as plt

# REF [book] >> "3.3.2. Character size estimation" (p. 368) in "Handbook of Character Recognition and Document Image Analysis", 1997.
def estimate_character_size_by_cca():
	#image_filepaths = glob.glob('/work/dataset/sample_image/*.png')  # Clean images.
	image_filepaths = glob.glob('/work/dataset/sample_image/receipt/*.png')  # Noisy images.

	window_size = 5
	for img_fpath in image_filepaths:
		# Read a gray image.
		img = skimage.io.imread(img_fpath, as_gray=True)
		if img is None:
			raise ValueError('Failed to load an image, {}.'.format(img_fpath))
		img = skimage.util.invert(img)

		thresh = skimage.filters.threshold_otsu(img)
		#thresh = skimage.filters.threshold_sauvola(img, window_size=window_size)
		#thresh = skimage.filters.threshold_niblack(img, window_size=window_size, k=0.8)
		bw_img = img > thresh

		# Connected component analysis/labeling (CCA/CCL).
		#label_img = skimage.measure.label(bw_img, background=None)  # All labels.
		label_img = skimage.measure.label(bw_img, background=0)  # Except background.

		props = skimage.measure.regionprops(label_img)
		#area_thresh = 0
		area_thresh = 50  # For noisy images. 50% of area 10 x 10.
		char_sizes = list(prop.bbox[3] - prop.bbox[1] for prop in props if prop.area > area_thresh)
		#char_sizes = list(math.ceil(prop.major_axis_length) for prop in props if prop.area > area_thresh)
		#char_sizes = list(math.ceil(prop.minor_axis_length) for prop in props if prop.area > area_thresh)  # Not good.

		bins = max(char_sizes) - min(char_sizes)
		hist, bin_edges  = np.histogram(char_sizes, bins)
		print('Character size =', round(bin_edges[np.argmax(hist)]))

		# actual_char_size = np.argmax(hist) * 2.

		plt.figure(figsize=(10, 10))
		plt.subplot(221)
		plt.imshow(img, cmap='gray')
		plt.axis('off')
		plt.subplot(222)
		plt.imshow(bw_img, cmap='gray')
		plt.axis('off')
		plt.subplot(223)
		plt.imshow(label_img, cmap='nipy_spectral')
		plt.axis('off')
		plt.subplot(224)
		plt.hist(char_sizes, bins=bins)
		#plt.axis('off')
		plt.tight_layout()
		plt.show()

def main():
	estimate_character_size_by_cca()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
