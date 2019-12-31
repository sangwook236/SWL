#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import glob
import numpy as np
import skimage, skimage.morphology
import cv2

def analyze_document_layout_based_on_morphology():
	#image_filepaths = glob.glob('/work/dataset/text/receipt_sminds/rotated/*.png')
	image_filepaths = glob.glob('/work/dataset/text/receipt_epapyrus/epapyrus_20191203/image_labelme/*.png')

	for image_filepath in image_filepaths:
		# Read gray image.
		img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
		if img is None:
			raise ValueError('Failed to load an image, {}.'.format(image_filepath))

		img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))

		cv2.imshow('Input', img)

		img = skimage.util.invert(img)
		cv2.imshow('Inverted Image', img)

		if True:
			#thresh = skimage.filters.threshold_otsu(img)
			#block_size = 35
			#thresh = skimage.filters.threshold_local(img, block_size, offset=10)
			#window_size = 25
			#thresh = skimage.filters.threshold_niblack(img, window_size=window_size, k=0.8)
			#thresh = skimage.filters.threshold_sauvola(img, window_size=window_size)
			#img = img > thresh
			#img = (img * 255).astype(np.uint8)

			selem = skimage.morphology.disk(5)
			img = skimage.filters.rank.otsu(img, selem)

			cv2.imshow('Thresholding', img)

		selem = skimage.morphology.disk(3)
		num_iterations = 1
		for _ in range(num_iterations):
			img = skimage.morphology.dilation(img, selem)
		for _ in range(num_iterations):
			img = skimage.morphology.erosion(img, selem)

		if True:
			#thresh = skimage.filters.threshold_isodata(img)
			#thresh = skimage.filters.threshold_li(img)
			#thresh = skimage.filters.threshold_mean(img)
			#thresh = skimage.filters.threshold_minimum(img)
			thresh = skimage.filters.threshold_otsu(img)
			#thresh = skimage.filters.threshold_triangle(img)
			#thresh = skimage.filters.threshold_yen(img)
			#thresh = 0
			img = img > thresh
			img = (img * 255).astype(np.uint8)

		cv2.imshow('Result', img)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	analyze_document_layout_based_on_morphology()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
