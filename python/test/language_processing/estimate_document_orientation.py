#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math, copy, glob
import numpy as np
import scipy, scipy.fftpack
import skimage, skimage.feature
import cv2

def estimate_orientation_based_on_fft():
	image_filepaths = glob.glob('/work/dataset/text/receipt_sminds/rotated/*.png')

	for image_filepath in image_filepaths:
		# Read gray image.
		img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
		if img is None:
			raise ValueError('Failed to load an image, {}.'.format(image_filepath))

		cv2.imshow('Input', img)

		#--------------------
		# FFT.
		# REF [function] >> image_fft_example() in ${SWDT_PYTHON_HOME}/rnd/test/signal_processing/scipy_fftpack_fft.py
		if True:
			img_fft = scipy.fftpack.fft2(img)
		else:
			# FIXME [error] >> Some artifacts (vertical and horizontal lines) are generated.
			cols, rows = img.shape[:2]
			m, n = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
			img_padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, 0)

			img_fft = scipy.fftpack.fft2(img_padded)
			img_fft = img_fft[:cols,:rows]
		img_fft_mag = np.log10(np.abs(img_fft))

		# Crop the spectrum, if it has an odd number of rows or columns.
		img_fft_mag = img_fft_mag[:(img_fft_mag.shape[0] & -2),:(img_fft_mag.shape[1] & -2)]

		# Rearrange the quadrants of Fourier image  so that the origin is at the image center.
		cy, cx = img_fft_mag.shape[0] // 2, img_fft_mag.shape[1] // 2
		top_left = copy.deepcopy(img_fft_mag[:cy,:cx])
		top_right = copy.deepcopy(img_fft_mag[:cy,cx:])
		bottom_left = copy.deepcopy(img_fft_mag[cy:,:cx])
		bottom_right = copy.deepcopy(img_fft_mag[cy:,cx:])
		img_fft_mag[:cy,:cx] = bottom_right
		img_fft_mag[:cy,cx:] = bottom_left
		img_fft_mag[cy:,:cx] = top_right
		img_fft_mag[cy:,cx:] = top_left
		del top_left, top_right, bottom_left, bottom_right

		# Transform the matrix with float values into a viewable image form (float between values 0 and 1).
		img_fft_mag_normalized = cv2.normalize(img_fft_mag, None, 0, 1, cv2.NORM_MINMAX)

		cv2.imshow('FFT', img_fft_mag_normalized)

		#--------------------
		# Find peaks.
		#peaks = skimage.feature.peak_local_max(img_fft_mag_normalized, min_distance=5, num_peaks=50)
		#peaks = skimage.feature.peak_local_max(img_fft_mag_normalized, min_distance=10, num_peaks=50)
		peaks = skimage.feature.peak_local_max(img_fft_mag_normalized, min_distance=1, num_peaks=500)
		peaks = peaks[:,::-1]  # (y, x) -> (x, y).

		rgb = cv2.cvtColor(np.round(img_fft_mag_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
		for pt in peaks:
			cv2.circle(rgb, tuple(pt), 1, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)

		#--------------------
		# Eigen analysis.
		try:
			peaks_centered = peaks - [cx, cy]
			u, s, vh = np.linalg.svd(peaks_centered, full_matrices=True)
			print('Document orientation (SVD) =', math.degrees(math.atan2(vh[0,1], vh[0,0])))

			ll = min(img_fft_mag.shape[:2]) * 0.1
			vh = np.round(vh * ll).astype(np.int32)
			cv2.line(rgb, (cx, cy), tuple(vh[0] + (cx, cy)), (0, 255, 255), 3, cv2.LINE_AA)
			cv2.line(rgb, (cx, cy), tuple(vh[1] + (cx, cy)), (255, 255, 0), 3, cv2.LINE_AA)
		except np.linalg.LinAlgError:
			print('np.linalg.LinAlgError raised.')
			raise

		#--------------------
		peaks = np.expand_dims(peaks, axis=1)
		if False:
			#--------------------
			# Hough transform.
			lines_max = 3
			threshold = 5
			# TODO [check] >> I guess that rho has relations with image size.
			rhoMin, rhoMax, rhoStep = 0, max(img_fft_mag_normalized.shape), 16
			#thetaMin, thetaMax, thetaStep = 0, math.pi, math.pi / 18
			thetaMin, thetaMax, thetaStep = -math.pi / 2, math.pi / 2, math.pi / 18
			# TODO [check] >> OpenCV implementation of Hough line transform may have some bugs.
			lines = cv2.HoughLinesPointSet(peaks, lines_max, threshold, rhoMin, rhoMax, rhoStep, thetaMin, thetaMax, thetaStep)

			if lines is not None:
				print('#detected lines =', len(lines))

				rgb = cv2.cvtColor(np.round(img_fft_mag_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
				offset = 1000
				for idx, line in enumerate(lines):
					votes, rho, theta = line[0]
					votes, rho, theta = int(votes), float(rho), float(theta)

					print('\t#{}: votes = {}, rho = {}, theta = {}.'.format(idx, votes, rho, math.degrees(theta)))

					cos_theta, sin_theta = math.cos(theta), math.sin(theta)
					x0, y0 = rho * cos_theta, rho * sin_theta
					dx, dy = offset * sin_theta, offset * cos_theta
					pt1 = (round(x0 - dx), round(y0 + dy))
					pt2 = (round(x0 + dx), round(y0 - dy))

					cv2.line(rgb, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA);

				cv2.imshow('Result', rgb)
			else:
				print('No detected line.')
		elif False:
			peak_img = np.zeros(img_fft_mag_normalized.shape, dtype=np.uint8)
			peak_img[peaks[:,0,1],peaks[:,0,0]] = 255

			#--------------------
			# Hough transform.
			# TODO [check] >> OpenCV implementation of Hough line transform may have some bugs.
			#lines = cv2.HoughLines(peak_img, rho=5, theta=math.pi / 18, threshold=5, srn=0, stn=0, min_theta=0, max_theta=math.pi)
			lines = cv2.HoughLines(peak_img, rho=5, theta=math.pi / 18, threshold=5, srn=0, stn=0, min_theta=-math.pi / 2, max_theta=math.pi / 2)

			if lines is not None:
				print('#detected lines =', len(lines))
				offset = 1000
				for idx, line in enumerate(lines):
					# NOTE [info] >> Rho can be negative.
					rho, theta = line[0]
					rho, theta = float(rho), float(theta)

					print('\t#{}: rho = {}, theta = {}.'.format(idx, rho, math.degrees(theta)))

					cos_theta, sin_theta = math.cos(theta), math.sin(theta)
					x0, y0 = rho * cos_theta, rho * sin_theta
					dx, dy = offset * sin_theta, offset * cos_theta
					pt1 = (round(x0 - dx), round(y0 + dy))
					pt2 = (round(x0 + dx), round(y0 - dy))

					cv2.line(rgb, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA);

				cv2.imshow('Result', rgb)
			else:
				print('No detected line.')
		elif True:
			peak_img = np.zeros(img_fft_mag_normalized.shape, dtype=np.uint8)
			peak_img[peaks[:,0,1],peaks[:,0,0]] = 255

			#--------------------
			# Image statistics.
			label_img = (peak_img > 0).astype(np.int)
			props = skimage.measure.regionprops(label_img, intensity_image=None, cache=True)
			#print('***', props[0].label, props[0].moments, props[0].moments_central, props[0].moments_hu, props[0].moments_normalized)
			#print('===', props[0].label, props[0].eccentricity, props[0].minor_axis_length, props[0].major_axis_length)
			print('Document orientation (statistics) =', math.degrees(props[0].orientation))

			#--------------------
			# Classic straight-line Hough transform.
			tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)  # Set a precision of 1.0 degree.
			hspace, theta, rho = skimage.transform.hough_line(peak_img, theta=tested_angles)

			origin = np.array((0, peak_img.shape[1]))
			for idx, (accum, angle, dist) in enumerate(zip(*skimage.transform.hough_line_peaks(hspace, theta, rho, min_distance=9, min_angle=10, threshold=None, num_peaks=2))):
			#for idx, (accum, angle, dist) in enumerate(zip(*skimage.transform.hough_line_peaks(hspace, theta, rho, min_distance=9, min_angle=10, threshold=3, num_peaks=5))):
				y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
				y0, y1 = float(y0), float(y1)

				print('#{}: vote = {}.'.format(idx, accum))

				if 0 == idx:
					cv2.line(rgb, (origin[0], round(y0)), (origin[1], round(y1)), (0, 0, 255), 1, cv2.LINE_AA)
				else:
					cv2.line(rgb, (origin[0], round(y0)), (origin[1], round(y1)), (0, 255, 0), 1, cv2.LINE_AA)
			cv2.imshow('Result', rgb)

			"""
			# Generate figure.
			fig, axes = plt.subplots(1, 3, figsize=(15, 6))
			ax = axes.ravel()

			ax[0].imshow(peak_img, cmap=matplotlib.cm.gray)
			ax[0].set_title('Input image')
			ax[0].set_axis_off()

			ax[1].imshow(np.log(1 + hspace),
						 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]],
						 cmap=matplotlib.cm.gray, aspect=1 / 1.5)
			ax[1].set_title('Hough transform')
			ax[1].set_xlabel('Angles (degrees)')
			ax[1].set_ylabel('Distance (pixels)')
			ax[1].axis('image')

			ax[2].imshow(peak_img, cmap=matplotlib.cm.gray)
			origin = np.array((0, peak_img.shape[1]))
			for _, angle, dist in zip(*skimage.transform.hough_line_peaks(hspace, theta, rho)):
				y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
				ax[2].plot(origin, (y0, y1), '-r')
			ax[2].set_xlim(origin)
			ax[2].set_ylim((peak_img.shape[0], 0))
			ax[2].set_axis_off()
			ax[2].set_title('Detected lines')

			plt.tight_layout()
			plt.show()
			"""

		cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	estimate_orientation_based_on_fft()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
