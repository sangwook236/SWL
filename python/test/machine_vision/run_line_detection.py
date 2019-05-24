#!/usr/bin/env python

import math, random
import numpy as np
import cv2 as cv
from scipy.optimize import minimize

# REF [site] >> https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator/32857432#32857432
def createLineIterator(P1, P2, img):
	"""
	Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

	Parameters:
		- P1: a numpy array that consists of the coordinate of the first point (x, y)
		- P2: a numpy array that consists of the coordinate of the second point (x, y)
		- img: the image being processed

	Returns:
		- it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
	"""

	# Define local variables for readability.
	imageH = img.shape[0]
	imageW = img.shape[1]
	P1X = P1[0]
	P1Y = P1[1]
	P2X = P2[0]
	P2Y = P2[1]

	# Difference and absolute difference between points used to calculate slope and relative location between points.
	dX = P2X - P1X
	dY = P2Y - P1Y
	dXa = np.abs(dX)
	dYa = np.abs(dY)

	# Predefine numpy array for output based on distance between points.
	#itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
	itbuffer = np.empty(shape=(np.maximum(dYa + 1, dXa + 1).astype(np.int), 3), dtype=np.float32)
	itbuffer.fill(np.nan)

	# Obtain coordinates along the line using a form of Bresenham's algorithm.
	negY = P1Y > P2Y
	negX = P1X > P2X
	if P1X == P2X:  # Vertical line segment.
		itbuffer[:,0] = P1X
		if negY:
			#itbuffer[:,1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1).astype(np.int)
			itbuffer[:,1] = np.arange(P1Y, P1Y - dYa - 1, -1).astype(np.int)
		else:
			#itbuffer[:,1] = np.arange(P1Y + 1, P1Y + dYa + 1).astype(np.int)
			itbuffer[:,1] = np.arange(P1Y, P1Y + dYa + 1).astype(np.int)          
	elif P1Y == P2Y:  # Horizontal line segment.
		itbuffer[:,1] = P1Y
		if negX:
			#itbuffer[:,0] = np.arange(P1X - 1, P1X - dXa - 1, -1).astype(np.int)
			itbuffer[:,0] = np.arange(P1X, P1X - dXa - 1, -1).astype(np.int)
		else:
			#itbuffer[:,0] = np.arange(P1X + 1, P1X + dXa + 1).astype(np.int)
			itbuffer[:,0] = np.arange(P1X, P1X + dXa + 1).astype(np.int)
	else:  # Diagonal line segment.
		steepSlope = dYa > dXa
		if steepSlope:
			slope = dX.astype(np.float32) / dY.astype(np.float32)
			if negY:
				#itbuffer[:,1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1).astype(np.int)
				itbuffer[:,1] = np.arange(P1Y, P1Y - dYa - 1, -1).astype(np.int)
			else:
				#itbuffer[:,1] = np.arange(P1Y + 1, P1Y + dYa + 1).astype(np.int)
				itbuffer[:,1] = np.arange(P1Y, P1Y + dYa + 1).astype(np.int)
			itbuffer[:,0] = (slope * (itbuffer[:,1] - P1Y)).astype(np.int) + P1X
		else:
			slope = dY.astype(np.float32) / dX.astype(np.float32)
			if negX:
				#itbuffer[:,0] = np.arange(P1X - 1, P1X - dXa - 1, -1).astype(np.int)
				itbuffer[:,0] = np.arange(P1X, P1X - dXa - 1, -1).astype(np.int)
			else:
				#itbuffer[:,0] = np.arange(P1X + 1, P1X + dXa + 1).astype(np.int)
				itbuffer[:,0] = np.arange(P1X, P1X + dXa + 1).astype(np.int)
			itbuffer[:,1] = (slope * (itbuffer[:,0] - P1X)).astype(np.int) + P1Y

	# Remove points outside of image.
	colX = itbuffer[:,0]
	colY = itbuffer[:,1]
	itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

	# Get intensities from img ndarray.
	itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint), itbuffer[:,0].astype(np.uint)]

	return itbuffer

def detect_line_based_on_superpixel():
	#image_filepath = '../../../data/machine_vision/build.png'
	image_filepath = 'D:/work_biz/silicon_minds/DataAnalysis_bitbucket/data/id_images/rrc_00.jpg'

	# Read gray image.
	img = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if img is None:
		raise ValueError('Failed to load an image, {}.'.format(image_filepath))

	#--------------------
	# Linear Spectral Clustering (LSC) superpixels algorithm.
	superpixel = cv.ximgproc.createSuperpixelLSC(img, region_size=20, ratio=0.075)

	# Calculate the superpixel segmentation.
	superpixel.iterate(num_iterations=10)

	#superpixel.enforceLabelConnectivity(min_element_size=20)

	print('#superpixels =', superpixel.getNumberOfSuperpixels())

	superpixel_label = superpixel.getLabels()  # CV_32UC1. [0, getNumberOfSuperpixels()].
	superpixel_contour_mask = superpixel.getLabelContourMask(thick_line=True)  # CV_8UC1.

	superpixel = img.copy()
	superpixel[superpixel_contour_mask > 0] = (0, 0, 255)
	cv.imshow('Superpixel', superpixel)
	cv.imwrite('./superpixel.png', superpixel)

	#--------------------
	dist = cv.distanceTransform(255 - superpixel_contour_mask, cv.DIST_L2, 3)

	# Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it.
	cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
	cv.imshow('Distance Transform', dist)

	#--------------------
	if False:
		pt1, pt2 = (0, 0), img.shape[:2]
		#pt1, pt2 = (10, 10), (300, 300)
		#pt1, pt2 = (300, 300), (10, 10)
		#lineIt = cv.LineIterator(dist, pt1, pt2, 8)  # Error.
		#for idx in range(it.count):
		#	print('{}: {}.'.format(idx, pos()))
		lineIt = createLineIterator(np.array(pt1), np.array(pt2), dist)
		level_on_line = np.sum(lineIt[:,-1])

	#--------------------
	alpha, beta = 0.001, 2.0
	def cost_func(x):
		#pt1, pt2 = (x[0], x[1]), (x[2], x[3])
		pt1, pt2 = (int(round(x[0])), int(round(x[1]))), (int(round(x[2])), int(round(x[3])))

		length2 = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

		lineIt = createLineIterator(np.array(pt1), np.array(pt2), dist * beta)
		level = np.sum(lineIt[:,-1])

		print('***', pt1, pt2, alpha * length2, level)

		#return alpha * length2 - level
		return -alpha * length2 + level

	#--------------------
	#x0 = [43, 105, 277, 117]
	x0 = [random.randrange(0, img.shape[1]), random.randrange(0, img.shape[1]), random.randrange(0, img.shape[0]), random.randrange(0, img.shape[0])]

	res = minimize(cost_func, x0, method='Nelder-Mead', tol=1e-6)
	#res = minimize(cost_func, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
	print('Solution = {}, initial = {}'.format(res.x, x0))

	rgb = superpixel.copy()
	pt1, pt2 = (int(round(res.x[0])), int(round(res.x[1]))), (int(round(res.x[2])), int(round(res.x[3])))
	cv.line(rgb, pt1, pt2, (255, 0, 0), 2, cv.LINE_AA)
	cv.line(rgb, (x0[0], x0[1]), (x0[2], x0[3]), (0, 255, 0), 2, cv.LINE_AA)

	cv.imshow('Line Found', rgb)

	cv.waitKey(0)
	cv.destroyAllWindows()

def main():
	detect_line_based_on_superpixel()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
