#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

def intersection_of_two_line_segments_test():
	from swl.math.geometry import intersection_of_two_line_segments

	pt1, pt2 = (1, 1), (4, 4)
	pt3, pt4 = (1, 8), (3, 0)

	# The intersection of the given lines is (2.4, 2.4).
	int_pt, is_valid = intersection_of_two_line_segments(pt1, pt2, pt3, pt4)
	if is_valid:
		print("Intersection point = {}.".format(int_pt))
	else:
		print("No valid intersection point, {}.".format(int_pt))

	#--------------------
	pt1, pt2 = (1, 1), (4, 4)
	pt3, pt4 = (1, 8), (2, 4)

	# The intersection of the given lines is (2.4, 2.4), but the coordinates of the intersection point is out of range.
	int_pt, is_valid = intersection_of_two_line_segments(pt1, pt2, pt3, pt4)
	if is_valid:
		print("Intersection point = {}.".format(int_pt))
	else:
		print("No valid intersection point, {}.".format(int_pt))

	#--------------------
	pt1, pt2 = (0, 1), (0, 4)
	pt3, pt4 = (1, 8), (1, 4)

	# The given lines are parallel.
	int_pt, is_valid = intersection_of_two_line_segments(pt1, pt2, pt3, pt4)
	if is_valid:
		print("Intersection point = {}.".format(int_pt))
	else:
		print("No valid intersection point, {}.".format(int_pt))

def sort_points_along_vector_test():
	from swl.math.geometry import sort_points_along_vector

	points = [(0, 0), (1, 1), (-1, 3), (-11, 0), (5, 3), (3, 5), (0, 13), (-3, 20), (4, 4), (-3, -2), (15, 0), (100, 1), (-7, -7), (-2, 0.1), (0, -2), (2, -50), (23, -23), (-3.7, 3.7)]
	#points = [(4, 5), (1, 1), (-1, 3), (-11, 0), (5, 3), (3, 5), (0, 13), (-3, 20), (4, 4), (-3, -2), (15, 0), (100, 1), (-7, -7), (-2, 0.1), (0, -2), (2, -50), (23, -23), (-3.7, 3.7)]

	u = (1, 0)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (123, 0)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (-1, 0)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (0, 1)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (0, 123)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (0, -1)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (1, 1)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (-1, -1)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (1, -1)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

	u = (-1, 1)
	points_sorted = sort_points_along_vector(points, u)
	print('Points sorted along {} = {}.'.format(u, points_sorted))

def main():
	#intersection_of_two_line_segments_test()
	sort_points_along_vector_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
