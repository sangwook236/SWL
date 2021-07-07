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

def main():
	intersection_of_two_line_segments_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
