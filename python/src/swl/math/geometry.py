import math

# REF [site] >> https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
def intersection_of_two_line_segments(pt1, pt2, pt3, pt4, tol=None):
	"""
	line segment 1: pt1 -- pt2.
	line segment 2: pt3 -- pt4.
	"""

	# Line 1 represented as a1 * x + b1 * y = c1.
	a1 = pt2[1] - pt1[1]  # y2 - y1.
	b1 = pt1[0] - pt2[0]  # x1 - x2.
	c1 = a1 * pt1[0] + b1 * pt1[1]

	# Line 2 represented as a2 * x + b2 * y = c2.
	a2 = pt4[1] - pt3[1]  # y4 - y3.
	b2 = pt3[0] - pt4[0]  # x3 - x4.
	c2 = a2 * pt3[0] + b2 * pt3[1]

	determinant = a1 * b2 - a2 * b1

	#if determinant == 0:
	if math.isclose(determinant, 0):
		# The lines are parallel. This is simplified by returning a pair of float("inf").
		return (float("inf"), float("inf")), False
	else:
		x = (b2 * c1 - b1 * c2) / determinant
		y = (a1 * c2 - a2 * c1) / determinant

		if tol:
			is_valid =  min(pt1[0], pt2[0]) - tol <= x <= max(pt1[0], pt2[0]) + tol and \
				min(pt1[1], pt2[1]) - tol <= y <= max(pt1[1], pt2[1]) + tol and \
				min(pt3[0], pt4[0]) - tol <= x <= max(pt3[0], pt4[0]) + tol and \
				min(pt3[1], pt4[1]) - tol <= y <= max(pt3[1], pt4[1]) + tol
		else:
			is_valid =  min(pt1[0], pt2[0]) <= x <= max(pt1[0], pt2[0]) and \
				min(pt1[1], pt2[1]) <= y <= max(pt1[1], pt2[1]) and \
				min(pt3[0], pt4[0]) <= x <= max(pt3[0], pt4[0]) and \
				min(pt3[1], pt4[1]) <= y <= max(pt3[1], pt4[1])
		return (x, y), is_valid

def sort_points_along_vector(points, u):
	def inner_product(u, v):
		"""u * v."""
		return sum(ui * vi for ui, vi in zip(u, v))

	return sorted(points, key=lambda pt: inner_product((p2 - p1 for p1, p2 in zip(points[0], pt)), u))
