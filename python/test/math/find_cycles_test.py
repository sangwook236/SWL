#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import time
from swl.math.find_cycles import count_cycles, find_cycles, find_cycles_in_graph_with_collinear_vertices

def main():
	# NOTE [info] >> Find n-cycles in a graph ==> Polygon and quadrilateral detection.
	#	(1) Find polygons from a set of line segments.
	#		Find cycles in a graph.
	#			Build graph with line segment ends and intersection points as vertices and line segments as edges, then find cycles using DFS.
	#			https://stackoverflow.com/questions/41245408/how-to-find-polygons-in-a-given-set-of-points-and-edges
	#			"Finding and Counting Given Length Cycles", Algorithmica 1997.
	#			https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
	#	(2) Finds polygons from a set of infinite lines (a simplified version of (1)).
	#		Find all possible combinations of adjacent atomic polygons constructed by the infinite lines
	#	(3) Find quadrilaterals from a set of line segments (a simplified version of (1)).
	#		Find cycles with four edges (4-cycles) in a graph.
	#	(4) Find quadrilaterals from a set of infinite lines (a simplified version of (2) & (3)).
	#		https://stackoverflow.com/questions/45248205/finding-all-quadrilaterals-in-a-set-of-intersections

	if False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0]
		]
		collinear_vertex_sets = None
	elif False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0],
			[1, 0, 1, 0, 1, 1],
			[0, 1, 0, 1, 0, 0],
			[1, 0, 1, 0, 1, 1],
			[0, 1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5}
		]
	elif False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 0],
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1],
			[0, 1, 0, 1, 0, 0, 0],
			[0, 1, 0, 1, 0, 0, 1],
			[0, 0, 0, 1, 0, 1, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5},
			{3, 4, 6}
		]
	elif False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1],
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 1],
			[0, 1, 0, 1, 0, 1, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5},
			{3, 4, 6}
		]
	elif True:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 1, 1],
			[0, 1, 0, 1, 0, 1, 0, 1],
			[0, 0, 0, 0, 0, 1, 1, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5},
			{3, 4, 6, 7}
		]

	# n-cycle.
	#n = 3  
	n = 4
	#n = 5

	print('Start counting n-cycles in a graph...')
	start_time = time.time()
	num_cycles = count_cycles(graph, n)
	print('End counting n-cycles in a graph: {} secs.'.format(time.time() - start_time))
	print("#cycles of length {} = {}.".format(n, num_cycles))

	print('Start finding n-cycles in a graph...')
	start_time = time.time()
	cycles = find_cycles(graph, n)
	print('End finding n-cycles in a graph: {} secs.'.format(time.time() - start_time))
	print("Cycles of length {} = {}.".format(n, cycles))

	assert num_cycles == len(cycles)

	if collinear_vertex_sets:
		print('Start finding n-cycles in a graph with collinear vertices...')
		start_time = time.time()
		cycles = find_cycles_in_graph_with_collinear_vertices(graph, collinear_vertex_sets, n)
		print('End finding n-cycles in a graph with collinear vertices: {} secs.'.format(time.time() - start_time))
		print("Cycles of length {} = {}.".format(n, cycles))

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
