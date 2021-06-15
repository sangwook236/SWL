#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../../src")
sys.path.append("./src")

import time
from swl.math.find_cycles import count_cycles, find_cycles, find_edge_cycles, find_cycles_in_graph_with_collinear_vertices

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

	print("Start counting {}-cycles in a graph...".format(n))
	start_time = time.time()
	num_cycles = count_cycles(graph, n)
	print("End counting {}-cycles in a graph: {} secs.".format(n, time.time() - start_time))
	print("#cycles of length {} = {}.".format(n, num_cycles))

	print("Start finding {}-cycles in a graph...".format(n))
	start_time = time.time()
	cycles = find_cycles(graph, n)
	print("End finding {}-cycles in a graph: {} secs.".format(n, time.time() - start_time))
	print("Cycles of length {} = {}.".format(n, cycles))

	assert num_cycles == len(cycles)

	# Find n-cycles in a graph with collinear vertices. 
	#	Use the line graph L(G) of the original graph G.
	if collinear_vertex_sets:
		# Constructs a set of line segments.
		line_segment_sets = collinear_vertex_sets.copy()
		for ith, row in enumerate(graph):
			#for jth, flag in enumerate(row):
			#	if flag:
			for jth in range(ith):
				if row[jth]:
					new_ls = {ith, jth}
					is_new = True
					for line_segment_set in line_segment_sets:
						if new_ls.issubset(line_segment_set):
							is_new = False
							break
					if is_new: line_segment_sets.append(new_ls)

		# Number of line segments.
		LS = len(line_segment_sets)

		# Constructs an adjacency matrix of the line graph L(G) of the original graph G.
		#	REF [site] >> https://en.wikipedia.org/wiki/Line_graph
		line_graph = [[0] * LS for _ in range(LS)]
		for ith, ith_ls in enumerate(line_segment_sets):
			for jth, jth_ls in enumerate(line_segment_sets):
				if ith != jth and ith_ls.intersection(jth_ls):
					line_graph[ith][jth] = line_graph[jth][ith] = 1

		print("Start finding {}-cycles in the line graph of a graph...".format(n))
		start_time = time.time()
		edge_cycles, vertex_cycles= find_edge_cycles(line_graph, line_segment_sets, n)
		print("End finding {}-cycles in the line graph of a graph: {} secs.".format(n, time.time() - start_time))
		print("Edge cycles of length {} = {}.".format(n, edge_cycles))
		print("Vertex cycles of length {} = {}.".format(n, vertex_cycles))

	# FIXME [fix] >> Incorrect results.
	# Find n-cycles in a graph with collinear vertices.
	if False:
		if collinear_vertex_sets:
			print("Start finding {}-cycles in a graph with collinear vertices...".format(n))
			start_time = time.time()
			cycles = find_cycles_in_graph_with_collinear_vertices(graph, collinear_vertex_sets, n)
			print("End finding {}-cycles in a graph with collinear vertices: {} secs.".format(n, time.time() - start_time))
			print("Cycles of length {} = {}.".format(n, cycles))

	# NOTE [info] >> Apply some constraints to filter valid polygons (n-gons, n-cycles).
	#	REF [function] >> detect_quadrilaterals_by_lines() in ${DataAnalysis_HOME}/app/id/detect_region.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
