# Counts cycles of length N (N-cycles) in a connected undirected graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def count_cycles(graph, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, marked, n, curr, start, count):
		# Mark the vertex curr as visited.
		marked[curr] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark curr as un-visited to make it usable again.
			marked[curr] = False

			# Check if vertex curr can end with vertex start.
			if graph[curr][start] == 1:
				count = count + 1
				return count
			else:
				return count

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[curr][i] == 1:
				# DFS for searching path by decreasing length by 1.
				count = DFS(graph, marked, n - 1, i, start, count)

		# Marking curr as unvisited to make it usable again.
		marked[curr] = False
		return count

	# All vertices are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	count = 0
	for i in range(V - (n - 1)):
		count = DFS(graph, marked, n - 1, i, i, count)

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return count // 2

# Finds cycles of length N (N-cycles) in a connected undirected graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_cycles(graph, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, marked, n, curr, start, curr_path, cycles):
		# Mark the vertex curr as visited.
		marked[curr] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark curr as un-visited to make it usable again.
			marked[curr] = False

			# Check if vertex curr can end with vertex start.
			if graph[curr][start] == 1 and \
				[curr_path[0]] + curr_path[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
				cycles.append(curr_path.copy())
			return

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[curr][i] == 1:
				curr_path.append(i)

				# DFS for searching path by decreasing length by 1.
				DFS(graph, marked, n - 1, i, start, curr_path, cycles)

				curr_path.pop()

		# Marking curr as unvisited to make it usable again.
		marked[curr] = False

	# All vertices are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	cycles = list()
	for i in range(V - (n - 1)):
		DFS(graph, marked, n - 1, i, i, curr_path=[i], cycles=cycles)

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return cycles

# Finds cycles of length N (N-cycles) in a connected undirected edge graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_edge_cycles(graph, line_sets, n):
	# Number of edges.
	E = len(graph)

	def DFS(graph, line_sets, marked, n, curr, start, edge_path, vertex_path, edge_cycles, vertex_cycles=None):
		# Mark the edge curr as visited.
		marked[curr] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark curr as un-visited to make it usable again.
			marked[curr] = False

			# Check if edge curr can end with edge start.
			if graph[curr][start] == 1 and \
				[edge_path[0]] + edge_path[1:][::-1] not in edge_cycles:  # Check if the reversed current path exists in the set of cycles or not.
				intersection_vertex = line_sets[curr].intersection(line_sets[start])
				# NOTE [info] >> More than 2 adjacent points can be included in two more sets.
				#assert len(intersection_vertex) == 1, "{} intersection {} = {}".format(line_sets[curr], line_sets[start], intersection_vertex)
				intersection_vertex = intersection_vertex.pop()

				if intersection_vertex not in vertex_path:
					edge_cycles.append(edge_path.copy())
					vertex_cycles.append(vertex_path + [intersection_vertex])
			return

		# For searching every possible path of length (n - 1).
		for i in range(E):
			if marked[i] == False and graph[curr][i] == 1:
				intersection_vertex = line_sets[curr].intersection(line_sets[i])
				# NOTE [info] >> More than 2 adjacent points can be included in two more sets.
				#assert len(intersection_vertex) == 1, "{} intersection {} = {}".format(line_sets[curr], line_sets[i], intersection_vertex)
				intersection_vertex = intersection_vertex.pop()

				#if intersection_vertex in vertex_path:
				if intersection_vertex in vertex_path or any([intersection_vertex in line_sets[eg] for eg in edge_path[:-1]]):
					continue

				edge_path.append(i)
				vertex_path.append(intersection_vertex)

				# DFS for searching path by decreasing length by 1.
				DFS(graph, line_sets, marked, n - 1, i, start, edge_path, vertex_path, edge_cycles, vertex_cycles)

				edge_path.pop()
				vertex_path.pop()

		# Marking curr as unvisited to make it usable again.
		marked[curr] = False

	# All edges are marked un-visited initially.
	marked = [False] * E

	# Searching for cycle by using v - n + 1 edges.
	edge_cycles, vertex_cycles = list(), list()
	for i in range(E - (n - 1)):
		DFS(graph, line_sets, marked, n - 1, i, i, edge_path=[i], vertex_path=[], edge_cycles=edge_cycles, vertex_cycles=vertex_cycles)

		# i-th edge is marked as visited and will not be visited again.
		marked[i] = True
	
	return edge_cycles, vertex_cycles

# Finds cycles of length N (N-cycles) in a connected undirected graph with collinear vertices.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_cycles_in_graph_with_collinear_vertices(graph, collinear_vertex_sets, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, collinear_vertex_sets, marked, n, curr, start, curr_path, collinear_flags, cycles):
		# Mark the vertex curr as visited.
		marked[curr] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark curr as un-visited to make it usable again.
			marked[curr] = False

			# Check if vertex curr can end with vertex start.
			if graph[curr][start] == 1:
				if len(collinear_flags) > n:  # Collinear vertices exist in the cycle.
					new_cycle = list(vtx for flag, vtx in zip(collinear_flags[1:], curr_path) if not flag) + [curr_path[-1]]  # Removes collinear vertices.
					if [new_cycle[0]] + new_cycle[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
						cycles.append(new_cycle)
				else:
					if [curr_path[0]] + curr_path[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
						cycles.append(curr_path.copy())
			return

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[curr][i] == 1:
				curr_path.append(i)
				is_collinear = False
				if len(curr_path) == 3:
					vertex_set = set(curr_path)
					for collinear_vertex_set in collinear_vertex_sets:
						if vertex_set.issubset(collinear_vertex_set):  # It is sensitive to the order of vertices in curr_path.
							is_collinear = True
							break
				elif len(curr_path) > 3:
					vertex_sets = [set(curr_path[-3:]), set(curr_path[-2:] + [curr_path[0]]), set([curr_path[-1]] + curr_path[:2])]
					for collinear_vertex_set in collinear_vertex_sets:
						for vertex_set in vertex_sets:
					#for vertex_set in [set(curr_path[-3:]), set(curr_path[-2:] + [curr_path[0]]), set([curr_path[-1]] + curr_path[:2])]:
					#	for collinear_vertex_set in collinear_vertex_sets:
							if vertex_set.issubset(collinear_vertex_set):  # It is sensitive to the order of vertices in curr_path.
						#for collinear_vertex_set in collinear_vertex_sets:
						#	if len(collinear_vertex_set.intersection(vertex_set)) > 2:  # Error.
								is_collinear = True
								break
						if is_collinear: break
				collinear_flags.append(is_collinear)

				# DFS for searching path by decreasing length by 1.
				DFS(graph, collinear_vertex_sets, marked, n if is_collinear else n - 1, i, start, curr_path, collinear_flags, cycles)

				curr_path.pop()
				collinear_flags.pop()

		# Marking curr as unvisited to make it usable again.
		marked[curr] = False

	# All vertices are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	cycles = list()
	for i in range(V - (n - 1)):
		DFS(graph, collinear_vertex_sets, marked, n - 1, i, i, curr_path=[i], collinear_flags=[False], cycles=cycles)

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return cycles
