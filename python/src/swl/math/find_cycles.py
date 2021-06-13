# Counts cycles of length N (N-cycles) in a connected undirected graph graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def count_cycles(graph, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, marked, n, vert, start, count):
		# Mark the vertex vert as visited.
		marked[vert] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark vert as un-visited to make it usable again.
			marked[vert] = False

			# Check if vertex vert can end with vertex start.
			if graph[vert][start] == 1:
				count = count + 1
				return count
			else:
				return count

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[vert][i] == 1:
				# DFS for searching path by decreasing length by 1.
				count = DFS(graph, marked, n - 1, i, start, count)

		# Marking vert as unvisited to make it usable again.
		marked[vert] = False
		return count

	# All vertex are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	count = 0
	for i in range(V - (n - 1)):
		count = DFS(graph, marked, n - 1, i, i, count)

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return count // 2

# Finds cycles of length N (N-cycles) in a connected undirected graph graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_cycles(graph, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, marked, n, vert, start, cycles, curr_path):
		# Mark the vertex vert as visited.
		marked[vert] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark vert as un-visited to make it usable again.
			marked[vert] = False

			# Check if vertex vert can end with vertex start.
			if graph[vert][start] == 1 and \
				[curr_path[0]] + curr_path[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
				cycles.append(curr_path.copy())
			return

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[vert][i] == 1:
				curr_path.append(i)

				# DFS for searching path by decreasing length by 1.
				DFS(graph, marked, n - 1, i, start, cycles, curr_path)

				curr_path.pop()

		# Marking vert as unvisited to make it usable again.
		marked[vert] = False

	# All vertex are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	cycles = list()
	for i in range(V - (n - 1)):
		DFS(graph, marked, n - 1, i, i, cycles, curr_path=[i])

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return cycles

# Finds cycles of length N (N-cycles) in a connected undirected graph with collinear vertices.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_cycles_in_graph_with_collinear_vertices(graph, collinear_vertex_sets, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, collinear_vertex_sets, marked, n, vert, start, cycles, curr_path, collinear_flags):
		# Mark the vertex vert as visited.
		marked[vert] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark vert as un-visited to make it usable again.
			marked[vert] = False

			# Check if vertex vert can end with vertex start.
			if graph[vert][start] == 1:
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
			if marked[i] == False and graph[vert][i] == 1:
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
				DFS(graph, collinear_vertex_sets, marked, n if is_collinear else n - 1, i, start, cycles, curr_path, collinear_flags)

				curr_path.pop()
				collinear_flags.pop()

		# Marking vert as unvisited to make it usable again.
		marked[vert] = False

	# All vertex are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	cycles = list()
	for i in range(V - (n - 1)):
		DFS(graph, collinear_vertex_sets, marked, n - 1, i, i, cycles, curr_path=[i], collinear_flags=[False])

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return cycles
