#pragma once

#if !defined(__SWL_RND_UTIL__GRAPH_ALGORITHM__H_)
#define __SWL_RND_UTIL__GRAPH_ALGORITHM__H_ 1


#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <list>


namespace swl {

//--------------------------------------------------------------------------
// Graph Algorithm.

// REF [site] >> http://www.geeksforgeeks.org/find-paths-given-source-destination/
template <typename Graph, typename Vertex = Graph::vertex_descriptor>
void findAllPathsInDirectedGraph(const Graph& graph, const typename Vertex& start, const typename Vertex& target, std::map<Vertex, bool>& visited, std::list<Vertex>& path, std::list<std::list<Vertex> >& paths)
{
	// Mark the current node and store it in path.
	visited[start] = true;
	path.push_back(start);

	// If the current vertex is same as destination, then print current path.
	if (start == target)
		paths.push_back(path);
	else  // If the current vertex is not destination.
	{
		// Recur for all the vertices adjacent to the current vertex.
		// For directed graphs.
		typename boost::graph_traits<Graph>::out_edge_iterator out_i, out_end;
		for (boost::tie(out_i, out_end) = boost::out_edges(start, graph); out_i != out_end; ++out_i)
		{
			Vertex targ = boost::target(*out_i, graph);
			if (!visited[targ])
				findAllPathsInDirectedGraph(graph, targ, target, visited, path, paths);
		}
	}

	// Remove the current vertex from path and mark it as unvisited.
	path.pop_back();
	visited[start] = false;
}

// REF [site] >> http://www.geeksforgeeks.org/find-paths-given-source-destination/
template <typename Graph, typename Vertex = Graph::vertex_descriptor>
void findAllPathsInUndirectedGraph(const Graph& graph, const typename Vertex& start, const typename Vertex& target, std::map<Vertex, bool>& visited, std::list<Vertex>& path, std::list<std::list<Vertex> >& paths)
{
	// Mark the current node and store it in path.
	visited[start] = true;
	path.push_back(start);

	// If the current vertex is same as destination, then print current path.
	if (start == target)
		paths.push_back(path);
	else  // If the current vertex is not destination.
	{
		// Recur for all the vertices adjacent to the current vertex.
		// For undirected graphs.
		typename boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;
		for (boost::tie(ai, ai_end) = boost::adjacent_vertices(start, graph); ai != ai_end; ++ai)
			if (!visited[*ai])
				findAllPathsInUndirectedGraph(graph, *ai, target, visited, path, paths);
	}

	// Remove the current vertex from path and mark it as unvisited.
	path.pop_back();
	visited[start] = false;
}

template <typename Graph, typename Vertex = Graph::vertex_descriptor>
void findAllPathsInDirectedGraph(const Graph& graph, const Vertex& start, const Vertex& target, std::list<std::list<Vertex> >& paths)
{
	std::map<Vertex, bool> visited;
	{
		typename Graph::vertex_iterator vertexIt, vertexBegin, vertexEnd;
		boost::tie(vertexBegin, vertexEnd) = boost::vertices(graph);
		for (vertexIt = vertexBegin; vertexIt != vertexEnd; ++vertexIt)
			visited[*vertexIt] = false;
	}
	std::list<Vertex> path;
	findAllPathsInDirectedGraph(graph, start, target, visited, path, paths);
}

template <typename Graph, typename Vertex = Graph::vertex_descriptor>
void findAllPathsInUndirectedGraph(const Graph& graph, const Vertex& start, const Vertex& target, std::list<std::list<Vertex> >& paths)
{
	std::map<Vertex, bool> visited;
	{
		typename Graph::vertex_iterator vertexIt, vertexBegin, vertexEnd;
		boost::tie(vertexBegin, vertexEnd) = boost::vertices(graph);
		for (vertexIt = vertexBegin; vertexIt != vertexEnd; ++vertexIt)
			visited[*vertexIt] = false;
	}
	std::list<Vertex> path;
	findAllPathsInUndirectedGraph(graph, start, target, visited, path, paths);
}

}  // namespace swl


#endif  // __SWL_RND_UTIL__GRAPH_ALGORITHM__H_
