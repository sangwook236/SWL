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
template <typename Graph>
void findAllPaths(const Graph& graph, const typename Graph::vertex_descriptor& start, const typename Graph::vertex_descriptor& target, std::map<typename Graph::vertex_descriptor, bool>& visited, std::list<typename Graph::vertex_descriptor>& path, std::list<std::list<typename Graph::vertex_descriptor> >& paths)
{
	// Mark the current node and store it in path.
	visited[start] = true;
	path.push_back(start);

	// If current vertex is same as destination, then print current path.
	if (start == target)
	{
		paths.push_back(path);
	}
	else  // If current vertex is not destination.
	{
		// Recur for all the vertices adjacent to current vertex.
#if 1
		// For directed graphs.
		typename boost::graph_traits<Graph>::out_edge_iterator out_i, out_end;
		for (boost::tie(out_i, out_end) = boost::out_edges(start, graph); out_i != out_end; ++out_i)
		{
			typename Graph::vertex_descriptor targ = boost::target(*out_i, graph);
			if (!visited[targ])
				findAllPaths(graph, targ, target, visited, path, paths);
		}
#else
		// For undirected graphs.
		typename boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;
		for (boost::tie(ai, ai_end) = boost::adjacent_vertices(start, graph); ai != ai_end; ++ai)
		{
			if (!visited[*ai])
				findAllPaths(graph, *ai, target, visited, path, paths);
		}
#endif
	}

	// Remove the current vertex from path and mark it as unvisited.
	path.pop_back();
	visited[start] = false;
}

template <typename Graph>
void findAllPaths(const Graph& graph, const typename Graph::vertex_descriptor& start, const typename Graph::vertex_descriptor& target, std::list<std::list<typename Graph::vertex_descriptor> >& paths)
{
	std::map<typename Graph::vertex_descriptor, bool> visited;
	{
		typename Graph::vertex_iterator vertexIt, vertexBegin, vertexEnd;
		boost::tie(vertexBegin, vertexEnd) = boost::vertices(graph);
		for (vertexIt = vertexBegin; vertexIt != vertexEnd; ++vertexIt)
			visited[*vertexIt] = false;
	}
	std::list<typename Graph::vertex_descriptor> path;
	findAllPaths(graph, start, target, visited, path, paths);
}

}  // namespace swl


#endif  // __SWL_RND_UTIL__GRAPH_ALGORITHM__H_
