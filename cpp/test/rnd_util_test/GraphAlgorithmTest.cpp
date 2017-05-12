#include "swl/Config.h"
#include "swl/rnd_util/GraphAlgorithm.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

template <typename VertexDescriptor>
void displayPaths(const std::list<std::list<VertexDescriptor> >& paths, const bool reverse = false)
{
	for (typename std::list<std::list<VertexDescriptor> >::const_iterator itPath = paths.begin(); itPath != paths.end(); ++itPath)
	{
		std::cout << "\t";
		if (reverse)
			for (typename std::list<VertexDescriptor>::const_reverse_iterator rit = itPath->rbegin(); rit != itPath->rend(); ++rit)
				std::cout << *rit << " ";
		else
			for (typename std::list<VertexDescriptor>::const_iterator it = itPath->begin(); it != itPath->end(); ++it)
				std::cout << *it << " ";
		std::cout << std::endl;
	}
}

void find_all_possible_paths_in_directed_graph_example()
{
	//typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> graph_type;
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> graph_type;

	// Construct a graph.
	// REF [site] >> http://www.geeksforgeeks.org/find-paths-given-source-destination/
	graph_type graph;
	graph_type::vertex_descriptor v0 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v1 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v2 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v3 = boost::add_vertex(graph);

	boost::add_edge(v0, v2, graph);
	boost::add_edge(v2, v0, graph);
	boost::add_edge(v2, v1, graph);
	boost::add_edge(v0, v1, graph);
	boost::add_edge(v0, v3, graph);
	boost::add_edge(v1, v3, graph);

	std::cout << "#vertices = " << boost::num_vertices(graph) << ", #edges = " << boost::num_edges(graph) << std::endl;

	// Find all candidate paths.
	std::list<std::list<graph_type::vertex_descriptor> > paths;
	{
		boost::timer::auto_cpu_timer timer;
		swl::findAllPathsInDirectedGraph(graph, v2, v3, paths);
	}

	// Output.
	//if (!paths.empty())
	{
		std::cout << "All possible paths: start vertex " << v2 << " --> target vertex " << v3 << std::endl;
		displayPaths(paths);
	}
}

void find_all_possible_paths_toward_leaf_nodes_in_directed_graph_example()
{
	//typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> graph_type;
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> graph_type;

	// Construct a graph.
	// REF [site] >> http://www.geeksforgeeks.org/find-paths-given-source-destination/
	graph_type graph;
	graph_type::vertex_descriptor v0 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v1 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v2 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v3 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v4 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v5 = boost::add_vertex(graph);

	boost::add_edge(v0, v2, graph);
	boost::add_edge(v2, v0, graph);
	boost::add_edge(v2, v1, graph);
	boost::add_edge(v0, v1, graph);
	boost::add_edge(v0, v3, graph);
	boost::add_edge(v1, v3, graph);
	boost::add_edge(v1, v4, graph);  // Leaf node.
	boost::add_edge(v3, v5, graph);  // Leaf node.

	std::cout << "#vertices = " << boost::num_vertices(graph) << ", #edges = " << boost::num_edges(graph) << std::endl;

	// Find all candidate paths.
	std::list<std::list<graph_type::vertex_descriptor> > paths;
#if 0
	{
		boost::timer::auto_cpu_timer timer;
		swl::findAllPathsInDirectedGraph(graph, v2, v4, paths);
		swl::findAllPathsInDirectedGraph(graph, v2, v5, paths);
	}

	// Output.
	//if (!paths.empty())
	{
		std::cout << "All possible paths: start vertex " << v2 << " --> target vertex " << v5 << std::endl;
		displayPaths(paths);
	}
#else
	{
		// NOTICE [info] >> targets must be leaf nodes.
		std::set<graph_type::vertex_descriptor> targets;
		targets.insert(v4);
		targets.insert(v5);

		boost::timer::auto_cpu_timer timer;
		swl::findAllPathsInDirectedGraph(graph, v2, targets, paths);
	}

	// Output.
	//if (!paths.empty())
	{
		std::cout << "All possible paths: start vertex " << v2 << " --> target vertices (" << v4 << ',' << v5 << ')' << std::endl;
		displayPaths(paths);
	}
#endif
}

void find_all_possible_paths_in_undirected_graph_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

	// Construct a graph.
	// REF [site] >> http://www.geeksforgeeks.org/find-paths-given-source-destination/
	graph_type graph;
	graph_type::vertex_descriptor v0 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v1 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v2 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v3 = boost::add_vertex(graph);

	boost::add_edge(v0, v2, graph);
	boost::add_edge(v2, v0, graph);
	boost::add_edge(v2, v1, graph);
	boost::add_edge(v0, v1, graph);
	boost::add_edge(v0, v3, graph);
	boost::add_edge(v1, v3, graph);

	std::cout << "#vertices = " << boost::num_vertices(graph) << ", #edges = " << boost::num_edges(graph) << std::endl;

	// Find all candidate paths.
	std::list<std::list<graph_type::vertex_descriptor> > paths;
	{
		boost::timer::auto_cpu_timer timer;
		swl::findAllPathsInUndirectedGraph(graph, v2, v3, paths);
	}

	// Output.
	//if (!paths.empty())
	{
		std::cout << "All possible paths: start vertex " << v2 << " --> target vertex " << v3 << std::endl;
		displayPaths(paths);
	}
}

void find_all_possible_paths_toward_leaf_nodes_in_undirected_graph_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

	// Construct a graph.
	// REF [site] >> http://www.geeksforgeeks.org/find-paths-given-source-destination/
	graph_type graph;
	graph_type::vertex_descriptor v0 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v1 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v2 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v3 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v4 = boost::add_vertex(graph);
	graph_type::vertex_descriptor v5 = boost::add_vertex(graph);

	boost::add_edge(v0, v2, graph);
	boost::add_edge(v2, v0, graph);
	boost::add_edge(v2, v1, graph);
	boost::add_edge(v0, v1, graph);
	boost::add_edge(v0, v3, graph);
	boost::add_edge(v1, v3, graph);
	boost::add_edge(v1, v4, graph);  // Leaf node.
	boost::add_edge(v3, v5, graph);  // Leaf node.

	std::cout << "#vertices = " << boost::num_vertices(graph) << ", #edges = " << boost::num_edges(graph) << std::endl;

	// Find all candidate paths.
	std::list<std::list<graph_type::vertex_descriptor> > paths;
#if 0
	{
		boost::timer::auto_cpu_timer timer;
		swl::findAllPathsInUndirectedGraph(graph, v2, v4, paths);
		swl::findAllPathsInUndirectedGraph(graph, v2, v5, paths);
	}

	// Output.
	//if (!paths.empty())
	{
		std::cout << "All possible paths: start vertex " << v2 << " --> target vertices (" << v4 << ',' << v5 << ')' << std::endl;
		displayPaths(paths);
	}
#else
	{
		// NOTICE [info] >> targets must be leaf nodes.
		std::set<graph_type::vertex_descriptor> targets;
		targets.insert(v4);
		targets.insert(v5);

		boost::timer::auto_cpu_timer timer;
		swl::findAllPathsInUndirectedGraph(graph, v2, targets, paths);
	}

	// Output.
	//if (!paths.empty())
	{
		std::cout << "All possible paths: start vertex " << v2 << " --> target vertices (" << v4 << ',' << v5 << ')' << std::endl;
		displayPaths(paths);
	}
#endif
}

}  // namespace local
}  // unnamed namespace

void graph_algorithm()
{
	local::find_all_possible_paths_in_directed_graph_example();
	local::find_all_possible_paths_toward_leaf_nodes_in_directed_graph_example();

	local::find_all_possible_paths_in_undirected_graph_example();
	local::find_all_possible_paths_toward_leaf_nodes_in_undirected_graph_example();
}
