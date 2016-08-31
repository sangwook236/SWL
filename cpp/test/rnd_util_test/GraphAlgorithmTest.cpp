#include "swl/Config.h"
#include "swl/rnd_util/GraphAlgorithm.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

template <typename VertexDescriptor>
void displayPaths(const std::list<std::list<VertexDescriptor> >& paths)
{
	for (std::list<std::list<VertexDescriptor> >::const_iterator itPath = paths.begin(); itPath != paths.end(); ++itPath)
	{
		std::cout << "\t";
		for (std::list<VertexDescriptor>::const_iterator it = itPath->begin(); it != itPath->end(); ++it)
			std::cout << *it << " ";
		std::cout << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void graph_algorithm()
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
	swl::findAllPaths(graph, v2, v3, paths);

	// Output.
	//if (!paths.empty())
	{
		std::cout << "start vertex " << v2 << " --> target vertex " << v3 << std::endl;
		local::displayPaths(paths);
	}
}
