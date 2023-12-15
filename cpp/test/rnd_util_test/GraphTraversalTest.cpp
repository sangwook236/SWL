#include "swl/Config.h"
#include "swl/rnd_util/GraphTraversal.h"
#include <iostream>
#include <string>
#include <list>
#include <memory>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

class TreeNode
{
public:
	TreeNode(const std::string& name)
	: name_(name), left_(nullptr), right_(nullptr)
	{}

public:
	void addLeft(TreeNode* left)  { left_ = left; }
	void addRight(TreeNode* right)  { right_ = right; }

	bool hasLeft() const { return nullptr != left_; }
	bool hasRight() const { return nullptr != right_; }

	TreeNode& getLeft() { return *left_; }
	const TreeNode& getLeft() const { return *left_; }
	TreeNode& getRight() { return *right_; }
	const TreeNode& getRight() const { return *right_; }

	const std::string& getName() const { return name_; }

private:
	const std::string name_;

	TreeNode* left_;
	TreeNode* right_;
};

template<typename TreeNode>
struct TreeVisitor
{
	void operator()(const TreeNode& node) const
	{
		std::cout << node.getName() << " -> ";
	}
};

class GraphVertex
{
public:
	GraphVertex(const std::string& name)
	: name_(name), isVisited_(false)
	{}

private:
	GraphVertex(const GraphVertex& rhs);
	GraphVertex& operator=(const GraphVertex& rhs);

public:
	void setVisited()  { isVisited_= true; }
	void resetVisited()  { isVisited_= false; }
	bool isVisited() const  { return isVisited_; }

	const std::string& getName() const { return name_; }

private:
	const std::string name_;

	bool isVisited_;
};

class UndirectedGraph
{
public:
	typedef GraphVertex vertex_type;
	typedef std::pair<std::size_t, std::size_t> edge_type;

public:
	UndirectedGraph()
	{}

public:
	void addVertex(vertex_type* v)  { vertices_.push_back(v); }
	void addEdge(const std::size_t& v1, const std::size_t& v2)  { edges_.push_back(std::make_pair(v1, v2)); }

	std::list<vertex_type*> getAdjacents(vertex_type* v) const
	{
		std::list<vertex_type*> vertices;
		for (std::list<edge_type>::const_iterator cit = edges_.begin(); cit != edges_.end(); ++cit)
		{
			vertex_type* u(*std::next(vertices_.begin(), cit->first));
			vertex_type* w(*std::next(vertices_.begin(), cit->second));
			if (u == v) vertices.push_back(w);
			else if (w == v) vertices.push_back(u);
		}
		return vertices;
	}

	void resetVisited() { for (auto v : vertices_) v->resetVisited(); }

private:
	std::list<vertex_type*> vertices_;
	std::list<edge_type> edges_;
};

template<typename Vertex>
struct GraphVisitor
{
	void operator()(const Vertex& v) const
	{
		std::cout << v.getName() << ", ";
	}
};

}  // namespace local
}  // unnamed namespace

void tree_traversal()
{
	// REF [site] >> https://en.wikipedia.org/wiki/Tree_traversal

	// create tree.
	std::unique_ptr<local::TreeNode> nA(new local::TreeNode("A"));
	std::unique_ptr<local::TreeNode> nB(new local::TreeNode("B"));
	std::unique_ptr<local::TreeNode> nC(new local::TreeNode("C"));
	std::unique_ptr<local::TreeNode> nD(new local::TreeNode("D"));
	std::unique_ptr<local::TreeNode> nE(new local::TreeNode("E"));
	std::unique_ptr<local::TreeNode> nF(new local::TreeNode("F"));
	std::unique_ptr<local::TreeNode> nG(new local::TreeNode("G"));
	std::unique_ptr<local::TreeNode> nH(new local::TreeNode("H"));
	std::unique_ptr<local::TreeNode> nI(new local::TreeNode("I"));

	nF->addLeft(nB.get()); nF->addRight(nG.get());
	nB->addLeft(nA.get()); nB->addRight(nD.get());
	nD->addLeft(nC.get()); nD->addRight(nE.get());
	nG->addRight(nI.get());
	nI->addLeft(nH.get());

	// DFS.
	std::cout << "Tree DFS pre-order : ";
	swl::DFS_preorder(*nF, local::TreeVisitor<local::TreeNode>());
	std::cout << std::endl;
	std::cout << "Tree DFS in-order : ";
	swl::DFS_inorder(*nF, local::TreeVisitor<local::TreeNode>());
	std::cout << std::endl;
	std::cout << "Tree DFS post-order : ";
	swl::DFS_postorder(*nF, local::TreeVisitor<local::TreeNode>());
	std::cout << std::endl;

	// BFS.
	std::cout << "Tree BFS : ";
	swl::BFS(*nF, local::TreeVisitor<local::TreeNode>());
	std::cout << std::endl;
}

void graph_traversal()
{
	// REF [site] >> https://en.wikipedia.org/wiki/Depth-first_search

	// create graph.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vA(new local::UndirectedGraph::vertex_type("A"));  // 0.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vB(new local::UndirectedGraph::vertex_type("B"));  // 1.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vC(new local::UndirectedGraph::vertex_type("C"));  // 2.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vD(new local::UndirectedGraph::vertex_type("D"));  // 3.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vE(new local::UndirectedGraph::vertex_type("E"));  // 4.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vF(new local::UndirectedGraph::vertex_type("F"));  // 5.
	std::unique_ptr<local::UndirectedGraph::vertex_type> vG(new local::UndirectedGraph::vertex_type("G"));  // 6.

	local::UndirectedGraph g;
	g.addVertex(vA.get());
	g.addVertex(vB.get());
	g.addVertex(vC.get());
	g.addVertex(vD.get());
	g.addVertex(vE.get());
	g.addVertex(vF.get());
	g.addVertex(vG.get());
	g.addEdge(0, 1);  // A - B.
	g.addEdge(0, 2);  // A - C.
	g.addEdge(0, 4);  // A - E.
	g.addEdge(1, 3);  // B - D.
	g.addEdge(1, 5);  // B - F.
	g.addEdge(2, 6);  // C - G.
	g.addEdge(4, 5);  // E - F.

	// DFS.
	std::cout << "Graph DFS : ";
	g.resetVisited();
	swl::DFS(g, vA.get(), local::GraphVisitor<local::UndirectedGraph::vertex_type>());
	std::cout << std::endl;

	// BFS.
	std::cout << "Graph BFS : ";
	g.resetVisited();
	swl::BFS(g, vA.get(), local::GraphVisitor<local::UndirectedGraph::vertex_type>());
	std::cout << std::endl;
}
