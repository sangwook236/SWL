#pragma once

#if !defined(__SWL_RND_UTIL__GRAPH_TRAVERSAL__H_)
#define __SWL_RND_UTIL__GRAPH_TRAVERSAL__H_ 1


#include <queue>
#include <list>


namespace swl {

//--------------------------------------------------------------------------
// Tree Traversal Algorithm.

// Depth-first search (DFS) : pre-order.
template<typename TreeNode, class Visitor>
void DFS_preorder(TreeNode& n, Visitor visitor)
{
	visitor(n);
	if (n.hasLeft()) DFS_preorder(n.getLeft(), visitor);
	if (n.hasRight()) DFS_preorder(n.getRight(), visitor);
}

// Depth-first search (DFS) : in-order.
template<typename TreeNode, class Visitor>
void DFS_inorder(TreeNode& n, Visitor visitor)
{
	if (n.hasLeft()) DFS_inorder(n.getLeft(), visitor);
	visitor(n);
	if (n.hasRight()) DFS_inorder(n.getRight(), visitor);
}

// Depth-first search (DFS) : post-order.
template<typename TreeNode, class Visitor>
void DFS_postorder(TreeNode& n, Visitor visitor)
{
	if (n.hasLeft()) DFS_postorder(n.getLeft(), visitor);
	if (n.hasRight()) DFS_postorder(n.getRight(), visitor);
	visitor(n);
}

// Breadth-first search (BFS).
template<typename TreeNode, class Visitor>
void BFS(TreeNode& n, Visitor visitor)
{
	std::queue<TreeNode> que;
	que.push(n);

	while (!que.empty())
	{
		TreeNode& m = que.front();
		que.pop();

		visitor(m);

		if (m.hasLeft()) que.push(m.getLeft());
		if (m.hasRight()) que.push(m.getRight());
	}
}

//--------------------------------------------------------------------------
// Graph Traversal Algorithm.

// Depth-first search (DFS).
template<typename Graph, typename Vertex, class Visitor>
void DFS(const Graph& g, Vertex* v, Visitor visitor)
{
	v->setVisited();
	visitor(*v);

	const std::list<Vertex*>& adjacents = g.getAdjacents(v);
	for (auto u : adjacents)
		if (!u->isVisited()) DFS(g, u, visitor);
}

// Breadth-first search (BFS).
template<typename Graph, typename Vertex, class Visitor>
void BFS(const Graph& g, Vertex* v, Visitor visitor)
{
	std::queue<Vertex*> que;
	que.push(v);

	while (!que.empty())
	{
		Vertex* u = que.front();
		que.pop();

        if (!u->isVisited())
        {
            u->setVisited();
            visitor(*u);

			const std::list<Vertex*>& adjacents = g.getAdjacents(u);
			for (auto w : adjacents)
				if (!w->isVisited()) que.push(w);
		}
	}
}

}  // namespace swl


#endif  // __SWL_RND_UTIL__GRAPH_TRAVERSAL__H_
