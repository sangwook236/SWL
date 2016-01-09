#if !defined(__SWL_RND_UTIL__SORT__H_)
#define __SWL_RND_UTIL__SORT__H_ 1


#include <queue>
#include <list>


namespace swl {

//--------------------------------------------------------------------------
// Tree Traversal Algorithm

// depth-first search (DFS) : pre-order
template<typename TreeNode, class Visitor>
void DFS_preorder(TreeNode& n, Visitor visitor)
{
	visitor(n);
	if (n.hasLeft()) DFS_preorder(n.getLeft(), visitor);
	if (n.hasRight()) DFS_preorder(n.getRight(), visitor);
}

// depth-first search (DFS) : in-order
template<typename TreeNode, class Visitor>
void DFS_inorder(TreeNode& n, Visitor visitor)
{
	if (n.hasLeft()) DFS_inorder(n.getLeft(), visitor);
	visitor(n);
	if (n.hasRight()) DFS_inorder(n.getRight(), visitor);
}

// depth-first search (DFS) : post-order
template<typename TreeNode, class Visitor>
void DFS_postorder(TreeNode& n, Visitor visitor)
{
	if (n.hasLeft()) DFS_postorder(n.getLeft(), visitor);
	if (n.hasRight()) DFS_postorder(n.getRight(), visitor);
	visitor(n);
}

// breadth-first search (BFS)
template<typename TreeNode, class Visitor>
void BFS(TreeNode& n, Visitor visitor)
{
	std::queue<TreeNode> q;
	q.push(n);

	while (!q.empty())
	{
		TreeNode& m = q.front();
		q.pop();

		visitor(m);

		if (m.hasLeft()) q.push(m.getLeft());
		if (m.hasRight()) q.push(m.getRight());
	}
}

//--------------------------------------------------------------------------
// Graph Traversal Algorithm

// depth-first search (DFS)
template<typename Graph, typename Vertex, class Visitor>
void DFS(const Graph& g, Vertex* v, Visitor visitor)
{
	v->setVisited();
	visitor(*v);

	const std::list<Vertex*>& adjacents = g.getAdjacents(v);
	for (auto u : adjacents)
		if (!u->isVisited()) DFS(g, u, visitor);
}

// breadth-first search (BFS)
template<typename Graph, typename Vertex, class Visitor>
void BFS(const Graph& g, Vertex* v, Visitor visitor)
{
	std::queue<Vertex*> q;
	q.push(v);

	while (!q.empty())
	{
		Vertex* u = q.front();
		q.pop();

        if (!u->isVisited())
        {
            u->setVisited();
            visitor(*u);
        }

		const std::list<Vertex*>& adjacents = g.getAdjacents(u);
		for (auto w : adjacents)
			if (!w->isVisited()) q.push(w);
	}
}

}  // namespace swl


#endif  // __SWL_RND_UTIL__SORT__H_
