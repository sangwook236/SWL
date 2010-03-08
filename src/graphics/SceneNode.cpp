#include "swl/Config.h"
#include "swl/graphics/SceneNode.h"
#include "swl/base/LogException.h"
#include <boost/foreach.hpp>
#include <algorithm>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// struct ISceneNode

//--------------------------------------------------------------------------
// class SceneComponentNode

SceneComponentNode::SceneComponentNode()
: //base_type(),
  parent_()
{
}

SceneComponentNode::SceneComponentNode(const SceneComponentNode &rhs)
: //base_type(rhs),
  parent_(rhs.parent_)
{
}

SceneComponentNode::~SceneComponentNode()
{
}

SceneComponentNode & SceneComponentNode::operator=(const SceneComponentNode &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	parent_ = rhs.parent_;
	return *this;
}

//--------------------------------------------------------------------------
// class SceneCompositeNode

SceneCompositeNode::SceneCompositeNode()
: base_type(),
  children_()
{
}

SceneCompositeNode::SceneCompositeNode(const SceneCompositeNode &rhs)
: base_type(rhs),
  children_(rhs.children_)
{
}

SceneCompositeNode::~SceneCompositeNode()
{
}

SceneCompositeNode & SceneCompositeNode::operator=(const SceneCompositeNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	children_.assign(children_.begin(), children_.end());
	return *this;
}

void SceneCompositeNode::draw()
{
	BOOST_FOREACH(node_type node, children_)
	{
		node->draw();
	}
}

void SceneCompositeNode::addChild(const SceneLeafNode::node_type &node)
{
	children_.push_back(node);
}

void SceneCompositeNode::removeChild(const SceneLeafNode::node_type &node)
{
	children_.remove(node);
}

void SceneCompositeNode::clearChildren()
{
	children_.clear();
}

void SceneCompositeNode::replace(const SceneLeafNode::node_type &oldNode, const SceneLeafNode::node_type &newNode)
{
	std::replace(children_.begin(), children_.end(), oldNode, newNode);
}

//--------------------------------------------------------------------------
// class SceneLeafNode

SceneLeafNode::SceneLeafNode()
: base_type()
{
}

SceneLeafNode::SceneLeafNode(const SceneLeafNode &rhs)
: base_type(rhs)
{
}

SceneLeafNode::~SceneLeafNode()
{
}

SceneLeafNode & SceneLeafNode::operator=(const SceneLeafNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	return *this;
}

void SceneLeafNode::addChild(const SceneLeafNode::node_type &node)
{
	throw LogException(LogException::L_INFO, "can't add a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

void SceneLeafNode::removeChild(const SceneLeafNode::node_type &node)
{
	throw LogException(LogException::L_INFO, "can't remove a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

void SceneLeafNode::clearChildren()
{
	throw LogException(LogException::L_INFO, "can't clear all children to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

}  // namespace swl
