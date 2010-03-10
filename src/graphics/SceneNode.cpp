#include "swl/Config.h"
#include "swl/graphics/SceneNode.h"
#include "swl/graphics/ISceneVisitor.h"
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

ISceneNode::~ISceneNode()
{
}

//--------------------------------------------------------------------------
// class ComponentSceneNode

ComponentSceneNode::ComponentSceneNode()
: base_type(),
  parent_()
{
}

ComponentSceneNode::ComponentSceneNode(const ComponentSceneNode &rhs)
: base_type(rhs),
  parent_(rhs.parent_)
{
}

ComponentSceneNode::~ComponentSceneNode()
{
}

ComponentSceneNode & ComponentSceneNode::operator=(const ComponentSceneNode &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	parent_ = rhs.parent_;
	return *this;
}

//--------------------------------------------------------------------------
// class GroupSceneNode

GroupSceneNode::GroupSceneNode()
: base_type(),
  children_()
{
}

GroupSceneNode::GroupSceneNode(const GroupSceneNode &rhs)
: base_type(rhs),
  children_(rhs.children_)
{
}

GroupSceneNode::~GroupSceneNode()
{
}

GroupSceneNode & GroupSceneNode::operator=(const GroupSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	children_.assign(children_.begin(), children_.end());
	return *this;
}

void GroupSceneNode::accept(const ISceneVisitor &visitor) const
{
	traverse(visitor);
	//visitor.visit(*this);
}

void GroupSceneNode::addChild(const GroupSceneNode::node_type &node)
{
	children_.push_back(node);
}

void GroupSceneNode::removeChild(const GroupSceneNode::node_type &node)
{
	children_.remove(node);
}

void GroupSceneNode::clearChildren()
{
	children_.clear();
}

void GroupSceneNode::traverse(const ISceneVisitor &visitor) const
{
	BOOST_FOREACH(node_type node, children_)
	{
		if (node) node->accept(visitor);
	}
}

void GroupSceneNode::replace(const GroupSceneNode::node_type &oldNode, const GroupSceneNode::node_type &newNode)
{
	std::replace(children_.begin(), children_.end(), oldNode, newNode);
}

//--------------------------------------------------------------------------
// class LeafSceneNode

LeafSceneNode::LeafSceneNode()
: base_type()
{
}

LeafSceneNode::LeafSceneNode(const LeafSceneNode &rhs)
: base_type(rhs)
{
}

LeafSceneNode::~LeafSceneNode()
{
}

LeafSceneNode & LeafSceneNode::operator=(const LeafSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	return *this;
}

void LeafSceneNode::addChild(const LeafSceneNode::node_type &node)
{
	throw LogException(LogException::L_INFO, "can't add a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

void LeafSceneNode::removeChild(const LeafSceneNode::node_type &node)
{
	throw LogException(LogException::L_INFO, "can't remove a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

void LeafSceneNode::clearChildren()
{
	throw LogException(LogException::L_INFO, "can't clear all children to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

}  // namespace swl
