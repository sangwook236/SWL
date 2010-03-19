#include "swl/Config.h"
#include "swl/graphics/SceneNode.h"
#include "swl/graphics/ISceneVisitor.h"
#include "swl/base/LogException.h"
#include <algorithm>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// struct ISceneNode

template<typename SceneVisitor>
ISceneNode<SceneVisitor>::~ISceneNode()
{
}

//--------------------------------------------------------------------------
// class ComponentSceneNode

template<typename SceneVisitor>
#if defined(UNICODE) || defined(_UNICODE)
ComponentSceneNode<SceneVisitor>::ComponentSceneNode(const std::wstring &name /*= std::wstring()*/)
#else
ComponentSceneNode<SceneVisitor>::ComponentSceneNode(const std::string &name /*= std::string()*/);
#endif
: base_type(),
  parent_(), name_(name)
{
}

template<typename SceneVisitor>
ComponentSceneNode<SceneVisitor>::ComponentSceneNode(const ComponentSceneNode &rhs)
: base_type(rhs),
  parent_(rhs.parent_), name_(rhs.name_)
{
}

template<typename SceneVisitor>
ComponentSceneNode<SceneVisitor>::~ComponentSceneNode()
{
}

template<typename SceneVisitor>
ComponentSceneNode<SceneVisitor> & ComponentSceneNode<SceneVisitor>::operator=(const ComponentSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	parent_ = rhs.parent_;
	name_ = rhs.name_;
	return *this;
}

//--------------------------------------------------------------------------
// class GroupSceneNode

template<typename SceneVisitor>
#if defined(UNICODE) || defined(_UNICODE)
GroupSceneNode<SceneVisitor>::GroupSceneNode(const std::wstring &name /*= std::wstring()*/)
#else
GroupSceneNode<SceneVisitor>::GroupSceneNode(const std::string &name /*= std::string()*/);
#endif
: base_type(name),
  children_()
{
}

template<typename SceneVisitor>
GroupSceneNode<SceneVisitor>::GroupSceneNode(const GroupSceneNode &rhs)
: base_type(rhs),
  children_(rhs.children_)
{
}

template<typename SceneVisitor>
GroupSceneNode<SceneVisitor>::~GroupSceneNode()
{
}

template<typename SceneVisitor>
GroupSceneNode<SceneVisitor> & GroupSceneNode<SceneVisitor>::operator=(const GroupSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	children_.assign(children_.begin(), children_.end());
	return *this;
}

template<typename SceneVisitor>
void GroupSceneNode<SceneVisitor>::accept(const visitor_type &visitor) const
{
	traverse(visitor);
	//visitor.visit(*this);
}

template<typename SceneVisitor>
void GroupSceneNode<SceneVisitor>::addChild(const node_type &node)
{
	children_.push_back(node);
}

template<typename SceneVisitor>
void GroupSceneNode<SceneVisitor>::removeChild(const node_type &node)
{
	children_.remove(node);
}

template<typename SceneVisitor>
void GroupSceneNode<SceneVisitor>::clearChildren()
{
	children_.clear();
}

template<typename SceneVisitor>
void GroupSceneNode<SceneVisitor>::traverse(const visitor_type &visitor) const
{
	for (std::list<node_type>::const_iterator it = children_.begin(); it != children_.end(); ++it)
		if (*it) (*it)->accept(visitor);
}

template<typename SceneVisitor>
void GroupSceneNode<SceneVisitor>::replace(const node_type &oldNode, const node_type &newNode)
{
	std::replace(children_.begin(), children_.end(), oldNode, newNode);
}

//--------------------------------------------------------------------------
// class LeafSceneNode

template<typename SceneVisitor>
#if defined(UNICODE) || defined(_UNICODE)
LeafSceneNode<SceneVisitor>::LeafSceneNode(const std::wstring &name /*= std::wstring()*/)
#else
LeafSceneNode<SceneVisitor>::LeafSceneNode(const std::string &name /*= std::string()*/);
#endif
: base_type(name)
{
}

template<typename SceneVisitor>
LeafSceneNode<SceneVisitor>::LeafSceneNode(const LeafSceneNode &rhs)
: base_type(rhs)
{
}

template<typename SceneVisitor>
LeafSceneNode<SceneVisitor>::~LeafSceneNode()
{
}

template<typename SceneVisitor>
LeafSceneNode<SceneVisitor> & LeafSceneNode<SceneVisitor>::operator=(const LeafSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	return *this;
}

template<typename SceneVisitor>
void LeafSceneNode<SceneVisitor>::addChild(const node_type &node)
{
	throw LogException(LogException::L_INFO, "can't add a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

template<typename SceneVisitor>
void LeafSceneNode<SceneVisitor>::removeChild(const node_type &node)
{
	throw LogException(LogException::L_INFO, "can't remove a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

template<typename SceneVisitor>
void LeafSceneNode<SceneVisitor>::clearChildren()
{
	throw LogException(LogException::L_INFO, "can't clear all children to a leaf", __FILE__, __LINE__, __FUNCTION__);
}

}  // namespace swl
