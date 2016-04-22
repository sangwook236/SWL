#if !defined(__SWL_GRAPHICS__SCENE_NODE__H_)
#define __SWL_GRAPHICS__SCENE_NODE__H_ 1


#include "swl/base/IVisitable.h"
#include "swl/base/LogException.h"
#include <boost/smart_ptr.hpp>
#include <list>


namespace swl {

//--------------------------------------------------------------------------
// struct ISceneNode

template<typename SceneVisitor>
struct ISceneNode: public IVisitable<SceneVisitor>
{
public:
	//typedef ISceneNode					base_type;
	typedef boost::shared_ptr<ISceneNode>	node_type;

public:
	virtual ~ISceneNode()  {}

public:
	virtual void addChild(const node_type &node) = 0;
 	virtual void removeChild(const node_type &node) = 0;
	virtual void clearChildren() = 0;
	virtual size_t countChildren() const = 0;
	virtual bool containChildren() const = 0;

	virtual node_type getParent() = 0;
	virtual const node_type getParent() const = 0;

	virtual bool isRoot() const = 0;
	virtual bool isLeaf() const = 0;
};

//--------------------------------------------------------------------------
// class ComponentSceneNode

template<typename SceneVisitor>
class ComponentSceneNode: public ISceneNode<SceneVisitor>
{
public:
	typedef ISceneNode<SceneVisitor> base_type;
	typedef typename base_type::node_type node_type;

protected:
#if defined(_UNICODE) || defined(UNICODE)
	ComponentSceneNode(const std::wstring &name = std::wstring())
#else
	ComponentSceneNode(const std::string &name = std::string())
#endif
	: base_type(),
	  parent_(), name_(name)
	{}
	ComponentSceneNode(const ComponentSceneNode &rhs)
	: base_type(rhs),
	  parent_(rhs.parent_), name_(rhs.name_)
	{}
public:
	virtual ~ComponentSceneNode()
	{}

	ComponentSceneNode & operator=(const ComponentSceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		parent_ = rhs.parent_;
		name_ = rhs.name_;
		return *this;
	}

public:
	/*final*/ /*virtual*/ node_type getParent()  {  return parent_;  }
	/*final*/ /*virtual*/ const node_type getParent() const  {  return parent_;  }

	/*final*/ /*virtual*/ bool isRoot() const  {  return NULL == parent_.get();  }

#if defined(_UNICODE) || defined(UNICODE)
	void setName(const std::wstring &name)  {  name_ = name;  }
	std::wstring & getName()  {  return name_;  }
	const std::wstring & getName() const  {  return name_;  }
#else
	void setName(const std::string name)  {  name_ = name;  }
	std::string & getName()  {  return name_;  }
	const std::string & getName() const  {  return name_;  }
#endif

private:
	node_type parent_;

#if defined(_UNICODE) || defined(UNICODE)
	std::wstring name_;
#else
	std::string name_;
#endif
};

//--------------------------------------------------------------------------
// class GroupSceneNode

template<typename SceneVisitor>
class GroupSceneNode: public ComponentSceneNode<SceneVisitor>
{
public:
	typedef ComponentSceneNode<SceneVisitor> base_type;
	typedef typename base_type::node_type node_type;
	typedef typename base_type::visitor_type visitor_type;

public:
#if defined(_UNICODE) || defined(UNICODE)
	GroupSceneNode(const std::wstring &name = std::wstring())
#else
	GroupSceneNode(const std::string &name = std::string())
#endif
	: base_type(name),
	  children_()
	{}
	GroupSceneNode(const GroupSceneNode &rhs)
	: base_type(rhs),
	  children_(rhs.children_)
	{}
	virtual ~GroupSceneNode()
	{}

	GroupSceneNode & operator=(const GroupSceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		children_.assign(children_.begin(), children_.end());
		return *this;
	}

public:
	/*virtual*/ void accept(const visitor_type &visitor) const
	{
		traverse(visitor);
		//visitor.visit(*this);
	}

	/*final*/ /*virtual*/ void addChild(const node_type &node)
	{
		children_.push_back(node);
	}
	/*final*/ /*virtual*/ void removeChild(const node_type &node)
	{
		children_.remove(node);
	}
	/*final*/ /*virtual*/ void clearChildren()
	{
		children_.clear();
	}
	/*final*/ /*virtual*/ size_t countChildren() const  {  return children_.size();  }
	/*final*/ /*virtual*/ bool containChildren() const  {  return !children_.empty();  }

	/*final*/ /*virtual*/ bool isLeaf() const  {  return children_.empty();  }

	//
	void traverse(const visitor_type &visitor) const
	{
		for (typename std::list<node_type>::const_iterator it = children_.begin(); it != children_.end(); ++it)
			if (*it) (*it)->accept(visitor);
	}
	void replace(const node_type &oldNode, const node_type &newNode)
	{
		std::replace(children_.begin(), children_.end(), oldNode, newNode);
	}

private:
	std::list<node_type> children_;
};

//--------------------------------------------------------------------------
// class LeafSceneNode

template<typename SceneVisitor>
class LeafSceneNode: public ComponentSceneNode<SceneVisitor>
{
public:
	typedef ComponentSceneNode<SceneVisitor> base_type;
	typedef typename base_type::node_type node_type;

protected:
#if defined(_UNICODE) || defined(UNICODE)
	LeafSceneNode(const std::wstring &name = std::wstring())
#else
	LeafSceneNode(const std::string &name = std::string())
#endif
	: base_type(name)
	{}
	LeafSceneNode(const LeafSceneNode &rhs)
	: base_type(rhs)
	{}
public:
	virtual ~LeafSceneNode()
	{}

	LeafSceneNode & operator=(const LeafSceneNode &rhs)
	{
		if (this == &rhs) return *this;
		static_cast<base_type &>(*this) = rhs;
		return *this;
	}

public:
	/*final*/ /*virtual*/ void addChild(const node_type &node)
	{
		throw LogException(LogException::L_INFO, "can't add a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
	}
	/*final*/ /*virtual*/ void removeChild(const node_type &node)
	{
		throw LogException(LogException::L_INFO, "can't remove a child to a leaf", __FILE__, __LINE__, __FUNCTION__);
	}
	/*final*/ /*virtual*/ void clearChildren()
	{
		throw LogException(LogException::L_INFO, "can't clear all children to a leaf", __FILE__, __LINE__, __FUNCTION__);
	}
	/*final*/ /*virtual*/ size_t countChildren() const  {  return 0;  }
	/*final*/ /*virtual*/ bool containChildren() const  {  return false;  }

	/*final*/ /*virtual*/ bool isLeaf() const  {  return true;  }
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SCENE_NODE__H_
