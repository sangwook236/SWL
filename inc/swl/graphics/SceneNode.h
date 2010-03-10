#if !defined(__SWL_GRAPHICS__SCENE_NODE__H_)
#define __SWL_GRAPHICS__SCENE_NODE__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include <boost/smart_ptr.hpp>
#include <list>


namespace swl {

struct ISceneVisitor;

//--------------------------------------------------------------------------
// struct ISceneNode

struct ISceneNode
{
public:
	//typedef ISceneNode					base_type;
	typedef boost::shared_ptr<ISceneNode>	node_type;

public:
	virtual ~ISceneNode();

public:
	virtual void accept(const ISceneVisitor &visitor) const = 0;
 
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

class SWL_GRAPHICS_API ComponentSceneNode: public ISceneNode
{
public:
	typedef ISceneNode base_type;

protected:
	ComponentSceneNode();
	ComponentSceneNode(const ComponentSceneNode &rhs);
public:
	virtual ~ComponentSceneNode();

	ComponentSceneNode & operator=(const ComponentSceneNode &rhs);

public:
	/*final*/ /*virtual*/ node_type getParent()  {  return parent_;  }
	/*final*/ /*virtual*/ const node_type getParent() const  {  return parent_;  }

	/*final*/ /*virtual*/ bool isRoot() const  {  return NULL == parent_.get();  }

private:
	node_type parent_;
};

//--------------------------------------------------------------------------
// class GroupSceneNode

class SWL_GRAPHICS_API GroupSceneNode: public ComponentSceneNode
{
public:
	typedef ComponentSceneNode base_type;

public:
	GroupSceneNode();
	GroupSceneNode(const GroupSceneNode &rhs);
	virtual ~GroupSceneNode();

	GroupSceneNode & operator=(const GroupSceneNode &rhs);

public:
	/*virtual*/ void accept(const ISceneVisitor &visitor) const;

	/*final*/ /*virtual*/ void addChild(const node_type &node);
 	/*final*/ /*virtual*/ void removeChild(const node_type &node);
	/*final*/ /*virtual*/ void clearChildren();
	/*final*/ /*virtual*/ size_t countChildren() const  {  return children_.size();  }
	/*final*/ /*virtual*/ bool containChildren() const  {  return !children_.empty();  }

	/*final*/ /*virtual*/ bool isLeaf() const  {  return children_.empty();  }

	void traverse(const ISceneVisitor &visitor) const;
	void replace(const node_type &oldNode, const node_type &newNode);

private:
	std::list<node_type> children_;
};

//--------------------------------------------------------------------------
// class LeafSceneNode

class SWL_GRAPHICS_API LeafSceneNode: public ComponentSceneNode
{
public:
	typedef ComponentSceneNode base_type;

protected:
	LeafSceneNode();
	LeafSceneNode(const LeafSceneNode &rhs);
public:
	virtual ~LeafSceneNode();

	LeafSceneNode & operator=(const LeafSceneNode &rhs);
 
public:
	/*final*/ /*virtual*/ void addChild(const node_type &node);
	/*final*/ /*virtual*/ void removeChild(const node_type &node);
	/*final*/ /*virtual*/ void clearChildren();
	/*final*/ /*virtual*/ size_t countChildren() const  {  return 0;  }
	/*final*/ /*virtual*/ bool containChildren() const  {  return false;  }

	/*final*/ /*virtual*/ bool isLeaf() const  {  return true;  }
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SCENE_NODE__H_
