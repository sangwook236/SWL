#if !defined(__SWL_GRAPHICS__SCENE_NODE__H_)
#define __SWL_GRAPHICS__SCENE_NODE__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include <boost/smart_ptr.hpp>
#include <list>


namespace swl {

//--------------------------------------------------------------------------
// struct ISceneNode

struct ISceneNode
{
public:
	//typedef ISceneNode					base_type;
	typedef boost::shared_ptr<ISceneNode>	node_type;

public:
	virtual void draw() = 0;
 
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
// class SceneComponentNode

class SceneComponentNode: public ISceneNode
{
public:
	//typedef ISceneNode base_type;

protected:
	SceneComponentNode();
	SceneComponentNode(const SceneComponentNode &rhs);
public:
	virtual ~SceneComponentNode();

	SceneComponentNode & operator=(const SceneComponentNode &rhs);

public:
	/*final*/ /*virtual*/ node_type getParent()  {  return parent_;  }
	/*final*/ /*virtual*/ const node_type getParent() const  {  return parent_;  }

	/*final*/ /*virtual*/ bool isRoot() const  {  return NULL == parent_.get();  }

private:
	node_type parent_;
};

//--------------------------------------------------------------------------
// class SceneCompositeNode

class SWL_GRAPHICS_API SceneCompositeNode: public SceneComponentNode
{
public:
	typedef SceneComponentNode base_type;

protected:
	SceneCompositeNode();
	SceneCompositeNode(const SceneCompositeNode &rhs);
public:
	virtual ~SceneCompositeNode();

	SceneCompositeNode & operator=(const SceneCompositeNode &rhs);

public:
	/*virtual*/ void draw();

	/*final*/ /*virtual*/ void addChild(const node_type &node);
 	/*final*/ /*virtual*/ void removeChild(const node_type &node);
	/*final*/ /*virtual*/ void clearChildren();
	/*final*/ /*virtual*/ size_t countChildren() const  {  return children_.size();  }
	/*final*/ /*virtual*/ bool containChildren() const  {  return !children_.empty();  }

	/*virtual*/ bool isLeaf() const  {  return children_.empty();  }

	void replace(const node_type &oldNode, const node_type &newNode);

private:
	std::list<node_type> children_;
};

//--------------------------------------------------------------------------
// class SceneLeafNode

class SWL_GRAPHICS_API SceneLeafNode: public SceneComponentNode
{
public:
	typedef SceneComponentNode base_type;

protected:
	SceneLeafNode();
	SceneLeafNode(const SceneLeafNode &rhs);
public:
	virtual ~SceneLeafNode();

	SceneLeafNode & operator=(const SceneLeafNode &rhs);
 
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
