#include "swl/Config.h"
#include "swl/graphics/TransformSceneNode.h"
#include "swl/graphics/ISceneVisitor.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class TransformSceneNode

TransformSceneNode::TransformSceneNode()
: base_type(),
  transform_()
{
}

TransformSceneNode::TransformSceneNode(const TransformSceneNode &rhs)
: base_type(rhs),
  transform_(rhs.transform_)
{
}

TransformSceneNode::~TransformSceneNode()
{
}

TransformSceneNode & TransformSceneNode::operator=(const TransformSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	transform_ = rhs.transform_;
	return *this;
}

void TransformSceneNode::accept(const ISceneVisitor &visitor) const
{
	//traverse(visitor);
	visitor.visit(*this);
}

}  // namespace swl
