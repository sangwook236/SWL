#include "swl/Config.h"
#include "swl/graphics/ShapeSceneNode.h"
#include "swl/graphics/ISceneVisitor.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class ShapeSceneNode

ShapeSceneNode::ShapeSceneNode(const ShapeSceneNode::shape_type &shape)
: base_type(),
  shape_(shape)
{
}

ShapeSceneNode::ShapeSceneNode(const ShapeSceneNode &rhs)
: base_type(rhs),
  shape_(rhs.shape_)
{
}

ShapeSceneNode::~ShapeSceneNode()
{
}

ShapeSceneNode & ShapeSceneNode::operator=(const ShapeSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	shape_ = rhs.shape_;
	return *this;
}

void ShapeSceneNode::accept(const ISceneVisitor &visitor) const
{
	visitor.visit(*this);
}

}  // namespace swl
