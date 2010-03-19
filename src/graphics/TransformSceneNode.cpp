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

template<typename SceneVisitor>
TransformSceneNode<SceneVisitor>::TransformSceneNode()
: base_type(),
  transform_()
{
}

template<typename SceneVisitor>
TransformSceneNode<SceneVisitor>::TransformSceneNode(const TransformSceneNode &rhs)
: base_type(rhs),
  transform_(rhs.transform_)
{
}

template<typename SceneVisitor>
TransformSceneNode<SceneVisitor>::~TransformSceneNode()
{
}

template<typename SceneVisitor>
TransformSceneNode<SceneVisitor> & TransformSceneNode<SceneVisitor>::operator=(const TransformSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	transform_ = rhs.transform_;
	return *this;
}

template<typename SceneVisitor>
void TransformSceneNode<SceneVisitor>::accept(const visitor_type &visitor) const
{
	visitor.visit(*this);
}

}  // namespace swl
