#include "swl/Config.h"
#include "swl/graphics/GeometrySceneNode.h"
#include "swl/graphics/ISceneVisitor.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GeometrySceneNode

GeometrySceneNode::GeometrySceneNode(const geometry_id_type &geometryId)
: base_type(),
  geometryId_(geometryId) //geometryId_(GeometryPool::UNDEFINED_GEOMETRY_ID)
{
}

GeometrySceneNode::GeometrySceneNode(const GeometrySceneNode &rhs)
: base_type(rhs),
  geometryId_(rhs.geometryId_)
{
}

GeometrySceneNode::~GeometrySceneNode()
{
}

GeometrySceneNode & GeometrySceneNode::operator=(const GeometrySceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	geometryId_ = rhs.geometryId_;
	return *this;
}

void GeometrySceneNode::accept(const ISceneVisitor &visitor) const
{
	visitor.visit(*this);
}

GeometrySceneNode::geometry_type GeometrySceneNode::getGeometry() const
{
	return GeometryPool::getInstance().getGeometry(geometryId_);
}

}  // namespace swl
