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

template<typename SceneVisitor>
#if defined(UNICODE) || defined(_UNICODE)
GeometrySceneNode<SceneVisitor>::GeometrySceneNode(const geometry_id_type &geometryId, const std::wstring &name /*= std::wstring()*/)
#else
GeometrySceneNode<SceneVisitor>::GeometrySceneNode(const geometry_id_type &geometryId, const std::string &name /*= std::string()*/);
#endif
: base_type(name),
  geometryId_(geometryId) //geometryId_(GeometryPool::UNDEFINED_GEOMETRY_ID)
{
}

template<typename SceneVisitor>
GeometrySceneNode<SceneVisitor>::GeometrySceneNode(const GeometrySceneNode &rhs)
: base_type(rhs),
  geometryId_(rhs.geometryId_)
{
}

template<typename SceneVisitor>
GeometrySceneNode<SceneVisitor>::~GeometrySceneNode()
{
}

template<typename SceneVisitor>
GeometrySceneNode<SceneVisitor> & GeometrySceneNode<SceneVisitor>::operator=(const GeometrySceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	geometryId_ = rhs.geometryId_;
	return *this;
}

template<typename SceneVisitor>
void GeometrySceneNode<SceneVisitor>::accept(const visitor_type &visitor) const
{
	visitor.visit(*this);
}

template<typename SceneVisitor>
typename GeometrySceneNode<SceneVisitor>::geometry_type GeometrySceneNode<SceneVisitor>::getGeometry() const
{
	return GeometryPool::getInstance().getGeometry(geometryId_);
}

}  // namespace swl
