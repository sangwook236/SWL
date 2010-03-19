#include "swl/Config.h"
#include "swl/graphics/Shape.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Shape

Shape::Shape(const bool isTransparent /*= false*/, const bool isPrintable /*= true*/, const bool isPickable /*= true*/, const attrib::PolygonMode polygonMode /*= attrib::POLYGON_FILL*/, const attrib::PolygonFace drawingFace /*= attrib::POLYGON_FACE_FRONT*/)
: base_type(isPrintable, isPickable),
  geometryId_(GeometryPool::UNDEFINED_GEOMETRY_ID), appearance_(true, isTransparent, polygonMode, drawingFace)
{
}

Shape::Shape(const Shape &rhs)
: base_type(rhs),
  geometryId_(rhs.geometryId_), appearance_(rhs.appearance_)
{
}

Shape::~Shape()
{
}

Shape & Shape::operator=(const Shape &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	geometryId_ = rhs.geometryId_;
	appearance_ = rhs.appearance_;
	return *this;
}

Shape::geometry_type Shape::getGeometry() const
{
	return GeometryPool::getInstance().getGeometry(geometryId_);
}

}  // namespace swl
