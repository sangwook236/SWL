#include "swl/Config.h"
#include "swl/graphics/Shape.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Shape

Shape::Shape()
: base_type(),
  geometryId_(GeometryPool::UNDEFINED_GEOMETRY_ID), appearance_()
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
