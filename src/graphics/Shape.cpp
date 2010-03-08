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
  geometry_(), appearance_()
{
}

Shape::Shape(const Shape &rhs)
: base_type(rhs),
  geometry_(rhs.geometry_), appearance_(rhs.appearance_)
{
}

Shape::~Shape()
{
}

Shape & Shape::operator=(const Shape &rhs)
{
	if (this == &rhs) return *this;
	static_cast<Shape::base_type &>(*this) = rhs;
	geometry_ = rhs.geometry_;
	appearance_ = rhs.appearance_;
	return *this;
}

}  // namespace swl
