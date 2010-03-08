#include "swl/Config.h"
#include "swl/graphics/Geometry.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Geometry

Geometry::Geometry()
//: base_type()
{
}

Geometry::Geometry(const Geometry &rhs)
//: base_type(rhs)
{
}

Geometry::~Geometry()
{
}

Geometry & Geometry::operator=(const Geometry &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	return *this;
}

}  // namespace swl
