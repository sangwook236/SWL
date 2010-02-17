#include "swl/Config.h"
#include "swl/graphics/Appearance.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Appearance

Appearance::Appearance()
//: base_type()
{
}

Appearance::Appearance(const Appearance &rhs)
//: base_type(rhs)
{
}

Appearance::~Appearance()
{
}

Appearance& Appearance::operator=(const Appearance &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<Appearance::base_type &>(*this) = rhs;
	color_ = rhs.color_;
	transparent_ = rhs.transparent_;
	return *this;
}

}  // namespace swl
