#include "swl/Config.h"
#include "swl/kinematics/Link.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Link

Link::Link()
//: base_type()
{
}

Link::Link(const Link &rhs)
//: base_type(rhs)
{
}

Link::~Link()
{
}

Link & Link::operator=(const Link &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	return *this;
}

}  // namespace swl
