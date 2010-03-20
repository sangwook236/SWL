#include "swl/Config.h"
#include "swl/graphics/PickableInterface.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class PickableInterface

PickableInterface::PickableInterface(const bool isPickable)
: isPickable_(isPickable)
{}

PickableInterface::PickableInterface(const PickableInterface &rhs)
: isPickable_(rhs.isPickable_)
{}

PickableInterface::~PickableInterface()
{}

PickableInterface & PickableInterface::operator=(const PickableInterface &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	isPickable_ = rhs.isPickable_;
	return *this;
}

}  // namespace swl
