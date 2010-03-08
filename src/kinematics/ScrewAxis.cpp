#include "swl/Config.h"
#include "swl/kinematics/ScrewAxis.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class ScrewAxis

ScrewAxis & ScrewAxis::operator=(const ScrewAxis &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	dir_ = rhs.dir_;
	pos_ = rhs.pos_;
	pitch_ = rhs.pitch_;
	return *this;
}

}  // namespace swl
