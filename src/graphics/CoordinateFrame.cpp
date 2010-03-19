#include "swl/Config.h"
#include "swl/graphics/CoordinateFrame.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class CoordinateFrame

CoordinateFrame::CoordinateFrame(const bool isPrintable, const bool isPickable)
: base_type(isPrintable, isPickable), 
  frame_()
{
}

CoordinateFrame::CoordinateFrame(const CoordinateFrame &rhs)
: base_type(rhs),
  frame_(rhs.frame_)
{
}

CoordinateFrame::~CoordinateFrame()
{
}

CoordinateFrame & CoordinateFrame::operator=(const CoordinateFrame &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	frame_ = rhs.frame_;
	return *this;
}

}  // namespace swl
