#include "swl/Config.h"
#include "swl/graphics/CoordinateFrame.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class CoordinateFrame

#if defined(_UNICODE) || defined(UNICODE)
CoordinateFrame::CoordinateFrame(const std::wstring &name /*= std::wstring()*/)
#else
CoordinateFrame::CoordinateFrame(const std::string &name /*= std::string()*/)
#endif
: base_type(), 
  name_(name), frame_()
{
}

CoordinateFrame::CoordinateFrame(const CoordinateFrame &rhs)
: base_type(rhs),
  name_(rhs.name_), frame_(rhs.frame_)
{
}

CoordinateFrame::~CoordinateFrame()
{
}

CoordinateFrame & CoordinateFrame::operator=(const CoordinateFrame &rhs)
{
	if (this == &rhs) return *this;
	static_cast<CoordinateFrame::base_type &>(*this) = rhs;
	name_ = rhs.name_;
	frame_ = rhs.frame_;
	return *this;
}

}  // namespace swl
