#include "swl/Config.h"
#include "swl/machine_vision/SIFT.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

SIFT::SIFT()
{
}

SIFT::~SIFT()
{
}

bool SIFT::extractFeature()
{
	// FIXME [add] >>
	throw std::runtime_error("not yet implemented");
}

bool SIFT::matchFeature()
{
	// FIXME [add] >>
	throw std::runtime_error("not yet implemented");
}

}  // namespace swl
