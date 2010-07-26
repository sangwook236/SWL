#include "swl/Config.h"
#include "swl/machine_vision/SURF.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace swl {

SURF::SURF()
{
}

SURF::~SURF()
{
}

bool SURF::extractFeature()
{
	// FIXME [add] >>
	throw std::runtime_error("not yet implemented");
}

bool SURF::matchFeature()
{
	// FIXME [add] >>
	throw std::runtime_error("not yet implemented");
}

}  // namespace swl
