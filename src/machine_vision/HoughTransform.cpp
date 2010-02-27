#include "swl/Config.h"
#include "swl/machine_vision/HoughTransform.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace swl {

GeneralizedHoughTransform::GeneralizedHoughTransform(const size_t tangentAngleCount)
: tangentAngleCount_(tangentAngleCount)
{
}

GeneralizedHoughTransform::~GeneralizedHoughTransform()
{
}

bool GeneralizedHoughTransform::run()
{
	// FIXME [add] >>
	return false;
}

}  // namespace swl
