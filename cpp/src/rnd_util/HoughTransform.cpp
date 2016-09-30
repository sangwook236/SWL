#include "swl/Config.h"
#include "swl/rnd_util/HoughTransform.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace swl {

GeneralizedHoughTransform::GeneralizedHoughTransform(const std::size_t tangentAngleCount)
: tangentAngleCount_(tangentAngleCount)
{
}

GeneralizedHoughTransform::~GeneralizedHoughTransform()
{
}

bool GeneralizedHoughTransform::run()
{
	// FIXME [add] >>
	//	REF [file] >> rnd_util_test/HoughTransformTest.cpp
	throw std::runtime_error("Not yet implemented");
}

}  // namespace swl
