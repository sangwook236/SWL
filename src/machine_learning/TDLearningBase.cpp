#include "swl/Config.h"
#include "swl/machine_learning/TDLearningBase.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace swl {

//-----------------------------------------------------------------------------
//

/*static*/ double TDLearningBase::defaultEpsilonFunction(const std::size_t episodeTrial)
{
	// variable exploration rate (epsilon)
	return 0 >= episodeTrial ? 1.0 : (1.0 / episodeTrial);
}

/*static*/ double TDLearningBase::defaultStepSizeFunction(const std::size_t iterationStep)
{
	// variable learning rate (step-size. alpha)
	return 0 >= iterationStep ? 1.0 : (1.0 / iterationStep);
}

}  // namespace swl
