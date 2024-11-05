#if !defined(__SWL_MACHINE_LEARNING__TD_LEARNING_BASE__H_)
#define __SWL_MACHINE_LEARNING__TD_LEARNING_BASE__H_ 1


#include "swl/machine_learning/ExportMachineLearning.h"
#include <map>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_MACHINE_LEARNING_API TDLearningBase
{
public:
	//typedef TDLearningBase base_type;

	typedef double (*epsilon_function_type)(const std::size_t episodeTrial);
	typedef double (*step_size_function_type)(const std::size_t iterationStep);

protected:
	explicit TDLearningBase(const double gamma)
	: gamma_(gamma), epsilonFunction_(&defaultEpsilonFunction), stepSizeFunction_(&defaultStepSizeFunction)
	{}
	explicit TDLearningBase(const double gamma, epsilon_function_type epsilonFunc, step_size_function_type stepSizeFunc)
	: gamma_(gamma), epsilonFunction_(epsilonFunc), stepSizeFunction_(stepSizeFunc)
	{}
	explicit TDLearningBase(const TDLearningBase &rhs)
	: gamma_(rhs.gamma_), epsilonFunction_(rhs.epsilonFunction_), stepSizeFunction_(rhs.stepSizeFunction_)
	{}
public:
	virtual ~TDLearningBase()  {}

private:
	TDLearningBase & operator=(const TDLearningBase &rhs);

private:
	static double defaultEpsilonFunction(const std::size_t episodeTrial);
	static double defaultStepSizeFunction(const std::size_t iterationStep);

protected:
	const double gamma_;

	epsilon_function_type epsilonFunction_;
	step_size_function_type stepSizeFunction_;
};

}  // namespace swl


#endif  // __SWL_MACHINE_LEARNING__TD_LEARNING_BASE__H_
