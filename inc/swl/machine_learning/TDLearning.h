#if !defined(__SWL_MACHINE_LEARNING__TD_LEARNING__H_)
#define __SWL_MACHINE_LEARNING__TD_LEARNING__H_ 1


#include "swl/machine_learning/TDLearningBase.h"
#include <map>


namespace swl {

//--------------------------------------------------------------------------
//

template<typename StateActionPair>
class TDLearning: public TDLearningBase
{
public:
	typedef TDLearningBase base_type;
	typedef StateActionPair state_action_pair_type;
	typedef typename state_action_pair_type::state_type state_type;
	typedef typename state_action_pair_type::reward_type reward_type;
	typedef typename state_action_pair_type::action_type action_type;
	typedef typename state_action_pair_type::policy_type policy_type;

protected:
	explicit TDLearning(const double gamma)
	: base_type(gamma)
	{}
	explicit TDLearning(const double gamma, epsilon_function_type epsilonFunc, step_size_function_type stepSizeFunc)
	: base_type(gamma, epsilonFunc, stepSizeFunc)
	{}
	explicit TDLearning(const TDLearning &rhs)
	: base_type(rhs)
	{}
public:
	virtual ~TDLearning()  {}

private:
	TDLearning & operator=(const TDLearning &rhs);

public:
	virtual void train(const size_t maxEpisodeCount, const policy_type &policy, std::map<const state_action_pair_type, double> &Q) const = 0;
};

}  // namespace swl


#endif  // __SWL_MACHINE_LEARNING__TD_LEARNING__H_
