#if !defined(__SWL_MACHINE_LEARNING__TD_LAMBDA__H_)
#define __SWL_MACHINE_LEARNING__TD_LAMBDA__H_ 1


#include "swl/machine_learning/TDLearningBase.h"
#include <map>


namespace swl {

//--------------------------------------------------------------------------
//

template<typename StateActionPair>
class TDLambda: public TDLearningBase
{
public:
	typedef TDLearningBase base_type;
	typedef StateActionPair state_action_pair_type;
	typedef typename state_action_pair_type::state_type state_type;
	typedef typename state_action_pair_type::reward_type reward_type;
	typedef typename state_action_pair_type::action_type action_type;
	typedef typename state_action_pair_type::policy_type policy_type;

protected:
	explicit TDLambda(const double gamma, const double lambda, const bool isReplacingTrace)
	: base_type(gamma), lambda_(lambda), isReplacingTrace_(isReplacingTrace)
	{}
	explicit TDLambda(const double gamma, epsilon_function_type epsilonFunc, step_size_function_type stepSizeFunc, const double lambda, const bool isReplacingTrace)
	: base_type(gamma, epsilonFunc, stepSizeFunc), lambda_(lambda), isReplacingTrace_(isReplacingTrace)
	{}
	explicit TDLambda(const TDLambda &rhs)
	: base_type(rhs), lambda_(rhs.lambda_), isReplacingTrace_(rhs.isReplacingTrace_)
	{}
public:
	virtual ~TDLambda()  {}

private:
	TDLambda & operator=(const TDLambda &rhs);

public:
	virtual void train(const size_t maxEpisodeCount, const policy_type &policy, std::map<const state_action_pair_type, double> &Q, std::map<const state_action_pair_type, double> &eligibility) const = 0;

protected:
	void updateQAndEligibility(const size_t iterationStep, const double delta, std::map<const state_action_pair_type, double> &Q, std::map<const state_action_pair_type, double> &eligibility, const bool *isGreedyAction) const
	{
		if (isReplacingTrace_)  // use an replacing trace
		{
			// FIXME [add] >>
			throw std::logic_error("not yet implemented");
		}
		else  // use an accumulating trace
		{
			for (typename std::map<const state_action_pair_type, double>::iterator it = Q.begin(); it != Q.end(); ++it)
			{
				it->second += (*stepSizeFunction_)(iterationStep) * delta * eligibility[it->first];
				if (isGreedyAction)
				{
					if (*isGreedyAction) eligibility[it->first] *= gamma_ * lambda_;
					else eligibility[it->first] = 0.0;
				}
				else eligibility[it->first] *= gamma_ * lambda_;
			}
		}
	}

private:
	const double lambda_;
	const bool isReplacingTrace_;
};

}  // namespace swl


#endif  // __SWL_MACHINE_LEARNING__TD_LAMBDA__H_
