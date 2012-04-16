#if !defined(__SWL_MACHINE_LEARNING__SARSA_LAMBDA__H_)
#define __SWL_MACHINE_LEARNING__SARSA_LAMBDA__H_ 1


#include "swl/machine_learning/TDLambda.h"
#include <map>


namespace swl {

//--------------------------------------------------------------------------
//

template<typename StateActionPair>
class SarsaLambda: public TDLambda<StateActionPair>
{
public:
	typedef TDLambda<StateActionPair> base_type;
	typedef typename base_type::state_type state_type;
	typedef typename base_type::action_type action_type;
	typedef typename base_type::policy_type policy_type;
	typedef typename base_type::reward_type reward_type;
	typedef typename base_type::epsilon_function_type epsilon_function_type;
	typedef typename base_type::step_size_function_type step_size_function_type;
	typedef typename base_type::state_action_pair_type state_action_pair_type;

public:
	explicit SarsaLambda(const double gamma, const double lambda, const bool isReplacingTrace)
	: base_type(gamma, lambda, isReplacingTrace)
	{}
	explicit SarsaLambda(const double gamma, epsilon_function_type epsilonFunc, step_size_function_type stepSizeFunc, const double lambda, const bool isReplacingTrace)
	: base_type(gamma, epsilonFunc, stepSizeFunc, lambda, isReplacingTrace)
	{}
	explicit SarsaLambda(const SarsaLambda &rhs)
	: base_type(rhs)
	{}

private:
	SarsaLambda & operator=(const SarsaLambda &rhs);

public:
	// Sarsa(lambda): a tabular on-policy TD(lambda) control algorithm
	/*virtual*/ void train(const size_t maxEpisodeCount, const policy_type &policy, std::map<const state_action_pair_type, double> &Q, std::map<const state_action_pair_type, double> &eligibility) const
	{
		//std::cout << "processing .";

		size_t episode = 1;
		while (true)
		{
			const size_t step = runSingleEpisoide(episode, policy, Q, eligibility);

			//std::cout << "episode #" << episode << ": step = " << step << std::endl;
			//if (0 == episode % 500)
			//	std::cout << '.';

			++episode;

			// check termination
			if (episode > maxEpisodeCount)
				break;
		}

		//std::cout << std::endl;
	}

private:
	size_t runSingleEpisoide(const size_t episodeTrial, const policy_type &policy, std::map<const state_action_pair_type, double> &Q, std::map<const state_action_pair_type, double> &eligibility) const
	{
		size_t step = 1;

		// initialize a state s
		state_type currState;
		if (!currState.isValidState())
			throw std::runtime_error("invalid state");
		// initialize an action a
		action_type currAction;

		while (true)
		{
			// take action & get a next state s'
			const state_type &nextState(currState.takeAction(currAction));
			if (!nextState.isValidState())
				throw std::runtime_error("invalid state");
			// get a reward r
			// TODO [check] >> the current state or the next state
			const reward_type &r(nextState.getReward());
			// choose a' from s' using policy derived from Q
			const action_type &nextAction(state_action_pair_type::getActionFromPolicy(nextState, Q, policy, (*epsilonFunction_)(episodeTrial)));

			const state_action_pair_type currSA(currState, currAction);
			const state_action_pair_type nextSA(nextState, nextAction);
			const double delta = r + gamma_ * Q[nextSA] - Q[currSA];
			eligibility[currSA] += 1.0;

			updateQAndEligibility(step, delta, Q, eligibility, NULL);

			// update the current state & action
			currState = nextState;
			currAction = nextAction;

			++step;

			// check termination
			// TODO [check] >> step limitation
			if (currState.isTerminalState())
			//if (currState.isTerminalState() || step > 1000)
				break;
		}

		return step;
	}
};

}  // namespace swl


#endif  // __SWL_MACHINE_LEARNING__SARSA_LAMBDA__H_
