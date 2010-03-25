#if !defined(__SWL_MACHINE_LEARNING__SARSA__H_)
#define __SWL_MACHINE_LEARNING__SARSA__H_ 1


#include "swl/machine_learning/TDLearning.h"
#include <map>


namespace swl {

//--------------------------------------------------------------------------
//

template<typename StateActionPair>
class Sarsa: public TDLearning<StateActionPair>
{
public:
	typedef TDLearning<StateActionPair> base_type;

public:
	explicit Sarsa(const double gamma)
	: base_type(gamma)
	{}
	explicit Sarsa(const double gamma, epsilon_function_type epsilonFunc, step_size_function_type stepSizeFunc)
	: base_type(gamma, epsilonFunc, stepSizeFunc)
	{}
	explicit Sarsa(const Sarsa &rhs)
	: base_type(rhs)
	{}

private:
	Sarsa & operator=(const Sarsa &rhs);

public:
	// Sarsa: a tabular on-policy TD control algorithm
	/*virtual*/ void train(const size_t maxEpisodeCount, const policy_type &policy, std::map<const state_action_pair_type, double> &Q) const
	{
		//std::cout << "processing .";

		size_t episode = 1;
		while (true)
		{
			const size_t step = runSingleEpisoide(episode, policy, Q);

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
	size_t runSingleEpisoide(const size_t episodeTrial, const policy_type &policy, std::map<const state_action_pair_type, double> &Q) const
	{
		size_t step = 1;

		// initialize a state s
		state_type currState;
		if (!currState.isValidState())
			throw std::runtime_error("invalid state");
		// choose a from s using policy derived from Q
		action_type &currAction(state_action_pair_type::getActionFromPolicy(currState, Q, policy, (*epsilonFunction_)(episodeTrial)));

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
			Q[currSA] += (*stepSizeFunction_)(step) * (r + gamma_ * Q[nextSA] - Q[currSA]);

			// update the current state & action
			currState = nextState;
			currAction = nextAction;

			++step;

			// check termination
			if (currState.isTerminalState())
				break;
		}

		return step;
	}
};

}  // namespace swl


#endif  // __SWL_MACHINE_LEARNING__SARSA__H_
