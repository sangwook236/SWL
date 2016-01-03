#if !defined(__SWL_MACHINE_LEARNING__Q_LEARNING__H_)
#define __SWL_MACHINE_LEARNING__Q_LEARNING__H_ 1


#include "swl/machine_learning/QLearning.h"
#include <map>


namespace swl {

//--------------------------------------------------------------------------
//

template<typename StateActionPair>
class QLearning: public TDLearning<StateActionPair>
{
public:
	typedef TDLearning<StateActionPair> base_type;
	typedef typename base_type::state_type state_type;
	typedef typename base_type::action_type action_type;
	typedef typename base_type::policy_type policy_type;
	typedef typename base_type::reward_type reward_type;
	typedef typename base_type::epsilon_function_type epsilon_function_type;
	typedef typename base_type::step_size_function_type step_size_function_type;
	typedef typename base_type::state_action_pair_type state_action_pair_type;

public:
	explicit QLearning(const double gamma)
	: base_type(gamma)
	{}
	explicit QLearning(const double gamma, epsilon_function_type epsilonFunc, step_size_function_type stepSizeFunc)
	: base_type(gamma, epsilonFunc, stepSizeFunc)
	{}
	explicit QLearning(const QLearning &rhs)
	: base_type(rhs)
	{}

private:
	QLearning & operator=(const QLearning &rhs);

public:
	// Q-learning: a tabular off-policy TD control algorithm
	/*virtual*/ void train(const size_t maxEpisodeCount, const policy_type &policy, std::map<const state_action_pair_type, double> &Q) const
	{
		//std::cout << "processing .";

		size_t episode = 1;
		while (true)
		{
			//const size_t step = runSingleEpisoide(episode, policy, Q);

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

		while (true)
		{
			// choose a from s using policy derived from Q
			const action_type &action(state_action_pair_type::getActionFromPolicy(currState, Q, policy, (*base_type::epsilonFunction_)(episodeTrial)));

			// take action & get a next state s'
			const state_type &nextState(currState.takeAction(action));
			if (!nextState.isValidState())
				throw std::runtime_error("invalid state");
			// get a reward r
			// TODO [check] >> the current state or the next state
			const reward_type &r(nextState.getReward());

			const action_type &nextGreedyAction(state_action_pair_type::getGreedyAction(nextState, Q));

			const state_action_pair_type currSA(currState, action);
			const state_action_pair_type nextSA(nextState, nextGreedyAction);
			Q[currSA] += (*base_type::stepSizeFunction_)(step) * (r + base_type::gamma_ * Q[nextSA] - Q[currSA]);

			// update the current state
			currState = nextState;

			++step;

			// check termination
			if (currState.isTerminalState())
				break;
		}

		return step;
	}
};

}  // namespace swl


#endif  // __SWL_MACHINE_LEARNING__Q_LEARNING__H_
