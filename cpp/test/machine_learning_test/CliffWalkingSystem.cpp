//#include "stdafx.h"
#include "CliffWalkingSystem.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>


//-----------------------------------------------------------------------------
//

/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::Up(CliffWalkingStateActionPair::action_type::UP);
/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::Down(CliffWalkingStateActionPair::action_type::DOWN);
/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::Right(CliffWalkingStateActionPair::action_type::RIGHT);
/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::Left(CliffWalkingStateActionPair::action_type::LEFT);

bool CliffWalkingStateActionPair::State::isTerminalState() const
{
	return 4 == row_ && 12 == col_;
}

bool CliffWalkingStateActionPair::State::isValidState() const
{
	return 1 <= row_ && row_ <= GRID_ROW_SIZE && 1 <= col_ && col_ <= GRID_COL_SIZE;
}

CliffWalkingStateActionPair::State::reward_type CliffWalkingStateActionPair::State::getReward() const
{
	if (4 == row_ && (2 <= col_ && col_ <= 11))  // the cliff region
		return -100;
	else if (4 == row_ && 12 == col_)  // the goal state
		return 0;
	else
		return -1;
}

CliffWalkingStateActionPair::State CliffWalkingStateActionPair::State::takeAction(const action_type &action) const
{
	if (isTerminalState()) return *this;

	if ((4 == row_ && 1 == col_ && Right == action) || (3 == row_ && 2 <= col_ && col_ <= 11 && Down == action))  // step into the cliff region
		return State();
	if ((1 == row_ && Up == action) || (4 == row_ && Down == action) ||
		(1 == col_ && Left == action) || (12 == col_ && Right == action))  // stay the current state
		return *this;

	state_type newState(*this);
	switch (action.getValue())
	{
	case action_type::UP:
		--newState.row_;
		break;
	case action_type::DOWN:
		++newState.row_;
		break;
	case action_type::RIGHT:
		++newState.col_;
		break;
	case action_type::LEFT:
		--newState.col_;
		break;
	default:
		throw std::runtime_error("invalid action");
	}

	return newState;
}

/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::getGreedyAction(const state_type &state, const std::map<const CliffWalkingStateActionPair, double> &Q)
{
	std::map<const CliffWalkingStateActionPair, double>::const_iterator cit1 = Q.find(CliffWalkingStateActionPair(state, Up));
	std::map<const CliffWalkingStateActionPair, double>::const_iterator cit2 = Q.find(CliffWalkingStateActionPair(state, Down));
	std::map<const CliffWalkingStateActionPair, double>::const_iterator cit3 = Q.find(CliffWalkingStateActionPair(state, Right));
	std::map<const CliffWalkingStateActionPair, double>::const_iterator cit4 = Q.find(CliffWalkingStateActionPair(state, Left));
	if (Q.end() != cit1 && Q.end() != cit2 && Q.end() != cit3 && Q.end() != cit4)
	{
		std::vector<double> q;
		q.reserve(action_type::ACTION_SIZE);
		q.push_back(cit1->second);
		q.push_back(cit2->second);
		q.push_back(cit3->second);
		q.push_back(cit4->second);

		std::vector<double>::iterator it = std::max_element(q.begin(), q.end());
		const size_t diff = std::distance(q.begin(), it);
		switch (diff)
		{
		case 0:
			return cit1->first.action_;
		case 1:
			return cit2->first.action_;
		case 2:
			return cit3->first.action_;
		case 3:
			return cit4->first.action_;
		default:
			throw std::runtime_error("index error");
		}
	}
	else
		throw std::runtime_error("undefined Q-value");
}

/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::getRandomAction()
{
	const double prob = double(std::rand() % RAND_MAX) / double(RAND_MAX);
	return prob <= 0.25 ? Up : (prob <= 0.50 ? Down : (prob <= 0.75 ? Right : Left));
}

/*static*/ CliffWalkingStateActionPair::action_type CliffWalkingStateActionPair::getActionFromPolicy(const state_type &state, const std::map<const CliffWalkingStateActionPair, double> &Q, const policy_type &policy, const double epsilon)
{
	const double prob = double(std::rand() % RAND_MAX) / double(RAND_MAX);

	switch (policy)
	{
	case GREEDY_POLICY:  // greedy policy
		return getGreedyAction(state, Q);
	case EPSILON_GREEDY_POLICY:  // epsilon-greedy policy
		return prob < (1.0 - epsilon) ? getGreedyAction(state, Q) : getRandomAction();
	case EPSILON_SOFT_POLICY:  // epsilon-soft policy
		return prob < (1.0 - epsilon + epsilon / (double)action_type::ACTION_SIZE) ? getGreedyAction(state, Q) : getRandomAction();
	default:
		throw std::runtime_error("undefined policy");
	}
}

/*static*/ double CliffWalkingStateActionPair::epsilonFunction(const size_t episodeTrial)
{
	// constant exploration rate (epsilon)
	return 0.1;

	// variable exploration rate (epsilon)
	//return 0 >= episodeTrial ? 1.0 : (1.0 / std::sqrt((double)episodeTrial));
	//return 0 >= episodeTrial ? 1.0 : (1.0 / episodeTrial);
	//return 0 >= episodeTrial ? 1.0 : (1.0 / (episodeTrial * episodeTrial));
	//return 0 >= episodeTrial ? 1.0 : (1.0 / std::exp((double)episodeTrial));
}

/*static*/ double CliffWalkingStateActionPair::stepSizeFunction(const size_t iterationStep)
{
	// constant learning rate (step-size, alpha)
	//return 0.2;

	// variable learning rate (step-size, alpha)
	return 0 >= iterationStep ? 1.0 : (1.0 / iterationStep);
}
