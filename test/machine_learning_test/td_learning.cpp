#include "stdafx.h"
#include "CliffWalkingSystem.h"
#include "swl/machine_learning/Sarsa.h"
#include "swl/machine_learning/QLearning.h"
#include "swl/machine_learning/SarsaLambda.h"
#include "swl/machine_learning/QLambda.h"
#include "swl/winutil/WinTimer.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>


namespace {

typedef CliffWalkingStateActionPair state_action_pair_type;


char getActionSymbol(const state_action_pair_type::action_type &action)
{
	switch (action.getValue())
	{
	case state_action_pair_type::action_type::UP:
		return 'U';
	case state_action_pair_type::action_type::DOWN:
		return 'D';
	case state_action_pair_type::action_type::RIGHT:
		return 'R';
	case state_action_pair_type::action_type::LEFT:
		return 'L';
	default:
		return ' ';
	}
}

// Sarsa
void sarsa(const size_t maxEpisodeCount, const state_action_pair_type::policy_type policy, const double gamma)
{
	std::cout << "<<-- Sarsa -->>" << std::endl;

	std::map<const state_action_pair_type, double> Q;

	// initialize Q(state, action)
	for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
		{
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Up)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Down)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Right)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Left)] = 0.0;
		}

	if (Q.size() != state_action_pair_type::state_type::STATE_SIZE * state_action_pair_type::action_type::ACTION_SIZE)
	{
		std::cerr << "state-action pair generation error" << std::endl;
		return;
	}

	try
	{
		typedef swl::Sarsa<state_action_pair_type> learner_type;

		swl::WinTimer aTimer;
		const learner_type learner(gamma, &state_action_pair_type::epsilonFunction, &state_action_pair_type::stepSizeFunction);
		learner.train(maxEpisodeCount, policy, Q);
		std::cout << "elapsed time: " << aTimer.getElapsedTimeInMilliSecond() << " msec" << std::endl;

		for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		{
			for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
				std::cout << getActionSymbol(state_action_pair_type::getGreedyAction(state_action_pair_type::state_type(row, col), Q)) << "  ";
			std::cout << std::endl;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

// Q-learning
void q_learning(const size_t maxEpisodeCount, const state_action_pair_type::policy_type policy, const double gamma)
{
	std::cout << "<<-- Q-learning -->>" << std::endl;

	std::map<const state_action_pair_type, double> Q;

	// initialize Q(state, action)
	for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
		{
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Up)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Down)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Right)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Left)] = 0.0;
		}

	if (Q.size() != state_action_pair_type::state_type::STATE_SIZE * state_action_pair_type::action_type::ACTION_SIZE)
	{
		std::cerr << "state-action pair generation error" << std::endl;
		return;
	}

	try
	{
		typedef swl::QLearning<state_action_pair_type> learner_type;

		swl::WinTimer aTimer;
		const learner_type learner(gamma, &state_action_pair_type::epsilonFunction, &state_action_pair_type::stepSizeFunction);
		learner.train(maxEpisodeCount, policy, Q);
		std::cout << "elapsed time: " << aTimer.getElapsedTimeInMilliSecond() << " msec" << std::endl;

		for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		{
			for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
				std::cout << getActionSymbol(state_action_pair_type::getGreedyAction(state_action_pair_type::state_type(row, col), Q)) << "  ";
			std::cout << std::endl;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

// Sarsa(lambda)
void sarsa_lambda(const size_t maxEpisodeCount, const state_action_pair_type::policy_type policy, const double gamma, const double lambda, const bool isReplacingTrace)
{
	std::cout << "<<-- Sarsa(lambda) -->>" << std::endl;

	std::map<const state_action_pair_type, double> Q;
	std::map<const state_action_pair_type, double> eligibility;

	// initialize Q(state, action)
	for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
		{
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Up)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Down)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Right)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Left)] = 0.0;

			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Up)] = 0.0;
			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Down)] = 0.0;
			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Right)] = 0.0;
			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Left)] = 0.0;
		}

	if (Q.size() != state_action_pair_type::state_type::STATE_SIZE * state_action_pair_type::action_type::ACTION_SIZE)
	{
		std::cerr << "state-action pair generation error" << std::endl;
		return;
	}

	try
	{
		typedef swl::SarsaLambda<state_action_pair_type> learner_type;

		swl::WinTimer aTimer;
		const learner_type learner(gamma, &state_action_pair_type::epsilonFunction, &state_action_pair_type::stepSizeFunction, lambda, isReplacingTrace);
		learner.train(maxEpisodeCount, policy, Q, eligibility);
		std::cout << "elapsed time: " << aTimer.getElapsedTimeInMilliSecond() << " msec" << std::endl;

		for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		{
			for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
				std::cout << getActionSymbol(state_action_pair_type::getGreedyAction(state_action_pair_type::state_type(row, col), Q)) << "  ";
			std::cout << std::endl;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

// Q(lambda)
void q_lambda(const size_t maxEpisodeCount, const state_action_pair_type::policy_type policy, const double gamma, const double lambda, const bool isReplacingTrace)
{
	std::cout << "<<-- Q(lambda) -->>" << std::endl;

	std::map<const state_action_pair_type, double> Q;
	std::map<const state_action_pair_type, double> eligibility;

	// initialize Q(state, action)
	for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
		{
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Up)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Down)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Right)] = 0.0;
			Q[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Left)] = 0.0;

			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Up)] = 0.0;
			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Down)] = 0.0;
			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Right)] = 0.0;
			eligibility[state_action_pair_type(state_action_pair_type::state_type(row, col), state_action_pair_type::Left)] = 0.0;
		}

	if (Q.size() != state_action_pair_type::state_type::STATE_SIZE * state_action_pair_type::action_type::ACTION_SIZE)
	{
		std::cerr << "state-action pair generation error" << std::endl;
		return;
	}

	try
	{
		typedef swl::QLambda<state_action_pair_type> learner_type;

		swl::WinTimer aTimer;
		const learner_type learner(gamma, &state_action_pair_type::epsilonFunction, &state_action_pair_type::stepSizeFunction, lambda, isReplacingTrace);
		learner.train(maxEpisodeCount, policy, Q, eligibility);
		std::cout << "elapsed time: " << aTimer.getElapsedTimeInMilliSecond() << " msec" << std::endl;

		for (size_t row = 1; row <= state_action_pair_type::state_type::GRID_ROW_SIZE; ++row)
		{
			for (size_t col = 1; col <= state_action_pair_type::state_type::GRID_COL_SIZE; ++col)
				std::cout << getActionSymbol(state_action_pair_type::getGreedyAction(state_action_pair_type::state_type(row, col), Q)) << "  ";
			std::cout << std::endl;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

}  // unnamed namespace

void td_learning()
{
	std::srand((unsigned int)std::time(NULL));

	//
	const state_action_pair_type::policy_type policy = state_action_pair_type::EPSILON_GREEDY_POLICY;

	const double gamma = 0.001;  // the discount rate: 0 <= gamma <= 1.0

	const size_t maxEpisodeCount = 20000;

	sarsa(maxEpisodeCount, policy, gamma);
	q_learning(maxEpisodeCount, policy, gamma);
}

void td_lambda()
{
	std::srand((unsigned int)std::time(NULL));

	//
	const state_action_pair_type::policy_type policy = state_action_pair_type::EPSILON_GREEDY_POLICY;

	const double gamma = 0.001;  // the discount rate: 0 <= gamma <= 1.0

	// for the eligibility trace
	const double lambda = 0.1;  // 0 <= lambda <= 1.0
	const bool isReplacingTrace = false;

	const size_t maxEpisodeCount = 20000;

	sarsa_lambda(maxEpisodeCount, policy, gamma, lambda, isReplacingTrace);
	q_lambda(maxEpisodeCount, policy, gamma, lambda, isReplacingTrace);
}
