#if !defined(__SWL_MACHINE_LEARNING_TEST__WINDY_GRID_WORLD_SYSTEM__H_)
#define __SWL_MACHINE_LEARNING_TEST__WINDY_GRID_WORLD_SYSTEM__H_ 1


#include <map>



//-----------------------------------------------------------------------------
//

struct WindyGridWorldStateActionPair
{
public:
	enum POLICY { GREEDY_POLICY = 0, EPSILON_GREEDY_POLICY, EPSILON_SOFT_POLICY };

	struct State;
	struct Action;

	struct State
	{
	public:
		static const size_t GRID_ROW_SIZE = 7;
		static const size_t GRID_COL_SIZE = 10;
		static const size_t STATE_SIZE = GRID_ROW_SIZE * GRID_COL_SIZE;

		typedef size_t value_type;
		typedef Action action_type;
		typedef int reward_type;

	public:
		explicit State()
		: row_(4), col_(1)  // the start state
		{}
		explicit State(const size_t row, const size_t col)
		: row_(row), col_(col)
		{}

		State & operator=(const State &rhs)
		{
			if (this == &rhs) return *this;
			row_ = rhs.row_;
			col_ = rhs.col_;
			return *this;
		}

	public:
		bool isTerminalState() const;
		bool isValidState() const;

		reward_type getReward() const;

		State takeAction(const action_type &action) const;

		//value_type getValue() const  {  return *this;  }
		value_type getValue() const  {  return (row_ - 1) * GRID_COL_SIZE + (col_ - 1);  }

	private:
		size_t row_;  // [1, 7]
		size_t col_;  // [1, 10]
	};

	struct Action
	{
	public:
		enum ACTION { UP = 0, DOWN, RIGHT, LEFT };
		static const size_t ACTION_SIZE = 4;

		typedef ACTION value_type;

	public:
		explicit Action()
		: action_(UP)  // the initial action_
		{}
		explicit Action(const ACTION action)
		: action_(action)
		{}

		Action & operator=(const Action &rhs)
		{
			if (this == &rhs) return *this;
			action_ = rhs.action_;
			return *this;
		}

	public:
		value_type getValue() const  {  return action_;  }

		bool operator==(const Action &rhs) const
		{  return action_ == rhs.action_;  }
		bool operator!=(const Action &rhs) const
		{  return action_ != rhs.action_;  }

	private:
		ACTION action_;
	};

public:
	//typedef WindyGridWorldStateActionPair base_type;
	typedef State state_type;
	typedef state_type::reward_type reward_type;
	typedef Action action_type;
	typedef POLICY policy_type;

	static action_type Up;
	static action_type Down;
	static action_type Right;
	static action_type Left;

public:
	WindyGridWorldStateActionPair(const state_type &state, const action_type &action_)
	: state_(state), action_(action_)
	{}
	explicit WindyGridWorldStateActionPair(const WindyGridWorldStateActionPair &rhs)
	: state_(rhs.state_), action_(rhs.action_)
	{}

	WindyGridWorldStateActionPair & operator=(const WindyGridWorldStateActionPair &rhs)
	{
		if (this == &rhs) return *this;
		state_ = rhs.state_;
		action_ = rhs.action_;
		return *this;
	}

public:
	bool operator<(const WindyGridWorldStateActionPair &rhs) const
	{
		return state_.getValue() * action_type::ACTION_SIZE + action_.getValue() < rhs.state_.getValue() * action_type::ACTION_SIZE + rhs.action_.getValue();
	}

	static action_type getGreedyAction(const state_type &state, const std::map<const WindyGridWorldStateActionPair, double> &Q);
	static action_type getRandomAction();
	static action_type getActionFromPolicy(const state_type &state, const std::map<const WindyGridWorldStateActionPair, double> &Q, const policy_type &policy, const double epsilon);

	static double epsilonFunction(const size_t episodeTrial);
	static double stepSizeFunction(const size_t iterationStep);

private:
	state_type state_;
	action_type action_;
};


#endif  // __SWL_MACHINE_LEARNING_TEST__WINDY_GRID_WORLD_SYSTEM__H_
