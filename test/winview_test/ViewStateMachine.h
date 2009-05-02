#if !defined(__SWL_WIN_VIEW_TEST__VIEW_STATE_MACHINE__H_)
#define __SWL_WIN_VIEW_TEST__VIEW_STATE_MACHINE__H_ 1


#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/simple_state.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/statechart/custom_reaction.hpp>
#include <boost/mpl/list.hpp>


namespace swl {

struct IdleState;

//-----------------------------------------------------------------------------
//

struct EvtIdle: public boost::statechart::event<EvtIdle> {};
struct EvtPan: public boost::statechart::event<EvtPan> {};
struct EvtRotate: public boost::statechart::event<EvtRotate> {};
struct EvtZoomAll: public boost::statechart::event<EvtZoomAll> {};
struct EvtZoomRegion: public boost::statechart::event<EvtZoomRegion> {};
struct EvtZoomIn: public boost::statechart::event<EvtZoomIn> {};
struct EvtZoomOut: public boost::statechart::event<EvtZoomOut> {};

//-----------------------------------------------------------------------------------
// 

struct ViewStateMachine: public boost::statechart::state_machine<ViewStateMachine, IdleState>
{
};

//-----------------------------------------------------------------------------
//

struct PanState;
struct RotateState;
struct ZoomRegionState;

struct IdleState: public boost::statechart::simple_state<IdleState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>
	> reactions;


public:
	IdleState()
	{
	}
	~IdleState()
	{
	}

private:
};

//-----------------------------------------------------------------------------
//

struct PanState: public boost::statechart::simple_state<PanState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>
	> reactions;

public:
	PanState()
	{
	}
	~PanState()
	{
	}

private:
};

//-----------------------------------------------------------------------------
//

struct RotateState: public boost::statechart::simple_state<RotateState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>
	> reactions;

public:
	RotateState()
	{
	}
	~RotateState()
	{
	}

private:
};

//-----------------------------------------------------------------------------
//

struct ZoomRegionState: public boost::statechart::simple_state<ZoomRegionState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>
	> reactions;

public:
	ZoomRegionState()
	{
	}
	~ZoomRegionState()
	{
	}

private:
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW_TEST__VIEW_STATE_MACHINE__H_
