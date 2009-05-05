#if !defined(__SWL_OGL_VIEW_TEST__VIEW_STATE_MACHINE__H_)
#define __SWL_OGL_VIEW_TEST__VIEW_STATE_MACHINE__H_ 1


#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/simple_state.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/statechart/custom_reaction.hpp>
#include <boost/mpl/list.hpp>


namespace swl {

struct MouseEvent;
struct KeyEvent;

//-----------------------------------------------------------------------------
//

struct IViewEventHandler
{
	virtual void pressMouse(const MouseEvent &evt) = 0;
	virtual void releaseMouse(const MouseEvent &evt) = 0;
	virtual void moveMouse(const MouseEvent &evt) = 0;
	virtual void wheelMouse(const MouseEvent &evt) = 0;

	virtual void clickMouse(const MouseEvent &evt) = 0;
	virtual void doubleClickMouse(const MouseEvent &evt) = 0;

	virtual void pressKey(const KeyEvent &evt) = 0;
	virtual void releaseKey(const KeyEvent &evt) = 0;

	virtual void hitKey(const KeyEvent &evt) = 0;
};

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

struct ViewBase;
struct ViewContext;
class ViewCamera2;

struct IdleState;
struct ViewStateMachine: public boost::statechart::state_machine<ViewStateMachine, IdleState>
{
public:
	ViewStateMachine(ViewBase &view, ViewContext &context, ViewCamera2 &camera);

private:
	ViewStateMachine(const ViewStateMachine &);
	ViewStateMachine & operator=(const ViewStateMachine &);

public:
	void pressMouse(const MouseEvent &evt);
	void releaseMouse(const MouseEvent &evt);
	void moveMouse(const MouseEvent &evt);
	void wheelMouse(const MouseEvent &evt);

	void clickMouse(const MouseEvent &evt);
	void doubleClickMouse(const MouseEvent &evt);

	void pressKey(const KeyEvent &evt);
	void releaseKey(const KeyEvent &evt);

	void hitKey(const KeyEvent &evt);

	ViewBase & getView()  {  return view_;  }
	const ViewBase & getView() const  {  return view_;  }
	ViewContext & getViewContext()  {  return context_;  }
	const ViewContext & getViewContext() const  {  return context_;  }
	ViewCamera2 & getViewCamera()  {  return camera_;  }
	const ViewCamera2 & getViewCamera() const  {  return camera_;  }
	
private:
	ViewBase &view_;
	ViewContext &context_;
	ViewCamera2 &camera_;
};

//-----------------------------------------------------------------------------
//

struct PanState;
struct RotateState;
struct ZoomRegionState;
struct IdleState: public IViewEventHandler, public boost::statechart::simple_state<IdleState, ViewStateMachine>
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

public:
	/*virtual*/ void pressMouse(const MouseEvent &evt)  {}
	/*virtual*/ void releaseMouse(const MouseEvent &evt)  {}
	/*virtual*/ void moveMouse(const MouseEvent &evt)  {}
	/*virtual*/ void wheelMouse(const MouseEvent &evt)  {}

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}
};

//-----------------------------------------------------------------------------
//

struct PanState: public IViewEventHandler, public boost::statechart::simple_state<PanState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>
	> reactions;

public:
	PanState();
	~PanState();

public:
	/*virtual*/ void pressMouse(const MouseEvent &evt);
	/*virtual*/ void releaseMouse(const MouseEvent &evt);
	/*virtual*/ void moveMouse(const MouseEvent &evt);
	/*virtual*/ void wheelMouse(const MouseEvent &evt)  {}

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}

private:
	bool isDragging_;
	int oldX_, oldY_;
};

//-----------------------------------------------------------------------------
//

struct RotateState: public IViewEventHandler, public boost::statechart::simple_state<RotateState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>
	> reactions;

public:
	RotateState();
	~RotateState();

public:
	/*virtual*/ void pressMouse(const MouseEvent &evt);
	/*virtual*/ void releaseMouse(const MouseEvent &evt);
	/*virtual*/ void moveMouse(const MouseEvent &evt);
	/*virtual*/ void wheelMouse(const MouseEvent &evt)  {}

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}

private:
	bool isDragging_;
	int oldX_, oldY_;
};

//-----------------------------------------------------------------------------
//

struct ZoomRegionState: public IViewEventHandler, public boost::statechart::simple_state<ZoomRegionState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>
	> reactions;

public:
	ZoomRegionState();
	~ZoomRegionState();

public:
	/*virtual*/ void pressMouse(const MouseEvent &evt);
	/*virtual*/ void releaseMouse(const MouseEvent &evt);
	/*virtual*/ void moveMouse(const MouseEvent &evt);
	/*virtual*/ void wheelMouse(const MouseEvent &evt)  {}

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}

private:
	bool isDragging_;
	int oldX_, oldY_;
};

//-----------------------------------------------------------------------------
//

struct ZoomAllState: public IViewEventHandler, public boost::statechart::simple_state<ZoomAllState, ViewStateMachine>
{
};

//-----------------------------------------------------------------------------
//

struct ZoomInState: public IViewEventHandler, public boost::statechart::simple_state<ZoomInState, ViewStateMachine>
{
};

//-----------------------------------------------------------------------------
//

struct ZoomOutState: public IViewEventHandler, public boost::statechart::simple_state<ZoomOutState, ViewStateMachine>
{
};

}  // namespace swl


#endif  // __SWL_OGL_VIEW_TEST__VIEW_STATE_MACHINE__H_
