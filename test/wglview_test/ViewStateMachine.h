#if !defined(__SWL_WGL_VIEW_TEST__VIEW_STATE_MACHINE__H_)
#define __SWL_WGL_VIEW_TEST__VIEW_STATE_MACHINE__H_ 1


#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/simple_state.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/statechart/deep_history.hpp>
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
struct EvtZoomRegion: public boost::statechart::event<EvtZoomRegion> {};
struct EvtZoomAll: public boost::statechart::event<EvtZoomAll> {};
struct EvtZoomIn: public boost::statechart::event<EvtZoomIn> {};
struct EvtZoomOut: public boost::statechart::event<EvtZoomOut> {};
struct EvtBackToPreviousState: public boost::statechart::event<EvtBackToPreviousState> {};

//-----------------------------------------------------------------------------------
// 

struct IView;
struct ViewContext;
class ViewCamera3;

struct NotTransientState;
struct ViewStateMachine: public boost::statechart::state_machine<ViewStateMachine, NotTransientState>
{
public:
	ViewStateMachine(IView &view, ViewContext &context, ViewCamera3 &camera);

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

	IView & getView()  {  return view_;  }
	const IView & getView() const  {  return view_;  }
	ViewContext & getViewContext()  {  return context_;  }
	const ViewContext & getViewContext() const  {  return context_;  }
	ViewCamera3 & getViewCamera()  {  return camera_;  }
	const ViewCamera3 & getViewCamera() const  {  return camera_;  }
	
private:
	IView &view_;
	ViewContext &context_;
	ViewCamera3 &camera_;
};

//-----------------------------------------------------------------------------
//

struct IdleState;
struct PanState;
struct RotateState;
struct ZoomRegionState;
struct ZoomAllState;
struct ZoomInState;
struct ZoomOutState;

struct NotTransientState: public boost::statechart::simple_state<NotTransientState, ViewStateMachine, IdleState, boost::statechart::has_deep_history>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtZoomAll, ZoomAllState>,
		boost::statechart::transition<EvtZoomIn, ZoomInState>,
		boost::statechart::transition<EvtZoomOut, ZoomOutState>
	> reactions;
};

//-----------------------------------------------------------------------------
//

struct IdleState: public IViewEventHandler, public boost::statechart::simple_state<IdleState, NotTransientState>
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

struct PanState: public IViewEventHandler, public boost::statechart::simple_state<PanState, NotTransientState>
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
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct RotateState: public IViewEventHandler, public boost::statechart::simple_state<RotateState, NotTransientState>
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
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct ZoomRegionState: public IViewEventHandler, public boost::statechart::simple_state<ZoomRegionState, NotTransientState>
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
	/*virtual*/ void wheelMouse(const MouseEvent &evt);

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}

private:
	void drawRubberBand(const MouseEvent &evt, HDC hdc) const;

private:
	bool isDragging_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct ZoomAllState: public boost::statechart::state<ZoomAllState, ViewStateMachine>
{
public:
	typedef boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> > reactions;

public:
	ZoomAllState(my_context ctx);

private:
	void handleEvent();
};

//-----------------------------------------------------------------------------
//

struct ZoomInState: public boost::statechart::state<ZoomInState, ViewStateMachine>
{
public:
	typedef boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> > reactions;

public:
	ZoomInState(my_context ctx);

private:
	void handleEvent();
};

//-----------------------------------------------------------------------------
//

struct ZoomOutState: public boost::statechart::state<ZoomOutState, ViewStateMachine>
{
public:
	typedef boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> > reactions;

public:
	ZoomOutState(my_context ctx);

private:
	void handleEvent();
};

}  // namespace swl


#endif  // __SWL_WGL_VIEW_TEST__VIEW_STATE_MACHINE__H_
