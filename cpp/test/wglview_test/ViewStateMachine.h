#if !defined(__SWL_WGL_VIEW_TEST__VIEW_STATE_MACHINE__H_)
#define __SWL_WGL_VIEW_TEST__VIEW_STATE_MACHINE__H_ 1


#include "swl/util/RegionOfInterest.h"
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
struct EvtPickObject: public boost::statechart::event<EvtPickObject> {};
struct EvtPickAndDragObject: public boost::statechart::event<EvtPickAndDragObject> {};
struct EvtHandleROI: public boost::statechart::event<EvtHandleROI> {};
struct EvtManipulateROI: public boost::statechart::event<EvtManipulateROI> {};
struct EvtCreateLineROI: public boost::statechart::event<EvtCreateLineROI> {};
struct EvtCreateRectangleROI: public boost::statechart::event<EvtCreateRectangleROI> {};
struct EvtCreatePolylineROI: public boost::statechart::event<EvtCreatePolylineROI> {};
struct EvtCreatePolygonROI: public boost::statechart::event<EvtCreatePolygonROI> {};
struct EvtBackToPreviousState: public boost::statechart::event<EvtBackToPreviousState> {};

//-----------------------------------------------------------------------------------
// 

struct HandleViewState;
struct ZoomAllState;
struct ZoomInState;
struct ZoomOutState;
struct HandleROIState;

// sub-states in HandleViewState
struct IdleState;
struct PanState;
struct RotateState;
struct ZoomRegionState;
struct PickObjectState;
struct PickAndDragObjectState;

// sub-states in HandleROIState
struct ManipulateROIState;
struct CreateLineROIState;
struct CreateRectangleROIState;
struct CreatePolylineROIState;
struct CreatePolygonROIState;

//-----------------------------------------------------------------------------------
// 

struct IView;
struct ViewContext;
class ViewCamera3;

struct ViewStateMachine: public boost::statechart::state_machine<ViewStateMachine, HandleViewState>
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

struct HandleViewState: public boost::statechart::simple_state<HandleViewState, ViewStateMachine, IdleState, boost::statechart::has_deep_history>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtZoomAll, ZoomAllState>,
		boost::statechart::transition<EvtZoomIn, ZoomInState>,
		boost::statechart::transition<EvtZoomOut, ZoomOutState>,
		boost::statechart::transition<EvtHandleROI, HandleROIState>
	> reactions;
};

//-----------------------------------------------------------------------------
//

struct IdleState: public IViewEventHandler, public boost::statechart::simple_state<IdleState, HandleViewState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>,
		boost::statechart::transition<EvtPickObject, PickObjectState>,
		boost::statechart::transition<EvtPickAndDragObject, PickAndDragObjectState>
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

struct PanState: public IViewEventHandler, public boost::statechart::simple_state<PanState, HandleViewState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>,
		boost::statechart::transition<EvtPickObject, PickObjectState>,
		boost::statechart::transition<EvtPickAndDragObject, PickAndDragObjectState>
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

struct RotateState: public IViewEventHandler, public boost::statechart::simple_state<RotateState, HandleViewState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>,
		boost::statechart::transition<EvtPickObject, PickObjectState>,
		boost::statechart::transition<EvtPickAndDragObject, PickAndDragObjectState>
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

struct ZoomRegionState: public IViewEventHandler, public boost::statechart::simple_state<ZoomRegionState, HandleViewState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtPickObject, PickObjectState>,
		boost::statechart::transition<EvtPickAndDragObject, PickAndDragObjectState>
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
	bool isDragging_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct ZoomAllState: public boost::statechart::state<ZoomAllState, ViewStateMachine>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> >
	> reactions;

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
	typedef boost::mpl::list<
		boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> >
	> reactions;

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
	typedef boost::mpl::list<
		boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> >
	> reactions;

public:
	ZoomOutState(my_context ctx);

private:
	void handleEvent();
};

//-----------------------------------------------------------------------------
//

struct PickObjectState: public IViewEventHandler, public boost::statechart::simple_state<PickObjectState, HandleViewState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>,
		boost::statechart::transition<EvtPickAndDragObject, PickAndDragObjectState>
	> reactions;

public:
	PickObjectState();
	~PickObjectState();

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
	bool isJustPressed_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct PickAndDragObjectState: public IViewEventHandler, public boost::statechart::simple_state<PickAndDragObjectState, HandleViewState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtIdle, IdleState>,
		boost::statechart::transition<EvtPan, PanState>,
		boost::statechart::transition<EvtRotate, RotateState>,
		boost::statechart::transition<EvtZoomRegion, ZoomRegionState>,
		boost::statechart::transition<EvtPickObject, PickObjectState>
	> reactions;

public:
	PickAndDragObjectState();
	~PickAndDragObjectState();

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
	bool isDraggingObject_;
	bool isJustPressed_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct HandleROIState: public boost::statechart::simple_state<HandleROIState, ViewStateMachine, ManipulateROIState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtBackToPreviousState, boost::statechart::deep_history<IdleState> >
	> reactions;
};

//-----------------------------------------------------------------------------
//

struct ManipulateROIState: public IViewEventHandler, public boost::statechart::simple_state<ManipulateROIState, HandleROIState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtCreateLineROI, CreateLineROIState>,
		boost::statechart::transition<EvtCreateRectangleROI, CreateRectangleROIState>,
		boost::statechart::transition<EvtCreatePolylineROI, CreatePolylineROIState>,
		boost::statechart::transition<EvtCreatePolygonROI, CreatePolygonROIState>
	> reactions;

public:
	ManipulateROIState();
	~ManipulateROIState();

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
	typedef LineROI roi_type;

private:
	bool isDragging_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct CreateLineROIState: public IViewEventHandler, public boost::statechart::simple_state<CreateLineROIState, HandleROIState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtManipulateROI, ManipulateROIState>,
		boost::statechart::transition<EvtCreateRectangleROI, CreateRectangleROIState>,
		boost::statechart::transition<EvtCreatePolylineROI, CreatePolylineROIState>,
		boost::statechart::transition<EvtCreatePolygonROI, CreatePolygonROIState>
	> reactions;

public:
	CreateLineROIState();
	~CreateLineROIState();

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
	typedef LineROI roi_type;

private:
	bool isDragging_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct CreateRectangleROIState: public IViewEventHandler, public boost::statechart::simple_state<CreateRectangleROIState, HandleROIState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtManipulateROI, ManipulateROIState>,
		boost::statechart::transition<EvtCreateLineROI, CreateLineROIState>,
		boost::statechart::transition<EvtCreatePolylineROI, CreatePolylineROIState>,
		boost::statechart::transition<EvtCreatePolygonROI, CreatePolygonROIState>
	> reactions;

public:
	CreateRectangleROIState();
	~CreateRectangleROIState();

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
	typedef RectangleROI roi_type;

private:
	bool isDragging_;
	bool isJustPressed_;
	int initX_, initY_;
	int prevX_, prevY_;
};

//-----------------------------------------------------------------------------
//

struct CreatePolylineROIState: public IViewEventHandler, public boost::statechart::simple_state<CreatePolylineROIState, HandleROIState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtManipulateROI, ManipulateROIState>,
		boost::statechart::transition<EvtCreateLineROI, CreateLineROIState>,
		boost::statechart::transition<EvtCreateRectangleROI, CreateRectangleROIState>,
		boost::statechart::transition<EvtCreatePolygonROI, CreatePolygonROIState>
	> reactions;

public:
	CreatePolylineROIState();
	~CreatePolylineROIState();

public:
	/*virtual*/ void pressMouse(const MouseEvent &evt)  {}
	/*virtual*/ void releaseMouse(const MouseEvent &evt);
	/*virtual*/ void moveMouse(const MouseEvent &evt);
	/*virtual*/ void wheelMouse(const MouseEvent &evt)  {}

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}

private:
	typedef PolylineROI roi_type;

private:
	bool isSelectingRegion_;
	int initX_, initY_;
	int prevX_, prevY_;

	roi_type roi_;
};

//-----------------------------------------------------------------------------
//

struct CreatePolygonROIState: public IViewEventHandler, public boost::statechart::simple_state<CreatePolygonROIState, HandleROIState>
{
public:
	typedef boost::mpl::list<
		boost::statechart::transition<EvtManipulateROI, ManipulateROIState>,
		boost::statechart::transition<EvtCreateLineROI, CreateLineROIState>,
		boost::statechart::transition<EvtCreateRectangleROI, CreateRectangleROIState>,
		boost::statechart::transition<EvtCreatePolylineROI, CreatePolylineROIState>
	> reactions;

public:
	CreatePolygonROIState();
	~CreatePolygonROIState();

public:
	/*virtual*/ void pressMouse(const MouseEvent &evt)  {}
	/*virtual*/ void releaseMouse(const MouseEvent &evt);
	/*virtual*/ void moveMouse(const MouseEvent &evt);
	/*virtual*/ void wheelMouse(const MouseEvent &evt)  {}

	/*virtual*/ void clickMouse(const MouseEvent &evt)  {}
	/*virtual*/ void doubleClickMouse(const MouseEvent &evt)  {}

	/*virtual*/ void pressKey(const KeyEvent &evt)  {}
	/*virtual*/ void releaseKey(const KeyEvent &evt)  {}

	/*virtual*/ void hitKey(const KeyEvent &evt)  {}

private:
	typedef PolygonROI roi_type;

private:
	bool isSelectingRegion_;
	int initX_, initY_;
	int prevX_, prevY_;

	roi_type roi_;
};

}  // namespace swl


#endif  // __SWL_WGL_VIEW_TEST__VIEW_STATE_MACHINE__H_
