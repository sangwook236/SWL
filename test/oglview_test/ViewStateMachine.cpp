#include "stdafx.h"
#include "ViewStateMachine.h"
#include "swl/view/ViewBase.h"
#include "swl/view/ViewContext.h"
#include "swl/view/ViewCamera2.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include <iostream>

#if defined(WIN32) && defined(_DEBUG)
#include "swl/common/Config.h"
#define new new(__FILE__, __LINE__)
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// 

ViewStateMachine::ViewStateMachine(ViewBase &view, ViewContext &context, ViewCamera2 &camera)
: view_(view), context_(context), camera_(camera)
{
}

void ViewStateMachine::pressMouse(const MouseEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).pressMouse(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::releaseMouse(const MouseEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).releaseMouse(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::moveMouse(const MouseEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).moveMouse(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::wheelMouse(const MouseEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).wheelMouse(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::clickMouse(const MouseEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).clickMouse(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::doubleClickMouse(const MouseEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).doubleClickMouse(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::pressKey(const KeyEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).pressKey(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::releaseKey(const KeyEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).releaseKey(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ViewStateMachine::hitKey(const KeyEvent &evt)
{
	try
	{
		const IViewEventHandler &handler = state_cast<const IViewEventHandler &>();
		const_cast<IViewEventHandler &>(handler).hitKey(evt);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

//-----------------------------------------------------------------------------------
// 

PanState::PanState()
: isDragging_(false), oldX_(0), oldY_(0)
{
}

PanState::~PanState()
{
}

void PanState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
	oldX_ = evt.x;
	oldY_ = evt.y;
}

void PanState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;

	//const int dX = evt.x - oldX_, dY = evt.y - oldY_;  // upward y-axis
	const int dX = evt.x - oldX_, dY = oldY_ - evt.y;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		ViewBase &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		context.activate();
			camera.moveView(dX, dY);
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		context.deactivate();
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void PanState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;

	//const int dX = evt.x - oldX_, dY = evt.y - oldY_;  // upward y-axis
	const int dX = evt.x - oldX_, dY = oldY_ - evt.y;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		ViewBase &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		context.activate();
			camera.moveView(dX, dY);
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		context.deactivate();
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

//-----------------------------------------------------------------------------------
// 

RotateState::RotateState()
: isDragging_(false), oldX_(0), oldY_(0)
{
}

RotateState::~RotateState()
{
}

void RotateState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
}


void RotateState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;
}


void RotateState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;
}

//-----------------------------------------------------------------------------------
// 

ZoomRegionState::ZoomRegionState()
: isDragging_(false), oldX_(0), oldY_(0)
{
}

ZoomRegionState::~ZoomRegionState()
{
}

void ZoomRegionState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
}


void ZoomRegionState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;
}

void ZoomRegionState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;
}

}  // namespace swl
