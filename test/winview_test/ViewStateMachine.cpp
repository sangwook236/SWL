#include "stdafx.h"
#include "ViewStateMachine.h"
#include "swl/view/ViewBase.h"
#include "swl/view/ViewContext.h"
#include "swl/view/ViewCamera2.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include <gdiplus.h>
#include <iostream>


#if defined(WIN32) && defined(_DEBUG)
#include "swl/common/Config.h"
#define new new(__FILE__, __LINE__)
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// 

ViewStateMachine::ViewStateMachine(IView &view, ViewContext &context, ViewCamera2 &camera)
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
: isDragging_(false), prevX_(0), prevY_(0)
{
}

PanState::~PanState()
{
}

void PanState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
	prevX_ = evt.x;
	prevY_ = evt.y;
}

void PanState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;
	if (evt.x == prevX_ && evt.y == prevY_) return;

	//const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
	const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			camera.moveView(dX, dY);
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void PanState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;
	if (evt.x == prevX_ && evt.y == prevY_) return;

	//const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
	const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			camera.moveView(dX, dY);
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}

	prevX_ = evt.x;
	prevY_ = evt.y;
}

//-----------------------------------------------------------------------------------
// 

ZoomRegionState::ZoomRegionState()
: isDragging_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
}

ZoomRegionState::~ZoomRegionState()
{
}

void ZoomRegionState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
	initX_ = prevX_ = evt.x;
	initY_ = prevY_ = evt.y;
}

void ZoomRegionState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;
	if (evt.x == initX_ && evt.y == initY_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			camera.setView(initX_, initY_, evt.x, evt.y);
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ZoomRegionState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;
	if (evt.x == prevX_ && evt.y == prevY_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		CView *vw = dynamic_cast<CView *>(&view);
		if (vw)
		{
			CClientDC dc(vw);
			drawRubberBand(evt, dc.GetSafeHdc());
		}
/*
		// this implementation is not working
		boost::any nativeCtx;
		{
			ViewContextGuard guard(context);
			nativeCtx = context.getNativeContext();
		}

		if (!nativeCtx.empty())
		{
			try
			{
				HDC *hdc = boost::any_cast<HDC *>(nativeCtx);
				if (hdc)
				{
					drawRubberBand(evt, *hdc);
					//view.updateScrollBar();
				}
			}
			catch (const boost::bad_any_cast &)
			{
			}

			try
			{
				Gdiplus::Graphics *graphics = boost::any_cast<Gdiplus::Graphics *>(nativeCtx);
				if (graphics)
				{
					HDC hdc = graphics->GetHDC();
					drawRubberBand(evt, hdc);
					//view.updateScrollBar();
					graphics->ReleaseHDC(hdc);
				}
			}
			catch (const boost::bad_any_cast &)
			{
			}
		}
*/
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}

	prevX_ = evt.x;
	prevY_ = evt.y;
}

void ZoomRegionState::wheelMouse(const MouseEvent &evt)
{
	if (0 == evt.scrollAmount) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			if (evt.scrollAmount > 0)
				camera.scaleViewRegion(/*1.0 / 0.8 =*/ 1.25 * evt.scrollAmount);  // zoom-out
			else
				camera.scaleViewRegion(0.8 * -evt.scrollAmount);  // zoom-in
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void ZoomRegionState::drawRubberBand(const MouseEvent &evt, HDC hdc) const
{
	{
		const int left = prevX_ <= initX_ ? prevX_ : initX_;
		const int right = prevX_ > initX_ ? prevX_ : initX_;
		const int top = prevY_ <= initY_ ? prevY_ : initY_;  // downward y-axis
		const int bottom = prevY_ > initY_ ? prevY_ : initY_;  // downward y-axis

		DrawFocusRect(hdc, &CRect(left, top, right, bottom));
	}

	{
		const int left = evt.x <= initX_ ? evt.x : initX_;
		const int right = evt.x > initX_ ? evt.x : initX_;
		const int top = evt.y <= initY_ ? evt.y : initY_;  // downward y-axis
		const int bottom = evt.y > initY_ ? evt.y : initY_;  // downward y-axis

		DrawFocusRect(hdc, &CRect(left, top, right, bottom));
	}
}

//-----------------------------------------------------------------------------
//

ZoomAllState::ZoomAllState(my_context ctx)
: my_base(ctx)
{
	handleEvent();
	post_event(EvtBackToPreviousState());
}

void ZoomAllState::handleEvent()
{
	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			camera.restoreViewRegion();
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

//-----------------------------------------------------------------------------
//

ZoomInState::ZoomInState(my_context ctx)
: my_base(ctx)
{
	handleEvent();
	post_event(EvtBackToPreviousState());
}

void ZoomInState::handleEvent()
{
	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			camera.scaleViewRegion(0.8);
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

//-----------------------------------------------------------------------------
//

ZoomOutState::ZoomOutState(my_context ctx)
: my_base(ctx)
{
	handleEvent();
	post_event(EvtBackToPreviousState());
}

void ZoomOutState::handleEvent()
{
	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContextGuard guard(context);
			camera.scaleViewRegion(/*1.0 / 0.8 =*/ 1.25);;
			view.raiseDrawEvent(false);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

}  // namespace swl
