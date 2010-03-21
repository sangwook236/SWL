#include "stdafx.h"
#include "swl/Config.h"
#include "ViewStateMachine.h"
#include "swl/winview/WglViewBase.h"
#include "swl/view/ViewContext.h"
#include "swl/view/ViewCamera3.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include "swl/graphics/ObjectPickerMgr.h"
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

void drawRubberBandUsingGdi(HDC hdc, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase = true, const bool doesDraw = true)
{
	if (doesErase)
	{
		const int left = prevX <= initX ? prevX : initX;
		const int right = prevX > initX ? prevX : initX;
		const int top = prevY <= initY ? prevY : initY;  // downward y-axis
		const int bottom = prevY > initY ? prevY : initY;  // downward y-axis

		RECT rect;
		rect.left = left;
		rect.right = right;
		rect.top = top;
		rect.bottom = bottom;
		DrawFocusRect(hdc, &rect);
	}

	if (doesDraw)
	{
		const int left = currX <= initX ? currX : initX;
		const int right = currX > initX ? currX : initX;
		const int top = currY <= initY ? currY : initY;  // downward y-axis
		const int bottom = currY > initY ? currY : initY;  // downward y-axis

		RECT rect;
		rect.left = left;
		rect.right = right;
		rect.top = top;
		rect.bottom = bottom;
		DrawFocusRect(hdc, &rect);
	}
}

}  // unnamed namespace

//-----------------------------------------------------------------------------------
// 

ViewStateMachine::ViewStateMachine(IView &view, ViewContext &context, ViewCamera3 &camera)
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

	const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
	//const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera3 &camera = fsm.getViewCamera();

		{
			ViewContext::guard_type guard(context);
			camera.moveView(dX, dY);
			view.raiseDrawEvent(true);
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

	const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
	//const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera3 &camera = fsm.getViewCamera();

		{
			ViewContext::guard_type guard(context);
			camera.moveView(dX, dY);
			view.raiseDrawEvent(true);
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

RotateState::RotateState()
: isDragging_(false), prevX_(0), prevY_(0)
{
}

RotateState::~RotateState()
{
}

void RotateState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
	prevX_ = evt.x;
	prevY_ = evt.y;
}

void RotateState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;
	if (evt.x == prevX_ && evt.y == prevY_) return;

	const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
	//const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera3 &camera = fsm.getViewCamera();

		{
			ViewContext::guard_type guard(context);
			camera.rotateView(dX, dY);
			view.raiseDrawEvent(true);
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void RotateState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;
	if (evt.x == prevX_ && evt.y == prevY_) return;

	const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
	//const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera3 &camera = fsm.getViewCamera();

		{
			ViewContext::guard_type guard(context);
			camera.rotateView(dX, dY);
			view.raiseDrawEvent(true);
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

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContext::guard_type guard(context);
			const swl::Region2<int> vp = camera.getViewport();
			camera.setView(initX_, vp.getHeight() - initY_, evt.x, vp.getHeight() - evt.y);
			view.raiseDrawEvent(true);
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

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

#if 1
		CView *vw = dynamic_cast<CView *>(&view);
		if (vw)
		{
			CClientDC dc(vw);
			drawRubberBandUsingGdi(dc.GetSafeHdc(), initX_, initY_, prevX_, prevY_, evt.x, evt.y);
		}
#else
		WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
		if (vw && !vw->isContextStackEmpty())
		{
			const boost::shared_ptr<WglViewBase::context_type> &ctx(vw->topContext());
			if (NULL != ctx.get())
			{
				const boost::any nativeHandle = ctx->getNativeWindowHandle();
				if (!nativeHandle.empty())
				{
					HWND *hwnd = NULL;
					try
					{
						hwnd = boost::any_cast<HWND *>(nativeHandle);
					}
					catch (const boost::bad_any_cast &)
					{
						hwnd = NULL;
					}

					if (hwnd)
					{
#if 1
						HDC hdc = GetDC(*hwnd);
						drawRubberBandUsingGdi(*hdc, initX_, initY_, prevX_, prevY_, evt.x, evt.y);
						//view.updateScrollBar();
						ReleaseDC(*hwnd, hdc);
#else
						Gdiplus::Graphics *graphics = Gdiplus::Graphics::FromHWND(*hwnd);
						if (graphics)
						{
							HDC hdc = graphics->GetHDC();
							drawRubberBandUsingGdi(hdc, initX_, initY_, prevX_, prevY_, evt.x, evt.y);
							//view.updateScrollBar();
							graphics->ReleaseHDC(hdc);
							delete graphics;
						}
#endif
					}
				}
			}
		}
#endif
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
			ViewContext::guard_type guard(context);
			if (evt.scrollAmount > 0)
				camera.scaleViewRegion(/*1.0 / 0.8 =*/ 1.25 * evt.scrollAmount);  // zoom-out
			else
				camera.scaleViewRegion(0.8 * -evt.scrollAmount);  // zoom-in
			view.raiseDrawEvent(true);
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
			ViewContext::guard_type guard(context);
			camera.restoreViewRegion();
			view.raiseDrawEvent(true);
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
			ViewContext::guard_type guard(context);
			camera.scaleViewRegion(0.8);
			view.raiseDrawEvent(true);
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
			ViewContext::guard_type guard(context);
			camera.scaleViewRegion(/*1.0 / 0.8 =*/ 1.25);
			view.raiseDrawEvent(true);
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

PickObjectState::PickObjectState()
: isDragging_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
	swl::ObjectPickerMgr::getInstance().clearAllPickedObjects();
	swl::ObjectPickerMgr::getInstance().startPicking();
}

PickObjectState::~PickObjectState()
{
	swl::ObjectPickerMgr::getInstance().stopPicking();
}

void PickObjectState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
	initX_ = prevX_ = evt.x;
	initY_ = prevY_ = evt.y;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawRubberBand(view, evt.x, evt.y, false, true);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void PickObjectState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawRubberBand(view, evt.x, evt.y, true, false);

		WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
		if (vw)
		{
			if (evt.x == initX_ && evt.y == initY_)
				vw->pickObject(evt.x, evt.y, false);
			else
				vw->pickObject(initX_, initY_, evt.x, evt.y, false);
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void PickObjectState::moveMouse(const MouseEvent &evt)
{
	if (isDragging_)
	{
		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			drawRubberBand(view, evt.x, evt.y, true, true);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}

		prevX_ = evt.x;
		prevY_ = evt.y;
	}
	else
	{
		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
			if (vw)
				vw->pickObject(evt.x, evt.y, true);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

void PickObjectState::drawRubberBand(IView &view, const int currX, const int currY, const bool doesErase, const bool doesDraw) const
{
#if 1
	CView *vw = dynamic_cast<CView *>(&view);
	if (vw)
	{
		CClientDC dc(vw);
		drawRubberBandUsingGdi(dc.GetSafeHdc(), initX_, initY_, prevX_, prevY_, currX, currY, doesErase, doesDraw);
	}
#else
	WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
	if (vw && !vw->isContextStackEmpty())
	{
		const boost::shared_ptr<WglViewBase::context_type> &ctx(vw->topContext());
		if (NULL != ctx.get())
		{
			const boost::any nativeHandle = ctx->getNativeWindowHandle();
			if (!nativeHandle.empty())
			{
				HWND *hwnd = NULL;
				try
				{
					hwnd = boost::any_cast<HWND *>(nativeHandle);
				}
				catch (const boost::bad_any_cast &)
				{
					hwnd = NULL;
				}

				if (hwnd)
				{
#if 1
					HDC hdc = GetDC(*hwnd);
					drawRubberBandUsingGdi(*hdc, initX_, initY_, prevX_, prevY_, currX, currY, doesErase, doesDraw);
					//view.updateScrollBar();
					ReleaseDC(*hwnd, hdc);
#else
					Gdiplus::Graphics *graphics = Gdiplus::Graphics::FromHWND(*hwnd);
					if (graphics)
					{
						HDC hdc = graphics->GetHDC();
						drawRubberBandUsingGdi(hdc, initX_, initY_, prevX_, prevY_, currX, currY, doesErase, doesDraw);
						//view.updateScrollBar();
						graphics->ReleaseHDC(hdc);
						delete graphics;
					}
#endif
				}
			}
		}
	}
#endif
}

//-----------------------------------------------------------------------------
//

DragObjectState::DragObjectState()
: isDragging_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
	swl::ObjectPickerMgr::getInstance().clearAllPickedObjects();
	swl::ObjectPickerMgr::getInstance().startPicking();
}

DragObjectState::~DragObjectState()
{
	swl::ObjectPickerMgr::getInstance().stopPicking();
}

void DragObjectState::pressMouse(const MouseEvent &evt)
{
	isDragging_ = true;
	initX_ = prevX_ = evt.x;
	initY_ = prevY_ = evt.y;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawRubberBand(view, evt.x, evt.y, false, true);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void DragObjectState::releaseMouse(const MouseEvent &evt)
{
	isDragging_ = false;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawRubberBand(view, evt.x, evt.y, true, false);

		WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
		if (vw)
		{
			if (evt.x == initX_ && evt.y == initY_)
				vw->pickObject(evt.x, evt.y, false);
			else
				vw->pickObject(initX_, initY_, evt.x, evt.y, false);
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void DragObjectState::moveMouse(const MouseEvent &evt)
{
	if (isDragging_)
	{
		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			drawRubberBand(view, evt.x, evt.y, true, true);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}

		prevX_ = evt.x;
		prevY_ = evt.y;
	}
	else
	{
		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
			if (vw)
				vw->pickObject(evt.x, evt.y, true);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

void DragObjectState::drawRubberBand(IView &view, const int currX, const int currY, const bool doesErase, const bool doesDraw) const
{
#if 1
	CView *vw = dynamic_cast<CView *>(&view);
	if (vw)
	{
		CClientDC dc(vw);
		drawRubberBandUsingGdi(dc.GetSafeHdc(), initX_, initY_, prevX_, prevY_, currX, currY, doesErase, doesDraw);
	}
#else
	WglViewBase *vw = dynamic_cast<WglViewBase *>(&view);
	if (vw && !vw->isContextStackEmpty())
	{
		const boost::shared_ptr<WglViewBase::context_type> &ctx(vw->topContext());
		if (NULL != ctx.get())
		{
			const boost::any nativeHandle = ctx->getNativeWindowHandle();
			if (!nativeHandle.empty())
			{
				HWND *hwnd = NULL;
				try
				{
					hwnd = boost::any_cast<HWND *>(nativeHandle);
				}
				catch (const boost::bad_any_cast &)
				{
					hwnd = NULL;
				}

				if (hwnd)
				{
#if 1
					HDC hdc = GetDC(*hwnd);
					drawRubberBandUsingGdi(*hdc, initX_, initY_, prevX_, prevY_, currX, currY, doesErase, doesDraw);
					//view.updateScrollBar();
					ReleaseDC(*hwnd, hdc);
#else
					Gdiplus::Graphics *graphics = Gdiplus::Graphics::FromHWND(*hwnd);
					if (graphics)
					{
						HDC hdc = graphics->GetHDC();
						drawRubberBandUsingGdi(hdc, initX_, initY_, prevX_, prevY_, currX, currY, doesErase, doesDraw);
						//view.updateScrollBar();
						graphics->ReleaseHDC(hdc);
						delete graphics;
					}
#endif
				}
			}
		}
	}
#endif
}

}  // namespace swl
