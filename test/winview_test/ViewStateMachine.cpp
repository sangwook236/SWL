#include "stdafx.h"
#include "swl/Config.h"
#include "ViewStateMachine.h"
#include "swl/winview/WinViewBase.h"
#include "swl/winview/GdiRubberBand.h"
#include "swl/view/ViewContext.h"
#include "swl/view/ViewCamera2.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include "swl/graphics/ObjectPickerMgr.h"
#include "swl/util/RegionOfInterestMgr.h"
#include <gdiplus.h>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

void drawLineRubberBand(IView &view, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase, const bool doesDraw)
{
#if 1
	CView *vw = dynamic_cast<CView *>(&view);
	if (vw)
	{
		CClientDC dc(vw);
		GdiRubberBand::drawLine(dc.GetSafeHdc(), initX, initY, prevX, prevY, currX, currY, doesErase, doesDraw);
	}
#else
	WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
	if (vw && !vw->isContextStackEmpty())
	{
		const boost::shared_ptr<WinViewBase::context_type> &ctx(vw->topContext());
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
#if 0
					HDC hdc = GetDC(*hwnd);
					GdiRubberBand::drawLine(hdc, initX, initY, prevX, prevY, currX, currY, doesErase, doesDraw);
					//view.updateScrollBar();
					ReleaseDC(*hwnd, hdc);
#else
					Gdiplus::Graphics *graphics = Gdiplus::Graphics::FromHWND(*hwnd);
					if (graphics)
					{
						HDC hdc = graphics->GetHDC();
						GdiRubberBand::drawLine(hdc, initX, initY, prevX, prevY, currX, currY, doesErase, doesDraw);
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

void drawRectangleRubberBand(IView &view, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase, const bool doesDraw)
{
#if 1
	CView *vw = dynamic_cast<CView *>(&view);
	if (vw)
	{
		CClientDC dc(vw);
		GdiRubberBand::drawRectangle(dc.GetSafeHdc(), initX, initY, prevX, prevY, currX, currY, doesErase, doesDraw);
	}
#else
	WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
	if (vw && !vw->isContextStackEmpty())
	{
		const boost::shared_ptr<WinViewBase::context_type> &ctx(vw->topContext());
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
#if 0
					HDC hdc = GetDC(*hwnd);
					GdiRubberBand::drawRectangle(hdc, initX, initY, prevX, prevY, currX, currY, doesErase, doesDraw);
					//view.updateScrollBar();
					ReleaseDC(*hwnd, hdc);
#else
					Gdiplus::Graphics *graphics = Gdiplus::Graphics::FromHWND(*hwnd);
					if (graphics)
					{
						HDC hdc = graphics->GetHDC();
						GdiRubberBand::drawRectangle(hdc, initX, initY, prevX, prevY, currX, currY, doesErase, doesDraw);
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

}  // unnamed namespace

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
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;

	isDragging_ = true;
	prevX_ = evt.x;
	prevY_ = evt.y;
}

void PanState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;
	if (!isDragging_) return;

	isDragging_ = false;
	if (evt.x == prevX_ && evt.y == prevY_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			//const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
			const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

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

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			//const int dX = evt.x - prevX_, dY = prevY_ - evt.y;  // upward y-axis
			const int dX = evt.x - prevX_, dY = evt.y - prevY_;  // downward y-axis

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

ZoomRegionState::ZoomRegionState()
: isDragging_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
}

ZoomRegionState::~ZoomRegionState()
{
}

void ZoomRegionState::pressMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;

	isDragging_ = true;
	initX_ = prevX_ = evt.x;
	initY_ = prevY_ = evt.y;
}

void ZoomRegionState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;
	if (!isDragging_) return;

	isDragging_ = false;
	if (evt.x == initX_ && evt.y == initY_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();
		ViewContext &context = fsm.getViewContext();
		ViewCamera2 &camera = fsm.getViewCamera();

		{
			ViewContext::guard_type guard(context);
			camera.setView(initX_, initY_, evt.x, evt.y);  // downward y-axis
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
	if (evt.x == prevX_ && evt.y == prevY_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, true);
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
			camera.scaleViewRegion(/*1.0 / 0.8 =*/ 1.25);;
			view.raiseDrawEvent(true);
			//view.updateScrollBar();
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

//-----------------------------------------------------------------------------------
// 

PickObjectState::PickObjectState()
: isDragging_(false), isJustPressed_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
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
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;

	isDragging_ = true;
	isJustPressed_ = true;
	initX_ = prevX_ = evt.x;
	initY_ = prevY_ = evt.y;
}

void PickObjectState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;
	if (!isDragging_) return;

	const bool isJustPressed = isJustPressed_;
	isDragging_ = false;
	isJustPressed_ = false;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		if (!isJustPressed) drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, false);

		WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
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
		//if (evt.x == initX_ && evt.y == initY_) return;
		if (evt.x == prevX_ && evt.y == prevY_) return;

		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, !isJustPressed_, true);
			if (isJustPressed_) isJustPressed_ = false;
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

			WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
			if (vw)
				vw->pickObject(evt.x, evt.y, true);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

//-----------------------------------------------------------------------------------
// 

PickAndDragObjectState::PickAndDragObjectState()
: isDragging_(false), isDraggingObject_(false), isJustPressed_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
	swl::ObjectPickerMgr::getInstance().clearAllPickedObjects();
	swl::ObjectPickerMgr::getInstance().startPicking();
}

PickAndDragObjectState::~PickAndDragObjectState()
{
	swl::ObjectPickerMgr::getInstance().stopPicking();
}

void PickAndDragObjectState::pressMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;

	isDragging_ = true;
	isDraggingObject_ = false;
	isJustPressed_ = true;
	initX_ = prevX_ = evt.x;
	initY_ = prevY_ = evt.y;

	try
	{
		if (swl::ObjectPickerMgr::getInstance().containTemporarilyPickedObject())
			isDraggingObject_ = true;
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void PickAndDragObjectState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;
	if (!isDragging_) return;

	const bool isDraggingObject = isDraggingObject_;
	const bool isJustPressed = isJustPressed_;
	isDragging_ = false;
	isDraggingObject_ = false;
	isJustPressed_ = false;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		if (isDraggingObject && (evt.x != initX_ || evt.y != initY_))
		{
			WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
			if (vw) vw->dragObject(prevX_, prevY_, evt.x, evt.y);
		}
		else
		{
			if (!isJustPressed) drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_,evt.x, evt.y, true, false);

			WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
			if (vw)
			{
				if (evt.x == initX_ && evt.y == initY_)
					vw->pickObject(evt.x, evt.y, false);
				else
					vw->pickObject(initX_, initY_, evt.x, evt.y, false);
			}
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void PickAndDragObjectState::moveMouse(const MouseEvent &evt)
{
	if (isDragging_)
	{
		//if (evt.x == initX_ && evt.y == initY_) return;
		if (evt.x == prevX_ && evt.y == prevY_) return;

		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			if (isDraggingObject_)
			{
				WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
				if (vw) vw->dragObject(prevX_, prevY_, evt.x, evt.y);
			}
			else
			{
				drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, !isJustPressed_, true);
				if (isJustPressed_) isJustPressed_ = false;
			}
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

			WinViewBase *vw = dynamic_cast<WinViewBase *>(&view);
			if (vw)
				vw->pickObject(evt.x, evt.y, true);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

//-----------------------------------------------------------------------------------
// 

HandleLineROIState::HandleLineROIState()
: isDragging_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
}

HandleLineROIState::~HandleLineROIState()
{
}

void HandleLineROIState::pressMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		//if (RegionOfInterestMgr::getInstance().containROI())
		//{
		//	RegionOfInterestMgr::getInstance().clearAllROIs();
		//	view.raiseDrawEvent(false);
		//}

		if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
		{
			isDragging_ = true;
			initX_ = prevX_ = evt.x;
			initY_ = prevY_ = evt.y;
		}
	}
	catch (const std::bad_cast &)
	{std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void HandleLineROIState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;
	if (!isDragging_) return;

	isDragging_ = false;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		//RegionOfInterestMgr::getInstance().clearAllROIs();

		if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
		{
			roi_type roi(roi_type::point_type(
				(roi_type::real_type)initX_, (roi_type::real_type)initY_),
				roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y),
				// TODO [fix] >> color
				true, roi_type::color_type(1.0f, 1.0f, 1.0f, 1.0f)
			);
			RegionOfInterestMgr::getInstance().addROI(roi);
			view.raiseDrawEvent(false);
		}
		else
		{
			//RegionOfInterestMgr::getInstance().clearAllROIs();
			drawLineRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, false);
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void HandleLineROIState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawLineRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, true);
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

HandleRectangleROIState::HandleRectangleROIState()
: isDragging_(false), isJustPressed_(false), initX_(0), initY_(0), prevX_(0), prevY_(0)
{
}

HandleRectangleROIState::~HandleRectangleROIState()
{
}

void HandleRectangleROIState::pressMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		//if (RegionOfInterestMgr::getInstance().containROI())
		//{
		//	RegionOfInterestMgr::getInstance().clearAllROIs();
		//	view.raiseDrawEvent(false);
		//}

		if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
		{
			isDragging_ = true;
			isJustPressed_ = true;
			initX_ = prevX_ = evt.x;
			initY_ = prevY_ = evt.y;
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void HandleRectangleROIState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) != swl::MouseEvent::BT_LEFT) return;
	if (!isDragging_) return;

	const bool isJustPressed = isJustPressed_;
	isDragging_ = false;
	isJustPressed_ = false;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		//RegionOfInterestMgr::getInstance().clearAllROIs();

		if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
		{
			roi_type roi(roi_type::point_type(
				(roi_type::real_type)initX_, (roi_type::real_type)initY_),
				roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y),
				// TODO [fix] >> color
				true, roi_type::color_type(1.0f, 1.0f, 1.0f, 1.0f)
			);
			RegionOfInterestMgr::getInstance().addROI(roi);
			view.raiseDrawEvent(false);
		}
		else
		{
			//RegionOfInterestMgr::getInstance().clearAllROIs();
			if (!isJustPressed) drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, false);
		}
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}
}

void HandleRectangleROIState::moveMouse(const MouseEvent &evt)
{
	if (!isDragging_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawRectangleRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, !isJustPressed_, true);
		if (isJustPressed_) isJustPressed_ = false;
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

HandlePolylineROIState::HandlePolylineROIState()
// TODO [fix] >> color
: isSelectingRegion_(false), initX_(0), initY_(0), prevX_(0), prevY_(0), roi_(true, roi_type::color_type(1.0f, 1.0f, 1.0f, 1.0f))
{
}

HandlePolylineROIState::~HandlePolylineROIState()
{
}

void HandlePolylineROIState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) == swl::MouseEvent::BT_LEFT)
	{
		if (isSelectingRegion_)
		{
			try
			{
				ViewStateMachine &fsm = context<ViewStateMachine>();
				IView &view = fsm.getView();

				if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
				{
					roi_.addPoint(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y));
					drawLineRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, true);

					initX_ = prevX_ = evt.x;
					initY_ = prevY_ = evt.y;
				}
				else
				{
					prevX_ = evt.x;
					prevY_ = evt.y;
					return;
				}
			}
			catch (const std::bad_cast &)
			{
				std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
		else  // start selecting ROI
		{
			initX_ = prevX_ = evt.x;
			initY_ = prevY_ = evt.y;

			try
			{
				ViewStateMachine &fsm = context<ViewStateMachine>();
				IView &view = fsm.getView();

				if (roi_.containPoint())
				{
					roi_.clearAllPoints();
					view.raiseDrawEvent(false);
				}
	
				if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
				{
					roi_.addPoint(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y));
					isSelectingRegion_ = true;
				}
			}
			catch (const std::bad_cast &)
			{
				std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}
	else if ((evt.button | swl::MouseEvent::BT_RIGHT) == swl::MouseEvent::BT_RIGHT)
	{
		isSelectingRegion_ = false;

		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			if (roi_.countPoint() < 2) roi_.clearAllPoints();
			else RegionOfInterestMgr::getInstance().addROI(roi_);
			view.raiseDrawEvent(false);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

void HandlePolylineROIState::moveMouse(const MouseEvent &evt)
{
	if (!isSelectingRegion_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawLineRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, true);
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

HandlePolygonROIState::HandlePolygonROIState()
// TODO [fix] >> color
: isSelectingRegion_(false), initX_(0), initY_(0), prevX_(0), prevY_(0), roi_(true, roi_type::color_type(1.0f, 1.0f, 1.0f, 1.0f))
{
}

HandlePolygonROIState::~HandlePolygonROIState()
{
}

void HandlePolygonROIState::releaseMouse(const MouseEvent &evt)
{
	if ((evt.button | swl::MouseEvent::BT_LEFT) == swl::MouseEvent::BT_LEFT)
	{
		if (isSelectingRegion_)
		{
			try
			{
				ViewStateMachine &fsm = context<ViewStateMachine>();
				IView &view = fsm.getView();

				if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
				{
					roi_.addPoint(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y));
					drawLineRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, true);

					initX_ = prevX_ = evt.x;
					initY_ = prevY_ = evt.y;
				}
				else
				{
					prevX_ = evt.x;
					prevY_ = evt.y;
					return;
				}
			}
			catch (const std::bad_cast &)
			{
				std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
		else  // start selecting ROI
		{
			initX_ = prevX_ = evt.x;
			initY_ = prevY_ = evt.y;

			try
			{
				ViewStateMachine &fsm = context<ViewStateMachine>();
				IView &view = fsm.getView();

				if (roi_.containPoint())
				{
					roi_.clearAllPoints();
					view.raiseDrawEvent(false);
				}

				if (RegionOfInterestMgr::getInstance().isInValidRegion(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y)))
				{
					roi_.addPoint(roi_type::point_type((roi_type::real_type)evt.x, (roi_type::real_type)evt.y));
					isSelectingRegion_ = true;
				}
			}
			catch (const std::bad_cast &)
			{
				std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}
	else if ((evt.button | swl::MouseEvent::BT_RIGHT) == swl::MouseEvent::BT_RIGHT)
	{
		isSelectingRegion_ = false;

		try
		{
			ViewStateMachine &fsm = context<ViewStateMachine>();
			IView &view = fsm.getView();

			if (roi_.countPoint() < 3) roi_.clearAllPoints();
			else RegionOfInterestMgr::getInstance().addROI(roi_);
			view.raiseDrawEvent(false);
		}
		catch (const std::bad_cast &)
		{
			std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

void HandlePolygonROIState::moveMouse(const MouseEvent &evt)
{
	if (!isSelectingRegion_) return;

	try
	{
		ViewStateMachine &fsm = context<ViewStateMachine>();
		IView &view = fsm.getView();

		drawLineRubberBand(view, initX_, initY_, prevX_, prevY_, evt.x, evt.y, true, true);
	}
	catch (const std::bad_cast &)
	{
		std::cerr << "caught bad_cast at " << __LINE__ << " in " << __FILE__ << std::endl;
	}

	prevX_ = evt.x;
	prevY_ = evt.y;
}

}  // namespace swl
