// WinViewTestView.cpp : implementation of the CWinViewTestView class
//

#include "stdafx.h"
#include "WinViewTest.h"

#include "WinViewTestDoc.h"
#include "WinViewTestView.h"

#include "ViewStateMachine.h"
#include "ViewEventHandler.h"
#include "swl/winview/GdiContext.h"
#include "swl/winview/GdiBitmapBufferedContext.h"
#include "swl/winview/GdiplusContext.h"
#include "swl/winview/GdiplusBitmapBufferedContext.h"
#include "swl/winview/GdiPrintContext.h"
#include "swl/winview/WinViewPrintApi.h"
#include "swl/winview/WinViewCaptureApi.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include "swl/view/ViewCamera2.h"
#include "swl/winutil/WinTimer.h"
#include <gdiplus.h>
#include <cmath>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CWinViewTestView

IMPLEMENT_DYNCREATE(CWinViewTestView, CView)

BEGIN_MESSAGE_MAP(CWinViewTestView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_WM_DESTROY()
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_WM_TIMER()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_LBUTTONDBLCLK()
	ON_WM_MBUTTONDOWN()
	ON_WM_MBUTTONUP()
	ON_WM_MBUTTONDBLCLK()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_WM_RBUTTONDBLCLK()
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEWHEEL()
	ON_WM_KEYDOWN()
	ON_WM_KEYUP()
	ON_WM_CHAR()
	ON_COMMAND(ID_VIEWHANDLING_PAN, &CWinViewTestView::OnViewhandlingPan)
	ON_COMMAND(ID_VIEWHANDLING_ROTATE, &CWinViewTestView::OnViewhandlingRotate)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMREGION, &CWinViewTestView::OnViewhandlingZoomregion)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMALL, &CWinViewTestView::OnViewhandlingZoomall)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMIN, &CWinViewTestView::OnViewhandlingZoomin)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMOUT, &CWinViewTestView::OnViewhandlingZoomout)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_PAN, &CWinViewTestView::OnUpdateViewhandlingPan)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ROTATE, &CWinViewTestView::OnUpdateViewhandlingRotate)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMREGION, &CWinViewTestView::OnUpdateViewhandlingZoomregion)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMALL, &CWinViewTestView::OnUpdateViewhandlingZoomall)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMIN, &CWinViewTestView::OnUpdateViewhandlingZoomin)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMOUT, &CWinViewTestView::OnUpdateViewhandlingZoomout)
	ON_COMMAND(ID_PRINTANDCAPTURE_PRINTVIEWUSINGGDI, &CWinViewTestView::OnPrintandcapturePrintviewusinggdi)
	ON_COMMAND(ID_PRINTANDCAPTURE_CAPTUREVIEWUSINGGDI, &CWinViewTestView::OnPrintandcaptureCaptureviewusinggdi)
	ON_COMMAND(ID_PRINTANDCAPTURE_CAPTUREVIEWUSINGGDIPLUS, &CWinViewTestView::OnPrintandcaptureCaptureviewusinggdiplus)
	ON_COMMAND(ID_PRINTANDCAPTURE_COPYTOCLIPBOARD, &CWinViewTestView::OnPrintandcaptureCopytoclipboard)
	ON_COMMAND(ID_EDIT_COPY, &CWinViewTestView::OnEditCopy)
END_MESSAGE_MAP()

// CWinViewTestView construction/destruction

CWinViewTestView::CWinViewTestView()
: viewStateFsm_()
{
}

CWinViewTestView::~CWinViewTestView()
{
}

BOOL CWinViewTestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CWinViewTestView drawing

void CWinViewTestView::OnDraw(CDC* pDC)
{
	CWinViewTestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	if (pDC && pDC->IsPrinting())
	{
		const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));

		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!camera) return;

		const int oldMapMode = pDC->SetMapMode(MM_TEXT);
		const CRect rctPage(0, 0, pDC->GetDeviceCaps(HORZRES), pDC->GetDeviceCaps(VERTRES));

		swl::GdiPrintContext printContext(pDC->GetSafeHdc(), rctPage);
		const std::auto_ptr<camera_type> printCamera(camera->cloneCamera());
		if (printCamera.get() && printContext.isActivated())
		{
			initializeView();
			printCamera->setViewRegion(camera->getCurrentViewRegion());
			printCamera->setViewport(rctPage.left, rctPage.top, rctPage.right, rctPage.bottom);
			renderScene(printContext, *printCamera);
		}

		pDC->SetMapMode(oldMapMode);

		DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
	}
	else
	{
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!camera) return;

		// using a locally-created context
		if (useLocallyCreatedContext_)
		{
			CRect rect;
			GetClientRect(&rect);

			boost::scoped_ptr<context_type> context;
			if (1 == drawMode_)
				context.reset(new swl::GdiContext(GetSafeHwnd()));
			else if (2 == drawMode_)
				context.reset(new swl::GdiBitmapBufferedContext(GetSafeHwnd(), rect));
			else if (3 == drawMode_)
				context.reset(new swl::GdiplusContext(GetSafeHwnd()));
			else if (4 == drawMode_)
				context.reset(new swl::GdiplusBitmapBufferedContext(GetSafeHwnd(), rect));

			if (context.get() && context->isActivated())
			{
				initializeView();
				camera->setViewport(0, 0, rect.Width(), rect.Height());
				renderScene(*context, *camera);
			}
		}
		else
		{
			const boost::shared_ptr<context_type> &context = topContext();
			if (context.get() && context->isActivated())
				renderScene(*context, *camera);
		}
	}
}


// CWinViewTestView printing

BOOL CWinViewTestView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CWinViewTestView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CWinViewTestView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CWinViewTestView diagnostics

#ifdef _DEBUG
void CWinViewTestView::AssertValid() const
{
	CView::AssertValid();
}

void CWinViewTestView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CWinViewTestDoc* CWinViewTestView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CWinViewTestDoc)));
	return (CWinViewTestDoc*)m_pDocument;
}
#endif //_DEBUG


// CWinViewTestView message handlers

void CWinViewTestView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	//
	idx_ = 0;
	timeInterval_ = 50;
	SetTimer(1, timeInterval_, NULL);

	for (int i = 0; i < 5000; ++i)
	{
		const double x = (double)i * timeInterval_ / 10000.0;
		const double y = std::sin(x) * 100.0 + 100.0;
		data1_.push_back(std::make_pair(i, (int)std::floor(y + 0.5)));
	}

	drawMode_ = 4;  // [1, 4]
	useLocallyCreatedContext_ = false;

	CRect rect;
	GetClientRect(&rect);

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
/*
	viewController_.addMousePressHandler(swl::MousePressHandler());
	viewController_.addMouseReleaseHandler(swl::MouseReleaseHandler());
	viewController_.addMouseMoveHandler(swl::MouseMoveHandler());
	viewController_.addMouseWheelHandler(swl::MouseWheelHandler());
	viewController_.addMouseClickHandler(swl::MouseClickHandler());
	viewController_.addMouseDoubleClickHandler(swl::MouseDoubleClickHandler());
	viewController_.addKeyPressHandler(swl::KeyPressHandler());
	viewController_.addKeyReleaseHandler(swl::KeyReleaseHandler());
	viewController_.addKeyHitHandler(swl::KeyHitHandler());
*/
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine
	
	// create a context
	if (1 == drawMode_)
		pushContext(boost::shared_ptr<context_type>(new swl::GdiContext(GetSafeHwnd(), false)));
	else if (2 == drawMode_)
		pushContext(boost::shared_ptr<context_type>(new swl::GdiBitmapBufferedContext(GetSafeHwnd(), rect, false)));
	else if (3 == drawMode_)
		pushContext(boost::shared_ptr<context_type>(new swl::GdiplusContext(GetSafeHwnd(), false)));
	else if (4 == drawMode_)
		pushContext(boost::shared_ptr<context_type>(new swl::GdiplusBitmapBufferedContext(GetSafeHwnd(), rect, false)));

	// create a camera
	pushCamera(boost::shared_ptr<camera_type>(new swl::ViewCamera2()));

	const boost::shared_ptr<context_type> &viewContext = topContext();
	const boost::shared_ptr<camera_type> &viewCamera = topCamera();

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state

	if (!useLocallyCreatedContext_ && NULL == viewStateFsm_.get() && viewContext.get() && viewCamera.get())
	{
		viewStateFsm_.reset(new swl::ViewStateMachine(*this, *viewContext, *viewCamera));
		if (viewStateFsm_.get()) viewStateFsm_->initiate();
	}

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	// initialize a view
	if (viewContext.get())
	{
		// guard the context
		context_type::guard_type guard(*viewContext);

		// set the view
		initializeView();

		// set the camera
		if (viewCamera.get())
		{
			//viewCamera->setViewBound(-500, -500, 1500, 1500);
			viewCamera->setViewBound(0, 0, rect.Width(), rect.Height());
			viewCamera->setViewport(0, 0, rect.Width(), rect.Height());
		}

		raiseDrawEvent(false);
	}

	// using a locally-created context
	if (useLocallyCreatedContext_)
		popContext();
}

void CWinViewTestView::OnDestroy()
{
	CView::OnDestroy();

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	popContext();
	popCamera();
}

void CWinViewTestView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	// using a locally-created context
	if (useLocallyCreatedContext_)
		raiseDrawEvent(false);
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (context.get())
		{
			if (context->isOffScreenUsed())
			{
				//context_type::guard_type guard(*context);
				context->swapBuffer();
			}
			else raiseDrawEvent(true);
		}
	}

	// Do not call CView::OnPaint() for painting messages
}

void CWinViewTestView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	if (cx <= 0 || cy <= 0) return;
	resizeView(0, 0, cx, cy);
}

void CWinViewTestView::OnTimer(UINT_PTR nIDEvent)
{
	const double x = (double)idx_ * timeInterval_ / 1000.0;
	const double y = std::cos(x) * 100.0 + 100.0;
	data2_.push_back(std::make_pair(idx_, (int)std::floor(y + 0.5)));

	++idx_;

	raiseDrawEvent(useLocallyCreatedContext_ ? false : true);

	CView::OnTimer(nIDEvent);
}

//-------------------------------------------------------------------------
// This code is required for SWL.WinView: basic routine

bool CWinViewTestView::raiseDrawEvent(const bool isContextActivated)
{
	if (isContextActivated)
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (!context.get() || context->isDrawing())
			return false;

		context_type::guard_type guard(*context);
		OnDraw(0L);
	}
	else OnDraw(0L);

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WinView: basic routine

bool CWinViewTestView::initializeView()
{
	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WinView: basic routine

bool CWinViewTestView::resizeView(const int x1, const int y1, const int x2, const int y2)
{
	const boost::shared_ptr<context_type> &context = topContext();
	if (context.get() && context->resize(x1, y1, x2, y2))
	{
		context_type::guard_type guard(*context);
		initializeView();
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (camera.get()) camera->setViewport(x1, y1, x2, y2);	
		raiseDrawEvent(false);

		return true;
	}
	else return false;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WinView: basic routine

bool CWinViewTestView::doPrepareRendering(const context_type &context, const camera_type &camera)
{
	// clear the background
	try
	{
		const HDC *dc = boost::any_cast<const HDC *>(context.getNativeContext());
		if (dc)
		{
			CDC *pDC = CDC::FromHandle(*dc);
/*
			const swl::Region2<double> &viewRgn = context.getViewingRegion();
			//const CRect rect((int)std::floor(viewRgn.left + 0.5), (int)std::floor(viewRgn.bottom + 0.5), (int)std::floor(viewRgn.right + 0.5), (int)std::floor(viewRgn.top + 0.5));
			const CRect rect(0, 0, (int)std::floor(viewRgn.getWidth() + 0.5), (int)std::floor(viewRgn.getHeight() + 0.5));
/*
			CRect rect;
			GetClientRect(&rect);
*/
			const swl::Region2<int> &viewport = camera.getViewport();
			const CRect rect(0, 0, viewport.getWidth(), viewport.getHeight());
			pDC->FillRect(rect, &CBrush(RGB(255, 255, 255)));
		}
	}
	catch (const boost::bad_any_cast &)
	{
	}

	try
	{
		Gdiplus::Graphics *graphics = boost::any_cast<Gdiplus::Graphics *>(context.getNativeContext());
		if (graphics)
			graphics->Clear(Gdiplus::Color(255, 255, 255, 255));
	}
	catch (const boost::bad_any_cast &)
	{
	}

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WinView: basic routine

bool CWinViewTestView::doRenderStockScene(const context_type &context, const camera_type &camera)
{
    return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WinView: basic routine

bool CWinViewTestView::doRenderScene(const context_type &context, const camera_type &camera)
{
	const int lineWidth1 = 6;
	const int lineWidth2 = 4;
	const int lineWidth3 = 2;

	if (false)  //if (context.isOffScreenUsed())
	{
		try
		{
			const HDC *dc = boost::any_cast<const HDC *>(context.getNativeContext());
			if (dc)
			{
				CDC *pDC = CDC::FromHandle(*dc);

				// draw contents
				{
					CPen pen(PS_SOLID, lineWidth1, RGB(255, 0, 255));
					CPen *oldPen = pDC->SelectObject(&pen);
					CBrush brush(RGB(240, 240, 240));
					CBrush *oldBrush = pDC->SelectObject(&brush);
					const swl::Region2<double> bound = camera.getViewBound();
					pDC->Rectangle(bound.left, bound.bottom, bound.right, bound.top);
					pDC->SelectObject(oldBrush);
					pDC->SelectObject(oldPen);
				}
/*
				const swl::Region2<double> bound = camera.getViewBound();
				CRgn rgn;
				const BOOL ret = rgn.CreateRectRgn(bound.left, bound.bottom, bound.right, bound.top);
				//rgn.SetRectRgn(bound.left, bound.bottom, bound.right, bound.top);
				const int ret1 = pDC->SelectClipRgn(&rgn);
				const int ret2 = pDC->GetClipBox(&rc);
				//const UINT ret2 = pDC->GetBoundsRect(&rc, 0);
*/
				{
					CPen pen(PS_SOLID, lineWidth1, RGB(255, 0, 0));
					CPen *oldPen = pDC->SelectObject(&pen);
					pDC->MoveTo(100, 200);
					pDC->LineTo(300, 300);
					pDC->MoveTo(-500, -500);
					pDC->LineTo(1500, 1500);
				}

				if (data1_.size() > 1)
				{
					CPen pen(PS_SOLID, lineWidth2, RGB(0, 255, 0));
					CPen *oldPen = pDC->SelectObject(&pen);
					data_type::iterator it = data1_.begin();
					pDC->MoveTo(it->first, it->second);
					for (++it; it != data1_.end(); ++it)
						pDC->LineTo(it->first, it->second);
					pDC->SelectObject(oldPen);
				}

				if (data2_.size() > 1)
				{
					CPen pen(PS_SOLID, lineWidth3, RGB(0, 0, 255));
					CPen *oldPen = pDC->SelectObject(&pen);
					data_type::iterator it = data2_.begin();
					pDC->MoveTo(it->first, it->second);
					for (++it; it != data2_.end(); ++it)
						pDC->LineTo(it->first, it->second);
					pDC->SelectObject(oldPen);
				}
			}
		}
		catch (const boost::bad_any_cast &)
		{
		}

		try
		{
			Gdiplus::Graphics *graphics = boost::any_cast<Gdiplus::Graphics *>(context.getNativeContext());
			if (graphics)
			{
				// draw contents
				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 255, 0, 255), lineWidth1);
					Gdiplus::SolidBrush brush(Gdiplus::Color(255, 240, 240, 240));
					const swl::Region2<double> bound = camera.getViewBound();
					graphics->DrawRectangle(&pen, (Gdiplus::REAL)bound.left, (Gdiplus::REAL)bound.bottom, (Gdiplus::REAL)bound.getWidth(), (Gdiplus::REAL)bound.getHeight());
					graphics->FillRectangle(&brush, (Gdiplus::REAL)bound.left, (Gdiplus::REAL)bound.bottom, (Gdiplus::REAL)bound.getWidth(), (Gdiplus::REAL)bound.getHeight());
				}

				const swl::Region2<double> rgn = camera.getCurrentViewRegion();

				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 255, 0, 0), lineWidth1);
					graphics->DrawLine(&pen, 100, 200, 300, 400);
					graphics->DrawLine(&pen, -500, -500, 1500, 1500);
				}

				if (data1_.size() > 1)
				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 0, 255, 0), lineWidth2);
					data_type::iterator prevIt = data1_.begin();
					data_type::iterator it = data1_.begin();
					for (++it; it != data1_.end(); ++prevIt, ++it)
						graphics->DrawLine(&pen, (Gdiplus::REAL)prevIt->first, (Gdiplus::REAL)prevIt->second, (Gdiplus::REAL)it->first, (Gdiplus::REAL)it->second);
				}

				if (data2_.size() > 1)
				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 0, 0, 255), lineWidth3);
					data_type::iterator prevIt = data2_.begin();
					data_type::iterator it = data2_.begin();
					for (++it; it != data2_.end(); ++prevIt, ++it)
						graphics->DrawLine(&pen, (Gdiplus::REAL)prevIt->first, (Gdiplus::REAL)prevIt->second, (Gdiplus::REAL)it->first, (Gdiplus::REAL)it->second);
				}
			}
		}
		catch (const boost::bad_any_cast &)
		{
		}
	}
	else
	{
		try
		{
			const HDC *dc = boost::any_cast<const HDC *>(context.getNativeContext());
			if (dc)
			{
				CDC *pDC = CDC::FromHandle(*dc);

				// draw contents
				int vx, vy;

				{
					CPen pen(PS_SOLID, lineWidth1, RGB(255, 0, 255));
					CPen *oldPen = pDC->SelectObject(&pen);
					const swl::Region2<double> bound = camera.getViewBound();
					int vx0, vy0;
					camera.mapCanvasToWindow(bound.left, bound.bottom, vx0, vy0);
					camera.mapCanvasToWindow(bound.right, bound.top, vx, vy);
					pDC->Rectangle(vx0, vy0, vx, vy);
					pDC->SelectObject(oldPen);
				}

				{
					CPen pen(PS_SOLID, lineWidth1, RGB(255, 0, 0));
					CPen *oldPen = pDC->SelectObject(&pen);
					camera.mapCanvasToWindow(100, 200, vx, vy);
					pDC->MoveTo(vx, vy);
					camera.mapCanvasToWindow(300, 400, vx, vy);
					pDC->LineTo(vx, vy);
					pDC->SelectObject(oldPen);
				}

				if (data1_.size() > 1)
				{
					CPen pen(PS_SOLID, lineWidth2, RGB(0, 255, 0));
					CPen *oldPen = pDC->SelectObject(&pen);
					data_type::iterator it = data1_.begin();
					camera.mapCanvasToWindow(it->first, it->second, vx, vy);
					pDC->MoveTo(vx, vy);
					for (++it; it != data1_.end(); ++it)
					{
						camera.mapCanvasToWindow(it->first, it->second, vx, vy);
						pDC->LineTo(vx, vy);
					}
					pDC->SelectObject(oldPen);
				}

				if (data2_.size() > 1)
				{
					CPen pen(PS_SOLID, lineWidth3, RGB(0, 0, 255));
					CPen *oldPen = pDC->SelectObject(&pen);
					data_type::iterator it = data2_.begin();
					camera.mapCanvasToWindow(it->first, it->second, vx, vy);
					pDC->MoveTo(vx, vy);
					for (++it; it != data2_.end(); ++it)
					{
						camera.mapCanvasToWindow(it->first, it->second, vx, vy);
						pDC->LineTo(vx, vy);
					}
					pDC->SelectObject(oldPen);
				}
			}
		}
		catch (const boost::bad_any_cast &)
		{
		}

		try
		{
			Gdiplus::Graphics *graphics = boost::any_cast<Gdiplus::Graphics *>(context.getNativeContext());
			if (graphics)
			{
				// draw contents
				int vx1, vy1, vx2, vy2;

				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 255, 0, 255), lineWidth1);
					const swl::Region2<double> bound = camera.getViewBound();
					camera.mapCanvasToWindow(bound.left, bound.bottom, vx1, vy1);
					camera.mapCanvasToWindow(bound.right, bound.top, vx2, vy2);
					graphics->DrawRectangle(&pen, vx1, vy1, vx2 - vx1, vy2 - vy1);
				}

				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 255, 0, 0), lineWidth1);
					camera.mapCanvasToWindow(100, 200, vx1, vy1);
					camera.mapCanvasToWindow(300, 400, vx2, vy2);
					graphics->DrawLine(&pen, vx1, vy1, vx2, vy2);
				}

				if (data1_.size() > 1)
				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 0, 255, 0), lineWidth2);
					data_type::iterator prevIt = data1_.begin();
					data_type::iterator it = data1_.begin();
					for (++it; it != data1_.end(); ++prevIt, ++it)
					{
						camera.mapCanvasToWindow(prevIt->first, prevIt->second, vx1, vy1);
						camera.mapCanvasToWindow(it->first, it->second, vx2, vy2);
						graphics->DrawLine(&pen, vx1, vy1, vx2, vy2);
					}
				}

				if (data2_.size() > 1)
				{
					Gdiplus::Pen pen(Gdiplus::Color(255, 0, 0, 255), lineWidth3);
					data_type::iterator prevIt = data2_.begin();
					data_type::iterator it = data2_.begin();
					for (++it; it != data2_.end(); ++prevIt, ++it)
					{
						camera.mapCanvasToWindow(prevIt->first, prevIt->second, vx1, vy1);
						camera.mapCanvasToWindow(it->first, it->second, vx2, vy2);
						graphics->DrawLine(&pen, vx1, vy1, vx2, vy2);
					}
				}
			}
		}
		catch (const boost::bad_any_cast &)
		{
		}
	}

    return true;
}

void CWinViewTestView::OnLButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDown(nFlags, point);
}

void CWinViewTestView::OnLButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonUp(nFlags, point);
}

void CWinViewTestView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDblClk(nFlags, point);
}

void CWinViewTestView::OnMButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDown(nFlags, point);
}

void CWinViewTestView::OnMButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonUp(nFlags, point);
}

void CWinViewTestView::OnMButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDblClk(nFlags, point);
}

void CWinViewTestView::OnRButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDown(nFlags, point);
}

void CWinViewTestView::OnRButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonUp(nFlags, point);
}

void CWinViewTestView::OnRButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDblClk(nFlags, point);
}

void CWinViewTestView::OnMouseMove(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	const swl::MouseEvent::EButton btn = (swl::MouseEvent::EButton)(
		((nFlags & MK_LBUTTON) == MK_LBUTTON ? swl::MouseEvent::BT_LEFT : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_MBUTTON) == MK_MBUTTON ? swl::MouseEvent::BT_MIDDLE : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_RBUTTON) == MK_RBUTTON ? swl::MouseEvent::BT_RIGHT : swl::MouseEvent::BT_NONE)
	);
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.moveMouse(swl::MouseEvent(point.x, point.y, btn, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->moveMouse(swl::MouseEvent(point.x, point.y, btn, ckey));

	CView::OnMouseMove(nFlags, point);
}

BOOL CWinViewTestView::OnMouseWheel(UINT nFlags, short zDelta, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	const swl::MouseEvent::EButton btn = (swl::MouseEvent::EButton)(
		((nFlags & MK_LBUTTON) == MK_LBUTTON ? swl::MouseEvent::BT_LEFT : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_MBUTTON) == MK_MBUTTON ? swl::MouseEvent::BT_MIDDLE : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_RBUTTON) == MK_RBUTTON ? swl::MouseEvent::BT_RIGHT : swl::MouseEvent::BT_NONE)
	);
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.wheelMouse(swl::MouseEvent(point.x, point.y, btn, ckey, swl::MouseEvent::SC_VERTICAL, zDelta / WHEEL_DELTA));
	if (viewStateFsm_.get()) viewStateFsm_->wheelMouse(swl::MouseEvent(point.x, point.y, btn, ckey, swl::MouseEvent::SC_VERTICAL, zDelta / WHEEL_DELTA));

	return CView::OnMouseWheel(nFlags, zDelta, point);
}

void CWinViewTestView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	//viewController_.pressKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_.get()) viewStateFsm_->pressKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyDown(nChar, nRepCnt, nFlags);
}

void CWinViewTestView::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_.get()) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyUp(nChar, nRepCnt, nFlags);
}

void CWinViewTestView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling
	// TODO [check] >>
	const swl::KeyEvent::EControlKey ckey = ((nFlags >> 28) & 0x01) == 0x01 ? swl::KeyEvent::CK_ALT : swl::KeyEvent::CK_NONE;
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));

	CView::OnChar(nChar, nRepCnt, nFlags);
}

void CWinViewTestView::OnViewhandlingPan()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtPan());
}

void CWinViewTestView::OnViewhandlingRotate()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtRotate());
}

void CWinViewTestView::OnViewhandlingZoomregion()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomRegion());
}

void CWinViewTestView::OnViewhandlingZoomall()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomAll());
}

void CWinViewTestView::OnViewhandlingZoomin()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomIn());
}

void CWinViewTestView::OnViewhandlingZoomout()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomOut());
}

void CWinViewTestView::OnUpdateViewhandlingPan(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::PanState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWinViewTestView::OnUpdateViewhandlingRotate(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	pCmdUI->Enable(FALSE);

	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::RotateState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWinViewTestView::OnUpdateViewhandlingZoomregion(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomRegionState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWinViewTestView::OnUpdateViewhandlingZoomall(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomAllState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWinViewTestView::OnUpdateViewhandlingZoomin(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomInState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWinViewTestView::OnUpdateViewhandlingZoomout(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomOutState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWinViewTestView::OnPrintandcapturePrintviewusinggdi()
{
	// initialize a PRINTDLG structure
	PRINTDLG pd;
	memset(&pd, 0, sizeof(pd));
	pd.lStructSize = sizeof(pd);
	pd.hwndOwner = GetSafeHwnd();
	pd.Flags = PD_RETURNDC | PD_DISABLEPRINTTOFILE;
	pd.hInstance = NULL;
	if (!PrintDlg(&pd)) return;
	if (!pd.hDC) return;

	//
	const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));

	// each logical unit is mapped to one device pixel. Positive x is to the right. positive y is down.
	SetMapMode(pd.hDC, MM_TEXT);

	DOCINFO di;
	di.cbSize = sizeof(DOCINFO);
	di.lpszDocName = _T("GDI Print");
	di.lpszOutput = NULL;

	// start the print job
	StartDoc(pd.hDC, &di);
	StartPage(pd.hDC);

	//
	if (!swl::printWinViewUsingGdi(*this, pd.hDC))
		AfxMessageBox(_T("fail to print a view"), MB_OK | MB_ICONSTOP);

	// end the print job
	EndPage(pd.hDC);
	EndDoc(pd.hDC);
	DeleteDC(pd.hDC);

	DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
}

void CWinViewTestView::OnPrintandcaptureCaptureviewusinggdi()
{
	CFileDialog dlg(FALSE, _T("bmp"), _T("*.bmp"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("BMP Files (*.bmp)|*.bmp||"), NULL);
	dlg.m_ofn.lpstrTitle = _T("Capture View As");
	if (dlg.DoModal() == IDOK)
	{
		const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));

#if defined(_UNICODE) || defined(UNICODE)
		const std::wstring filePathName((wchar_t *)(LPCTSTR)dlg.GetPathName());
#else
		const std::string filePathName((char *)(LPCTSTR)dlg.GetPathName());
#endif
		if (!swl::captureWinViewUsingGdi(filePathName, *this, GetSafeHwnd()))
			AfxMessageBox(_T("fail to capture a view"), MB_OK | MB_ICONSTOP);

		DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
	}
}

void CWinViewTestView::OnPrintandcaptureCaptureviewusinggdiplus()
{
	CFileDialog dlg(FALSE, _T("bmp"), _T("*.bmp"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("BMP Files (*.bmp)|*.bmp|JPEG Files (*.jpg)|*.jpg|GIF Files (*.gif)|*.gif|PNG Files (*.png)|*.png|TIFF Files (*.tif)|*.tif|EMF Files (*.emf)|*.emf|WMF Files (*.wmf)|*.wmf||"), NULL);
	//CFileDialog dlg(FALSE, _T("bmp"), _T("*.bmp"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("BMP Files (*.bmp)|*.bmp|JPEG Files (*.jpg)|*.jpg|GIF Files (*.gif)|*.gif|PNG Files (*.png)|*.png|TIFF Files (*.tif)||"), NULL);
	dlg.m_ofn.lpstrTitle = _T("Capture View As");
	if (dlg.DoModal() == IDOK)
	{
		const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));

#if defined(_UNICODE) || defined(UNICODE)
		const std::wstring filePathName((wchar_t *)(LPCTSTR)dlg.GetPathName());
		const std::wstring fileExtName((wchar_t *)(LPCTSTR)dlg.GetFileExt());
#else
		const std::string filePathName((char *)(LPCTSTR)dlg.GetPathName());
		const std::string fileExtName((char *)(LPCTSTR)dlg.GetFileExt());
#endif
		if (!swl::captureWinViewUsingGdiplus(filePathName, fileExtName, *this, GetSafeHwnd()))
			AfxMessageBox(_T("fail to capture a view"), MB_OK | MB_ICONSTOP);

		DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
	}
}

void CWinViewTestView::OnPrintandcaptureCopytoclipboard()
{
	CClientDC dc(this);

	CDC memDC;
	memDC.CreateCompatibleDC(&dc);

	CRect rect;
	GetWindowRect(&rect);

	CBitmap bitmap;
	bitmap.CreateCompatibleBitmap(&dc, rect.Width(), rect.Height());

	CBitmap *oldBitmap = memDC.SelectObject(&bitmap);
	memDC.BitBlt(0, 0, rect.Width(), rect.Height(), &dc, 0, 0, SRCCOPY);

	// clipboard
	if (OpenClipboard())
	{
		EmptyClipboard();
		SetClipboardData(CF_BITMAP, bitmap.GetSafeHandle());
		CloseClipboard();
	}

	memDC.SelectObject(oldBitmap);
	bitmap.Detach();
}

void CWinViewTestView::OnEditCopy()
{
	OnPrintandcaptureCopytoclipboard();
}
