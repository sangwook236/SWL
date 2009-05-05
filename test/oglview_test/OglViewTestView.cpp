// OglViewTestView.cpp : implementation of the COglViewTestView class
//

#include "stdafx.h"
#include "OglViewTest.h"

#include "OglViewTestDoc.h"
#include "OglViewTestView.h"

#include "ViewStateMachine.h"
#include "ViewEventHandler.h"
#include "swl/winview/WglDoubleBufferedContext.h"
#include "swl/winview/WglBitmapBufferedContext.h"
#include "swl/oglview/OglCamera.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include <GL/glut.h>
#include <iostream>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// COglViewTestView

IMPLEMENT_DYNCREATE(COglViewTestView, CView)

BEGIN_MESSAGE_MAP(COglViewTestView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_WM_SIZE()
	ON_WM_PAINT()
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
	ON_COMMAND(ID_VIEWSTATE_PAN, &COglViewTestView::OnViewstatePan)
	ON_COMMAND(ID_VIEWSTATE_ROTATE, &COglViewTestView::OnViewstateRotate)
	ON_COMMAND(ID_VIEWSTATE_ZOOMREGION, &COglViewTestView::OnViewstateZoomregion)
	ON_COMMAND(ID_VIEWSTATE_ZOOMALL, &COglViewTestView::OnViewstateZoomall)
	ON_COMMAND(ID_VIEWSTATE_ZOOMIN, &COglViewTestView::OnViewstateZoomin)
	ON_COMMAND(ID_VIEWSTATE_ZOOMOUT, &COglViewTestView::OnViewstateZoomout)
	ON_UPDATE_COMMAND_UI(ID_VIEWSTATE_PAN, &COglViewTestView::OnUpdateViewstatePan)
	ON_UPDATE_COMMAND_UI(ID_VIEWSTATE_ROTATE, &COglViewTestView::OnUpdateViewstateRotate)
	ON_UPDATE_COMMAND_UI(ID_VIEWSTATE_ZOOMREGION, &COglViewTestView::OnUpdateViewstateZoomregion)
	ON_UPDATE_COMMAND_UI(ID_VIEWSTATE_ZOOMALL, &COglViewTestView::OnUpdateViewstateZoomall)
	ON_UPDATE_COMMAND_UI(ID_VIEWSTATE_ZOOMIN, &COglViewTestView::OnUpdateViewstateZoomin)
	ON_UPDATE_COMMAND_UI(ID_VIEWSTATE_ZOOMOUT, &COglViewTestView::OnUpdateViewstateZoomout)
END_MESSAGE_MAP()

// COglViewTestView construction/destruction

COglViewTestView::COglViewTestView()
: viewContext_(), viewCamera_(), viewStateFsm_()
{
}

COglViewTestView::~COglViewTestView()
{
}

BOOL COglViewTestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// COglViewTestView drawing

void COglViewTestView::OnDraw(CDC* pDC)
{
	COglViewTestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: basic routine

	if (pDC && pDC->IsPrinting())
	{
		// FIXME [add] >>
	}
	else
	{
		if (viewContext_.get() && viewCamera_.get() && viewContext_->isActivated())
			renderScene(*viewContext_, *viewCamera_);
	}
}


// COglViewTestView printing

BOOL COglViewTestView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void COglViewTestView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void COglViewTestView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// COglViewTestView diagnostics

#ifdef _DEBUG
void COglViewTestView::AssertValid() const
{
	CView::AssertValid();
}

void COglViewTestView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

COglViewTestDoc* COglViewTestView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(COglViewTestDoc)));
	return (COglViewTestDoc*)m_pDocument;
}
#endif //_DEBUG


// COglViewTestView message handlers

void COglViewTestView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	//
	CRect rect;
	GetClientRect(&rect);

	const int drawMode = 0x02;

	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling

	viewController_.addMousePressHandler(swl::MousePressHandler());
	viewController_.addMouseReleaseHandler(swl::MouseReleaseHandler());
	viewController_.addMouseMoveHandler(swl::MouseMoveHandler());
	viewController_.addMouseWheelHandler(swl::MouseWheelHandler());
	viewController_.addMouseClickHandler(swl::MouseClickHandler());
	viewController_.addMouseDoubleClickHandler(swl::MouseDoubleClickHandler());
	viewController_.addKeyPressHandler(swl::KeyPressHandler());
	viewController_.addKeyReleaseHandler(swl::KeyReleaseHandler());
	viewController_.addKeyHitHandler(swl::KeyHitHandler());

	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: basic routine

	// create a context
	if (NULL == viewContext_.get())
	{
		if ((0x01 & drawMode) == 0x01)
			viewContext_.reset(new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect, false));
		if ((0x02 & drawMode) == 0x02)
			viewContext_.reset(new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect, false));
	}

	// create a camera
	if (NULL == viewCamera_.get())
		viewCamera_.reset(new swl::OglCamera());

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state

	if (NULL == viewStateFsm_.get() && NULL != viewContext_.get() && NULL != viewCamera_.get())
	{
		viewStateFsm_.reset(new swl::ViewStateMachine(*this, *viewContext_, *viewCamera_));
		if (viewStateFsm_.get()) viewStateFsm_->initiate();
	}

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	// initialize a view
	if (viewContext_.get())
	{
		// activate the context
		viewContext_->activate();

		// set the view
		initializeView();

		// set the camera
		if (viewCamera_.get())
		{
			//viewCamera_->setViewBound(-1600.0, -1100.0, 2400.0, 2900.0, 1.0, 20000.0);
			viewCamera_->setViewBound(-1000.0, -1000.0, 1000.0, 1000.0, 4000.0, 12000.0);
			//viewCamera_->setViewBound(-50.0, -50.0, 50.0, 50.0, 1.0, 2000.0);

			viewCamera_->setViewport(0, 0, rect.Width(), rect.Height());
			viewCamera_->setEyePosition(1000.0, 1000.0, 1000.0, false);
			viewCamera_->setEyeDistance(8000.0, false);
			viewCamera_->setObjectPosition(0.0, 0.0, 0.0);
			//viewCamera_->setEyeDistance(1000.0, false);
			//viewCamera_->setObjectPosition(110.0, 110.0, 150.0);
		}

		raiseDrawEvent(false);

		// de-activate the context
		viewContext_->deactivate();
	}
}

void COglViewTestView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: basic routine

	if (viewContext_.get())
	{
		if (viewContext_->isOffScreenUsed())
		{
			//viewContext_->activate();
			viewContext_->swapBuffer();
			//viewContext_->deactivate();
		}
		else raiseDrawEvent(true);
	}

	// Do not call CView::OnPaint() for painting messages
}

void COglViewTestView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: basic routine

	if (cx <= 0 || cy <= 0) return;
	resizeView(0, 0, cx, cy);
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

bool COglViewTestView::raiseDrawEvent(const bool isContextActivated)
{
	if (NULL == viewContext_.get() || viewContext_->isDrawing())
		return false;

	if (isContextActivated)
	{
		viewContext_->activate();
		OnDraw(0L);
		viewContext_->deactivate();
	}
	else OnDraw(0L);

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

bool COglViewTestView::initializeView()
{
	// Can we put this in the constructor?
	// specify black(0.0f, 0.0f, 0.0f, 0.0f) or white(1.0f, 1.0f, 1.0f, 1.0f) as clear color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// specify the back of the buffer as clear depth
    glClearDepth(1.0f);
	// enable depth testing
    glEnable(GL_DEPTH_TEST);

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

bool COglViewTestView::resizeView(const int x1, const int y1, const int x2, const int y2)
{
	if (viewContext_.get() && viewContext_->resize(x1, y1, x2, y2))
	{
		viewContext_->activate();
		initializeView();
		if (viewCamera_.get()) viewCamera_->setViewport(x1, y1, x2, y2);
		raiseDrawEvent(false);
		viewContext_->deactivate();

		return true;
	}
	else return false;
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

bool COglViewTestView::doPrepareRendering()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

bool COglViewTestView::doRenderStockScene()
{
    return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

bool COglViewTestView::doRenderScene()
{
	glPushMatrix();
		//glLoadIdentity();
		glTranslatef(-250.0f, 250.0f, -250.0f);
		glColor3f(1.0f, 0.0f, 0.0f);
		glutWireSphere(500.0, 20, 20);
		//glutSolidSphere(500.0, 20, 20);
	glPopMatrix();

	glPushMatrix();
		//glLoadIdentity();
		glTranslatef(250.0f, -250.0f, 250.0f);
		glColor3f(0.5f, 0.5f, 1.0f);
		glutWireCube(500.0);
		//glutSolidCube(500.0);
	glPopMatrix();

    return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.OglView: basic routine

void COglViewTestView::renderScene(swl::WglContextBase &context, swl::ViewCamera3 &camera)
{
#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		if (GL_NO_ERROR != glErrorCode)
			std::cerr << "OpenGL error at " << __LINE__ << " in " << __FILE__ << ": " << gluErrorString(glErrorCode) << std::endl;
	}
#endif

	int oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);

	{
		glPushMatrix();
			//
			glLoadIdentity();
			camera.lookAt();

			//
			glPushMatrix();
				doPrepareRendering();
			glPopMatrix();

			glPushMatrix();
				doRenderStockScene();
			glPopMatrix();

			doRenderScene();
		glPopMatrix();
	}

	glFlush();

	// swap buffers
	context.swapBuffer();

	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		if (GL_NO_ERROR != glErrorCode)
			std::cerr << "OpenGL error at " << __LINE__ << " in " << __FILE__ << ": " << gluErrorString(glErrorCode) << std::endl;
	}
#endif
}

void COglViewTestView::OnLButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDown(nFlags, point);
}

void COglViewTestView::OnLButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonUp(nFlags, point);
}

void COglViewTestView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDblClk(nFlags, point);
}

void COglViewTestView::OnMButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDown(nFlags, point);
}

void COglViewTestView::OnMButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonUp(nFlags, point);
}

void COglViewTestView::OnMButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDblClk(nFlags, point);
}

void COglViewTestView::OnRButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDown(nFlags, point);
}

void COglViewTestView::OnRButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonUp(nFlags, point);
}

void COglViewTestView::OnRButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDblClk(nFlags, point);
}

void COglViewTestView::OnMouseMove(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.moveMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->moveMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnMouseMove(nFlags, point);
}

BOOL COglViewTestView::OnMouseWheel(UINT nFlags, short zDelta, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags | MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags | MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.wheelMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey, swl::MouseEvent::SC_HORIZONTAL, zDelta));
	if (viewStateFsm_.get()) viewStateFsm_->wheelMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey, swl::MouseEvent::SC_VERTICAL, zDelta));

	return CView::OnMouseWheel(nFlags, zDelta, point);
}

void COglViewTestView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	//viewController_.pressKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_.get()) viewStateFsm_->pressKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyDown(nChar, nRepCnt, nFlags);
}

void COglViewTestView::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_.get()) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyUp(nChar, nRepCnt, nFlags);
}

void COglViewTestView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: event handling
	const swl::KeyEvent::EControlKey ckey = ((nFlags >> 28) & 0x01) == 0x01 ? swl::KeyEvent::CK_ALT : swl::KeyEvent::CK_NONE;
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));

	CView::OnChar(nChar, nRepCnt, nFlags);
}

void COglViewTestView::OnViewstatePan()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtPan());
}

void COglViewTestView::OnViewstateRotate()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtRotate());
}

void COglViewTestView::OnViewstateZoomregion()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomRegion());
}

void COglViewTestView::OnViewstateZoomall()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomAll());
}

void COglViewTestView::OnViewstateZoomin()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomIn());
}

void COglViewTestView::OnViewstateZoomout()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomOut());
}

void COglViewTestView::OnUpdateViewstatePan(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get())
	{
		const swl::PanState *active = viewStateFsm_->state_cast<const swl::PanState *>();
		pCmdUI->SetCheck(active ? 1 : 0);
	}
	else pCmdUI->SetCheck(0);
}

void COglViewTestView::OnUpdateViewstateRotate(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get())
	{
		const swl::RotateState *active = viewStateFsm_->state_cast<const swl::RotateState *>();
		pCmdUI->SetCheck(active ? 1 : 0);
	}
	else pCmdUI->SetCheck(0);
}

void COglViewTestView::OnUpdateViewstateZoomregion(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get())
	{
		const swl::ZoomRegionState *active = viewStateFsm_->state_cast<const swl::ZoomRegionState *>();
		pCmdUI->SetCheck(active ? 1 : 0);
	}
	else pCmdUI->SetCheck(0);
}

void COglViewTestView::OnUpdateViewstateZoomall(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get())
	{
		const swl::ZoomAllState *active = viewStateFsm_->state_cast<const swl::ZoomAllState *>();
		pCmdUI->SetCheck(active ? 1 : 0);
	}
	else pCmdUI->SetCheck(0);
}

void COglViewTestView::OnUpdateViewstateZoomin(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get())
	{
		const swl::ZoomInState *active = viewStateFsm_->state_cast<const swl::ZoomInState *>();
		pCmdUI->SetCheck(active ? 1 : 0);
	}
	else pCmdUI->SetCheck(0);
}

void COglViewTestView::OnUpdateViewstateZoomout(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView: view state
	if (viewStateFsm_.get())
	{
		const swl::ZoomOutState *active = viewStateFsm_->state_cast<const swl::ZoomOutState *>();
		pCmdUI->SetCheck(active ? 1 : 0);
	}
	else pCmdUI->SetCheck(0);
}
