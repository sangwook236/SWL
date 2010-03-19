// WglSceneGraphView.cpp : implementation of the CWglSceneGraphView class
//

#include "stdafx.h"
#include "swl/Config.h"
#include "WglViewTest.h"

#include "WglViewTestDoc.h"
#include "WglSceneGraphView.h"

#include "ViewStateMachine.h"
#include "ViewEventHandler.h"
#include "SceneGraphShape.h"
#include "swl/winview/WglDoubleBufferedContext.h"
#include "swl/winview/WglBitmapBufferedContext.h"
#include "swl/winview/WglPrintContext.h"
#include "swl/winview/WglViewPrintApi.h"
#include "swl/winview/WglViewCaptureApi.h"
#include "swl/winview/WglFont.h"
#include "swl/glutil/GLCamera.h"
#include "swl/glutil/GLRenderSceneVisitor.h"
#include "swl/glutil/GLCreateDisplayListVisitor.h"
#include "swl/glutil/CWglSceneGraphView.h"
#include "swl/glutil/GLShapeSceneNode.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include "swl/math/MathConstant.h"
#include <boost/smart_ptr.hpp>
#include <boost/multi_array.hpp>
#include <GL/glu.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif

#define __USE_OPENGL_DISPLAY_LIST 1

namespace {

}  // unnamed namespace


// CWglSceneGraphView

IMPLEMENT_DYNCREATE(CWglSceneGraphView, CView)

BEGIN_MESSAGE_MAP(CWglSceneGraphView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_WM_DESTROY()
	ON_WM_PAINT()
	ON_WM_SIZE()
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
	ON_COMMAND(ID_VIEWHANDLING_PAN, &CWglSceneGraphView::OnViewhandlingPan)
	ON_COMMAND(ID_VIEWHANDLING_ROTATE, &CWglSceneGraphView::OnViewhandlingRotate)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMREGION, &CWglSceneGraphView::OnViewhandlingZoomregion)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMALL, &CWglSceneGraphView::OnViewhandlingZoomall)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMIN, &CWglSceneGraphView::OnViewhandlingZoomin)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMOUT, &CWglSceneGraphView::OnViewhandlingZoomout)
	ON_COMMAND(ID_VIEWHANDLING_PICKOBJECT, &CWglSceneGraphView::OnViewhandlingPickobject)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_PAN, &CWglSceneGraphView::OnUpdateViewhandlingPan)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ROTATE, &CWglSceneGraphView::OnUpdateViewhandlingRotate)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMREGION, &CWglSceneGraphView::OnUpdateViewhandlingZoomregion)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMALL, &CWglSceneGraphView::OnUpdateViewhandlingZoomall)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMIN, &CWglSceneGraphView::OnUpdateViewhandlingZoomin)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMOUT, &CWglSceneGraphView::OnUpdateViewhandlingZoomout)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_PICKOBJECT, &CWglSceneGraphView::OnUpdateViewhandlingPickobject)
	ON_COMMAND(ID_PRINTANDCAPTURE_PRINTVIEWUSINGGDI, &CWglSceneGraphView::OnPrintandcapturePrintviewusinggdi)
	ON_COMMAND(ID_PRINTANDCAPTURE_CAPTUREVIEWUSINGGDI, &CWglSceneGraphView::OnPrintandcaptureCaptureviewusinggdi)
	ON_COMMAND(ID_PRINTANDCAPTURE_CAPTUREVIEWUSINGGDIPLUS, &CWglSceneGraphView::OnPrintandcaptureCaptureviewusinggdiplus)
	ON_COMMAND(ID_PRINTANDCAPTURE_COPYTOCLIPBOARD, &CWglSceneGraphView::OnPrintandcaptureCopytoclipboard)
	ON_COMMAND(ID_EDIT_COPY, &CWglSceneGraphView::OnEditCopy)
END_MESSAGE_MAP()

// CWglSceneGraphView construction/destruction

CWglSceneGraphView::CWglSceneGraphView()
: swl::WglViewBase(MAX_OPENGL_DISPLAY_LIST_COUNT, swl::WglFont::FONT_DISPLAY_LIST_COUNT),
  viewStateFsm_(),
  isPerspective_(true), isWireFrame_(false),
  isGradientBackgroundUsed_(true), isFloorShown_(true), isColorBarShown_(true), isCoordinateFrameShown_(true),
  isPrinting_(false), isPickingObject_(false),
  pickedObj_(0u), temporarilyPickedObj_(0u),
  rootSceneNode_()
{
#if 0
	topGradientBackgroundColor_[0] = 0.776f;
	topGradientBackgroundColor_[1] = 0.835f;
	topGradientBackgroundColor_[2] = 0.980f;
	topGradientBackgroundColor_[3] = 1.0f;
	bottomGradientBackgroundColor_[0] = 0.243f;
	bottomGradientBackgroundColor_[1] = 0.443f;
	bottomGradientBackgroundColor_[2] = 0.968f;
	bottomGradientBackgroundColor_[3] = 1.0f;
#elif 0
	topGradientBackgroundColor_[0] = 0.780f;
	topGradientBackgroundColor_[1] = 0.988f;
	topGradientBackgroundColor_[2] = 0.910f;
	topGradientBackgroundColor_[3] = 1.0f;
	bottomGradientBackgroundColor_[0] = 0.302f;
	bottomGradientBackgroundColor_[1] = 0.969f;
	bottomGradientBackgroundColor_[2] = 0.712f;
	bottomGradientBackgroundColor_[3] = 1.0f;
#else
	topGradientBackgroundColor_[0] = 0.812f;
	topGradientBackgroundColor_[1] = 0.847f;
	topGradientBackgroundColor_[2] = 0.863f;
	topGradientBackgroundColor_[3] = 1.0f;
	bottomGradientBackgroundColor_[0] = 0.384f;
	bottomGradientBackgroundColor_[1] = 0.467f;
	bottomGradientBackgroundColor_[2] = 0.510f;
	bottomGradientBackgroundColor_[3] = 1.0f;
#endif

	floorColor_[0] = 0.5f;
	floorColor_[1] = 0.5f;
	floorColor_[2] = 0.5f;
	floorColor_[3] = 0.5f;

	pickedColor_[0] = 1.0f;
	pickedColor_[1] = 1.0f;
	pickedColor_[2] = 0.0f;
	pickedColor_[3] = 1.0f;
}

CWglSceneGraphView::~CWglSceneGraphView()
{
}

BOOL CWglSceneGraphView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CWglSceneGraphView drawing

void CWglSceneGraphView::OnDraw(CDC* pDC)
{
	CWglViewTestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	if (pDC && pDC->IsPrinting())
	{
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!camera) return;

		const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));
		const int oldMapMode = pDC->SetMapMode(MM_TEXT);

		// save view's states
		const bool isPrinting = isPrinting_;
		if (!isPrinting) isPrinting_ = true;

		//
		const double eps = 1.0e-20;

		const swl::Region2<int> rctPage(swl::Point2<int>(0, 0), swl::Point2<int>(pDC->GetDeviceCaps(HORZRES), pDC->GetDeviceCaps(VERTRES)));
		const swl::Region2<double> &currViewRegion = camera->getCurrentViewRegion();
		const double width = currViewRegion.getWidth() >= eps ? currViewRegion.getWidth() : 1.0;
		const double height = currViewRegion.getHeight() >= eps ? currViewRegion.getHeight() : 1.0;
		const double ratio = std::min(rctPage.getWidth() / width, rctPage.getHeight() / height);

		const double width0 = width * ratio, height0 = height * ratio;
		const int w0 = (int)std::floor(width0), h0 = (int)std::floor(height0);
		const int x0 = rctPage.left + (int)std::floor((rctPage.getWidth() - width0) * 0.5), y0 = rctPage.bottom + (int)std::floor((rctPage.getHeight() - height0) * 0.5);

		const boost::shared_ptr<context_type> &context = topContext();

		swl::WglPrintContext printContext(pDC->GetSafeHdc(), swl::Region2<int>(x0, y0, x0 + w0, y0 + h0));
		const std::auto_ptr<camera_type> printCamera(dynamic_cast<WglViewBase::camera_type *>(camera->cloneCamera()));

		const bool isDisplayListShared = !context ? false : printContext.shareDisplayList(*context);

		if (printCamera.get() && printContext.isActivated())
		{
			const bool doesRecreateDisplayListUsed = !isDisplayListShared && isDisplayListUsed();
			// create & push a new name base of OpenGL display list
			if (doesRecreateDisplayListUsed) pushDisplayList(true);

			initializeView();
			printCamera->setViewRegion(camera->getCurrentViewRegion());
			printCamera->setViewport(0, 0, w0, h0);

			// re-create a OpenGL display list
			if (doesRecreateDisplayListUsed) createDisplayList(true);

			renderScene(printContext, *printCamera);

			// pop & delete a new name base of OpenGL display list
			if (doesRecreateDisplayListUsed) popDisplayList(true);
		}

		// restore view's states
		if (!isPrinting) isPrinting_ = false;

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
				context.reset(new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect));
			else if (2 == drawMode_)
				context.reset(new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect));

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


// CWglSceneGraphView printing

BOOL CWglSceneGraphView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CWglSceneGraphView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CWglSceneGraphView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CWglSceneGraphView diagnostics

#ifdef _DEBUG
void CWglSceneGraphView::AssertValid() const
{
	CView::AssertValid();
}

void CWglSceneGraphView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CWglViewTestDoc* CWglSceneGraphView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CWglViewTestDoc)));
	return (CWglViewTestDoc*)m_pDocument;
}
#endif //_DEBUG


// CWglSceneGraphView message handlers

void CWglSceneGraphView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	//
	CRect rect;
	GetClientRect(&rect);

	drawMode_ = 2;  // [1, 2]
	useLocallyCreatedContext_ = false;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
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
	// This code is required for SWL.WglView: basic routine

	// create a context
	if (1 == drawMode_)
		// it is not working with clipboard in Windows.
		pushContext(boost::shared_ptr<context_type>(new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect, false)));
	else if (2 == drawMode_)
		pushContext(boost::shared_ptr<context_type>(new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect, false)));

	// create a camera
	pushCamera(boost::shared_ptr<camera_type>(new swl::GLCamera()));

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

#if defined(__USE_OPENGL_DISPLAY_LIST)
		if (!pushDisplayList(true))
		{
			// error: OpenGL display list cannot be initialized !!!
		}
#endif

		contructSceneGraph();

		// set the view
		initializeView();

		// set the camera
		if (viewCamera.get())
		{
			// set the size of viewing volume
			viewCamera->setEyePosition(1000.0, 1000.0, 1000.0, false);
			viewCamera->setEyeDistance(8000.0, false);
			viewCamera->setObjectPosition(0.0, 0.0, 0.0, false);
			//viewCamera->setEyeDistance(1000.0, false);
			//viewCamera->setObjectPosition(110.0, 110.0, 150.0, false);

			// (left, bottom, right, top) is set wrt a eye coordinates frame
			// (near, far) is the distances from the eye point(viewpoint) to the near & far clipping planes of viewing volume
			//viewCamera->setViewBound(-1600.0, -1100.0, 2400.0, 2900.0, 1.0, 20000.0);
			viewCamera->setViewBound(-1000.0, -1000.0, 1000.0, 1000.0, 4000.0, 12000.0);
			//viewCamera->setViewBound(-50.0, -50.0, 50.0, 50.0, 1.0, 2000.0);

			viewCamera->setViewport(0, 0, rect.Width(), rect.Height());
			
			viewCamera->setPerspective(isPerspective_);
		}

#if defined(__USE_OPENGL_DISPLAY_LIST)
		createDisplayList(true);
#endif

		raiseDrawEvent(true);
	}

	// using a locally-created context
	if (useLocallyCreatedContext_)
		popContext();
}

void CWglSceneGraphView::OnDestroy()
{
	CView::OnDestroy();

#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!popDisplayList(false))
	{
		// error: OpenGL display list cannot be finalized !!!
	}
#endif

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	popContext();
	popCamera();
}

void CWglSceneGraphView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	// using a locally-created context
	if (useLocallyCreatedContext_)
		raiseDrawEvent(true);
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
			else raiseDrawEvent(false);
		}
	}

	// Do not call CView::OnPaint() for painting messages
}

void CWglSceneGraphView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	if (cx <= 0 || cy <= 0) return;
	resizeView(0, 0, cx, cy);
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglSceneGraphView::raiseDrawEvent(const bool isContextActivated)
{
	if (isContextActivated)
		OnDraw(0L);
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (!context || context->isDrawing())
			return false;

		context_type::guard_type guard(*context);
		OnDraw(0L);
	}

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglSceneGraphView::initializeView()
{
	// can we put this in the constructor?
	// specify black(0.0f, 0.0f, 0.0f, 0.0f) or white(1.0f, 1.0f, 1.0f, 1.0f) as clear color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// specify the back of the buffer as clear depth
    glClearDepth(1.0f);
	// enable depth testing
    glEnable(GL_DEPTH_TEST);
	// the type of depth testing
	glDepthFunc(GL_LESS);

	// enable stencil testing
	//glEnable(GL_STENCIL_TEST);
	// the type of stencil testing
	//glStencilFunc(GL_ALWAYS, 0, 1);
	//
	//glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

	//
	//glEnable(GL_BLEND);
	////glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//
	//GLfloat pointSizes[2] = { 0.0f, };
	//glGetFloatv(GL_POINT_SIZE_RANGE, pointSizes);
	//GLfloat lineGradularities[2] = { 0.0f, };
	//glGetFloatv(GL_LINE_WIDTH_GRANULARITY, lineGradularities);
	//GLfloat lineWidths[2] = { 0.0f, };
	//glGetFloatv(GL_LINE_WIDTH_RANGE, lineWidths);

	//glEnable(GL_POINT_SMOOTH);
	//glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	//glEnable(GL_LINE_SMOOTH);
	//glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	//glEnable(GL_POLYGON_SMOOTH);
	//glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	//// really nice perspective calculations
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);

	// create light components
	const GLfloat ambientLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	const GLfloat diffuseLight[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	const GLfloat specularLight[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const GLfloat position0[] = { 0.2f, 0.2f, 1.0f, 0.0f };
	const GLfloat position1[] = { 0.2f, 0.2f, -1.0f, 0.0f };

	// assign created components to GL_LIGHT0
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
	glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specularLight);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);

	// polygon winding
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	//glEnable(GL_CULL_FACE);

	// surface normal
	glEnable(GL_NORMALIZE);
	glEnable(GL_AUTO_NORMAL);

	// shading model
	//glShadeModel(GL_FLAT);
	glShadeModel(GL_SMOOTH);

	// color tracking
	glEnable(GL_COLOR_MATERIAL);
	// set material properties which will be assigned by glColor
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	// clipping
	//int maxClipPlanes = 0;
	//glGetIntegerv(GL_MAX_CLIP_PLANES, &maxClipPlanes);

	glPolygonMode(GL_FRONT, GL_FILL);

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglSceneGraphView::resizeView(const int x1, const int y1, const int x2, const int y2)
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!popDisplayList(false))
	{
		// error: OpenGL display list cannot be finalized !!!
	}
#endif

	const boost::shared_ptr<context_type> &context = topContext();
	if (context.get() && context->resize(x1, y1, x2, y2))
	{
		context_type::guard_type guard(*context);

#if defined(__USE_OPENGL_DISPLAY_LIST)
		if (!pushDisplayList(true))
		{
			// error: OpenGL display list cannot be initialized !!!
		}
#endif

		initializeView();
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (camera.get())
		{
			camera->setViewport(x1, y1, x2, y2);

#if defined(__USE_OPENGL_DISPLAY_LIST)
			createDisplayList(true);
#endif
		}

		raiseDrawEvent(true);
		return true;
	}
	else return false;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglSceneGraphView::doPrepareRendering(const context_type &/*context*/, const camera_type &/*camera*/)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglSceneGraphView::doRenderStockScene(const context_type &/*context*/, const camera_type &/*camera*/)
{
	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglSceneGraphView::doRenderScene(const context_type &/*context*/, const camera_type &/*camera*/)
{
	// traverse a scene graph
	if (rootSceneNode_)
	{
		rootSceneNode_->accept(swl::GLRenderSceneVisitor(swl::GLRenderSceneVisitor::RENDER_OPAQUE_OBJECTS));
		rootSceneNode_->accept(swl::GLRenderSceneVisitor(swl::GLRenderSceneVisitor::RENDER_TRANSPARENT_OBJECTS));
	}

    return true;
}

void CWglSceneGraphView::contructSceneGraph()
{
	// the name base of OpenGL display list that is actually used
	const unsigned int currDiplayListNameBase = getCurrentDisplayListNameBase();
	if (0u == currDiplayListNameBase) return;

	rootSceneNode_.reset(new swl::GroupSceneNode<visitor_type>());

	// background
	GradientBackgroundShape *bgShape = new GradientBackgroundShape(currDiplayListNameBase + DLN_GRADIENT_BACKGROUND);
	bgShape->setTopColor(topGradientBackgroundColor_[0], topGradientBackgroundColor_[1], topGradientBackgroundColor_[2], topGradientBackgroundColor_[3]);
	bgShape->setBottomColor(bottomGradientBackgroundColor_[0], bottomGradientBackgroundColor_[1], bottomGradientBackgroundColor_[2], bottomGradientBackgroundColor_[3]);

	boost::shared_ptr<swl::GLShape> backgroundShape(bgShape);
#if defined(UNICODE) || defined(_UNICODE)
	boost::shared_ptr<scene_node_type> backgroundNode(new swl::GLShapeSceneNode<visitor_type>(backgroundShape, L"background"));
#else
	boost::shared_ptr<scene_node_type> backgroundNode(new swl::GLShapeSceneNode<visitor_type>(backgroundShape, "background"));
#endif
	rootSceneNode_->addChild(backgroundNode);

	// floor
	boost::shared_ptr<swl::GLShape> floorShape(new FloorShape(*this));
	floorShape->setColor(floorColor_[0], floorColor_[1], floorColor_[2], floorColor_[3]);
#if defined(UNICODE) || defined(_UNICODE)
	boost::shared_ptr<scene_node_type> floorNode(new swl::GLShapeSceneNode<visitor_type>(floorShape, L"floor"));
#else
	boost::shared_ptr<scene_node_type> floorNode(new swl::GLShapeSceneNode<visitor_type>(floorShape, "floor"));
#endif
	rootSceneNode_->addChild(floorNode);

	// main contents
	boost::shared_ptr<swl::GLShape> main1Shape(new Main1Shape(currDiplayListNameBase + DLN_MAIN_OBJECT_1));
	main1Shape->setColor(1.0f, 0.0f, 0.0f, 1.0f);
#if defined(UNICODE) || defined(_UNICODE)
	boost::shared_ptr<scene_node_type> main1Node(new swl::GLShapeSceneNode<visitor_type>(main1Shape, L"main #1"));
#else
	boost::shared_ptr<scene_node_type> main1Node(new swl::GLShapeSceneNode<visitor_type>(main1Shape, "main #1"));
#endif
	rootSceneNode_->addChild(main1Node);

	boost::shared_ptr<swl::GLShape> main2Shape(new Main2Shape(currDiplayListNameBase + DLN_MAIN_OBJECT_2));
	main2Shape->setColor(0.5f, 0.5f, 1.0f, 1.0f);
#if defined(UNICODE) || defined(_UNICODE)
	boost::shared_ptr<scene_node_type> main2Node(new swl::GLShapeSceneNode<visitor_type>(main2Shape, L"main #2"));
#else
	boost::shared_ptr<scene_node_type> main2Node(new swl::GLShapeSceneNode<visitor_type>(main2Shape, "main #2"));
#endif
	rootSceneNode_->addChild(main2Node);

	// color bar
	boost::shared_ptr<swl::GLShape> colorBarShape(new ColorBarShape(currDiplayListNameBase + DLN_COLOR_BAR));
#if defined(UNICODE) || defined(_UNICODE)
	boost::shared_ptr<scene_node_type> colorBarNode(new swl::GLShapeSceneNode<visitor_type>(colorBarShape, L"color bar"));
#else
	boost::shared_ptr<scene_node_type> colorBarNode(new swl::GLShapeSceneNode<visitor_type>(colorBarShape, "color bar"));
#endif
	rootSceneNode_->addChild(colorBarNode);

	// coordinate frame
	boost::shared_ptr<swl::GLShape> coordinateFrameShape(new CoordinateFrameShape(*this));
#if defined(UNICODE) || defined(_UNICODE)
	boost::shared_ptr<scene_node_type> coordinateFrameNode(new swl::GLShapeSceneNode<visitor_type>(coordinateFrameShape, L"coordinate frame"));
#else
	boost::shared_ptr<scene_node_type> coordinateFrameNode(new swl::GLShapeSceneNode<visitor_type>(coordinateFrameShape, "coordinate frame"));
#endif
	rootSceneNode_->addChild(coordinateFrameNode);
}

bool CWglSceneGraphView::createDisplayList(const bool isContextActivated)
{
	if (isDisplayListUsed())
	{
		// the name base of OpenGL display list for fonts that is actually used
		const unsigned int currFontDiplayListNameBase = getCurrentFontDisplayListNameBase();

		HDC *dc = NULL;

		const boost::shared_ptr<context_type> &context = topContext();
		if (context.get())
		{
			//context_type::guard_type guard(*context);
			try
			{
				dc = boost::any_cast<HDC *>(context->getNativeContext());
				if (NULL == dc) return false;
			}
			catch (const boost::bad_any_cast &)
			{
				return false;
			}
		}

		if (isContextActivated)
		{
			if (rootSceneNode_) rootSceneNode_->accept(swl::GLCreateDisplayListVisitor());
			swl::WglFont::getInstance().create(*dc, currFontDiplayListNameBase);
		}
		else
		{
			context_type::guard_type guard(*context);
			if (rootSceneNode_) rootSceneNode_->accept(swl::GLCreateDisplayListVisitor());
			swl::WglFont::getInstance().create(*dc, currFontDiplayListNameBase);
		}
	}

	return true;
}

void CWglSceneGraphView::pickObject(const int x, const int y, const bool isTemporary /*= false*/)
{
	if (rootSceneNode_)
	{
		const boost::shared_ptr<context_type> &context = topContext();
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!context || !camera) return;

		isPickingObject_ = true;

		context_type::guard_type guard(*context);
		rootSceneNode_->accept(swl::CWglSceneGraphView(/*camera, x, y, 2, 2, isTemporary*/));

		isPickingObject_ = false;
	}
}

void CWglSceneGraphView::pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary /*= false*/)
{
	if (rootSceneNode_)
	{
		const boost::shared_ptr<context_type> &context = topContext();
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!context || !camera) return;

		isPickingObject_ = true;

		context_type::guard_type guard(*context);
		const swl::Region2<int> region(swl::Point2<int>(x1, y1), swl::Point2<int>(x2, y2));
		rootSceneNode_->accept(swl::CWglSceneGraphView(/*camera, region.getCenterX(), region.getCenterY(), region.getWidth(), region.getHeight(), isTemporary*/));

		isPickingObject_ = false;
	}
}

{
	// save states
	GLint oldMatrixMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	glPushAttrib(GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT);

	// set attributes
	glDisable(GL_LIGHTING);
	glDepthFunc(GL_LEQUAL);

	// set selection buffer
	const GLsizei SELECT_BUFFER_SIZE = 64;
	GLuint selectBuffer[SELECT_BUFFER_SIZE] = { 0, };
	glSelectBuffer(SELECT_BUFFER_SIZE, selectBuffer);

	// change rendering mode
	glRenderMode(GL_SELECT);

	// initialize name stack
	glInitNames();
	//glPushName(0u);

	// render scene
#if 0
	renderScene(*context, *camera);
#else
	if (rootSceneNode_) rootSceneNode_->accept(swl::CWglSceneGraphView());

	processToPickMainContent(camera_, x_, y_, width_, height_);
	if (isCoordinateFrameShown_) processToPickCoordinateFrame(camera_, x_, y_, width_, height_);
#endif

	// gather hit records
	const GLint hitCount = glRenderMode(GL_RENDER);

	// pop original attributes
	glPopAttrib();  // GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT

	glMatrixMode(oldMatrixMode);

	// process hits
	if (isTemporary_)
	{
		const unsigned int pickedObj = hitCount > 0 ? processHits(hitCount, selectBuffer) : 0u;
		if (pickedObj != temporarilyPickedObj_)
		{
			temporarilyPickedObj_ = pickedObj;
			raiseDrawEvent(false);
		}
	}
	else
	{
		const unsigned int oldPickedObj = pickedObj_;

		temporarilyPickedObj_ = 0u;
		if (hitCount > 0)
		{
			pickedObj_ = processHits(hitCount, selectBuffer);

			// FIXME [modify] >>
			//switch (pickedObj_)
			//{
			//case PON_SPHERE:
			//	break;
			//case PON_CUBE:
			//	break;
			//case PON_X_AXIS:
			//	break;
			//case PON_Y_AXIS:
			//	break;
			//case PON_Z_AXIS:
			//	break;
			//}
		}
		else pickedObj_ = 0u;

		if (oldPickedObj != pickedObj_)
			raiseDrawEvent(false);
	}
}

void CWglSceneGraphView::processToPickMainContent(const boost::shared_ptr<camera_type> &camera, const int x, const int y, const int width, const int height) const
{
	// set attributes
	glEnable(GL_DEPTH_TEST);
	//glDepthRange(0.0, 1.0);

	//double modelviewMatrix[16];
	double projectionMatrix[16];
	int viewport[4];
	//glGetDoublev(GL_MODELVIEW_MATRIX, modelviewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projectionMatrix);
	glGetIntegerv(GL_VIEWPORT, viewport);

	// set projection matrix
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPickMatrix(x, viewport[3] - y, width, height, viewport);

	// need to load current projection matrix
	glMultMatrixd(projectionMatrix);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
		glLoadIdentity();
		// 1. need to load current modelview matrix
		//   e.g.) glLoadMatrixd(modelviewMatrix);
		// 2. need to be thought of viewing transformation
		//   e.g.) camera->lookAt();
		camera->lookAt();

#if 0
		doRenderScene(*context, *camera);
#else
		drawMainContent();
#endif
	glPopMatrix();

	// pop projection matrix
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

void CWglSceneGraphView::processToPickCoordinateFrame(const boost::shared_ptr<camera_type> &camera, const int x, const int y, const int width, const int height) const
{
	const swl::Region2<int> &oldViewport = camera->getViewport();
	const swl::Region2<double> &oldViewRegion = camera->getViewRegion();

	const int dX = int(oldViewport.getWidth() * 0.10);
	const int dY = int(oldViewport.getHeight() * 0.10);
	const int size = std::max(std::max(dX, dY), 100);

	camera->setViewport(swl::Region2<int>(oldViewport.left, oldViewport.bottom, size, size));
	camera->setViewRegion(static_cast<swl::ViewCamera2 *>(camera.get())->getViewBound());
	const swl::Region2<double> &currViewRegion = camera->getCurrentViewRegion();

	// save states
	glDisable(GL_DEPTH_TEST);

	//double modelviewMatrix[16];
	double projectionMatrix[16];
	int viewport[4];
	//glGetDoublev(GL_MODELVIEW_MATRIX, modelviewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projectionMatrix);
	glGetIntegerv(GL_VIEWPORT, viewport);

	// set projection matrix
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	//gluPickMatrix(x, viewport[3] - y, width, height, viewport);
	gluPickMatrix(x, oldViewport.getHeight() - y, width, height, viewport);

	// need to load current projection matrix
	glMultMatrixd(projectionMatrix);

	//
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
		glLoadIdentity();
		// 1. need to load current modelview matrix
		//   e.g.) glLoadMatrixd(modelviewMatrix);
		// 2. need to be thought of viewing transformation
		//   e.g.) camera->lookAt();
		camera->lookAt();

		// move origin
		double eyeX(0.0), eyeY(0.0), eyeZ(0.0), dirX(0.0), dirY(0.0), dirZ(0.0);
		camera->getEyePosition(eyeX, eyeY, eyeZ);
		camera->getEyeDirection(dirX, dirY, dirZ);
		const double eyeDist = camera->getEyeDistance();
		glTranslated(eyeX + eyeDist * dirX, eyeY + eyeDist * dirY, eyeZ + eyeDist * dirZ);

		std::multimap<double, int> vals;
		vals.insert(std::make_pair(std::acos(dirX), 0));
		vals.insert(std::make_pair(std::acos(dirY), 1));
		vals.insert(std::make_pair(std::acos(dirZ), 2));
		std::multimap<double, int>::iterator it = vals.begin();
		const int order1 = it->second;  ++it;
		const int order2 = it->second;  ++it;
		const int order3 = it->second;
		const int order[] = { order1, order2, order3 };

		float length = (float)std::min(currViewRegion.getHeight(), currViewRegion.getWidth()) * 0.25f;
		if (camera->isPerspective()) length *= 2.0f / std::sqrt(3.0f);
		drawCoordinateFrame(false, length, order);
	glPopMatrix();

	// pop projection matrix
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	// restore states
	camera->setViewRegion(oldViewRegion);
	camera->setViewport(oldViewport);
}

unsigned int CWglSceneGraphView::processHits(const int hitCount, const unsigned int *buffer) const
{
	const GLuint *ptr = (const GLuint *)buffer;

	GLuint selectedObj = 0u;
	bool isCoordinateFramePicked = false;
	//float minZ = 1.0f;
	unsigned int minZ = 0xffffffff;
	for (int i = 0; i < hitCount; ++i)
	{
		// number of names for each hit.
		const GLuint nameCount = *ptr;
		++ptr;
		// min. window-coordinate z values of all vertices of the primitives that intersectd the viewing volume since the last recorded hit.
		////const float mnZ = float(*ptr) / 0x7fffffff;
		//const float mnZ = float(*ptr) / 0xffffffff;  // 2^32 - 1
		const unsigned int mnZ = *ptr;
		++ptr;
		// max. window-coordinate z values of all vertices of the primitives that intersectd the viewing volume since the last recorded hit
		////const float mxZ = float(*ptr) / 0x7fffffff;
		//const float mxZ = float(*ptr) / 0xffffffff;  // 2^32 - 1
		const unsigned int mxZ = *ptr;
		++ptr;

		if (0 == nameCount) continue;

		const GLuint currObj = *(ptr + nameCount - 1);
		if (isCoordinateFramePicked)
		{
			switch (currObj)
			{
			case PON_X_AXIS:
			case PON_Y_AXIS:
			case PON_Z_AXIS:
				if (mnZ < minZ)
				{
					minZ = mnZ;
					selectedObj = currObj;
				}
				break;
			}
		}
		else
		{
			switch (currObj)
			{
			case PON_X_AXIS:
			case PON_Y_AXIS:
			case PON_Z_AXIS:
				minZ = mnZ;
				selectedObj = currObj;
				isCoordinateFramePicked = true;
				break;
			default:
				if (mnZ < minZ)
				{
					minZ = mnZ;
					selectedObj = currObj;
				}
				break;
			}
		}

		const GLuint *ptr2 = ptr;
		//TRACE("***** the number of names for each hit: %d, min z: %f, max z: %f\n", nameCount, mnZ, mxZ);
		TRACE("***** the number of names for each hit: %d, min z: %d, max z: %d\n", nameCount, mnZ, mxZ);
		// the contents of the name stack
		TRACE("\tthe contents of the name stack: %d", *ptr2);
		++ptr2;
		for (GLuint j = 1; j < nameCount; ++j)
		{
			const GLint name = *ptr2;
			++ptr2;
			TRACE(" - %d", name);
		}
		TRACE("\n");

		ptr += nameCount;
	}

	TRACE("=====> the picked object: %d\n", selectedObj);
	return selectedObj;
}

void CWglSceneGraphView::setPerspective(const bool isPerspective)
{
	if (isPerspective == isPerspective_) return;

	const boost::shared_ptr<context_type> &context = topContext();
	const boost::shared_ptr<camera_type> &camera = topCamera();
	if (context.get() && camera.get())
	{
		isPerspective_ = isPerspective;

		context_type::guard_type guard(*context);
		camera->setPerspective(isPerspective_);

//#if defined(__USE_OPENGL_DISPLAY_LIST)
//		createDisplayList(true);
//#endif
	}
}

void CWglSceneGraphView::setWireFrame(const bool isWireFrame)
{
	if (isWireFrame == isWireFrame_) return;

	isWireFrame_ = isWireFrame;

#if defined(__USE_OPENGL_DISPLAY_LIST)
	createDisplayList(false);
#endif
}

void CWglSceneGraphView::OnLButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDown(nFlags, point);
}

void CWglSceneGraphView::OnLButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonUp(nFlags, point);
}

void CWglSceneGraphView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDblClk(nFlags, point);
}

void CWglSceneGraphView::OnMButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDown(nFlags, point);
}

void CWglSceneGraphView::OnMButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonUp(nFlags, point);
}

void CWglSceneGraphView::OnMButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDblClk(nFlags, point);
}

void CWglSceneGraphView::OnRButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDown(nFlags, point);
}

void CWglSceneGraphView::OnRButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonUp(nFlags, point);
}

void CWglSceneGraphView::OnRButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDblClk(nFlags, point);
}

void CWglSceneGraphView::OnMouseMove(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
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

BOOL CWglSceneGraphView::OnMouseWheel(UINT nFlags, short zDelta, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
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

void CWglSceneGraphView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	//viewController_.pressKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_.get()) viewStateFsm_->pressKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyDown(nChar, nRepCnt, nFlags);
}

void CWglSceneGraphView::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_.get()) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyUp(nChar, nRepCnt, nFlags);
}

void CWglSceneGraphView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::KeyEvent::EControlKey ckey = ((nFlags >> 28) & 0x01) == 0x01 ? swl::KeyEvent::CK_ALT : swl::KeyEvent::CK_NONE;
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));
	if (viewStateFsm_.get()) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));

	CView::OnChar(nChar, nRepCnt, nFlags);
}

void CWglSceneGraphView::OnViewhandlingPan()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtPan());
}

void CWglSceneGraphView::OnViewhandlingRotate()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtRotate());
}

void CWglSceneGraphView::OnViewhandlingZoomregion()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomRegion());
}

void CWglSceneGraphView::OnViewhandlingZoomall()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomAll());
}

void CWglSceneGraphView::OnViewhandlingZoomin()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomIn());
}

void CWglSceneGraphView::OnViewhandlingZoomout()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtZoomOut());
}

void CWglSceneGraphView::OnViewhandlingPickobject()
{
	const bool isRedrawn = 0u != pickedObj_;
	pickedObj_ = temporarilyPickedObj_ = 0u;
	if (isRedrawn) raiseDrawEvent(false);

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_.get()) viewStateFsm_->process_event(swl::EvtPickObject());
}

void CWglSceneGraphView::OnUpdateViewhandlingPan(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::PanState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnUpdateViewhandlingRotate(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::RotateState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnUpdateViewhandlingZoomregion(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomRegionState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnUpdateViewhandlingZoomall(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomAllState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnUpdateViewhandlingZoomin(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomInState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnUpdateViewhandlingZoomout(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomOutState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnUpdateViewhandlingPickobject(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_.get())
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::PickObjectState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglSceneGraphView::OnPrintandcapturePrintviewusinggdi()
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
	di.lpszDocName = _T("OpenGL Print");
	di.lpszOutput = NULL;

	// start the print job
	StartDoc(pd.hDC, &di);
	StartPage(pd.hDC);

	//
#if 0
	// save view's states
	const bool isPrinting = isPrinting_;
	if (!isPrinting) isPrinting_ = true;

	if (!swl::printWglViewUsingGdi(*this, pd.hDC))
		AfxMessageBox(_T("fail to print a view"), MB_OK | MB_ICONSTOP);

	// restore view's states
	if (!isPrinting) isPrinting_ = false;
#else
	CDC *pDC = CDC::FromHandle(pd.hDC);
	if (pDC)
	{
		pDC->m_bPrinting = TRUE;;
		OnDraw(pDC);
	}
#endif

	// end the print job
	EndPage(pd.hDC);
	EndDoc(pd.hDC);
	DeleteDC(pd.hDC);

	DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
}

void CWglSceneGraphView::OnPrintandcaptureCaptureviewusinggdi()
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
		if (!swl::captureWglViewUsingGdi(filePathName, *this, GetSafeHwnd()))
			AfxMessageBox(_T("fail to capture a view"), MB_OK | MB_ICONSTOP);

		DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
	}
}

void CWglSceneGraphView::OnPrintandcaptureCaptureviewusinggdiplus()
{
	// FIXME [add] >>
	AfxMessageBox(_T("not yet implemented"), MB_OK | MB_ICONSTOP);
}

void CWglSceneGraphView::OnPrintandcaptureCopytoclipboard()
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

void CWglSceneGraphView::OnEditCopy()
{
	OnPrintandcaptureCopytoclipboard();
}
