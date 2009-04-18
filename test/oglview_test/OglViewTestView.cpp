// OglViewTestView.cpp : implementation of the COglViewTestView class
//

#include "stdafx.h"
#include "OglViewTest.h"

#include "OglViewTestDoc.h"
#include "OglViewTestView.h"

#include "swl/winview/WglDoubleBufferedContext.h"
#include "swl/winview/WglBitmapBufferedContext.h"
#include "swl/oglview/OglCamera.h"
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
	ON_WM_DESTROY()
	ON_WM_SIZE()
	ON_WM_PAINT()
END_MESSAGE_MAP()

// COglViewTestView construction/destruction

COglViewTestView::COglViewTestView()
: viewContext_(), viewCamera_()
{
	// TODO: add construction code here

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

void COglViewTestView::OnDraw(CDC* /*pDC*/)
{
	COglViewTestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
	if (viewContext_.get() && viewCamera_.get() && viewContext_->isActivated())
		draw(*viewContext_, *viewCamera_);
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

	// TODO: Add your specialized code here and/or call the base class
	const int drawMode = 0x01;

	// use double-buffered OpenGL context
	CRect rect;
	GetClientRect(&rect);

	if (NULL == viewContext_.get())
	{
		if ((0x01 & drawMode) == 0x01)
			viewContext_.reset(new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect, false));
		if ((0x02 & drawMode) == 0x02)
			viewContext_.reset(new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect, false));
	}

	if (NULL == viewCamera_.get())
		viewCamera_.reset(new swl::OglCamera());

	assert(viewContext_.get() && viewCamera_.get());

	// initialize a view
	{
		// activate a context
		viewContext_->activate();

		initializeView();
		//viewCamera_->setViewBound(-1600.0, -1100.0, 2400.0, 2900.0, 1.0, 20000.0);
		viewCamera_->setViewBound(-2000.0, -2000.0, 2000.0, 2000.0, 4000.0, 12000.0);
		//viewCamera_->setViewBound(-50.0, -50.0, 50.0, 50.0, 1.0, 2000.0);

		viewCamera_->setViewport(0, 0, rect.Width(), rect.Height());
		viewCamera_->setEyePosition(1000.0, 1000.0, 1000.0, false);
		viewCamera_->setEyeDistance(8000.0, false);
		viewCamera_->setObjectPosition(0.0, 0.0, 0.0);
		//viewCamera_->setEyeDistance(1000.0, false);
		//viewCamera_->setObjectPosition(110.0, 110.0, 150.0);

		raiseDrawEvent(false);

		// de-activate the context
		viewContext_->deactivate();
	}
}

void COglViewTestView::OnDestroy()
{
	CView::OnDestroy();

	// TODO: Add your message handler code here
}

void COglViewTestView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	if (viewContext_.get())
	{
		if (viewContext_->isOffScreenUsed())
		{
			viewContext_->activate();
			viewContext_->swapBuffer();
			viewContext_->deactivate();
		}
		else raiseDrawEvent(true);
	}

	// Do not call CView::OnPaint() for painting messages
}

void COglViewTestView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	if (cx <= 0 || cy <= 0) return;
	resize(0, 0, cx, cy);
}

bool COglViewTestView::raiseDrawEvent(const bool isContextActivated)
{
	if (NULL == viewContext_.get() || viewContext_->isDrawing())
		return false;

	if (isContextActivated)
	{
		if (viewContext_.get())
		{
			viewContext_->activate();
			OnDraw(0L);
			viewContext_->deactivate();
		}
	}
	else OnDraw(0L);

	return true;
}

bool COglViewTestView::resize(const int x1, const int y1, const int x2, const int y2)
{
	if (viewContext_.get() && viewContext_->resize(x1, y1, x2, y2))
	{
		//viewContext_->activate();
		//if (viewCamera_.get()) viewCamera_->setViewport(x1, y1, x2, y2);	
		//raiseDrawEvent(false);
		//viewContext_->deactivate();

		return true;
	}
	else return false;
}

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

bool COglViewTestView::doPrepareRendering()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    return true;
}

bool COglViewTestView::doRenderStockScene()
{
    return true;
}

bool COglViewTestView::doRenderScene()
{
	glPushMatrix();
		//glLoadIdentity();
		//glTranslatef(100.0f, 100.0f, 100.0f);
		glColor3f(1.0f, 0.0f, 0.0f);
		glutWireSphere(500.0, 20, 20);
		glColor3f(0.5f, 0.5f, 1.0f);
		glutWireCube(500.0);
	glPopMatrix();

    return true;
}

void COglViewTestView::draw(swl::WglContextBase &context, swl::ViewCamera3 &camera)
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
