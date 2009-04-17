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
END_MESSAGE_MAP()

// COglViewTestView construction/destruction

COglViewTestView::COglViewTestView()
: camera_(NULL), context_(NULL)
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
	if (context_)
	{
		context_->activate();
		draw(*context_);
		context_->deactivate();
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

	// TODO: Add your specialized code here and/or call the base class
	const int drawMode = 0x02;

	// use double-buffered OpenGL context
	CRect rect;
	GetClientRect(&rect);

	if (NULL == context_)
	{
		if ((0x01 & drawMode) == 0x01)
			context_ = new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect, false);
		if ((0x02 & drawMode) == 0x02)
			context_ = new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect, false);
	}

	if (NULL == camera_) camera_ = new swl::OglCamera;

	assert(context_ && camera_);

	// activate a context
	context_->activate();

	initializeView();
	//camera_->setViewBound(-1600.0, -1100.0, 2400.0, 2900.0, 1.0, 20000.0);
	camera_->setViewBound(-2000.0, -2000.0, 2000.0, 2000.0, 4000.0, 12000.0);
	//camera_->setViewBound(-50.0, -50.0, 50.0, 50.0, 1.0, 2000.0);

	//context_->resize(rect.Width(), rect.Height());
	camera_->setViewport(0, 0, rect.Width(), rect.Height());
	camera_->setEyeDistance(8000.0, false);
	camera_->setObjectPosition(400.0, 900.0, 500.0);
	camera_->setEyeDistance(1000.0, false);
	camera_->setObjectPosition(110.0, 110.0, 150.0);

	context_->redraw();

	// de-activate the context
	context_->deactivate();
}

void COglViewTestView::OnDestroy()
{
	CView::OnDestroy();

	// TODO: Add your message handler code here
	if (camera_)
	{
		delete camera_;
		camera_ = NULL;
	}

	if (context_)
	{
		delete context_;
		context_ = NULL;
	}
}

bool COglViewTestView::initializeView()
{
	// Can we put this in the constructor?
	// specify black as clear color
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
		glutWireSphere(50.0, 20, 20);
		glColor3f(0.5f, 0.5f, 1.0f);
		glutWireCube(100.0);
	glPopMatrix();

    return true;
}

void COglViewTestView::draw(swl::WglContextBase &ctx)
{
#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		const CString msg(gluErrorString(glErrorCode));
		if (GL_NO_ERROR != glErrorCode)
			TRACE(_T("OpenGL error at %d in %s: %s\n"), __LINE__, __FILE__, (LPCTSTR)CString(gluErrorString(glErrorCode)));
	}
#endif

	int oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);

	{
		glPushMatrix();
			//
			glLoadIdentity();
			if (camera_) camera_->lookAt();

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

	ctx.redraw();

	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		if (GL_NO_ERROR != glErrorCode)
			TRACE(_T("OpenGL error at %d in %s: %s\n"), __LINE__, __FILE__, (LPCTSTR)CString(gluErrorString(glErrorCode)));
	}
#endif
}
