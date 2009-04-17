// WinViewTestView.cpp : implementation of the CWinViewTestView class
//

#include "stdafx.h"
#include "WinViewTest.h"

#include "WinViewTestDoc.h"
#include "WinViewTestView.h"

#include "swl/winview/GdiContext.h"
#include "swl/winview/GdiBitmapBufferedContext.h"
#include "swl/winview/GdiplusContext.h"
#include "swl/winview/GdiplusBitmapBufferedContext.h"
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
	ON_WM_TIMER()
END_MESSAGE_MAP()

// CWinViewTestView construction/destruction

CWinViewTestView::CWinViewTestView()
{
	// TODO: add construction code here

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

void CWinViewTestView::OnDraw(CDC* /*pDC*/)
{
	CWinViewTestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
	//runTest1();
	runTest2();
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

	idx_ = 0;
	timeInterval_ = 50;
	SetTimer(1, timeInterval_, NULL);

	for (int i = 0; i < 5000; ++i)
	{
		const double x = (double)idx_ * timeInterval_ / 1000.0;
		const double y = std::sin(x) * 100.0 + 100.0;
		data_.push_back(std::make_pair(idx_, (int)std::floor(y + 0.5)));

		++idx_;
	}

	CRect rect;
	GetClientRect(&rect);

	// create a context
	viewContext_.reset(new swl::GdiplusBitmapBufferedContext(GetSafeHwnd(), rect, true));
}

void CWinViewTestView::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default
/*
	const double x = (double)idx_ * timeInterval_ / 1000.0;
	const double y = std::sin(x) * 100.0 + 100.0;
	data_.push_back(std::make_pair(idx_, (int)std::floor(y + 0.5)));

	++idx_;
*/
	if (viewContext_ && !viewContext_->isDrawing())
		Invalidate();

	CView::OnTimer(nIDEvent);
}

void CWinViewTestView::runTest1()
{
	CRect rect;
	GetClientRect(&rect);

	const int drawMode = 0x10;

	// use double(bitmap)-buffered GDI context
	if ((0x02 & drawMode) == 0x02)
	{
		// create a context
		swl::GdiBitmapBufferedContext ctx(GetSafeHwnd(), rect);

		HDC *dc = static_cast<HDC *>(ctx.getNativeContext());
		if (dc)
		{
			CDC *pDC = CDC::FromHandle(*dc);

			// clear the background
			//pDC->SetBkColor(RGB(192, 192, 192));  // not working ???
			//pDC->FillRect(rect, &CBrush(RGB(192, 192, 192)));
			//pDC->FillRect(rect, &CBrush(GetSysColor(COLOR_WINDOW)));

			// draw contents
			CPen pen(PS_SOLID, 3, RGB(0, 255, 0));
			pDC->SelectObject(&pen);
			pDC->MoveTo(100, 150);
			pDC->LineTo(300, 350);

			// redraw the context
			ctx.redraw();
		}
	}

	// use single-buffered GDI context
	if ((0x01 & drawMode) == 0x01)
	{
		// create a context
		swl::GdiContext ctx(GetSafeHwnd());

		HDC *dc = static_cast<HDC *>(ctx.getNativeContext());
		if (dc)
		{
			CDC *pDC = CDC::FromHandle(*dc);

			// clear the background
			//pDC->SetBkColor(RGB(192, 192, 0));  // not working ???
			//pDC->FillRect(rect, &CBrush(RGB(192, 192, 0)));

			// draw contents
			CPen pen(PS_SOLID, 2, RGB(255, 0, 0));
			pDC->SelectObject(&pen);
			pDC->MoveTo(100, 100);
			pDC->LineTo(300, 300);

			// redraw the context
			ctx.redraw();
		}
	}

	// use single-buffered GDI+ context
	if ((0x04 & drawMode) == 0x04)
	{
		// create a context
		swl::GdiplusContext ctx(GetSafeHwnd());

		Gdiplus::Graphics *graphics = static_cast<Gdiplus::Graphics *>(ctx.getNativeContext());
		if (graphics)
		{
			// clear the background
			//graphics->Clear(Gdiplus::Color(255, 192, 0, 192));

			// draw contents
			Gdiplus::Pen pen(Gdiplus::Color(255, 0, 0, 255), 4.0f);
			graphics->DrawLine(&pen, 100, 200, 300, 400);

			// redraw the context
			ctx.redraw();
		}
	}

	// use double(bitmap)-buffered GDI+ context
	if ((0x08 & drawMode) == 0x08)
	{
		// create a context
		swl::GdiplusBitmapBufferedContext ctx(GetSafeHwnd(), rect);

		Gdiplus::Graphics *graphics = static_cast<Gdiplus::Graphics *>(ctx.getNativeContext());
		if (graphics)
		{
			// clear the background
			//graphics->Clear(Gdiplus::Color(255, 0, 192, 192));

			// draw contents
			Gdiplus::Pen pen(Gdiplus::Color(255, 255, 255, 0), 5.0f);
			graphics->DrawLine(&pen, 100, 250, 300, 450);

			// redraw the context
			ctx.redraw();
		}
	}

	// use double(bitmap)-buffered GDI+ context
	if ((0x10 & drawMode) == 0x10)
	{
		// activate a context
		viewContext_->activate();

		Gdiplus::Graphics *graphics = static_cast<Gdiplus::Graphics *>(viewContext_->getNativeContext());
		if (graphics)
		{
			// clear the background
			//graphics->Clear(Gdiplus::Color(255, 0, 192, 192));

			// draw contents
			Gdiplus::Pen pen(Gdiplus::Color(255, 255, 0, 255), 5.0f);
			graphics->DrawLine(&pen, 100, 300, 300, 500);

			// redraw the context
			viewContext_->redraw();
		}

		// de-activate the context
		viewContext_->deactivate();
	}
}

void CWinViewTestView::runTest2()
{
	CRect rect;
	GetClientRect(&rect);

	const int drawMode = 0x10;

	// use double(bitmap)-buffered GDI context
	if ((0x02 & drawMode) == 0x02)
	{
		// create a context
		swl::GdiBitmapBufferedContext ctx(GetSafeHwnd(), rect);

		HDC *dc = static_cast<HDC *>(ctx.getNativeContext());
		if (dc && data_.size() > 1)
		{
			CDC *pDC = CDC::FromHandle(*dc);

			// clear the background
			//pDC->SetBkColor(RGB(192, 192, 192));  // not working ???
			//pDC->FillRect(rect, &CBrush(RGB(192, 192, 192)));
			//pDC->FillRect(rect, &CBrush(GetSysColor(COLOR_WINDOW)));

			// draw contents
			CPen pen(PS_SOLID, 3, RGB(0, 255, 0));
			data_type::iterator it = data_.begin();
			pDC->MoveTo(it->first, it->second);
			for (++it; it != data_.end(); ++it)
				pDC->LineTo(it->first, it->second);

			// redraw the context
			ctx.redraw();
		}
	}

	// use single-buffered GDI context
	if ((0x01 & drawMode) == 0x01)
	{
		// create a context
		swl::GdiContext ctx(GetSafeHwnd());

		HDC *dc = static_cast<HDC *>(ctx.getNativeContext());
		if (dc && data_.size() > 1)
		{
			CDC *pDC = CDC::FromHandle(*dc);

			// clear the background
			//pDC->SetBkColor(RGB(192, 192, 0));  // not working ???
			//pDC->FillRect(rect, &CBrush(RGB(192, 192, 0)));

			// draw contents
			CPen pen(PS_SOLID, 2, RGB(255, 0, 0));
			data_type::iterator it = data_.begin();
			pDC->MoveTo(it->first, it->second);
			for (++it; it != data_.end(); ++it)
				pDC->LineTo(it->first, it->second);

			// redraw the context
			ctx.redraw();
		}
	}

	// use single-buffered GDI+ context
	if ((0x04 & drawMode) == 0x04)
	{
		// create a context
		swl::GdiplusContext ctx(GetSafeHwnd());

		Gdiplus::Graphics *graphics = static_cast<Gdiplus::Graphics *>(ctx.getNativeContext());
		if (graphics && data_.size() > 1)
		{
			// clear the background
			//graphics->Clear(Gdiplus::Color(255, 192, 0, 192));

			// draw contents
			Gdiplus::Pen pen(Gdiplus::Color(255, 0, 0, 255), 4.0f);
			data_type::iterator prevIt = data_.begin();
			data_type::iterator it = data_.begin();
			for (++it; it != data_.end(); ++prevIt, ++it)
				graphics->DrawLine(&pen, (Gdiplus::REAL)prevIt->first, (Gdiplus::REAL)prevIt->second, (Gdiplus::REAL)it->first, (Gdiplus::REAL)it->second);

			// redraw the context
			ctx.redraw();
		}
	}

	// use double(bitmap)-buffered GDI+ context
	if ((0x08 & drawMode) == 0x08)
	{
		// create a context
		swl::GdiplusBitmapBufferedContext ctx(GetSafeHwnd(), rect);

		Gdiplus::Graphics *graphics = static_cast<Gdiplus::Graphics *>(ctx.getNativeContext());
		if (graphics && data_.size() > 1)
		{
			// clear the background
			//graphics->Clear(Gdiplus::Color(255, 0, 192, 192));

			// draw contents
			Gdiplus::Pen pen(Gdiplus::Color(255, 255, 255, 0), 5.0f);
			data_type::iterator prevIt = data_.begin();
			data_type::iterator it = data_.begin();
			for (++it; it != data_.end(); ++prevIt, ++it)
				graphics->DrawLine(&pen, (Gdiplus::REAL)prevIt->first, (Gdiplus::REAL)prevIt->second, (Gdiplus::REAL)it->first, (Gdiplus::REAL)it->second);

			// redraw the context
			ctx.redraw();
		}
	}

	// use double(bitmap)-buffered GDI+ context
	if ((0x10 & drawMode) == 0x10)
	{
		swl::WinTimer timer;

		// activate a context
		//viewContext_->activate();

		Gdiplus::Graphics *graphics = static_cast<Gdiplus::Graphics *>(viewContext_->getNativeContext());
		if (graphics && data_.size() > 1)
		{
			// clear the background
			//graphics->Clear(Gdiplus::Color(255, 0, 192, 192));

			// draw contents
			Gdiplus::Pen pen(Gdiplus::Color(255, 255, 0, 255), 5.0f);
			data_type::iterator prevIt = data_.begin();
			data_type::iterator it = data_.begin();
			for (++it; it != data_.end(); ++prevIt, ++it)
				graphics->DrawLine(&pen, (Gdiplus::REAL)prevIt->first, (Gdiplus::REAL)prevIt->second, (Gdiplus::REAL)it->first, (Gdiplus::REAL)it->second);

			// redraw the context
			viewContext_->redraw();
		}

		// de-activate the context
		//viewContext_->deactivate();

		TRACE(_T("***** elapsed time: %ld, %ld\n"), (long)timer.getElapsedTimeInMilliSecond(), (long)timer.getElapsedTimeInMicroSecond());
	}
}
