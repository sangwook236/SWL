// WinViewTestView.h : interface of the CWinViewTestView class
//


#pragma once

#include "swl/view/ViewEventController.h"
#include <boost/smart_ptr.hpp>
#include <deque>
#include <memory>

namespace swl {
class GdiplusBitmapBufferedContext;
class ViewCamera2;
struct ViewStateMachine;
}  // namespace swl


class CWinViewTestView : public CView
{
protected: // create from serialization only
	CWinViewTestView();
	DECLARE_DYNCREATE(CWinViewTestView)

// Attributes
public:
	CWinViewTestDoc* GetDocument() const;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// Implementation
public:
	virtual ~CWinViewTestView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

public:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView

	virtual bool raiseDrawEvent(const bool isContextActivated);

	virtual bool initializeView();
	virtual bool resizeView(const int x1, const int y1, const int x2, const int y2);

	//-------------------------------------------------------------------------
	// This code is required for view state

	void triggerPanEvent();
	void triggerRotateEvent();
	void triggerZoomAllEvent();
	void triggerZoomRegionEvent();
	void triggerZoomInEvent();
	void triggerZoomOutEvent();

private:
	void test1();
	void test2();
	void test3();
	void test4();
	void test5();

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView

	boost::scoped_ptr<swl::GdiplusBitmapBufferedContext> viewContext_;
	boost::scoped_ptr<swl::ViewCamera2> viewCamera_;

	typedef std::pair<int, int> datum_type;
	typedef std::deque<datum_type> data_type;

	data_type data1_;
	data_type data2_;
	size_t idx_;
	UINT timeInterval_;

	int drawMode_;

	//-------------------------------------------------------------------------
	// This code is required for event handling

	swl::ViewEventController viewController_;

	//-------------------------------------------------------------------------
	// This code is required for view state

	const std::auto_ptr<swl::ViewStateMachine> viewStateFsm_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnPaint();
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnLButtonDblClk(UINT nFlags, CPoint point);
	afx_msg void OnMButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMButtonDblClk(UINT nFlags, CPoint point);
	afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnRButtonDblClk(UINT nFlags, CPoint point);
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags);
};

#ifndef _DEBUG  // debug version in WinViewTestView.cpp
inline CWinViewTestDoc* CWinViewTestView::GetDocument() const
   { return reinterpret_cast<CWinViewTestDoc*>(m_pDocument); }
#endif

