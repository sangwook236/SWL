// WglViewTestView.h : interface of the CWglViewTestView class
//


#pragma once

#include "swl/winview/WglViewBase.h"
#include "swl/view/ViewEventController.h"
#include <boost/smart_ptr.hpp>

namespace swl {
class WglContextBase;
class ViewCamera3;
class OglCamera;
struct ViewStateMachine;
}

class CWglViewTestView : public CView, public swl::WglViewBase
{
protected: // create from serialization only
	CWglViewTestView();
	DECLARE_DYNCREATE(CWglViewTestView)

// Attributes
public:
	CWglViewTestDoc* GetDocument() const;

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
	virtual ~CWglViewTestView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

public:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	/*virtual*/ bool raiseDrawEvent(const bool isContextActivated);

	/*virtual*/ bool initializeView();
	/*virtual*/ bool resizeView(const int x1, const int y1, const int x2, const int y2);

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	/*virtual*/ bool doPrepareRendering();
	/*virtual*/ bool doRenderStockScene();
	/*virtual*/ bool doRenderScene();

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	boost::scoped_ptr<swl::WglContextBase> viewContext_;
	boost::scoped_ptr<swl::OglCamera> viewCamera_;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling

	//swl::ViewEventController viewController_;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state

	boost::scoped_ptr<swl::ViewStateMachine> viewStateFsm_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
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
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint point);
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnViewstatePan();
	afx_msg void OnViewstateRotate();
	afx_msg void OnViewstateZoomregion();
	afx_msg void OnViewstateZoomall();
	afx_msg void OnViewstateZoomin();
	afx_msg void OnViewstateZoomout();
	afx_msg void OnUpdateViewstatePan(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewstateRotate(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewstateZoomregion(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewstateZoomall(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewstateZoomin(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewstateZoomout(CCmdUI *pCmdUI);
};

#ifndef _DEBUG  // debug version in WglViewTestView.cpp
inline CWglViewTestDoc* CWglViewTestView::GetDocument() const
   { return reinterpret_cast<CWglViewTestDoc*>(m_pDocument); }
#endif

