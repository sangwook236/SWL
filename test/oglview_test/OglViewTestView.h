// OglViewTestView.h : interface of the COglViewTestView class
//


#pragma once

#include "swl/view/ViewEventController.h"
#include <boost/smart_ptr.hpp>
#include <memory>

namespace swl {
class WglContextBase;
class ViewCamera3;
class OglCamera;
struct ViewStateMachine;
}

class COglViewTestView : public CView
{
protected: // create from serialization only
	COglViewTestView();
	DECLARE_DYNCREATE(COglViewTestView)

// Attributes
public:
	COglViewTestDoc* GetDocument() const;

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
	virtual ~COglViewTestView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

public:
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView

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
	//-------------------------------------------------------------------------
	// This code is required for SWL.OglView

	void renderScene(swl::WglContextBase &context, swl::ViewCamera3 &camera);

	virtual bool doPrepareRendering();
	virtual bool doRenderStockScene();
	virtual bool doRenderScene();

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView

	boost::scoped_ptr<swl::WglContextBase> viewContext_;
	boost::scoped_ptr<swl::OglCamera> viewCamera_;

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

#ifndef _DEBUG  // debug version in OglViewTestView.cpp
inline COglViewTestDoc* COglViewTestView::GetDocument() const
   { return reinterpret_cast<COglViewTestDoc*>(m_pDocument); }
#endif

