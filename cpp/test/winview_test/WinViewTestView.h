// WinViewTestView.h : interface of the CWinViewTestView class
//


#pragma once

#include "swl/winview/WinViewBase.h"
#include "swl/view/ViewEventController.h"
#include <boost/smart_ptr.hpp>
#include <deque>

namespace swl {
class GdiplusBitmapBufferedContext;
class ViewCamera2;
struct ViewStateMachine;
}  // namespace swl


class CWinViewTestView : public CView, public swl::WinViewBase
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
	// This code is required for SWL.WinView: basic routine

	/*virtual*/ bool raiseDrawEvent(const bool isContextActivated);

	/*virtual*/ bool initializeView();
	/*virtual*/ bool resizeView(const int x1, const int y1, const int x2, const int y2);

	/*virtual*/ void pickObject(const int x, const int y, const bool isTemporary = false);
	/*virtual*/ void pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary = false);

	/*virtual*/ void dragObject(const int x1, const int y1, const int x2, const int y2);

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	/*virtual*/ bool doPrepareRendering(const context_type &context, const camera_type &camera);
	/*virtual*/ bool doRenderStockScene(const context_type &context, const camera_type &camera);
	/*virtual*/ bool doRenderScene(const context_type &context, const camera_type &camera);

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: event handling

	//swl::ViewEventController viewController_;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state

	boost::scoped_ptr<swl::ViewStateMachine> viewStateFsm_;

	//-------------------------------------------------------------------------
	//

	typedef std::pair<int, int> datum_type;
	typedef std::deque<datum_type> data_type;

	data_type data1_;
	data_type data2_;
	size_t idx_;
	UINT timeInterval_;

	int drawMode_;
	bool useLocallyCreatedContext_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnDestroy();
	afx_msg void OnPaint();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnTimer(UINT_PTR nIDEvent);
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
	afx_msg void OnViewhandlingPan();
	afx_msg void OnViewhandlingRotate();
	afx_msg void OnViewhandlingZoomregion();
	afx_msg void OnViewhandlingZoomall();
	afx_msg void OnViewhandlingZoomin();
	afx_msg void OnViewhandlingZoomout();
	afx_msg void OnViewhandlingPickobject();
	afx_msg void OnUpdateViewhandlingPan(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewhandlingRotate(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewhandlingZoomregion(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewhandlingZoomall(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewhandlingZoomin(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewhandlingZoomout(CCmdUI *pCmdUI);
	afx_msg void OnUpdateViewhandlingPickobject(CCmdUI *pCmdUI);
	afx_msg void OnPrintandcapturePrintviewusinggdi();
	afx_msg void OnPrintandcaptureCaptureviewusinggdi();
	afx_msg void OnPrintandcaptureCaptureviewusinggdiplus();
	afx_msg void OnPrintandcaptureCopytoclipboard();
	afx_msg void OnEditCopy();
};

#ifndef _DEBUG  // debug version in WinViewTestView.cpp
inline CWinViewTestDoc* CWinViewTestView::GetDocument() const
   { return reinterpret_cast<CWinViewTestDoc*>(m_pDocument); }
#endif

