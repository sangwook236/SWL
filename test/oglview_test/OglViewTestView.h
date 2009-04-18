// OglViewTestView.h : interface of the COglViewTestView class
//


#pragma once

#include <boost/smart_ptr.hpp>

namespace swl {
class WglContextBase;
class ViewCamera3;
class OglCamera;
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
	virtual bool raiseDrawEvent(const bool isContextActivated);
	virtual bool resize(const int x1, const int y1, const int x2, const int y2);

	virtual bool initializeView();

private:
	void draw(swl::WglContextBase &ctx, swl::ViewCamera3 &camera);

	virtual bool doPrepareRendering();
	virtual bool doRenderStockScene();
	virtual bool doRenderScene();

private:
	boost::scoped_ptr<swl::WglContextBase> viewContext_;
	boost::scoped_ptr<swl::OglCamera> viewCamera_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnPaint();
};

#ifndef _DEBUG  // debug version in OglViewTestView.cpp
inline COglViewTestDoc* COglViewTestView::GetDocument() const
   { return reinterpret_cast<COglViewTestDoc*>(m_pDocument); }
#endif

