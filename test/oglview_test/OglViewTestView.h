// OglViewTestView.h : interface of the COglViewTestView class
//


#pragma once

namespace swl {
class WglContextBase;
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

private:
	void draw(swl::WglContextBase &ctx);

	virtual bool initializeView();

	virtual bool doPrepareRendering();
	virtual bool doRenderStockScene();
	virtual bool doRenderScene();

private:
	swl::OglCamera *viewCamera_;
	swl::WglContextBase *viewContext_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
};

#ifndef _DEBUG  // debug version in OglViewTestView.cpp
inline COglViewTestDoc* COglViewTestView::GetDocument() const
   { return reinterpret_cast<COglViewTestDoc*>(m_pDocument); }
#endif

