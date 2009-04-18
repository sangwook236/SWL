// WinViewTestView.h : interface of the CWinViewTestView class
//


#pragma once

#include <boost/smart_ptr.hpp>
#include <deque>

namespace swl {
class GdiplusBitmapBufferedContext;
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
	virtual bool raiseDrawEvent(const bool isContextActivated);
	virtual bool resize(const int x1, const int y1, const int x2, const int y2);

private:
	void test1();
	void test2();
	void test3();
	void test4();
	void test5();

private:
	typedef std::pair<int, int> datum_type;
	typedef std::deque<datum_type> data_type;

	boost::scoped_ptr<swl::GdiplusBitmapBufferedContext> viewContext_;

	data_type data1_;
	data_type data2_;
	size_t idx_;
	UINT timeInterval_;

	int drawMode_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnSize(UINT nType, int cx, int cy);
};

#ifndef _DEBUG  // debug version in WinViewTestView.cpp
inline CWinViewTestDoc* CWinViewTestView::GetDocument() const
   { return reinterpret_cast<CWinViewTestDoc*>(m_pDocument); }
#endif

