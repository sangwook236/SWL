// WglViewTestView.h : interface of the CWglViewTestView class
//


#pragma once

#include "swl/winview/WglViewBase.h"
#include "swl/view/ViewEventController.h"
#include <boost/smart_ptr.hpp>

namespace swl {
class WglContextBase;
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

	/*virtual*/ bool createDisplayList(const bool isContextActivated);

	//-------------------------------------------------------------------------
	//

	void setPerspective(const bool isPerspective);
	bool isPerspective() const  {  return isPerspective_;  }
	void setWireFrame(const bool isWireFrame);
	bool isWireFrame() const  {  return isWireFrame_;  }

	void showGradientBackground(const bool isShown)  {  isGradientBackgroundUsed_ = isShown;  }
	bool isGradientBackgroundShown() const  {  return isGradientBackgroundUsed_;  }
	void showFloor(const bool isShown)  {  isFloorShown_ = isShown;  }
	bool isFloorShown() const  {  return isFloorShown_;  }
	void showColorBar(const bool isShown)  {  isColorBarShown_ = isShown;  }
	bool isColorBarShown() const  {  return isColorBarShown_;  }
	void showCoordinateFrame(const bool isShown)  {  isCoordinateFrameShown_ = isShown;  }
	bool isCoordinateFrameShown() const  {  return isCoordinateFrameShown_;  }

	void setFloorColor(const unsigned char r, const unsigned char g, const unsigned char b, const unsigned char a)
	{
		floorColor_[0] = r / 255.0f;  floorColor_[1] = g / 255.0f;
		floorColor_[2] = b / 255.0f;  floorColor_[3] = a / 255.0f;
	}
	void getFloorColor(unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a) const
	{
		r = (unsigned char)(floorColor_[0] * 255.0f + 0.5f);  g = (unsigned char)(floorColor_[1] * 255.0f + 0.5f);
		b = (unsigned char)(floorColor_[2] * 255.0f + 0.5f);  a = (unsigned char)(floorColor_[3] * 255.0f + 0.5f);
	}
	void setTopGradientBackgroundColor(const unsigned char r, const unsigned char g, const unsigned char b, const unsigned char a)
	{
		topGradientBackgroundColor_[0] = r / 255.0f;  topGradientBackgroundColor_[1] = g / 255.0f;
		topGradientBackgroundColor_[2] = b / 255.0f;  topGradientBackgroundColor_[3] = a / 255.0f;
	}
	void getTopGradientBackgroundColor(unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a) const
	{
		r = (unsigned char)(topGradientBackgroundColor_[0] * 255.0f + 0.5f);  g = (unsigned char)(topGradientBackgroundColor_[1] * 255.0f + 0.5f);
		b = (unsigned char)(topGradientBackgroundColor_[2] * 255.0f + 0.5f);  a = (unsigned char)(topGradientBackgroundColor_[3] * 255.0f + 0.5f);
	}
	void setBottomGradientBackgroundColor(const unsigned char r, const unsigned char g, const unsigned char b, const unsigned char a)
	{
		bottomGradientBackgroundColor_[0] = r / 255.0f;  bottomGradientBackgroundColor_[1] = g / 255.0f;
		bottomGradientBackgroundColor_[2] = b / 255.0f;  bottomGradientBackgroundColor_[3] = a / 255.0f;
	}
	void getBottomGradientBackgroundColor(unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a) const
	{
		r = (unsigned char)(bottomGradientBackgroundColor_[0] * 255.0f + 0.5f);  g = (unsigned char)(bottomGradientBackgroundColor_[1] * 255.0f + 0.5f);
		b = (unsigned char)(bottomGradientBackgroundColor_[2] * 255.0f + 0.5f);  a = (unsigned char)(bottomGradientBackgroundColor_[3] * 255.0f + 0.5f);
	}

	void setPrinting(const bool isPrinting)  {  isPrinting_ = isPrinting;  }
	bool isPrinting() const  {  return isPrinting_;  }

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	/*virtual*/ bool doPrepareRendering(const context_type &context, const camera_type &camera);
	/*virtual*/ bool doRenderStockScene(const context_type &context, const camera_type &camera);
	/*virtual*/ bool doRenderScene(const context_type &context, const camera_type &camera);

	void drawMainContent(const bool doesCreateDisplayList = false) const;
	void drawGradientBackground(const bool doesCreateDisplayList = false) const;
	void drawFloor(const bool doesCreateDisplayList = false) const;
	void drawColorBar(const bool doesCreateDisplayList = false) const;
	void drawCoordinateFrame(const bool doesCreateDisplayList = false) const;

	//
	void createDisplayLists(const unsigned int displayListNameBase) const;
#if defined(_UNICODE) || defined(UNICODE)
	bool createBitmapFonts(const std::wstring &fontName, const int fontSize) const;
	bool createOutlineFonts(const std::wstring &fontName, const int fontSize, const float depth) const;
#else
	bool createBitmapFonts(const std::string &fontName, const int fontSize) const;
	bool createOutlineFonts(const std::string &fontName, const int fontSize, const float depth) const;
#endif

	void drawCoordinateFrame(const float height, const int order[]) const;
#if defined(_UNICODE) || defined(UNICODE)
	void drawText(const float x, const float y, const float z, const std::wstring &str) const;
	void drawTextUsingGlutBitmapFonts(const float x, const float y, const float z, const std::wstring &str) const;
	void drawTextUsingGlutStrokeFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::wstring &str) const;
	void drawTextUsingBitmapFonts(const float x, const float y, const float z, const std::wstring &str) const;
	void drawTextUsingOutlineFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::wstring &str) const;
#else
	void drawText(const float x, const float y, const float z, const std::string &str) const;
	void drawTextUsingGlutBitmapFonts(const float x, const float y, const float z, const std::string &str) const;
	void drawTextUsingGlutStrokeFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::string &str) const;
	void drawTextUsingBitmapFonts(const float x, const float y, const float z, const std::string &str) const;
	void drawTextUsingOutlineFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::string &str) const;
#endif

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling

	//swl::ViewEventController viewController_;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state

	boost::scoped_ptr<swl::ViewStateMachine> viewStateFsm_;

	//-------------------------------------------------------------------------
	// OpenGL display list

	enum DisplayListNames { DLN_MAIN_CONTENT = 0, DLN_GRADIENT_BACKGROUND, DLN_COLOR_BAR };
	//enum DisplayListNames { DLN_MAIN_CONTENT = 0, DLN_FLOOR, DLN_GRADIENT_BACKGROUND, DLN_COLOR_BAR, DLN_COORDINATE_FRAME };
	static const int MAX_OPENGL_DISPLAY_LIST_COUNT = 4;
	static const int MAX_OPENGL_BITMAP_FONT_DISPLAY_LIST_COUNT = 96;
	static const int MAX_OPENGL_OUTLINE_FONT_DISPLAY_LIST_COUNT = 256;
	mutable GLYPHMETRICSFLOAT gmf_[256];

	//-------------------------------------------------------------------------
	//

	int drawMode_;
	bool useLocallyCreatedContext_;

	bool isPerspective_;
	bool isWireFrame_;

	bool isGradientBackgroundUsed_;
	bool isFloorShown_;
	bool isColorBarShown_;
	bool isCoordinateFrameShown_;

	float topGradientBackgroundColor_[4], bottomGradientBackgroundColor_[4];  // r, g, b, a
	float floorColor_[4];  // r, g, b, a

	bool isPrinting_;

	const int polygonFacing_;

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnDestroy();
	afx_msg void OnPaint();
	afx_msg void OnSize(UINT nType, int cx, int cy);
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

#ifndef _DEBUG  // debug version in WglViewTestView.cpp
inline CWglViewTestDoc* CWglViewTestView::GetDocument() const
   { return reinterpret_cast<CWglViewTestDoc*>(m_pDocument); }
#endif

