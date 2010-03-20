// WglSceneGraphView.h : interface of the CWglSceneGraphView class
//


#pragma once

#include "swl/winview/WglViewBase.h"
#include "swl/view/ViewEventController.h"
#include "swl/glutil/IGLSceneVisitor.h"
#include "swl/graphics/SceneNode.h"
#include <boost/smart_ptr.hpp>

namespace swl {
class WglContextBase;
class GLCamera;
struct ViewStateMachine;
}

class CWglSceneGraphView : public CView, public swl::WglViewBase
{
protected: // create from serialization only
	CWglSceneGraphView();
	DECLARE_DYNCREATE(CWglSceneGraphView)

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
	virtual ~CWglSceneGraphView();
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
	/*virtual*/ void generateDisplayListName(const bool isContextActivated);
	/*virtual*/ void deleteDisplayListName(const bool isContextActivated);
	/*virtual*/ bool isDisplayListUsed() const;

	/*virtual*/ void pickObject(const int x, const int y, const bool isTemporary = false);
	/*virtual*/ void pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary = false);

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

	//-------------------------------------------------------------------------
	//

	void contructSceneGraph();

	void processToPickObject(const int x, const int y, const int width, const int height, const bool isTemporary = false);
	unsigned int processHits(const int hitCount, const unsigned int *buffer) const;

private:
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling

	//swl::ViewEventController viewController_;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state

	boost::scoped_ptr<swl::ViewStateMachine> viewStateFsm_;

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

	//
	typedef swl::IGLSceneVisitor visitor_type;
	typedef swl::ISceneNode<visitor_type> scene_node_type;
	scene_node_type::node_type rootSceneNode_;

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

#ifndef _DEBUG  // debug version in WglSceneGraphView.cpp
inline CWglViewTestDoc* CWglSceneGraphView::GetDocument() const
   { return reinterpret_cast<CWglViewTestDoc*>(m_pDocument); }
#endif

