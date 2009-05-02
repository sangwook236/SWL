// ChildFrm.cpp : implementation of the CChildFrame class
//
#include "stdafx.h"
#include "WinViewTest.h"

#include "ChildFrm.h"
#include "WinViewTestDoc.h"
#include "WinViewTestView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CChildFrame

IMPLEMENT_DYNCREATE(CChildFrame, CMDIChildWnd)

BEGIN_MESSAGE_MAP(CChildFrame, CMDIChildWnd)
	ON_COMMAND(ID_VIEWMODE_PAN, &CChildFrame::OnViewmodePan)
	ON_COMMAND(ID_VIEWMODE_ROTATE, &CChildFrame::OnViewmodeRotate)
	ON_COMMAND(ID_VIEWMODE_ZOOMREGION, &CChildFrame::OnViewmodeZoomregion)
	ON_COMMAND(ID_VIEWMODE_ZOOMALL, &CChildFrame::OnViewmodeZoomall)
	ON_COMMAND(ID_VIEWMODE_ZOOMIN, &CChildFrame::OnViewmodeZoomin)
	ON_COMMAND(ID_VIEWMODE_ZOOMOUT, &CChildFrame::OnViewmodeZoomout)
END_MESSAGE_MAP()


// CChildFrame construction/destruction

CChildFrame::CChildFrame()
{
	// TODO: add member initialization code here
}

CChildFrame::~CChildFrame()
{
}


BOOL CChildFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying the CREATESTRUCT cs
	if( !CMDIChildWnd::PreCreateWindow(cs) )
		return FALSE;

	return TRUE;
}


// CChildFrame diagnostics

#ifdef _DEBUG
void CChildFrame::AssertValid() const
{
	CMDIChildWnd::AssertValid();
}

void CChildFrame::Dump(CDumpContext& dc) const
{
	CMDIChildWnd::Dump(dc);
}

#endif //_DEBUG


// CChildFrame message handlers

void CChildFrame::OnViewmodePan()
{
	//-------------------------------------------------------------------------
	// This code is required for view state
	if (GetActiveView())
		((CWinViewTestView*)GetActiveView())->triggerPanEvent();
}

void CChildFrame::OnViewmodeRotate()
{
	//-------------------------------------------------------------------------
	// This code is required for view state
	if (GetActiveView())
		((CWinViewTestView*)GetActiveView())->triggerRotateEvent();
}

void CChildFrame::OnViewmodeZoomregion()
{
	//-------------------------------------------------------------------------
	// This code is required for view state
	if (GetActiveView())
		((CWinViewTestView*)GetActiveView())->triggerZoomRegionEvent();
}

void CChildFrame::OnViewmodeZoomall()
{
	//-------------------------------------------------------------------------
	// This code is required for view state
	if (GetActiveView())
		((CWinViewTestView*)GetActiveView())->triggerZoomAllEvent();
}

void CChildFrame::OnViewmodeZoomin()
{
	//-------------------------------------------------------------------------
	// This code is required for view state
	if (GetActiveView())
		((CWinViewTestView*)GetActiveView())->triggerZoomInEvent();
}

void CChildFrame::OnViewmodeZoomout()
{
	//-------------------------------------------------------------------------
	// This code is required for view state
	if (GetActiveView())
		((CWinViewTestView*)GetActiveView())->triggerZoomOutEvent();
}
