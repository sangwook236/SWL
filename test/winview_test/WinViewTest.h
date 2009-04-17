// WinViewTest.h : main header file for the WinViewTest application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CWinViewTestApp:
// See WinViewTest.cpp for the implementation of this class
//

class CWinViewTestApp : public CWinApp
{
public:
	CWinViewTestApp();

private:
	ULONG_PTR gdiplusToken_;

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
	virtual int ExitInstance();
};

extern CWinViewTestApp theApp;