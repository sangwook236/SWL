// WglViewTest.h : main header file for the WglViewTest application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CWglViewTestApp:
// See WglViewTest.cpp for the implementation of this class
//

class CWglViewTestApp : public CWinApp
{
public:
	CWglViewTestApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CWglViewTestApp theApp;