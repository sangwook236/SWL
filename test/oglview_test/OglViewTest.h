// OglViewTest.h : main header file for the OglViewTest application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// COglViewTestApp:
// See OglViewTest.cpp for the implementation of this class
//

class COglViewTestApp : public CWinApp
{
public:
	COglViewTestApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern COglViewTestApp theApp;