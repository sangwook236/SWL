// UnitTestMfc.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CUnitTestMfcApp:
// See UnitTestMfc.cpp for the implementation of this class
//

class CUnitTestMfcApp : public CWinApp
{
public:
	CUnitTestMfcApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CUnitTestMfcApp theApp;