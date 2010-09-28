
// swl_gps_aided_imu_filter_app.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// Cswl_gps_aided_imu_filter_appApp:
// See swl_gps_aided_imu_filter_app.cpp for the implementation of this class
//

class Cswl_gps_aided_imu_filter_appApp : public CWinAppEx
{
public:
	Cswl_gps_aided_imu_filter_appApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern Cswl_gps_aided_imu_filter_appApp theApp;