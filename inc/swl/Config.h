#if !defined(__SWL__CONFIG__H_)
#define __SWL__CONFIG__H_ 1


//-----------------------------------------------------------------------------
//

#if defined(WIN32)

//	version			_MSC_VER
//-----------------------------
//	VC 2008			1500
//	VC 2005			1400
//	VC 2003			1310
//	VC 2002			1300
//	VC 6			1200
//	VC 5			1100
//	VC 4.x			1000

//	Minimum system required									Minimum value for _WIN32_WINNT and WINVER
//-----------------------------------------------------------------------------------------------------------
//	Windows Server 2008										0x0600
//	Windows Vista											0x0600
//	Windows Server 2003 with SP1, Windows XP with SP2		0x0502
//	Windows Server 2003, Windows XP							0x0501
//	Windows 2000											0x0500
//	Windows NT												0x0400

//	Minimum system required									Minimum value for _WIN32_WINDOWS and WINVER
//-----------------------------------------------------------------------------------------------------------
//	Windows Me												0x0500
//	Windows 98												0x0410
//	Windows 95												0x0400

//	Minimum system required							Value for NTDDI_VERSION
//-----------------------------------------------------------------------------
//	Windows Server 2008								NTDDI_WS08
//	Windows Vista with Service Pack 1 (SP1)			NTDDI_VISTASP1
//	Windows Vista									NTDDI_VISTA
//	Windows Server 2003 with Service Pack 1 (SP1)	NTDDI_WS03SP1
//	Windows Server 2003								NTDDI_WS03
//	Windows XP with Service Pack 2 (SP2)			NTDDI_WINXPSP2
//	Windows XP with Service Pack 1 (SP1)			NTDDI_WINXPSP1
//	Windows XP										NTDDI_WINXP
//	Windows 2000 with Service Pack 4 (SP4)			NTDDI_WIN2KSP4
//	Windows 2000 with Service Pack 3 (SP3)			NTDDI_WIN2KSP3
//	Windows 2000 with Service Pack 2 (SP2)			NTDDI_WIN2KSP2
//	Windows 2000 with Service Pack 1 (SP1)			NTDDI_WIN2KSP1
//	Windows 2000									NTDDI_WIN2K

//	Minimum version required				Minimum value of _WIN32_IE
//-----------------------------------------------------------------------------
//	Internet Explorer 7.0					0x0700
//	Internet Explorer 6.0 SP2				0x0603
//	Internet Explorer 6.0 SP1				0x0601
//	Internet Explorer 6.0					0x0600
//	Internet Explorer 5.5					0x0550
//	Internet Explorer 5.01					0x0501
//	Internet Explorer 5.0, 5.0a, 5.0b		0x0500
//	Internet Explorer 4.01					0x0401
//	Internet Explorer 4.0					0x0400
//	Internet Explorer 3.0, 3.01, 3.02		0x0300

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN		// Exclude rarely-used stuff from Windows headers
#endif

// Modify the following defines if you have to target a platform prior to the ones specified below.
// Refer to MSDN for the latest info on corresponding values for different platforms.
#ifndef WINVER				// Allow use of features specific to Windows XP or later.
#define WINVER 0x0501		// Change this to the appropriate value to target other versions of Windows.
#endif

#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.                   
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif						

#ifndef _WIN32_WINDOWS		// Allow use of features specific to Windows 98 or later.
#define _WIN32_WINDOWS 0x0410 // Change this to the appropriate value to target Windows Me or later.
#endif

#ifndef _WIN32_IE			// Allow use of features specific to IE 6.0 or later.
#define _WIN32_IE 0x0600	// Change this to the appropriate value to target other versions of IE.
#endif

#endif  // WIN32

//-----------------------------------------------------------------------------
//

#if defined(_MSC_VER)

#if (_MSC_VER >= 1500)  // VC2008
#elif (_MSC_VER >= 1400)  // VC2005
#elif (_MSC_VER >= 1310)  // VC2003
#elif (_MSC_VER >= 1300)  // VC2002
#elif (_MSC_VER >= 1200)  // VC6/eVC4
#elif (_MSC_VER >= 1100)  // VC5
#elif (_MSC_VER >= 1000)  // VC4
#else
#endif

#if (_MSC_VER < 1100)  // VC4 and before
#elif (_MSC_VER < 1200)  // VC5
#elif (_MSC_VER < 1300)  // VC6/eVC4 
#elif (_MSC_VER < 1310)  // VC2002
#elif (_MSC_VER < 1400)  // VC2003
#elif (_MSC_VER < 1500)  // VC2005
#elif (_MSC_VER >= 1500)  // VC2008 and higher
#endif

#endif  // _MSC_VER


#endif  // __SWL__CONFIG__H_
