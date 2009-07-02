#if !defined(__SWL__COMPILER_WARNING__H_)
#define __SWL__COMPILER_WARNING__H_ 1


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

#if (_MSC_VER < 1200)  // VC5 and before
#  pragma warning(disable : 4018)  // signed/unsigned mismatch
#  pragma warning(disable : 4290)  // c++ exception specification ignored
#  pragma warning(disable : 4389)  // '==' : signed/unsigned mismatch
#  pragma warning(disable : 4610)  // struct '...' can never be instantiated - user defined constructor required
#endif

#if (_MSC_VER < 1300)  // VC6/eVC4 
#  pragma warning(disable : 4097)  // typedef-name used as based class of (...)
#  pragma warning(disable : 4251)  // DLL interface needed
#  pragma warning(disable : 4284)  // for -> operator
#  pragma warning(disable : 4503)  // decorated name length exceeded, name was truncated
#  pragma warning(disable : 4514)  // unreferenced inline function has been removed
#  pragma warning(disable : 4660)  // template-class specialization '...' is already instantiated
#  pragma warning(disable : 4701)  // local variable 'base' may be used without having been initialized
#  pragma warning(disable : 4710)  // function (...) not inlined
#  pragma warning(disable : 4786)  // identifier truncated to 255 characters
#endif

#if (_MSC_VER <= 1310)
#  pragma warning(disable : 4511)  // copy constructor cannot be generated
#endif

#if (_MSC_VER < 1300) && defined(UNDER_CE)
#  pragma warning(disable : 4201)  // nonstandard extension used : nameless struct/union
#  pragma warning(disable : 4214)  // nonstandard extension used : bit field types other than int
#endif

#pragma warning(disable : 4075)  // initializers put in unrecognized initialization area
// This warning is disable only for the c_locale_win32.c file compilation:
#pragma warning(disable : 4100)  // unreferenced formal parameter
#pragma warning(disable : 4127)  // conditional expression is constant
#pragma warning(disable : 4146)  // unary minus applied to unsigned type
#pragma warning(disable : 4245)  // conversion from 'enum ' to 'unsigned int', signed/unsigned mismatch
#pragma warning(disable : 4244)  // implicit conversion: possible loss of data
#pragma warning(disable : 4512)  // assignment operator could not be generated
#pragma warning(disable : 4571)  // catch(...) blocks compiled with /EHs do not catch or re-throw Structured Exceptions
#pragma warning(disable : 4702)  // unreachable code (appears in release with warning level4)

// dums: This warning, signaling deprecated C functions like strncpy,
// will have to be fixed one day:
#pragma warning(disable : 4996)


#endif  // __SWL__COMPILER_WARNING__H_
