#include "stdafx.h"
#include "swl/Config.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


int wmain(int argc, wchar_t* argv[])
{
	void test_boost_serial_port();
	void test_windows_serial_port();

	test_boost_serial_port();
	//test_windows_serial_port();

	return 0;
}

