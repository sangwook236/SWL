//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include "swl/Config.h"
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


int main(int argc, char *argv[])
{
	void test_boost_serial_port();
	void test_windows_serial_port();

	try
	{
		test_boost_serial_port();
		//test_windows_serial_port();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		return -1;
	}

	//std::cout << "press any key to exit ..." << std::endl;
	//std::cin.get();

	return 0;
}

