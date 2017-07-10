//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <cstdlib>
#include <stdexcept>


int main(int argc, char *argv[])
{
	void wave_data();

	int retval = EXIT_SUCCESS;
	try
	{
		//-----------------------------------------------------------
		// Utility.
		wave_data();
	}
    catch (const std::bad_alloc &ex)
	{
		std::cout << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cout << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	//std::cout << "Press any key to exit ..." << std::endl;
	//std::cin.get();

	return retval;
}
