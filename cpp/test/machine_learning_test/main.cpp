//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <cstdlib>


int main(int argc, char *argv[])
{
	void td_learning();
	void td_lambda();

	int retval = EXIT_SUCCESS;
	try
	{
		td_learning();
		td_lambda();
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
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
