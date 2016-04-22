//#include "stdafx.h"
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#include <vld/vld.h>
#endif
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <cstdlib>


int main(int argc, char *argv[])
{
	void scale_space_test();
	
	int retval = EXIT_SUCCESS;
	try
	{
		bool canUseGPU = false;
		cv::theRNG();

#if 0
		if (cv::gpu::getCudaEnabledDeviceCount() > 0)
		{
			canUseGPU = true;
			std::cout << "GPU info:" << std::endl;
			cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
		}
		else
			std::cout << "GPU not found ..." << std::endl;
#endif
		
		// Scale space representation -------------------------------
		scale_space_test();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		retval = EXIT_FAILURE;
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
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	//std::cout << "press any key to exit ..." << std::endl;
	//std::cin.get();

	return retval;
}

