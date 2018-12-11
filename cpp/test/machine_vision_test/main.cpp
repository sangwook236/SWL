//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <cstdlib>


int main(int argc, char *argv[])
{
	void convolution_test();
	void morphology_test();
	void image_filter_test();
	void scale_space_test();

	void boundary_extraction();

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

		//-----------------------------------------------------------
		// Convolution & image filtering.
		convolution_test();
		morphology_test();

		//-----------------------------------------------------------
		// Image filter.
		//image_filter_test();

		//-----------------------------------------------------------
		// Scale space representation.
		//scale_space_test();

		//-----------------------------------------------------------
		// Application.

		// Boundary extraction and weighting.
		//boundary_extraction();
	}
	catch (const cv::Exception &ex)
	{
		//std::cout << "OpenCV exception caught: " << ex.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(ex.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tDescription: " << ex.err << std::endl
			<< "\tLine:        " << ex.line << std::endl
			<< "\tFunction:    " << ex.func << std::endl
			<< "\tFile:        " << ex.file << std::endl;

		retval = EXIT_FAILURE;
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

