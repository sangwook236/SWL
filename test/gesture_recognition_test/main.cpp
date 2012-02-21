#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	void gestureRecognitionByHistogram();
	void gestureRecognitionBasedTemporalOrientationHistogram();

	try
	{
		//gestureRecognitionByHistogram();
		gestureRecognitionBasedTemporalOrientationHistogram();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception occurred !!!: " << e.what() << std::endl;
		//std::cout << "OpenCV exception occurred !!!: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception occurred !!!:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

	return 0;
}
