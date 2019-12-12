#include "swl/Config.h"
#include "swl/machine_vision/NonMaximumSuppression.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
//#include <execution>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

void computeNonMaximumSuppression_test()
{
	const std::string image_filepath("../data/FFT.png");

	// Load an image.
	const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_GRAYSCALE));
	//const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load an image file: " << image_filepath << std::endl;
		return;
	}

	cv::Mat result_img;
	swl::NonMaximumSuppression::computeNonMaximumSuppression(img, result_img);

	cv::imshow("Non-Maximum Suppressed Image", result_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void computeNonMaximumSuppression2_test()
{
	const std::string image_filepath("../data/FFT.png");

	// Load an image.
	const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_GRAYSCALE));
	//const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load an image file: " << image_filepath << std::endl;
		return;
	}

	cv::Mat nms;
	swl::NonMaximumSuppression::computeNonMaximumSuppression(img, 5, nms, cv::Mat());

	cv::imshow("Non-Maximum Suppressed Image", nms);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void findMountainChain_test()
{
	const std::string image_filepath("../data/FFT.png");

	// Load an image.
	const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_GRAYSCALE));
	//const cv::Mat img(cv::imread(image_filepath, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Failed to load an image file: " << image_filepath << std::endl;
		return;
	}

	cv::Mat result_img;
	swl::NonMaximumSuppression::findMountainChain(img, result_img);

	cv::imshow("Mountain Chain Image", result_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

void non_maximum_suppression_test()
{
	local::computeNonMaximumSuppression_test();
	//local::computeNonMaximumSuppression2_test();

	//local::findMountainChain_test();  // FIXME [fix] >> Not correctly working.
}
