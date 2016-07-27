#include "swl/Config.h"
#include "swl/machine_vision/ImageFilter.h"
#include "swl/machine_vision/DerivativesOfGaussian.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

void very_simple_example()
{
	const std::string img_filename("./data/machine_vision/box_256x256_1.png");
	//const std::string img_filename("./data/machine_vision/box_256x256_2.png");

	// Load an image.
	std::cout << "Loading input image..." << std::endl;
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
	//const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Fail to load image file: " << img_filename << std::endl;
		return;
	}

	cv::Mat img_double;
	img.convertTo(img_double, CV_64FC1);

	// Filter the image.
	const double apertureSize = 3.0;
	const double baseScale = 0.3 * ((apertureSize - 1.0) * 0.5 - 1.0) + 0.8;

	//swl::ImageFilter::GaussianOperator operation;
	//swl::ImageFilter::DerivativeOfGaussianOperator operation;
	//swl::ImageFilter::LaplacianOfGaussianOperator operation;
	swl::ImageFilter::RidgenessOperator operation;
	//swl::ImageFilter::CornernessOperator operation;
	//swl::ImageFilter::IsophoteCurvatureOperator operation;
	//swl::ImageFilter::FlowlineCurvatureOperator operation;
	//swl::ImageFilter::UnflatnessOperator operation;
	//swl::ImageFilter::UmbilicityOperator operation;
	const cv::Mat ridge = operation(img_double, apertureSize, baseScale);

	// Show the result.
	const std::string windowName("Image Filtering");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName, ridge);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

void image_filter_test()
{
	local::very_simple_example();
}
