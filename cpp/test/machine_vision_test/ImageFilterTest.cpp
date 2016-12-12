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
	const cv::Mat img(cv::imread(img_filename, cv::IMREAD_GRAYSCALE));
	//const cv::Mat img(cv::imread(img_filename, cv::IMREAD_COLOR));
	if (img.empty())
	{
		std::cerr << "Fail to load image file: " << img_filename << std::endl;
		return;
	}

	cv::Mat img_double;
	img.convertTo(img_double, CV_64FC1);
	cv::normalize(img_double, img_double, 0.0, 1.0, cv::NORM_MINMAX);

	cv::imshow("Image", img_double);

	//
	//swl::ImageFilter::GaussianOperator operation;
	//swl::ImageFilter::DerivativeOfGaussianOperator operation;
	//swl::ImageFilter::LaplacianOfGaussianOperator operation;
	swl::ImageFilter::RidgenessOperator operation;  // Ridge and valley.
	//swl::ImageFilter::CornernessOperator operation;
	//swl::ImageFilter::IsophoteCurvatureOperator operation;
	//swl::ImageFilter::FlowlineCurvatureOperator operation;
	//swl::ImageFilter::UnflatnessOperator operation;
	//swl::ImageFilter::UmbilicityOperator operation;

	//const size_t apertureSize = 3;
	for (auto apertureSize : { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 })
	{
		const double sigma = 0.3 * ((double(apertureSize) - 1.0) * 0.5 - 1.0) + 0.8;
		//for (auto sigma : { 1.0, 3.0, 5.0, 7.0, 9.0 })
		{
			std::cout << "Aperture size = " << apertureSize << ", sigma = " << sigma << std::endl;

			// Filter the image.
			cv::Mat filtered(operation(img_double, apertureSize, sigma));

			// Show the result.
			cv::normalize(filtered, filtered, 0.0, 1.0, cv::NORM_MINMAX);
			cv::imshow("Filtered image", filtered);

			cv::waitKey(0);
		}
	}

	cv::destroyAllWindows();
}

void simple_example()
{
	const int IMG_WIDTH = 400, IMG_HEIGHT = 400;
	cv::Mat img = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);

	// Create an image.
	cv::rectangle(img, cv::Point(IMG_WIDTH / 2 - 50, 0), cv::Point(IMG_WIDTH / 2 + 50, IMG_HEIGHT), cv::Scalar::all(255), cv::FILLED, cv::LINE_8);
	cv::rectangle(img, cv::Point(0, IMG_HEIGHT / 2 - 50), cv::Point(IMG_WIDTH, IMG_HEIGHT / 2 + 50), cv::Scalar::all(255), cv::FILLED, cv::LINE_8);

	//
	cv::Mat dist;
	cv::distanceTransform(img, dist, cv::DIST_L2, 3);

	cv::Mat dist_double;
	dist.convertTo(dist_double, CV_64FC1);
	cv::normalize(dist_double, dist_double, 0.0, 1.0, cv::NORM_MINMAX);

	cv::imshow("Distance transform", dist_double);

	//
	//swl::ImageFilter::GaussianOperator operation;
	//swl::ImageFilter::DerivativeOfGaussianOperator operation;
	//swl::ImageFilter::LaplacianOfGaussianOperator operation;
	swl::ImageFilter::RidgenessOperator operation;  // Ridge and valley.
	//swl::ImageFilter::CornernessOperator operation;
	//swl::ImageFilter::IsophoteCurvatureOperator operation;
	//swl::ImageFilter::FlowlineCurvatureOperator operation;
	//swl::ImageFilter::UnflatnessOperator operation;
	//swl::ImageFilter::UmbilicityOperator operation;

	//const size_t apertureSize = 3;
	for (auto apertureSize : { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 })
	{
		const double sigma = 0.3 * ((double(apertureSize) - 1.0) * 0.5 - 1.0) + 0.8;
		//for (auto sigma : { 1.0, 3.0, 5.0, 7.0, 9.0 })
		{
			std::cout << "Aperture size = " << apertureSize << ", sigma = " << sigma << std::endl;

			// Filter the image.
			cv::Mat filtered(operation(dist_double, apertureSize, sigma));

			// Show the result.
			cv::normalize(filtered, filtered, 0.0, 1.0, cv::NORM_MINMAX);
			cv::imshow("Filtered image", filtered);

			cv::waitKey(0);
		}
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

void image_filter_test()
{
	//local::very_simple_example();
	local::simple_example();
}
