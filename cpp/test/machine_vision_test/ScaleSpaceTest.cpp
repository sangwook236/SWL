#include "swl/Config.h"
#include "swl/machine_vision/ScaleSpace.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

void scale_space(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale)
{
	const std::string output_filename_appendix(".scale_space.png");

	const std::string windowName("scale space");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const bool useScaleSpacePyramid = false;
	const long firstOctaveIndex = 0, lastOctaveIndex = 5;
	const std::size_t octaveResolution = 2;
	const long firstSublevelIndex = 0, lastSublevelIndex = octaveResolution - 1;
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			for (long sublevelIndex = firstSublevelIndex; sublevelIndex <= lastSublevelIndex; ++sublevelIndex)
			{
				std::cout << "scale space: " << octaveIndex << "-th octave, " << sublevelIndex << "-th sublevel" << std::endl;

				// calculate scale space.
				std::cout << "\tstart processing scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledImage(img, octaveIndex, sublevelIndex, useScaleSpacePyramid));
				std::cout << "\tend processing scale space..." << std::endl;

				if (scaled.empty()) continue;

				// show results.
				cv::imshow(windowName, scaled);

				// save results.
				std::cout << "\tsaving output image..." << std::endl;
				cv::imwrite(*cit + output_filename_appendix, scaled);

				const unsigned char key = cv::waitKey(0);
				if (27 == key)
					break;
			}
		}
	}

	cv::destroyAllWindows();
}

void gradient_scale_space(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale)
{
	const std::string output_filename_appendix(".gradient_scale_space.png");

	const std::string windowName("gradient scale space");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const bool useScaleSpacePyramid = false;
	const long firstOctaveIndex = 0, lastOctaveIndex = 5;
	const std::size_t octaveResolution = 2;
	const long firstSublevelIndex = 0, lastSublevelIndex = octaveResolution - 1;
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			for (long sublevelIndex = firstSublevelIndex; sublevelIndex <= lastSublevelIndex; ++sublevelIndex)
			{
				std::cout << "gradient scale space: " << octaveIndex << "-th octave, " << sublevelIndex << "-th sublevel" << std::endl;

				// calculate scale space.
				std::cout << "\tstart processing gradient scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledGradientImage(img, octaveIndex, sublevelIndex, useScaleSpacePyramid));
				std::cout << "\tend processing gradient scale space..." << std::endl;

				if (scaled.empty()) continue;

				// show results.
				cv::imshow(windowName, scaled);

				// save results.
				std::cout << "\tsaving output image..." << std::endl;
				cv::imwrite(*cit + output_filename_appendix, scaled);

				const unsigned char key = cv::waitKey(0);
				if (27 == key)
					break;
			}
		}
	}

	cv::destroyAllWindows();
}

void laplacian_scale_space(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale)
{
	const std::string output_filename_appendix(".laplacian_scale_space.png");

	const std::string windowName("Laplacian scale space");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const bool useScaleSpacePyramid = false;
	const long firstOctaveIndex = 0, lastOctaveIndex = 5;
	const std::size_t octaveResolution = 2;
	const long firstSublevelIndex = 0, lastSublevelIndex = octaveResolution - 1;
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			for (long sublevelIndex = firstSublevelIndex; sublevelIndex <= lastSublevelIndex; ++sublevelIndex)
			{
				std::cout << "Laplacian scale space: " << octaveIndex << "-th octave, " << sublevelIndex << "-th sublevel" << std::endl;

				// calculate scale space.
				std::cout << "\tstart processing Laplacian scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledGradientImage(img, octaveIndex, sublevelIndex, useScaleSpacePyramid));
				std::cout << "\tend processing Laplacian scale space..." << std::endl;

				if (scaled.empty()) continue;

				// show results.
				cv::imshow(windowName, scaled);

				// save results.
				std::cout << "\tsaving output image..." << std::endl;
				cv::imwrite(*cit + output_filename_appendix, scaled);

				const unsigned char key = cv::waitKey(0);
				if (27 == key)
					break;
			}
		}
	}

	cv::destroyAllWindows();
}

void derivative_scale_space(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale)
{
	const std::string output_filename_appendix(".derivative_scale_space.png");

	const std::string windowName("derivative scale space");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	// callback.
	auto derivativeOperator = [](const cv::Mat &img, std::size_t kernelSize, const double sigma) -> cv::Mat
	{
		// FIXME [check] >> when applying a filter, does its kernel size increase as its sigma increases?
		//	REF [site] >> cv::getGaussianKernel() in OpenCV.
		//	kernelSize = round(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.0)
		//const std::size_t kernelSize = (std::size_t)std::floor(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.5);  // "+ 0.5" for rounding.
		const int halfKernelSize = (int)kernelSize / 2;

#if 0
		// Derivative of Gaussian wrt x- & y-axes.

		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
		cv::Mat kernelX(kernelSize, kernelSize, CV_64F), kernelY(kernelSize, kernelSize, CV_64F);
		for (int r = -halfKernelSize, rr = 0; r <= halfKernelSize; ++r, ++rr)
			for (int c = -halfKernelSize, cc = 0; c <= halfKernelSize; ++c, ++cc)
			{
				const double exp = std::exp(-(double(r)*double(r) + double(c)*double(c)) / _2_sigma2);
				// TODO [check] >> x- & y-axis derivative.
				kernelX.at<double>(rr, cc) = -double(c) * exp / _2_pi_sigma4;
				kernelY.at<double>(rr, cc) = -double(r) * exp / _2_pi_sigma4;
			}

		cv::Mat filteredX, filteredY;
		cv::filter2D(img, filteredX, CV_64F, kernelX, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, filteredY, CV_64F, kernelY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat scaled;
		cv::magnitude(filteredX, filteredY, scaled);
#elif 0
		// Laplacian of Gaussian.

		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
		cv::Mat kernelXX(kernelSize, kernelSize, CV_64F), kernelYY(kernelSize, kernelSize, CV_64F);
		for (int r = -halfKernelSize, rr = 0; r <= halfKernelSize; ++r, ++rr)
			for (int c = -halfKernelSize, cc = 0; c <= halfKernelSize; ++c, ++cc)
			{
				const double r2 = double(r) * double(r), c2 = double(c) * double(c);
				const double exp = std::exp(-(r2 + c2) / _2_sigma2);
				// TODO [check] >> x- & y-axis derivative.
				kernelXX.at<double>(rr, cc) = (c2 / sigma2 - 1.0) * exp / _2_pi_sigma4;
				kernelYY.at<double>(rr, cc) = (r2 / sigma2 - 1.0) * exp / _2_pi_sigma4;
			}

		cv::Mat filteredX, filteredY;
		cv::filter2D(img, filteredX, CV_64F, kernelXX, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, filteredY, CV_64F, kernelYY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat scaled;
		cv::add(filteredX, filteredY, scaled);

		return scaled;
#elif 0
		// A local coordinate frame based on the gradient vector w and its right-handed normal vector v.
		//	REF [book] >> section 9.1.2 (p. 254) in "Digital and Medical Image Processing", 2005.
		//	REF [book] >> Figure 9.10 & 9.11 (p. 260) in "Digital and Medical Image Processing", 2005.

		// FIXME [implement] >> first we have to find v- & w-axes.

		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
		cv::Mat kernelVV(kernelSize, kernelSize, CV_64F), kernelWW(kernelSize, kernelSize, CV_64F);
		for (int w = -halfKernelSize, ww = 0; w <= halfKernelSize; ++w, ++ww)
			for (int v = -halfKernelSize, vv = 0; v <= halfKernelSize; ++v, ++vv)
			{
				const double w2 = double(w) * double(w), v2 = double(v) * double(v);
				const double exp = std::exp(-(w2 + v2) / _2_sigma2);
				// TODO [check] >> v- & w-axis derivative.
				kernelVV.at<double>(ww, vv) = (v2 / sigma2 - 1.0) * exp / _2_pi_sigma4;
				kernelWW.at<double>(ww, vv) = (w2 / sigma2 - 1.0) * exp / _2_pi_sigma4;
			}

		cv::Mat filteredV, filteredW;
		cv::filter2D(img, filteredV, CV_64F, kernelVV, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		//cv::filter2D(img, filteredW, CV_64F, kernelWW, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		return filteredV;
		//return cv::abs(filteredW);
#elif 1
		// A local coordinate frame based on the gradient vector w and its right-handed normal vector v.
		//	REF [book] >> Figure 9.12 (p. 261) in "Digital and Medical Image Processing", 2005.

		// FIXME [implement] >> first we have to find v- & w-axes.

		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
		cv::Mat kernelVV(kernelSize, kernelSize, CV_64F), kernelW(kernelSize, kernelSize, CV_64F);
		for (int w = -halfKernelSize, ww = 0; w <= halfKernelSize; ++w, ++ww)
			for (int v = -halfKernelSize, vv = 0; v <= halfKernelSize; ++v, ++vv)
			{
				const double w2 = double(w) * double(w), v2 = double(v) * double(v);
				const double exp = std::exp(-(w2 + v2) / _2_sigma2);
				// TODO [check] >> v- & w-axis derivative.
				kernelVV.at<double>(ww, vv) = (v2 / sigma2 - 1.0) * exp / _2_pi_sigma4;
				kernelW.at<double>(ww, vv) = -double(w) * exp / _2_pi_sigma4;
			}

		cv::Mat filteredV, filteredW;
		cv::filter2D(img, filteredV, CV_64F, kernelVV, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, filteredW, CV_64F, kernelW, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat scaled;
		cv::divide(filteredV, filteredW, scaled, -1.0);

		return scaled;
#endif
	};

	const bool useScaleSpacePyramid = false;
	const long firstOctaveIndex = 0, lastOctaveIndex = 5;
	const std::size_t octaveResolution = 2;
	const long firstSublevelIndex = 0, lastSublevelIndex = octaveResolution - 1;
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			for (long sublevelIndex = firstSublevelIndex; sublevelIndex <= lastSublevelIndex; ++sublevelIndex)
			{
				std::cout << "derivative scale space: " << octaveIndex << "-th octave, " << sublevelIndex << "-th sublevel" << std::endl;

				// calculate scale space.
				std::cout << "\tstart processing derivative scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledDerivativeImage(img, octaveIndex, sublevelIndex, derivativeOperator, useScaleSpacePyramid));
				std::cout << "\tend processing derivative scale space..." << std::endl;

				if (scaled.empty()) continue;

				// show results.
				cv::imshow(windowName, scaled);

				// save results.
				std::cout << "\tsaving output image..." << std::endl;
				cv::imwrite(*cit + output_filename_appendix, scaled);

				const unsigned char key = cv::waitKey(0);
				if (27 == key)
					break;
			}
		}
	}

	cv::destroyAllWindows();
}

void gaussian_pyramid(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale)
{
	const std::string output_filename_appendix(".gaussian_pyramid.png");

	const std::string windowName("Gaussian pyramid");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const long firstOctaveIndex = 0, lastOctaveIndex = 4;
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			std::cout << "Gaussian pyramid: " << octaveIndex << "-th octave" << std::endl;

			// calculate scale space.
			std::cout << "\tstart processing Gaussian pyramid..." << std::endl;
			const cv::Mat scaled(swl::ScaleSpace::getScaledImageInGaussianPyramid(img, kernelSize, baseScale, octaveIndex));
			std::cout << "\tend processing Gaussian pyramid..." << std::endl;

			if (scaled.empty()) continue;

			// show results.
			cv::imshow(windowName, scaled);

			// save results.
			std::cout << "\tsaving output image..." << std::endl;
			cv::imwrite(*cit + output_filename_appendix, scaled);

			const unsigned char key = cv::waitKey(0);
			if (27 == key)
				break;
		}
	}

	cv::destroyAllWindows();
}

void laplacian_pyramid(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale)
{
	const std::string output_filename_appendix(".laplacian_pyramid.png");

	const std::string windowName("Laplacian pyramid");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const long firstOctaveIndex = 0, lastOctaveIndex = 4;
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			std::cout << "Laplacian pyramid: " << octaveIndex << "-th octave" << std::endl;

			// calculate scale space.
			std::cout << "\tstart processing Laplacian pyramid..." << std::endl;
			const cv::Mat scaled(swl::ScaleSpace::getScaledImageInLaplacianPyramid(img, kernelSize, baseScale, octaveIndex));
			std::cout << "\tend processing Laplacian pyramid..." << std::endl;

			if (scaled.empty()) continue;

			// show results.
			cv::imshow(windowName, scaled);

			// save results.
			std::cout << "\tsaving output image..." << std::endl;
			cv::imwrite(*cit + output_filename_appendix, scaled);

			const unsigned char key = cv::waitKey(0);
			if (27 == key)
				break;
		}
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

void scale_space_test()
{
	// REF [site] >> cv::getGaussianKernel() in OpenCV.
	//	sigma = 0.3 * ((kernelSize - 1.0) * 0.5 - 1.0) + 0.8

#if 1
	// Figure 9.3 (p. 250) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/whale_256x256.png");

		const std::size_t kernelSize = 7;
		const double baseScale = 3.0;

		local::scale_space(img_filenames, kernelSize, baseScale);
	}
#endif

#if 0
	// REF [book] >> Figure 9.7 (p. 258) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/whale_256x256.png");

		const std::size_t kernelSize3 = 3;
		const double baseScale3 = 0.3 * ((kernelSize3 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize9 = 9;
		const double baseScale9 = 0.3 * ((kernelSize9 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize17 = 17;
		const double baseScale17 = 0.3 * ((kernelSize17 - 1.0) * 0.5 - 1.0) + 0.8;

		local::gradient_scale_space(img_filenames, kernelSize3, baseScale3);
		local::gradient_scale_space(img_filenames, kernelSize9, baseScale9);
		local::gradient_scale_space(img_filenames, kernelSize17, baseScale17);
		local::laplacian_scale_space(img_filenames, kernelSize3, baseScale3);
		local::laplacian_scale_space(img_filenames, kernelSize9, baseScale9);
		local::laplacian_scale_space(img_filenames, kernelSize17, baseScale17);
	}
#endif

#if 0
	// REF [book] >> Figure 9.8 (p. 259) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/box_256x256_1.png");
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/box_256x256_2.png");

		const std::size_t kernelSize3 = 3;
		const double baseScale3 = 0.3 * ((kernelSize3 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize33 = 33;
		const double baseScale33 = 0.3 * ((kernelSize33 - 1.0) * 0.5 - 1.0) + 0.8;

		local::gradient_scale_space(img_filenames, kernelSize3, baseScale3);
		local::gradient_scale_space(img_filenames, kernelSize33, baseScale33);
	}
#endif

#if 0
	{
		std::list<std::string> img_filenames;
		//img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/box_256x256_1.png");
		//img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/box_256x256_2.png");
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/brain_256x256_1.png");
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/brain_256x256_2.png");

		const std::size_t kernelSize3 = 3;
		const double baseScale3 = 0.3 * ((kernelSize3 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize9 = 9;
		const double baseScale9 = 0.3 * ((kernelSize9 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize33 = 33;
		const double baseScale33 = 0.3 * ((kernelSize33 - 1.0) * 0.5 - 1.0) + 0.8;

		local::derivative_scale_space(img_filenames, kernelSize3, baseScale3);
		local::derivative_scale_space(img_filenames, kernelSize33, baseScale33);
		local::derivative_scale_space(img_filenames, kernelSize9, baseScale9);
	}
#endif

#if 0
	// REF [book] >> Figure 9.21 (p. 271) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("D:/work/swl_github/cpp/bin/data/machine_vision/whale_256x256.png");

		const std::size_t kernelSize = 3;
		const double baseScale = 0.3 * ((kernelSize - 1.0) * 0.5 - 1.0) + 0.8;

		local::gaussian_pyramid(img_filenames, kernelSize, baseScale);
		local::laplacian_pyramid(img_filenames, kernelSize, baseScale);
	}
#endif
}
