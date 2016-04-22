#include "swl/Config.h"
#include "swl/machine_vision/ScaleSpace.h"
#include "swl/machine_vision/DerivativesOfGaussian.h"
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
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_GRAYSCALE));
		//const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
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

				// Calculate scale space.
				std::cout << "\tstart processing scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledImage(img, octaveIndex, sublevelIndex, useScaleSpacePyramid));
				std::cout << "\tend processing scale space..." << std::endl;

				if (scaled.empty()) continue;

				// Output result.
				{
					// Rescale image.
					cv::Mat rescaled;
					{
						double minVal = 0.0, maxVal = 0.0;
						cv::minMaxLoc(scaled, &minVal, &maxVal);
						const double scaleFactor = 255.0 / (maxVal - minVal);
						scaled.convertTo(rescaled, CV_8U, scaleFactor, -scaleFactor * minVal);
						//scaled.convertTo(rescaled, CV_8U, -scaleFactor, scaleFactor * maxVal);  // reversed image.
					}

					// Show image.
					cv::imshow(windowName, rescaled);

					// Save image.
					std::cout << "\tsaving output image..." << std::endl;
					cv::imwrite(*cit + output_filename_appendix, rescaled);
				}

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
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize, baseScale);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_GRAYSCALE));
		//const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
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

				// Calculate scale space.
				std::cout << "\tstart processing gradient scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledGradientImage(img, octaveIndex, sublevelIndex, useScaleSpacePyramid));
				std::cout << "\tend processing gradient scale space..." << std::endl;

				if (scaled.empty()) continue;

				// Output result.
				{
					// Rescale image.
					cv::Mat rescaled;
					{
						double minVal = 0.0, maxVal = 0.0;
						cv::minMaxLoc(scaled, &minVal, &maxVal);
						const double scaleFactor = 255.0 / (maxVal - minVal);
						scaled.convertTo(rescaled, CV_8U, scaleFactor, -scaleFactor * minVal);
						//scaled.convertTo(rescaled, CV_8U, -scaleFactor, scaleFactor * maxVal);  // reversed image.
					}

					// Show image.
					cv::imshow(windowName, rescaled);

					// Save image.
					std::cout << "\tsaving output image..." << std::endl;
					cv::imwrite(*cit + output_filename_appendix, rescaled);
				}

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
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize, baseScale);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_GRAYSCALE));
		//const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
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

				// Calculate scale space.
				std::cout << "\tstart processing Laplacian scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledLaplacianImage(img, octaveIndex, sublevelIndex, useScaleSpacePyramid));
				std::cout << "\tend processing Laplacian scale space..." << std::endl;

				if (scaled.empty()) continue;

				// Output result.
				{
					// Rescale image.
					cv::Mat rescaled;
					{
						double minVal = 0.0, maxVal = 0.0;
						cv::minMaxLoc(scaled, &minVal, &maxVal);
						const double scaleFactor = 255.0 / (maxVal - minVal);
						scaled.convertTo(rescaled, CV_8U, scaleFactor, -scaleFactor * minVal);
						//scaled.convertTo(rescaled, CV_8U, -scaleFactor, scaleFactor * maxVal);  // reversed image.
					}

					// Show image.
					cv::imshow(windowName, rescaled);

					// Save image.
					std::cout << "\tsaving output image..." << std::endl;
					cv::imwrite(*cit + output_filename_appendix, rescaled);
				}

				const unsigned char key = cv::waitKey(0);
				if (27 == key)
					break;
			}
		}
	}

	cv::destroyAllWindows();
}

template<class RidgenessOperator>
void ridgeness_scale_space(const std::list<std::string>& img_filenames, const std::size_t kernelSize, const double baseScale, RidgenessOperator ridgenessOperator)
{
	const std::string output_filename_appendix(".ridgeness_scale_space.png");

	const std::string windowName("ridgeness scale space");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const bool useScaleSpacePyramid = false;
	const long firstOctaveIndex = 0, lastOctaveIndex = 5;
	const std::size_t octaveResolution = 2;
	const long firstSublevelIndex = 0, lastSublevelIndex = octaveResolution - 1;
	swl::ScaleSpace scaleSpace(firstOctaveIndex, lastOctaveIndex, firstSublevelIndex, lastSublevelIndex, octaveResolution, kernelSize, baseScale);
	for (std::list<std::string>::const_iterator cit = img_filenames.begin(); cit != img_filenames.end(); ++cit)
	{
		std::cout << "loading input image..." << std::endl;
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_GRAYSCALE));
		//const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			for (long sublevelIndex = firstSublevelIndex; sublevelIndex <= lastSublevelIndex; ++sublevelIndex)
			{
				std::cout << "ridgeness scale space: " << octaveIndex << "-th octave, " << sublevelIndex << "-th sublevel" << std::endl;

				// Calculate scale space.
				std::cout << "\tstart processing ridgeness scale space..." << std::endl;
				const cv::Mat scaled(scaleSpace.getScaledDerivativeImage(img, octaveIndex, sublevelIndex, ridgenessOperator, useScaleSpacePyramid));
				std::cout << "\tend processing ridgeness scale space..." << std::endl;

				if (scaled.empty()) continue;

				// Output result.
				{
					// Rescale image.
					cv::Mat rescaled;
					{
						double minVal = 0.0, maxVal = 0.0;
						cv::minMaxLoc(scaled, &minVal, &maxVal);
						const double scaleFactor = 255.0 / (maxVal - minVal);
						scaled.convertTo(rescaled, CV_8U, scaleFactor, -scaleFactor * minVal);
						//scaled.convertTo(rescaled, CV_8U, -scaleFactor, scaleFactor * maxVal);  // reversed image.
					}

#if 1
					const cv::Mat processed(rescaled);
#else
					// If the measure(F_vv or F_vv / F_w) is low, it means ridges. If the measure is high, it means valleys.
					cv::Mat processed(rescaled.rows, rescaled.cols, CV_8UC1, cv::Scalar::all(128));
					processed.setTo(cv::Scalar::all(0), rescaled < 0.3);
					processed.setTo(cv::Scalar::all(255), rescaled > 0.7);
#endif

					// Show image.
					cv::imshow(windowName, processed);

					// Save image.
					std::cout << "\tsaving output image..." << std::endl;
					cv::imwrite(*cit + output_filename_appendix, processed);
				}

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
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_GRAYSCALE));
		//const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			std::cout << "Gaussian pyramid: " << octaveIndex << "-th octave" << std::endl;

			// Calculate scale space.
			std::cout << "\tstart processing Gaussian pyramid..." << std::endl;
			const cv::Mat scaled(swl::ScaleSpace::getScaledImageInGaussianPyramid(img, kernelSize, baseScale, octaveIndex));
			std::cout << "\tend processing Gaussian pyramid..." << std::endl;

			if (scaled.empty()) continue;

			// Output result.
			{
				// Rescale image.
				cv::Mat rescaled;
				{
					double minVal = 0.0, maxVal = 0.0;
					cv::minMaxLoc(scaled, &minVal, &maxVal);
					const double scaleFactor = 255.0 / (maxVal - minVal);
					scaled.convertTo(rescaled, CV_8U, scaleFactor, -scaleFactor * minVal);
					//scaled.convertTo(rescaled, CV_8U, -scaleFactor, scaleFactor * maxVal);  // reversed image.
				}

				// Show image.
				cv::imshow(windowName, rescaled);

				// Save image.
				std::cout << "\tsaving output image..." << std::endl;
				cv::imwrite(*cit + output_filename_appendix, rescaled);
			}

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
		const cv::Mat img(cv::imread(*cit, cv::IMREAD_GRAYSCALE));
		//const cv::Mat img(cv::imread(*cit, cv::IMREAD_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *cit << std::endl;
			continue;
		}

		for (long octaveIndex = firstOctaveIndex; octaveIndex <= lastOctaveIndex; ++octaveIndex)
		{
			std::cout << "Laplacian pyramid: " << octaveIndex << "-th octave" << std::endl;

			// Calculate scale space.
			std::cout << "\tstart processing Laplacian pyramid..." << std::endl;
			const cv::Mat scaled(swl::ScaleSpace::getScaledImageInLaplacianPyramid(img, kernelSize, baseScale, octaveIndex));
			std::cout << "\tend processing Laplacian pyramid..." << std::endl;

			if (scaled.empty()) continue;

			// Output result.
			{
				// Rescale image.
				cv::Mat rescaled;
				{
					double minVal = 0.0, maxVal = 0.0;
					cv::minMaxLoc(scaled, &minVal, &maxVal);
					const double scaleFactor = 255.0 / (maxVal - minVal);
					scaled.convertTo(rescaled, CV_8U, scaleFactor, -scaleFactor * minVal);
					//scaled.convertTo(rescaled, CV_8U, -scaleFactor, scaleFactor * maxVal);  // reversed image.
				}

				// Show image.
				cv::imshow(windowName, rescaled);

				// Save image.
				std::cout << "\tsaving output image..." << std::endl;
				cv::imwrite(*cit + output_filename_appendix, rescaled);
			}

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
		img_filenames.push_back("./data/machine_vision/whale_256x256.png");

		const std::size_t kernelSize = 7;
		const double baseScale = 3.0;

		local::scale_space(img_filenames, kernelSize, baseScale);
	}
#endif

#if 0
	// REF [book] >> Figure 9.7 (p. 258) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("./data/machine_vision/whale_256x256.png");

		const std::size_t kernelSize3 = 3;
		const double baseScale3 = 0.3 * ((kernelSize3 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize9 = 9;
		const double baseScale9 = 0.3 * ((kernelSize9 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize17 = 17;
		const double baseScale17 = 0.3 * ((kernelSize17 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize33 = 33;
		const double baseScale33 = 0.3 * ((kernelSize33 - 1.0) * 0.5 - 1.0) + 0.8;

		local::gradient_scale_space(img_filenames, kernelSize3, baseScale3);
		local::gradient_scale_space(img_filenames, kernelSize9, baseScale9);
		local::gradient_scale_space(img_filenames, kernelSize17, baseScale17);
		local::gradient_scale_space(img_filenames, kernelSize33, baseScale33);
		local::laplacian_scale_space(img_filenames, kernelSize3, baseScale3);
		local::laplacian_scale_space(img_filenames, kernelSize9, baseScale9);
		local::laplacian_scale_space(img_filenames, kernelSize17, baseScale17);
		local::laplacian_scale_space(img_filenames, kernelSize33, baseScale3);
	}
#endif

#if 0
	// REF [book] >> Figure 9.8 (p. 259) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("./data/machine_vision/box_256x256_1.png");
		img_filenames.push_back("./data/machine_vision/box_256x256_2.png");

		const std::size_t kernelSize3 = 3;
		const double baseScale3 = 0.3 * ((kernelSize3 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize33 = 33;
		const double baseScale33 = 0.3 * ((kernelSize33 - 1.0) * 0.5 - 1.0) + 0.8;

		local::gradient_scale_space(img_filenames, kernelSize3, baseScale3);
		local::gradient_scale_space(img_filenames, kernelSize33, baseScale33);
	}
#endif

#if 1
	{
		std::list<std::string> img_filenames;
		//img_filenames.push_back("./data/machine_vision/box_256x256_1.png");
		//img_filenames.push_back("./data/machine_vision/box_256x256_2.png");
		img_filenames.push_back("./data/machine_vision/brain_256x256_1.png");
		img_filenames.push_back("./data/machine_vision/brain_256x256_2.png");

		const std::size_t kernelSize3 = 3;
		const double baseScale3 = 0.3 * ((kernelSize3 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize9 = 9;
		const double baseScale9 = 0.3 * ((kernelSize9 - 1.0) * 0.5 - 1.0) + 0.8;
		const std::size_t kernelSize33 = 33;
		const double baseScale33 = 0.3 * ((kernelSize33 - 1.0) * 0.5 - 1.0) + 0.8;

#if 0
		// callback.
		auto ridgenessOperator = [](const cv::Mat &img, const std::size_t kernelSize, const double sigma) -> cv::Mat
		{
			swl::ScaleSpace::RidgenessOperator ridgeness;
			return ridgeness(img, kernelSize, sigma);
		};
		auto isophoteCurvatureOperator = [](const cv::Mat &img, const std::size_t kernelSize, const double sigma) -> cv::Mat
		{
			swl::ScaleSpace::IsophoteCurvatureOperator ridgeness;
			return ridgeness(img, kernelSize, sigma);
		};

		local::ridgeness_scale_space(img_filenames, kernelSize3, baseScale3, ridgenessOperator);
		local::ridgeness_scale_space(img_filenames, kernelSize9, baseScale9, ridgenessOperator);
		local::ridgeness_scale_space(img_filenames, kernelSize33, baseScale33, ridgenessOperator);
		local::ridgeness_scale_space(img_filenames, kernelSize3, baseScale3, isophoteCurvatureOperator);
		local::ridgeness_scale_space(img_filenames, kernelSize9, baseScale9, isophoteCurvatureOperator);
		local::ridgeness_scale_space(img_filenames, kernelSize33, baseScale33, isophoteCurvatureOperator);
#else
		local::ridgeness_scale_space(img_filenames, kernelSize3, baseScale3, swl::ScaleSpace::RidgenessOperator());
		local::ridgeness_scale_space(img_filenames, kernelSize9, baseScale9, swl::ScaleSpace::RidgenessOperator());
		local::ridgeness_scale_space(img_filenames, kernelSize33, baseScale33, swl::ScaleSpace::RidgenessOperator());
		local::ridgeness_scale_space(img_filenames, kernelSize3, baseScale3, swl::ScaleSpace::IsophoteCurvatureOperator());
		local::ridgeness_scale_space(img_filenames, kernelSize9, baseScale9, swl::ScaleSpace::IsophoteCurvatureOperator());
		local::ridgeness_scale_space(img_filenames, kernelSize33, baseScale33, swl::ScaleSpace::IsophoteCurvatureOperator());
#endif
	}
#endif

#if 0
	// REF [book] >> Figure 9.21 (p. 271) in "Digital and Medical Image Processing", 2005.
	{
		std::list<std::string> img_filenames;
		img_filenames.push_back("./data/machine_vision/whale_256x256.png");

		const std::size_t kernelSize = 3;
		const double baseScale = 0.3 * ((kernelSize - 1.0) * 0.5 - 1.0) + 0.8;

		local::gaussian_pyramid(img_filenames, kernelSize, baseScale);
		local::laplacian_pyramid(img_filenames, kernelSize, baseScale);
	}
#endif
}
