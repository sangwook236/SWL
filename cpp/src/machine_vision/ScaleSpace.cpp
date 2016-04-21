#include "swl/Config.h"
#include "swl/machine_vision/ScaleSpace.h"
#include "swl/machine_vision/DerivativesOfGaussian.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t kernelSize)
: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), kernelSize_(kernelSize), baseScale_(0.3 * ((kernelSize - 1.0) * 0.5 - 1.0) + 0.8)
{
	// REF [site] >> cv::getGaussianKernel() in OpenCV.
	//	sigma = 0.3 * ((kernelSize - 1.0) * 0.5 - 1.0) + 0.8

	assert(firstOctaveIndex_ <= lastOctaveIndex);
	assert(firstSublevelIndex_ <= lastSublevelIndex_);
	assert(octaveResolution_ > 0);
	assert(kernelSize_ > 0 && 1 == kernelSize_ % 2);
	assert(baseScale_ > 0.0);
}

#if 0
/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const double baseScale)
: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), kernelSize_(2 * (int)std::ceil(((((baseScale - 0.8) / 0.3 + 1.0) * 2.0 + 1.0) + 1.0) / 2.0) - 1), baseScale_(baseScale)
{
	// REF [site] >> cv::getGaussianKernel() in OpenCV.
	//	kernelSize = round(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.0)

	assert(firstOctaveIndex_ <= lastOctaveIndex);
	assert(firstSublevelIndex_ <= lastSublevelIndex_);
	assert(octaveResolution_ > 0);
	assert(kernelSize_ > 0 && 1 == kernelSize_ % 2);
	assert(baseScale_ > 0.0);
}
#endif

/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t kernelSize, const double baseScale)
: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), kernelSize_(kernelSize), baseScale_(baseScale)
{
	assert(firstOctaveIndex_ <= lastOctaveIndex);
	assert(firstSublevelIndex_ <= lastSublevelIndex_);
	assert(octaveResolution_ > 0);
	assert(kernelSize_ > 0 && 1 == kernelSize_ % 2);
	assert(baseScale_ > 0.0);
}

cv::Mat ScaleSpace::getScaledImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid /*= false*/) const
{
	if (octaveIndex < firstOctaveIndex_ || octaveIndex > lastOctaveIndex_ || sublevelIndex < firstSublevelIndex_ || sublevelIndex > lastSublevelIndex_)
		return cv::Mat();

	cv::Mat resized;
	if (!useScaleSpacePyramid || 0 == octaveIndex)
		img.copyTo(resized);
	else
	{
		const double ratio = std::pow(2.0, -octaveIndex);
		cv::resize(img, resized, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
	}

	// FIXME [check] >> which one is correct?
	//	REF [site] >> Fig 6 in http://darkpgmr.tistory.com/137
	//	Is there any relationship between reducing a image by half and doubling the sigma of Gaussian filter?
	//const double sigma(baseScale_ * std::pow(2.0, (double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_));
	const double sigma(baseScale_ * std::pow(2.0, useScaleSpacePyramid ? ((double)sublevelIndex / (double)octaveResolution_) : ((double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_)));

	// FIXME [check] >> when applying Gaussian filter, does its kernel size increase as its sigma increases?
	//	REF [site] >> cv::getGaussianKernel() in OpenCV.
	//	kernelSize = round(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.0)
	//const std::size_t kernelSize = (std::size_t)std::floor(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.5);  // "+ 0.5" for rounding.

	cv::Mat scaled;
	cv::GaussianBlur(resized, scaled, cv::Size(kernelSize_, kernelSize_), sigma, sigma, cv::BORDER_DEFAULT);

	return scaled;
}

cv::Mat ScaleSpace::getScaledGradientImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid /*= false*/) const
{
	if (octaveIndex < firstOctaveIndex_ || octaveIndex > lastOctaveIndex_ || sublevelIndex < firstSublevelIndex_ || sublevelIndex > lastSublevelIndex_)
		return cv::Mat();

	cv::Mat resized;
	if (!useScaleSpacePyramid || 0 == octaveIndex)
		img.copyTo(resized);
	else
	{
		const double ratio = std::pow(2.0, -octaveIndex);
		cv::resize(img, resized, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
	}

	// FIXME [check] >> which one is correct?
	//	REF [site] >> Fig 6 in http://darkpgmr.tistory.com/137
	//	Is there any relationship between reducing a image by half and doubling the sigma of Gaussian filter?
	//const double sigma(baseScale_ * std::pow(2.0, (double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_));
	const double sigma(baseScale_ * std::pow(2.0, useScaleSpacePyramid ? ((double)sublevelIndex / (double)octaveResolution_) : ((double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_)));

	// FIXME [check] >> when applying a filter, does its kernel size increase as its sigma increases?
	//	REF [site] >> cv::getGaussianKernel() in OpenCV.
	//	kernelSize = round(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.0)
	//const std::size_t kernelSize = (std::size_t)std::floor(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.5);  // "+ 0.5" for rounding.
	const int halfKernelSize = (int)kernelSize_ / 2;

	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
	cv::Mat kernelX(kernelSize_, kernelSize_, CV_64F), kernelY(kernelSize_, kernelSize_, CV_64F);
	for (int r = -halfKernelSize, rr = 0; r <= halfKernelSize; ++r, ++rr)
		for (int c = -halfKernelSize, cc = 0; c <= halfKernelSize; ++c, ++cc)
		{
			const double exp = std::exp(-(double(r)*double(r) + double(c)*double(c)) / _2_sigma2);
			// TODO [check] >> x- & y-axis derivative.
			kernelX.at<double>(rr, cc) = -double(c) * exp / _2_pi_sigma4;
			kernelY.at<double>(rr, cc) = -double(r) * exp / _2_pi_sigma4 ;
		}

	cv::Mat filteredX, filteredY;
	cv::filter2D(resized, filteredX, CV_64F, kernelX, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(resized, filteredY, CV_64F, kernelY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

 	cv::Mat scaled;
	cv::magnitude(filteredX, filteredY, scaled);

	return scaled;
}

cv::Mat ScaleSpace::getScaledLaplacianImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid /*= false*/) const
{
	if (octaveIndex < firstOctaveIndex_ || octaveIndex > lastOctaveIndex_ || sublevelIndex < firstSublevelIndex_ || sublevelIndex > lastSublevelIndex_)
		return cv::Mat();

	cv::Mat resized;
	if (!useScaleSpacePyramid || 0 == octaveIndex)
		img.copyTo(resized);
	else
	{
		const double ratio = std::pow(2.0, -octaveIndex);
		cv::resize(img, resized, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
	}

	// FIXME [check] >> which one is correct?
	//	REF [site] >> Fig 6 in http://darkpgmr.tistory.com/137
	//	Is there any relationship between reducing a image by half and doubling the sigma of Gaussian filter?
	//const double sigma(baseScale_ * std::pow(2.0, (double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_));
	const double sigma(baseScale_ * std::pow(2.0, useScaleSpacePyramid ? ((double)sublevelIndex / (double)octaveResolution_) : ((double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_)));

	// FIXME [check] >> when applying a filter, does its kernel size increase as its sigma increases?
	//	REF [site] >> cv::getGaussianKernel() in OpenCV.
	//	kernelSize = round(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.0)
	//const std::size_t kernelSize = (std::size_t)std::floor(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.5);  // "+ 0.5" for rounding.
	const int halfKernelSize = (int)kernelSize_ / 2;

	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
	cv::Mat kernelXX(kernelSize_, kernelSize_, CV_64F), kernelYY(kernelSize_, kernelSize_, CV_64F);
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
	cv::filter2D(resized, filteredX, CV_64F, kernelXX, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(resized, filteredY, CV_64F, kernelYY, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	cv::Mat scaled;
	cv::add(filteredX, filteredY, scaled);

	return scaled;
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInGaussianPyramid(const cv::Mat& img, const std::size_t baseKernelSize, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, baseKernelSize).getScaledImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInGaussianPyramid(const cv::Mat& img, const std::size_t baseKernelSize, const double baseScale, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, baseKernelSize, baseScale).getScaledImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInLaplacianPyramid(const cv::Mat& img, const std::size_t baseKernelSize, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, baseKernelSize).getScaledLaplacianImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInLaplacianPyramid(const cv::Mat& img, const std::size_t baseKernelSize, const double baseScale, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, baseKernelSize, baseScale).getScaledLaplacianImage(img, octaveIndex, 0, true);
}

}  // namespace swl
