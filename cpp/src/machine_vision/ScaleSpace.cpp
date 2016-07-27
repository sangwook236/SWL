#include "swl/Config.h"
#include "swl/machine_vision/ScaleSpace.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t baseApertureSize)
: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), baseApertureSize_(baseApertureSize), baseScale_(std::pow(0.3 * ((baseApertureSize - 1.0) * 0.5 - 1.0) + 0.8, 2.0)), useVariableApertureSize_(true)
{
	// REF [function] >> cv::getGaussianKernel() in OpenCV.
	//	sigma = 0.3 * ((apertureSize - 1.0) * 0.5 - 1.0) + 0.8.
	//	scale = sigma^2.

	assert(firstOctaveIndex_ <= lastOctaveIndex);
	assert(firstSublevelIndex_ <= lastSublevelIndex_);
	assert(octaveResolution_ > 0);
	assert(baseApertureSize_ > 0 && 1 == baseApertureSize_ % 2);
	assert(baseScale_ > 0.0);
}

#if 0
/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const double baseScale)
: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), baseApertureSize_(2 * (int)std::ceil(((((std::sqrt(baseScale) - 0.8) / 0.3 + 1.0) * 2.0 + 1.0) + 1.0) / 2.0) - 1), baseScale_(baseScale), useVariableApertureSize_(true)
{
	// REF [function] >> cv::getGaussianKernel() in OpenCV.
	//	sigma = sqrt(scale).
	//	apertureSize = round(((sigma - 0.8) / 0.3 + 1.0) * 2.0 + 1.0).

	assert(firstOctaveIndex_ <= lastOctaveIndex);
	assert(firstSublevelIndex_ <= lastSublevelIndex_);
	assert(octaveResolution_ > 0);
	assert(baseApertureSize_ > 0 && 1 == baseApertureSize_ % 2);
	assert(baseScale_ > 0.0);
}
#endif

/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t baseApertureSize, const double baseScale)
: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), baseApertureSize_(baseApertureSize), baseScale_(baseScale), useVariableApertureSize_(true)
{
	assert(firstOctaveIndex_ <= lastOctaveIndex);
	assert(firstSublevelIndex_ <= lastSublevelIndex_);
	assert(octaveResolution_ > 0);
	assert(baseApertureSize_ > 0 && 1 == baseApertureSize_ % 2);
	assert(baseScale_ > 0.0);
}

double ScaleSpace::getScaleFactor(const long octaveIndex, const long sublevelIndex, const bool useImagePyramid /*= false*/) const
{
	// FIXME [check] >> Which one is correct?
	//	REF [site] >> Fig 6 in http://darkpgmr.tistory.com/137
	//	Is there any relationship between reducing a image by half and doubling the sigma of Gaussian filter?
	//return std::pow(2.0, (double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_);
	return std::pow(2.0, useImagePyramid ? ((double)sublevelIndex / (double)octaveResolution_) : ((double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_));
}

/*static*/ cv::Mat ScaleSpace::getImageInGaussianPyramid(const cv::Mat& img, const std::size_t apertureSize, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, apertureSize).getScaledImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getImageInGaussianPyramid(const cv::Mat& img, const std::size_t apertureSize, const double baseScale, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, apertureSize, baseScale).getScaledImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getImageInLaplacianPyramid(const cv::Mat& img, const std::size_t apertureSize, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, apertureSize).getScaledLaplacianImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getImageInLaplacianPyramid(const cv::Mat& img, const std::size_t apertureSize, const double baseScale, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, apertureSize, baseScale).getScaledLaplacianImage(img, octaveIndex, 0, true);
}

}  // namespace swl
