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

// Derivative of Gaussian wrt x- & y-axes.
cv::Mat ScaleSpace::DerivativeOfGaussianOperator::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	const int halfKernelSize = (int)kernelSize / 2;

#if 0
	cv::Mat Gx(kernelSize, kernelSize, CV_64F), Gy(kernelSize, kernelSize, CV_64F);
	{
		//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
		for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
			for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
			{
				//const double factor = std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2) / _2_pi_sigma4;
				const double exp = std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2);
				// TODO [check] >> x- & y-axis derivative.
				//Gx.at<double>(yy, xx) = -double(x) * factor;
				//Gy.at<double>(yy, xx) = -double(y) * factor;
				Gx.at<double>(yy, xx) = -double(x) * exp;
				Gy.at<double>(yy, xx) = -double(y) * exp;
			}

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		Gx -= cv::Scalar(cv::sum(Gx) / ((double)kernelSize * (double)kernelSize));
		Gy -= cv::Scalar(cv::sum(Gy) / ((double)kernelSize * (double)kernelSize));
	}
#else
	// If a kernel has the same size in x- and y-directions.

	cv::Mat Gx(kernelSize, kernelSize, CV_64F);
	{
		//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
		for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
			for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
			{
				//Gx.at<double>(yy, xx) = -double(x) * std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2) / _2_pi_sigma4;
				Gx.at<double>(yy, xx) = -double(x) * std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2);
			}

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		Gx -= cv::Scalar(cv::sum(Gx) / ((double)kernelSize * (double)kernelSize));
	}

	cv::Mat Gy;
	cv::transpose(Gx, Gy);
#endif

	cv::Mat filteredX, filteredY;
	cv::filter2D(img, filteredX, CV_64F, Gx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(img, filteredY, CV_64F, Gy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	cv::Mat scaled;
	cv::magnitude(filteredX, filteredY, scaled);

	return scaled;
			}

// Laplacian of Gaussian (LoG).
cv::Mat ScaleSpace::LaplacianOfGaussianOperator::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	const int halfKernelSize = (int)kernelSize / 2;

	// Laplacian of Gaussian (LoG).
	cv::Mat kernel(kernelSize, kernelSize, CV_64F);
	{
		//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, pi_sigma4 = M_PI * sigma2 * sigma2;
		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
		for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
			for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
			{
				const double x2 = double(x) * double(x), y2 = double(y) * double(y), val = (x2 + y2) / _2_sigma2;
				const double exp = std::exp(-val);
				//kernel.at<double>(yy, xx) = (val - 1.0) * exp / pi_sigma4;
				kernel.at<double>(yy, xx) = (val - 1.0) * exp;
			}

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		kernel -= cv::Scalar(cv::sum(kernel) / ((double)kernelSize * (double)kernelSize));
	}

	cv::Mat scaled;
	cv::filter2D(img, scaled, CV_64F, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	return scaled;
}

// The second order derivative of Gaussian wrt the normal vector v: G_vv.
//	A local coordinate frame based on the gradient vector w and its right-handed normal vector v.
//	REF [book] >> section 9.1.2 (p. 254) in "Digital and Medical Image Processing", 2005.
//	REF [book] >> Figure 9.10 & 9.11 (p. 260) in "Digital and Medical Image Processing", 2005.
cv::Mat ScaleSpace::RidgenessOperator1::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	// Compute derivatives wrt xy-coordinate system.
	cv::Mat Gx(kernelSize, kernelSize, CV_64F), Gy(kernelSize, kernelSize, CV_64F);
	cv::Mat Gxx(kernelSize, kernelSize, CV_64F), Gyy(kernelSize, kernelSize, CV_64F), Gxy(kernelSize, kernelSize, CV_64F);
	DerivativesOfGaussian::getFirstOrderDerivatives(kernelSize, sigma, Gx, Gy);
	DerivativesOfGaussian::getSecondOrderDerivatives(kernelSize, sigma, Gxx, Gyy, Gxy);

	// Compute Gvv.
	// REF [book] >> p. 255 ~ 256 in "Digital and Medical Image Processing", 2005.
	cv::Mat Gvv;
	{
		cv::Mat Gx2, Gy2;
		cv::multiply(Gx, Gx, Gx2);
		cv::multiply(Gy, Gy, Gy2);

		cv::Mat num1, num2, num3, num;
		cv::multiply(Gy2, Gxx, num1);
		cv::multiply(Gx, Gy, num2);
		// TODO [check] >> num2 used at two places.
		cv::multiply(num2, Gxy, num2);
		cv::multiply(Gx2, Gyy, num3);
		cv::addWeighted(num1, 1.0, num2, -2.0, 0.0, num);
		// TODO [check] >> num used at two places.
		cv::add(num, num3, num);

		cv::Mat den;
		cv::add(Gx2, Gy2, den);

		cv::divide(num, den, Gvv);

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		Gvv -= cv::Scalar(cv::sum(Gvv) / ((double)kernelSize * (double)kernelSize));
	}

	cv::Mat scaled;
	cv::filter2D(img, scaled, CV_64F, Gvv, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	return scaled;
}

// The second-order derivative of Gaussian wrt the normal vector v over the derivative of Gaussian wrt the gradient vector w: G_vv / G_w.
//	A local coordinate frame based on the gradient vector w and its right-handed normal vector v.
//	REF [book] >> Figure 9.12 (p. 261) in "Digital and Medical Image Processing", 2005.
cv::Mat ScaleSpace::RidgenessOperator2::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	// Compute derivatives wrt xy-coordinate system.
	cv::Mat Gx(kernelSize, kernelSize, CV_64F), Gy(kernelSize, kernelSize, CV_64F);
	cv::Mat Gxx(kernelSize, kernelSize, CV_64F), Gyy(kernelSize, kernelSize, CV_64F), Gxy(kernelSize, kernelSize, CV_64F);
	DerivativesOfGaussian::getFirstOrderDerivatives(kernelSize, sigma, Gx, Gy);
	DerivativesOfGaussian::getSecondOrderDerivatives(kernelSize, sigma, Gxx, Gyy, Gxy);

	// Compute Gvv and Gw.
	// REF [book] >> p. 255 ~ 256 in "Digital and Medical Image Processing", 2005.
	cv::Mat GvvGw;
	{
		cv::Mat Gx2, Gy2;
		cv::multiply(Gx, Gx, Gx2);
		cv::multiply(Gy, Gy, Gy2);

		cv::Mat num1, num2, num3, num;
		cv::multiply(Gy2, Gxx, num1);
		cv::multiply(Gx, Gy, num2);
		// TODO [check] >> num2 used at two places.
		cv::multiply(num2, Gxy, num2);
		cv::multiply(Gx2, Gyy, num3);
		cv::addWeighted(num1, 1.0, num2, -2.0, 0.0, num);
		// TODO [check] >> num used at two places.
		cv::add(num, num3, num);

		cv::Mat den;
		cv::add(Gx2, Gy2, den);

		cv::Mat Gvv;
		cv::divide(num, den, Gvv);

		//
		cv::Mat Gw;
		cv::magnitude(Gx, Gy, Gw);

		cv::divide(Gvv, Gw, GvvGw);

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		GvvGw -= cv::Scalar(cv::sum(GvvGw) / ((double)kernelSize * (double)kernelSize));
	}

	cv::Mat scaled;
	cv::filter2D(img, scaled, CV_64F, GvvGw, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	return scaled;
}

/*explicit*/ ScaleSpace::ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t kernelSize)
	: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), kernelSize_(kernelSize), baseScale_(0.3 * ((kernelSize - 1.0) * 0.5 - 1.0) + 0.8), useVariableKernelSize_(false)
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
	: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), kernelSize_(2 * (int)std::ceil(((((baseScale - 0.8) / 0.3 + 1.0) * 2.0 + 1.0) + 1.0) / 2.0) - 1), baseScale_(baseScale), useVariableKernelSize_(false)
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
	: firstOctaveIndex_(firstOctaveIndex), lastOctaveIndex_(lastOctaveIndex), firstSublevelIndex_(firstSublevelIndex), lastSublevelIndex_(lastSublevelIndex), octaveResolution_(octaveResolution), kernelSize_(kernelSize), baseScale_(baseScale), useVariableKernelSize_(false)
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
	//const double scaleFactor = std::pow(2.0, (double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_);
	const double scaleFactor = std::pow(2.0, useScaleSpacePyramid ? ((double)sublevelIndex / (double)octaveResolution_) : ((double)octaveIndex + (double)sublevelIndex / (double)octaveResolution_));

	const double sigma(baseScale_ * scaleFactor);

	// FIXME [check] >> when applying Gaussian filter, does its kernel size increase as its sigma increases?
	const std::size_t kernelSize = useVariableKernelSize_ ? (kernelSize_ * scaleFactor) : kernelSize_;

	cv::Mat scaled;
	cv::GaussianBlur(resized, scaled, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);

	return scaled;
}

cv::Mat ScaleSpace::getScaledGradientImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid /*= false*/) const
{
	return getScaledDerivativeImage(img, octaveIndex, sublevelIndex, DerivativeOfGaussianOperator(), useScaleSpacePyramid);
}

cv::Mat ScaleSpace::getScaledLaplacianImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid /*= false*/) const
{
	return getScaledDerivativeImage(img, octaveIndex, sublevelIndex, LaplacianOfGaussianOperator(), useScaleSpacePyramid);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInGaussianPyramid(const cv::Mat& img, const std::size_t kernelSize, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, kernelSize).getScaledImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInGaussianPyramid(const cv::Mat& img, const std::size_t kernelSize, const double baseScale, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, kernelSize, baseScale).getScaledImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInLaplacianPyramid(const cv::Mat& img, const std::size_t kernelSize, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, kernelSize).getScaledLaplacianImage(img, octaveIndex, 0, true);
}

/*static*/ cv::Mat ScaleSpace::getScaledImageInLaplacianPyramid(const cv::Mat& img, const std::size_t kernelSize, const double baseScale, const long octaveIndex)
{
	// FIXME [check] >> is this implementation correct?
	return ScaleSpace(octaveIndex, octaveIndex, 0, 0, 1, kernelSize, baseScale).getScaledLaplacianImage(img, octaveIndex, 0, true);
}

}  // namespace swl
