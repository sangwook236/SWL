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
		Gx -= cv::sum(Gx) / ((double)kernelSize * (double)kernelSize);
		Gy -= cv::sum(Gy) / ((double)kernelSize * (double)kernelSize);
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
		Gx -= cv::sum(Gx) / ((double)kernelSize * (double)kernelSize);
	}

	cv::Mat Gy;
	cv::transpose(Gx, Gy);
#endif

	cv::Mat Fx, Fy;
	cv::filter2D(img, Fx, CV_64F, Gx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(img, Fy, CV_64F, Gy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	cv::Mat gradient;
	cv::magnitude(Fx, Fy, gradient);

	return gradient;
}

// Laplacian of Gaussian (LoG).
cv::Mat ScaleSpace::LaplacianOfGaussianOperator::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	const int halfKernelSize = (int)kernelSize / 2;

	// Laplacian of Gaussian (LoG).
	cv::Mat DoG(kernelSize, kernelSize, CV_64F);
	{
		//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, pi_sigma4 = M_PI * sigma2 * sigma2;
		const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
		for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
			for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
			{
				const double x2 = double(x) * double(x), y2 = double(y) * double(y), val = (x2 + y2) / _2_sigma2;
				const double exp = std::exp(-val);
				//DoG.at<double>(yy, xx) = (val - 1.0) * exp / pi_sigma4;
				DoG.at<double>(yy, xx) = (val - 1.0) * exp;
			}

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		DoG -= cv::sum(DoG) / ((double)kernelSize * (double)kernelSize);
	}

	cv::Mat deltaF;
	cv::filter2D(img, deltaF, CV_64F, DoG, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	return deltaF;
}

// The second order derivative of Gaussian wrt the normal vector v: F_vv.
//	A local coordinate frame based on the gradient vector w and its right-handed normal vector v.
//	REF [book] >> section 9.1.2 (p. 254) in "Digital and Medical Image Processing", 2005.
//	REF [book] >> Figure 9.10 & 9.11 (p. 260) in "Digital and Medical Image Processing", 2005.
cv::Mat ScaleSpace::RidgenessOperator::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	cv::Mat Fx, Fy, Fxx, Fyy, Fxy;
	{
		// Compute derivatives wrt xy-coordinate system.
		cv::Mat Gx(kernelSize, kernelSize, CV_64F), Gy(kernelSize, kernelSize, CV_64F);
		cv::Mat Gxx(kernelSize, kernelSize, CV_64F), Gyy(kernelSize, kernelSize, CV_64F), Gxy(kernelSize, kernelSize, CV_64F);
		DerivativesOfGaussian::getFirstOrderDerivatives(kernelSize, sigma, Gx, Gy);
		DerivativesOfGaussian::getSecondOrderDerivatives(kernelSize, sigma, Gxx, Gyy, Gxy);

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		const double kernelArea = (double)kernelSize * (double)kernelSize;
		Gx -= cv::sum(Gx) / kernelArea;
		Gy -= cv::sum(Gy) / kernelArea;
		Gxx -= cv::sum(Gxx) / kernelArea;
		Gyy -= cv::sum(Gyy) / kernelArea;
		Gxy -= cv::sum(Gxy) / kernelArea;

		// Compute Fx, Fy, Fxx, Fyy, Fxy.
		cv::filter2D(img, Fx, CV_64F, Gx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fy, CV_64F, Gy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fxx, CV_64F, Gxx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fyy, CV_64F, Gyy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fxy, CV_64F, Gxy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	}

	// Compute Fvv.
	// REF [book] >> p. 255 ~ 256 in "Digital and Medical Image Processing", 2005.
	cv::Mat Fvv;
	{
		cv::Mat Fx2, Fy2;
		cv::multiply(Fx, Fx, Fx2);
		cv::multiply(Fy, Fy, Fy2);

		cv::Mat num1, num2, num3, num;
		cv::multiply(Fy2, Fxx, num1);
		cv::multiply(Fx, Fy, num2);
		cv::multiply(num2, Fxy, num2);
		cv::addWeighted(num1, 1.0, num2, -2.0, 0.0, num2);
		cv::multiply(Fx2, Fyy, num3);
		cv::add(num2, num3, num);

		cv::Mat den;
		cv::add(Fx2, Fy2, den);

		cv::divide(num, den, Fvv);
	}

	return Fvv;
}

// The second-order derivative of Gaussian wrt the normal vector v over the derivative of Gaussian wrt the gradient vector w: F_vv / F_w.
//	A local coordinate frame based on the gradient vector w and its right-handed normal vector v.
//	REF [book] >> Figure 9.12 (p. 261) in "Digital and Medical Image Processing", 2005.
cv::Mat ScaleSpace::IsophoteCurvatureOperator::operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const
{
	cv::Mat Fx, Fy, Fxx, Fyy, Fxy;
	{
		// Compute derivatives wrt xy-coordinate system.
		cv::Mat Gx(kernelSize, kernelSize, CV_64F), Gy(kernelSize, kernelSize, CV_64F);
		cv::Mat Gxx(kernelSize, kernelSize, CV_64F), Gyy(kernelSize, kernelSize, CV_64F), Gxy(kernelSize, kernelSize, CV_64F);
		DerivativesOfGaussian::getFirstOrderDerivatives(kernelSize, sigma, Gx, Gy);
		DerivativesOfGaussian::getSecondOrderDerivatives(kernelSize, sigma, Gxx, Gyy, Gxy);

		// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
		//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
		const double kernelArea = (double)kernelSize * (double)kernelSize;
		Gx -= cv::sum(Gx) / kernelArea;
		Gy -= cv::sum(Gy) / kernelArea;
		Gxx -= cv::sum(Gxx) / kernelArea;
		Gyy -= cv::sum(Gyy) / kernelArea;
		Gxy -= cv::sum(Gxy) / kernelArea;

		// Compute Fx, Fy, Fxx, Fyy, Fxy.
		cv::filter2D(img, Fx, CV_64F, Gx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fy, CV_64F, Gy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fxx, CV_64F, Gxx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fyy, CV_64F, Gyy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(img, Fxy, CV_64F, Gxy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	}

	// Compute FvvFw.
	// REF [book] >> p. 255 ~ 256 in "Digital and Medical Image Processing", 2005.
	cv::Mat FvvFw;
	{
		cv::Mat Fx2, Fy2;
		cv::multiply(Fx, Fx, Fx2);
		cv::multiply(Fy, Fy, Fy2);

		cv::Mat num1, num2, num3, num;
		cv::multiply(Fy2, Fxx, num1);
		cv::multiply(Fx, Fy, num2);
		cv::multiply(num2, Fxy, num2);
		cv::addWeighted(num1, 1.0, num2, -2.0, 0.0, num2);
		cv::multiply(Fx2, Fyy, num3);
		cv::add(num2, num3, num);

		cv::Mat den;
		cv::add(Fx2, Fy2, den);
		cv::pow(den, 1.5, den);

		cv::divide(-num, den, FvvFw);
	}

	return FvvFw;
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
