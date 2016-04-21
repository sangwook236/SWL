#include "swl/Config.h"
#include "swl/machine_vision/DerivativesOfGaussian.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

// Compute first-order derivatives of Gaussian.
/*static*/ void DerivativesOfGaussian::getFirstOrderDerivatives(const size_t kernelSize, const double sigma, cv::Mat& Gx, cv::Mat& Gy)
{
	const int halfKernelSize = (int)kernelSize / 2;
	const double kernelArea = (double)kernelSize * (double)kernelSize;

#if 0
	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
	for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
		for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
		{
			const double factor = std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2) / _2_pi_sigma4;
			//const double factor = std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2);
			// TODO [check] >> x- & y-axis derivative.
			Gx.at<double>(yy, xx) = -double(x) * factor;
			Gy.at<double>(yy, xx) = -double(y) * factor;
		}

	// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
	//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
	Gx -= cv::Scalar(cv::sum(Gx) / kernelArea);
	Gy -= cv::Scalar(cv::sum(Gy) / kernelArea);
#else
	// If a kernel has the same size in x- and y-directions.

	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
	for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
		for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
		{
			Gx.at<double>(yy, xx) = -double(x) * std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2) / _2_pi_sigma4;
			//Gx.at<double>(yy, xx) = -double(x) * std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2);
		}

	// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
	//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
	Gx -= cv::Scalar(cv::sum(Gx) / ((double)kernelSize * (double)kernelSize));

	cv::transpose(Gx, Gy);
#endif
}

// Compute second-order derivatives of Gaussian.
/*static*/ void DerivativesOfGaussian::getSecondOrderDerivatives(const size_t kernelSize, const double sigma, cv::Mat& Gxx, cv::Mat& Gyy, cv::Mat& Gxy)
{
	const int halfKernelSize = (int)kernelSize / 2;
	const double kernelArea = (double)kernelSize * (double)kernelSize;

	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma6 = M_PI * _2_sigma2 * sigma2 * sigma2;
	for (int y = -halfKernelSize, yy = 0; y <= halfKernelSize; ++y, ++yy)
		for (int x = -halfKernelSize, xx = 0; x <= halfKernelSize; ++x, ++xx)
		{
			const double x2 = double(x) * double(x), y2 = double(y) * double(y);
			const double factor = std::exp(-(x2 + y2) / _2_sigma2) / _2_pi_sigma6;
			//const double factor = std::exp(-(x2 + y2) / _2_sigma2) / sigma2;
			Gxx.at<double>(yy, xx) = (x2 - sigma2) * factor;
			Gyy.at<double>(yy, xx) = (y2 - sigma2) * factor;
			Gxy.at<double>(yy, xx) = x * y * factor;
		}

	// Make sure that the sum (or average) of all elements of the kernel has to be zero (similar to the Laplace kernel) so that the convolution result of a homogeneous regions is always zero.
	//	REF [site] >> http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
	Gxx -= cv::Scalar(cv::sum(Gxx) / kernelArea);
	Gyy -= cv::Scalar(cv::sum(Gyy) / kernelArea);
	Gxy -= cv::Scalar(cv::sum(Gxy) / kernelArea);
}

}  // namespace swl
