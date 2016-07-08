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
/*static*/ void DerivativesOfGaussian::getFirstOrderDerivatives(const size_t apertureSize, const double sigma, cv::Mat& Gx, cv::Mat& Gy)
{
	const int halfApertureSize = (int)apertureSize / 2;

#if 0
	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
	//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
	for (int y = -halfApertureSize, yy = 0; y <= halfApertureSize; ++y, ++yy)
		for (int x = -halfApertureSize, xx = 0; x <= halfApertureSize; ++x, ++xx)
		{
			const double factor = std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2) / _2_pi_sigma4;
			//const double factor = std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2);
			// TODO [check] >> x- & y-axis derivative.
			Gx.at<double>(yy, xx) = -double(x) * factor;
			Gy.at<double>(yy, xx) = -double(y) * factor;
		}
#else
	// If a kernel has the same size in x- and y-directions.

	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma4 = M_PI * _2_sigma2 * sigma2;
	//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
	for (int y = -halfApertureSize, yy = 0; y <= halfApertureSize; ++y, ++yy)
		for (int x = -halfApertureSize, xx = 0; x <= halfApertureSize; ++x, ++xx)
		{
			Gx.at<double>(yy, xx) = -double(x) * std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2) / _2_pi_sigma4;
			//Gx.at<double>(yy, xx) = -double(x) * std::exp(-(double(x)*double(x) + double(y)*double(y)) / _2_sigma2);
		}

	cv::transpose(Gx, Gy);
#endif
}

// Compute second-order derivatives of Gaussian.
/*static*/ void DerivativesOfGaussian::getSecondOrderDerivatives(const size_t apertureSize, const double sigma, cv::Mat& Gxx, cv::Mat& Gyy, cv::Mat& Gxy)
{
	const int halfApertureSize = (int)apertureSize / 2;

	const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2, _2_pi_sigma6 = M_PI * _2_sigma2 * sigma2 * sigma2;
	//const double sigma2 = sigma * sigma, _2_sigma2 = 2.0 * sigma2;
	for (int y = -halfApertureSize, yy = 0; y <= halfApertureSize; ++y, ++yy)
		for (int x = -halfApertureSize, xx = 0; x <= halfApertureSize; ++x, ++xx)
		{
			const double x2 = double(x) * double(x), y2 = double(y) * double(y);
			const double factor = std::exp(-(x2 + y2) / _2_sigma2) / _2_pi_sigma6;
			//const double factor = std::exp(-(x2 + y2) / _2_sigma2) / sigma2;
			Gxx.at<double>(yy, xx) = (x2 - sigma2) * factor;
			Gyy.at<double>(yy, xx) = (y2 - sigma2) * factor;
			Gxy.at<double>(yy, xx) = x * y * factor;
		}
}

}  // namespace swl
