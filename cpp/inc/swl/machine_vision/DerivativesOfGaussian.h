#pragma once

#if !defined(__SWL_MACHINE_VISION__DERIVATIVES_OF_GAUSSIAN__H_)
#define __SWL_MACHINE_VISION__DERIVATIVES_OF_GAUSSIAN__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"


namespace cv {

class Mat;

}  // namespace cv


namespace swl {

//--------------------------------------------------------------------------
// Derivatives Of Gaussian

struct SWL_MACHINE_VISION_API DerivativesOfGaussian
{
public:
	// Compute first-order derivatives of Gaussian.
	static void getFirstOrderDerivatives(const size_t apertureSize, const double sigma, cv::Mat& Gx, cv::Mat& Gy);

	// Compute second-order derivatives of Gaussian.
	static void getSecondOrderDerivatives(const size_t apertureSize, const double sigma, cv::Mat& Gxx, cv::Mat& Gyy, cv::Mat& Gxy);
};

}  // namespace swl


#endif  // __DERIVATIVES_OF_GAUSSIAN_H_
