#pragma once

#if !defined(__SWL_MACHINE_VISION__IMAGE_FILTER__H_)
#define __SWL_MACHINE_VISION__IMAGE_FILTER__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Image Filter.

class SWL_MACHINE_VISION_API ImageFilter
{
public:
	// Gaussian operator.
	struct SWL_MACHINE_VISION_API GaussianOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Derivative-of-Gaussian (gradient) operator.
	struct SWL_MACHINE_VISION_API DerivativeOfGaussianOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Laplacian-of-Gaussian (LoG) operator.
	struct SWL_MACHINE_VISION_API LaplacianOfGaussianOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Ridgeness operator: F_vv.
	//	F_vv = G_vv * F is a measure of concavity.
	//	If F_vv is low, it means ridges. If F_vv is high, it means valleys.
	struct SWL_MACHINE_VISION_API RidgenessOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Cornerness operator: Fvv * Fw^2.
	struct SWL_MACHINE_VISION_API CornernessOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Isophote curvature operator: -F_vv / F_w.
	//	-F_vv / F_w is a measure of concavity, where F_vv = G_vv * F and F_w = G_w * F.
	//	If -F_vv / F_w is low, it means ridges. If -F_vv / F_w is high, it means valleys.
	struct SWL_MACHINE_VISION_API IsophoteCurvatureOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Flowline curvature operator: -F_vw / F_w.
	struct SWL_MACHINE_VISION_API FlowlineCurvatureOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Unflatness operator.
	struct SWL_MACHINE_VISION_API UnflatnessOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

	// Umbilicity operator.
	struct SWL_MACHINE_VISION_API UmbilicityOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t apertureSize, const double sigma) const;
	};

public:
	static void computeDerivativesOfImage(const cv::Mat& img, const std::size_t apertureSize, const double sigma, cv::Mat& Fx, cv::Mat& Fy, cv::Mat& Fxx, cv::Mat& Fyy, cv::Mat& Fxy);
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__IMAGE_FILTER__H_
