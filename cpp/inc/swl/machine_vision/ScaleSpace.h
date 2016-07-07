#pragma once

#if !defined(__SWL_MACHINE_VISION__SCALE_SPACE__H_)
#define __SWL_MACHINE_VISION__SCALE_SPACE__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Scale Space Representation

class SWL_MACHINE_VISION_API ScaleSpace
{
public:
	// Gaussian operator.
	struct SWL_MACHINE_VISION_API GaussianOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Derivative-of-Gaussian (gradient) operator.
	struct SWL_MACHINE_VISION_API DerivativeOfGaussianOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Laplacian-of-Gaussian (LoG) operator.
	struct SWL_MACHINE_VISION_API LaplacianOfGaussianOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Ridgeness operator: F_vv.
	//	F_vv = G_vv * F is a measure of concavity.
	//	If F_vv is low, it means ridges. If F_vv is high, it means valleys.
	struct SWL_MACHINE_VISION_API RidgenessOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Cornerness operator: Fvv * Fw^2.
	struct SWL_MACHINE_VISION_API CornernessOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Isophote curvature operator: -F_vv / F_w.
	//	-F_vv / F_w is a measure of concavity, where F_vv = G_vv * F and F_w = G_w * F.
	//	If -F_vv / F_w is low, it means ridges. If -F_vv / F_w is high, it means valleys.
	struct SWL_MACHINE_VISION_API IsophoteCurvatureOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Flowline curvature operator: -F_vw / F_w.
	struct SWL_MACHINE_VISION_API FlowlineCurvatureOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Unflatness operator.
	struct SWL_MACHINE_VISION_API UnflatnessOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

	// Umbilicity operator.
	struct SWL_MACHINE_VISION_API UmbilicityOperator
	{
	public:
		cv::Mat operator()(const cv::Mat& img, const std::size_t kernelSize, const double sigma) const;
	};

public:
	explicit ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t kernelSize);
	//explicit ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const double baseScale);
	explicit ScaleSpace(const long firstOctaveIndex, const long lastOctaveIndex, const long firstSublevelIndex, const long lastSublevelIndex, const std::size_t octaveResolution, const std::size_t kernelSize, const double baseScale);
	explicit ScaleSpace(const ScaleSpace& rhs)
	: firstOctaveIndex_(rhs.firstOctaveIndex_), lastOctaveIndex_(rhs.lastOctaveIndex_), firstSublevelIndex_(rhs.firstSublevelIndex_), lastSublevelIndex_(rhs.lastSublevelIndex_), octaveResolution_(rhs.octaveResolution_), kernelSize_(kernelSize_), baseScale_(rhs.baseScale_), useVariableKernelSize_(rhs.useVariableKernelSize_)
	{}

private:
	ScaleSpace & operator=(const ScaleSpace& rhs);  // Not implemeted.

public:
	template<class FilterOperation>
	cv::Mat getImageInScaleSpace(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, FilterOperation operation, const bool useScaleSpacePyramid = false) const
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

		// FIXME [check] >> when applying a filter, does its kernel size increase as its sigma increases?
		return operation(resized, (useVariableKernelSize_ ? (kernelSize_ * scaleFactor) : kernelSize_), baseScale_ * scaleFactor);
	}

	//
	cv::Mat getScaledImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid = false) const
	{
		return getImageInScaleSpace(img, octaveIndex, sublevelIndex, GaussianOperator(), useScaleSpacePyramid);
	}
	// the norm of gradient at a level.
	cv::Mat getScaledGradientImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid = false) const
	{
		return getImageInScaleSpace(img, octaveIndex, sublevelIndex, DerivativeOfGaussianOperator(), useScaleSpacePyramid);
	}
	// the Laplacian at a level.
	cv::Mat getScaledLaplacianImage(const cv::Mat& img, const long octaveIndex, const long sublevelIndex, const bool useScaleSpacePyramid = false) const
	{
		return getImageInScaleSpace(img, octaveIndex, sublevelIndex, LaplacianOfGaussianOperator(), useScaleSpacePyramid);
	}

public:
	// Gaussian pyramid.
	static cv::Mat getImageInGaussianPyramid(const cv::Mat& img, const std::size_t kernelSize, const long octaveIndex);
	static cv::Mat getImageInGaussianPyramid(const cv::Mat& img, const std::size_t kernelSize, const double baseScale, const long octaveIndex);
	// Laplacian pyramid.
	static cv::Mat getImageInLaplacianPyramid(const cv::Mat& img, const std::size_t kernelSize, const long octaveIndex);
	static cv::Mat getImageInLaplacianPyramid(const cv::Mat& img, const std::size_t kernelSize, const double baseScale, const long octaveIndex);

private:
	static void computeDerivativesOfImage(const cv::Mat& img, const std::size_t kernelSize, const double sigma, cv::Mat& Fx, cv::Mat& Fy, cv::Mat& Fxx, cv::Mat& Fyy, cv::Mat& Fxy);

private:
	const long firstOctaveIndex_, lastOctaveIndex_;
	const long firstSublevelIndex_, lastSublevelIndex_;
	const std::size_t octaveResolution_;
	const std::size_t kernelSize_;
	const double baseScale_;
	const bool useVariableKernelSize_;
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__SCALE_SPACE__H_
