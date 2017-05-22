#pragma once

#if !defined(__SWL_MACHINE_VISION__BOUNDARY_EXTRACTION__H_)
#define __SWL_MACHINE_VISION__BOUNDARY_EXTRACTION__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Boundary Extraction.

class SWL_MACHINE_VISION_API IBoundaryExtraction
{
public:
	virtual ~IBoundaryExtraction();

public:
	virtual void extractBoundary(const cv::Mat &label, cv::Mat &boundary) const = 0;
};

//--------------------------------------------------------------------------
// Naive Boundary Extraction.

class SWL_MACHINE_VISION_API NaiveBoundaryExtraction final : public IBoundaryExtraction
{
public:
	typedef IBoundaryExtraction base_type;

public:
	NaiveBoundaryExtraction(const bool use8connectivity = true);

public:
	/*virtual*/ void extractBoundary(const cv::Mat &label, cv::Mat &boundary) const override;

private:
	const bool use8connectivity_;
};

//--------------------------------------------------------------------------
// Contour Boundary Extraction.

class SWL_MACHINE_VISION_API ContourBoundaryExtraction final : public IBoundaryExtraction
{
public:
	typedef IBoundaryExtraction base_type;

public:
	ContourBoundaryExtraction();

public:
	/*virtual*/ void extractBoundary(const cv::Mat &label, cv::Mat &boundary) const override;
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__BOUNDARY_EXTRACTION__H_
