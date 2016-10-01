#pragma once

#if !defined(__SWL_MACHINE_VISION__NON_MAXIMUM_SUPPRESSION__H_)
#define __SWL_MACHINE_VISION__NON_MAXIMUM_SUPPRESSION__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
// Non-Maximum Suppression.

struct SWL_MACHINE_VISION_API NonMaximumSuppression
{
public:
	//
	static void computeNonMaximumSuppression(const cv::Mat &in_float, cv::Mat &out_uint8);
	static void findMountainChain(const cv::Mat &in_float, cv::Mat &out_uint8);

	//
	static void computeNonMaximumSuppression(const cv::Mat& src, const int sz, cv::Mat& dst, const cv::Mat mask);

private:
	static void checkMountainPeak(const int ridx, const int cidx, const int rows, const int cols, const cv::Mat &in_float, cv::Mat &peak_flag, cv::Mat &visit_flag, const bool start_flag);
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__NON_MAXIMUM_SUPPRESSION__H_
