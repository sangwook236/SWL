#include "swl/pattern_recognition/MotionSegmenter.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/optflow/motempl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>


namespace swl {

/*static*/ void MotionSegmenter::segmentUsingMHI(const double timestamp, const double mhiTimeDuration, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &processed_mhi, cv::Mat &component_label_map, std::vector<cv::Rect> &component_rects)
{
	cv::Mat silh;
	cv::absdiff(prev_gray_img, curr_gray_img, silh);  // get difference between frames

	const int diff_threshold = 8;
	cv::threshold(silh, silh, diff_threshold, 1.0, cv::THRESH_BINARY);  // threshold

	{
#if 1
		//const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
		//const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		cv::Mat processed_silh;
		const int iterations = 1;
#if 0
		cv::erode(silh, processed_silh, selement3, cv::Point(-1, -1), iterations);
		cv::dilate(processed_silh, silh, selement3, cv::Point(-1, -1), iterations);
#else
		cv::morphologyEx(silh, processed_silh, cv::MORPH_OPEN, selement3, cv::Point(-1, -1), iterations);
		cv::morphologyEx(processed_silh, silh, cv::MORPH_CLOSE, selement3, cv::Point(-1, -1), iterations);
#endif
#endif
	}

	cv::motempl::updateMotionHistory(silh, mhi, timestamp, mhiTimeDuration);  // update MHI

	//
	{
#if 1
		//const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));
		//const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		const int iterations = 1;
#if 0
		cv::erode(mhi, processed_mhi, selement3, cv::Point(-1, -1), iterations);
		cv::dilate(processed_mhi, processed_mhi, selement3, cv::Point(-1, -1), iterations);
#else
		cv::morphologyEx(mhi, processed_mhi, cv::MORPH_OPEN, selement3, cv::Point(-1, -1), iterations);
		cv::morphologyEx(processed_mhi, processed_mhi, cv::MORPH_CLOSE, selement3, cv::Point(-1, -1), iterations);
#endif

		mhi.copyTo(processed_mhi, processed_mhi > 0);
#else
		mhi.copyTo(processed_mhi);
#endif
	}

	// segment motion: get sequence of motion components.
	const double motion_segmentation_threshold = 0.5;  // recommended to be equal to the interval between motion history "steps" or greater.
	// segmask is marked motion components map. it is not used further.
	cv::Mat segmask;
	// TODO [check] >> have to diff revision before 2016/04/27.
	cv::motempl::segmentMotion(processed_mhi, segmask, component_rects, timestamp, motion_segmentation_threshold);

	//segmask.convertTo(component_label_map, CV_8SC1, 1.0, 0.0);  // Oops !!! error
	segmask.convertTo(component_label_map, CV_8UC1, 1.0, 0.0);
}

}  // namespace swl
