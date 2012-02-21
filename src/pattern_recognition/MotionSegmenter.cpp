#include "swl/pattern_recognition/MotionSegmenter.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>


namespace swl {

/*static*/ void MotionSegmenter::segmentUsingMHI(const double timestamp, const double mhiTimeDuration, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &processed_mhi, cv::Mat &component_label_map, std::vector<cv::Rect> &component_rects)
{
	cv::Mat silh;
	cv::absdiff(prev_gray_img, curr_gray_img, silh);  // get difference between frames

	const int diff_threshold = 8;
	cv::threshold(silh, silh, diff_threshold, 1.0, cv::THRESH_BINARY);  // threshold

	{
#if 1
		const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1)); 
		const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1)); 
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)); 
		cv::Mat processed_silh;
		cv::erode(silh, processed_silh, selement3);
		cv::dilate(processed_silh, silh, selement3);
#endif
	}

	cv::updateMotionHistory(silh, mhi, timestamp, mhiTimeDuration);  // update MHI

	//
	{
#if 1
		const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1)); 
		const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1)); 
		const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)); 
		cv::erode(mhi, processed_mhi, selement3);
		cv::dilate(processed_mhi, processed_mhi, selement3);

		mhi.copyTo(processed_mhi, processed_mhi > 0);
#else
		mhi.copyTo(processed_mhi);
#endif
	}

	//
	CvMemStorage *storage = cvCreateMemStorage(0);  // temporary storage

	// segment motion: get sequence of motion components.
	const double motion_segmentation_threshold = 0.5;  // recommended to be equal to the interval between motion history "steps" or greater.
	// segmask is marked motion components map. it is not used further.
	IplImage *segmask = cvCreateImage(cvSize(curr_gray_img.cols, curr_gray_img.rows), IPL_DEPTH_32F, 1);  // motion segmentation map
	CvSeq *seq = cvSegmentMotion(&(IplImage)processed_mhi, segmask, storage, timestamp, motion_segmentation_threshold);

	//cv::Mat(segmask, false).convertTo(component_label_map, CV_8SC1, 1.0, 0.0);  // Oops !!! error
	cv::Mat(segmask, false).convertTo(component_label_map, CV_8UC1, 1.0, 0.0);

	// iterate through the motion components
	component_rects.reserve(seq->total);
	for (int i = 0; i < seq->total; ++i)
	{
		const CvConnectedComp *comp = (CvConnectedComp *)cvGetSeqElem(seq, i);
		component_rects.push_back(cv::Rect(comp->rect));
	}

	cvReleaseImage(&segmask);

	//cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);
}

}  // namespace swl
