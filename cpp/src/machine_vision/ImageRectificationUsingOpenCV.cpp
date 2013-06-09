#include "swl/Config.h"
#include "swl/machine_vision/ImageRectificationUsingOpenCV.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

ImageRectificationUsingOpenCV::ImageRectificationUsingOpenCV(
	const cv::Size &imageSize_left, const cv::Mat &K_left, const cv::Mat &distCoeffs_left,
	const cv::Size &imageSize_right, const cv::Mat &K_right, const cv::Mat &distCoeffs_right,
	const cv::Mat &R, const cv::Mat &T
)
{
	cv::Rect validRoi_left, validRoi_right;
	cv::stereoRectify(
		K_left, distCoeffs_left,
		K_right, distCoeffs_right,
		imageSize_left, R, T, R_left_, R_right_, P_left_, P_right_, Q_,
		cv::CALIB_ZERO_DISPARITY, 1, imageSize_left, &validRoi_left, &validRoi_right
	);

	cv::initUndistortRectifyMap(K_left, distCoeffs_left, R_left_, P_left_, imageSize_left, CV_16SC2, rmap_left_[0], rmap_left_[1]);
	cv::initUndistortRectifyMap(K_right, distCoeffs_right, R_right_, P_right_, imageSize_right, CV_16SC2, rmap_right_[0], rmap_right_[1]);

	// OpenCV can handle left-right or up-down camera arrangements
	//const bool isVerticalStereo = std::fabs(P_right_.at<double>(1, 3)) > std::fabs(P_right_.at<double>(0, 3));
}

ImageRectificationUsingOpenCV::~ImageRectificationUsingOpenCV()
{
}

// [ref] rectify_images_using_opencv() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void ImageRectificationUsingOpenCV::rectify(const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right) const
{
	cv::remap(input_image_left, output_image_left, rmap_left_[0], rmap_left_[1], CV_INTER_LINEAR);
	cv::remap(input_image_right, output_image_right, rmap_right_[0], rmap_right_[1], CV_INTER_LINEAR);
}

}  // namespace swl
