#include "swl/Config.h"
#include "swl/machine_vision/ImageUndistortionUsingOpenCV.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#define __USE_OPENCV_REMAP 1

namespace swl {

ImageUndistortionUsingOpenCV::ImageUndistortionUsingOpenCV(const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs)
: imageSize_(imageSize), K_(K), distCoeffs_(distCoeffs)
{
#if __USE_OPENCV_REMAP
	cv::initUndistortRectifyMap(
		K_, distCoeffs_, cv::Mat(),
		cv::getOptimalNewCameraMatrix(K_, distCoeffs_, imageSize_, 1, imageSize_, 0),
		imageSize_, CV_16SC2, rmap_[0], rmap_[1]
	);
#endif
}

ImageUndistortionUsingOpenCV::~ImageUndistortionUsingOpenCV()
{
}

// [ref] undistort_images_using_opencv() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_undistortion.cpp
void ImageUndistortionUsingOpenCV::undistort(const cv::Mat &input_image, cv::Mat &output_image) const
{
#if __USE_OPENCV_REMAP
	cv::remap(input_image, output_image, rmap_[0], rmap_[1], cv::INTER_LINEAR);
#else
	cv::undistort(input_image, output_image, K_, distCoeffs_, K_);
#endif
}

}  // namespace swl
