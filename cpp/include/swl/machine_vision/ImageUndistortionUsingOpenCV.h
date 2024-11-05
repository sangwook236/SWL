#if !defined(__SWL_MACHINE_VISION__IMAGE_UNDISTORTION_USING_OPENCV__H_)
#define __SWL_MACHINE_VISION__IMAGE_UNDISTORTION_USING_OPENCV__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_MACHINE_VISION_API ImageUndistortionUsingOpenCV
{
public:
	//typedef ImageUndistortionUsingOpenCV base_type;

public:
	ImageUndistortionUsingOpenCV(const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs);
	~ImageUndistortionUsingOpenCV();

private:
	explicit ImageUndistortionUsingOpenCV(const ImageUndistortionUsingOpenCV &rhs);  // not implemented
	ImageUndistortionUsingOpenCV & operator=(const ImageUndistortionUsingOpenCV &rhs);  // not implemented

public:
	void undistort(const cv::Mat &input_image, cv::Mat &output_image) const;

private:
	const cv::Size imageSize_;
	const cv::Mat K_;
	const cv::Mat distCoeffs_;

	cv::Mat rmap_[2];
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__IMAGE_UNDISTORTION_USING_OPENCV__H_
