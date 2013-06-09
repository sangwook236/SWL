#if !defined(__SWL_MACHINE_VISION__IMAGE_RECTIFICATION_USING_OPENCV__H_)
#define __SWL_MACHINE_VISION__IMAGE_RECTIFICATION_USING_OPENCV__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_MACHINE_VISION_API ImageRectificationUsingOpenCV
{
public:
	//typedef ImageRectificationUsingOpenCV base_type;

public:
	ImageRectificationUsingOpenCV(
		const cv::Size &imageSize_left, const cv::Mat &K_left, const cv::Mat &distCoeffs_left,
		const cv::Size &imageSize_right, const cv::Mat &K_right, const cv::Mat &distCoeffs_right,
		const cv::Mat &R, const cv::Mat &T
	);
	~ImageRectificationUsingOpenCV();

private:
	explicit ImageRectificationUsingOpenCV(const ImageRectificationUsingOpenCV &rhs);  // not implemented
	ImageRectificationUsingOpenCV & operator=(const ImageRectificationUsingOpenCV &rhs);  // not implemented

public:
	void rectify(const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right) const;

private:
	cv::Mat R_left_, R_right_, P_left_, P_right_, Q_;
	cv::Mat rmap_left_[2], rmap_right_[2];
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__IMAGE_RECTIFICATION_USING_OPENCV__H_
