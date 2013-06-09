#if !defined(__SWL_MACHINE_VISION__KINECT_SENSOR__H_)
#define __SWL_MACHINE_VISION__KINECT_SENSOR__H_ 1


#include "swl/machine_vision/ExportMachineVision.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>


namespace swl {

class ImageUndistortionUsingOpenCV;
class ImageRectificationUsingOpenCV;

//--------------------------------------------------------------------------
//

class SWL_MACHINE_VISION_API KinectSensor
{
public:
	KinectSensor(
		const bool useIRtoRGB,
		const cv::Size &imageSize_ir, const cv::Mat &K_ir, const cv::Mat &distCoeffs_ir,
		const cv::Size &imageSize_rgb, const cv::Mat &K_rgb, const cv::Mat &distCoeffs_rgb,
		const cv::Mat &R, const cv::Mat &T
	);
	~KinectSensor();

private:
	explicit KinectSensor(const KinectSensor &rhs);  // not implemented
	KinectSensor & operator=(const KinectSensor &rhs);  // not implemented

public:
	void initialize();
	void rectifyImagePair(const cv::Mat &ir_input_image, const cv::Mat &rgb_input_image, cv::Mat &ir_output_image, cv::Mat &rgb_output_image) const;

private:
	void prepareRectification();

	void rectifyImagePairUsingDepth(const cv::Mat &ir_input_image, const cv::Mat &rgb_input_image, cv::Mat &ir_output_image, cv::Mat &rgb_output_image) const;
	void computeHomogeneousImageCoordinates(const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs, cv::Mat &IC_homo, cv::Mat &IC_homo_undist);

	void KinectSensor::rectifyImagePairFromIRToRGBUsingDepth(
		const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right,
		const cv::Size &imageSize_left, const cv::Size &imageSize_right,
		const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T,
		const cv::Mat &IC_homo_left
	) const;
	void KinectSensor::rectifyImagePairFromRGBToIRUsingDepth(
		const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right,
		const cv::Size &imageSize_left, const cv::Size &imageSize_right,
		const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T,
		const cv::Mat &IC_homo_left
	) const;

	// [ref] undistort_images_using_formula() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_undistortion.cpp
	template <typename T>
	void undistortImagePairUsingFormula(const cv::Mat &input_image, cv::Mat &output_image, const cv::Mat &IC_homo, const cv::Mat &IC_homo_undist) const
	{
		// [ref] "Joint Depth and Color Camera Calibration with Distortion Correction", D. Herrera C., J. Kannala, & J. Heikkila, TPAMI, 2012

		cv::Mat(input_image.size(), input_image.type(), cv::Scalar::all(0)).copyTo(output_image);
		//output_image = cv::Mat::zeros(input_image.size(), input_image.type());
#pragma omp parallel
//#pragma omp parallel shared(a, b, c, d, loop_count, chunk) private(i)
//#pragma omp parallel shared(a, b, c, d, loop_count, i, chunk)
		{
#pragma omp for
//#pragma omp for schedule(dynamic, 100) nowait
			for (int idx = 0; idx < input_image.rows*input_image.cols; ++idx)
			{
#if 0
				// don't apply interpolation.
				const int &cc_new = cvRound(IC_homo.at<double>(0,idx));
				const int &rr_new = cvRound(IC_homo.at<double>(1,idx));
				const int &cc = cvRound(IC_homo_undist.at<double>(0,idx));
				const int &rr = cvRound(IC_homo_undist.at<double>(1,idx));
				if (0 <= cc && cc < input_image.cols && 0 <= rr && rr < input_image.rows)
				{
					// TODO [check] >> why is the code below correctly working?
					//output_image.at<T>(rr, cc) = input_image.at<T>(rr_new, cc_new);
					output_image.at<T>(rr_new, cc_new) = input_image.at<T>(rr, cc);
				}
#else
				// apply interpolation.

				// TODO [enhance] >> speed up.

				const int &cc_new = cvRound(IC_homo.at<double>(0,idx));
				const int &rr_new = cvRound(IC_homo.at<double>(1,idx));

				const double &cc = IC_homo_undist.at<double>(0,idx);
				const double &rr = IC_homo_undist.at<double>(1,idx);
				const int cc_0 = cvFloor(cc), cc_1 = cc_0 + 1;
				const int rr_0 = cvFloor(rr), rr_1 = rr_0 + 1;
				const double alpha_cc = cc - cc_0, alpha_rr = rr - rr_0;
				if (0 <= cc_0 && cc_0 < input_image.cols - 1 && 0 <= rr_0 && rr_0 < input_image.rows - 1)
				{
					output_image.at<T>(rr_new, cc_new) =
						(T)((1.0 - alpha_rr) * (1.0 - alpha_cc) * input_image.at<T>(rr_0, cc_0) +
						(1.0 - alpha_rr) * alpha_cc * input_image.at<T>(rr_0, cc_1) +
						alpha_rr * (1.0 - alpha_cc) * input_image.at<T>(rr_1, cc_0) +
						alpha_rr * alpha_cc * input_image.at<T>(rr_1, cc_1));
				}
#endif
				}
		}
	}

private:
	const bool useIRtoRGB_;
	const bool useOpenCV_;

	const cv::Size imageSize_ir_, imageSize_rgb_;
	const cv::Mat K_ir_, K_rgb_;
	const cv::Mat distCoeffs_ir_, distCoeffs_rgb_;
	const cv::Mat R_, T_;

	// when using OpenCV
	boost::scoped_ptr<ImageRectificationUsingOpenCV> imageRectificationUsingOpenCV_;

	// when using Kinect depth
	cv::Mat IC_homo_ir_, IC_homo_rgb_;  // if sizes of images are the same, IC_homo_ir_ == IC_homo_rgb_.
	cv::Mat IC_homo_undist_ir_, IC_homo_undist_rgb_;

	boost::scoped_ptr<ImageUndistortionUsingOpenCV> imageUndistortionUsingOpenCV_ir_, imageUndistortionUsingOpenCV_rgb_;
};

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__KINECT_SENSOR__H_
