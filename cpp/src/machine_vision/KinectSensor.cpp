#include "swl/Config.h"
#include "swl/machine_vision/KinectSensor.h"
#include "swl/machine_vision/ImageUndistortionUsingOpenCV.h"
#include "swl/machine_vision/ImageRectificationUsingOpenCV.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>


#define __USE_OPENCV_REMAP 1
//#define __USE_OPENCV_UNDISTORTION 1

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace swl {

KinectSensor::KinectSensor(
	const bool useIRtoRGB,
	const cv::Size &imageSize_ir, const cv::Mat &K_ir, const cv::Mat &distCoeffs_ir,
	const cv::Size &imageSize_rgb, const cv::Mat &K_rgb, const cv::Mat &distCoeffs_rgb,
	const cv::Mat &R, const cv::Mat &T
)
: useIRtoRGB_(useIRtoRGB), useOpenCV_(false),
  imageSize_ir_(imageSize_ir), imageSize_rgb_(imageSize_rgb),
  K_ir_(K_ir), K_rgb_(K_rgb),
  distCoeffs_ir_(distCoeffs_ir), distCoeffs_rgb_(distCoeffs_rgb),
  R_(R), T_(T)
{
}

KinectSensor::~KinectSensor()
{
}

void KinectSensor::initialize()
{
	prepareRectification();
}

void KinectSensor::rectifyImagePair(const cv::Mat &ir_input_image, const cv::Mat &rgb_input_image, cv::Mat &ir_output_image, cv::Mat &rgb_output_image) const
{
	if (useOpenCV_)
	{
		if (useIRtoRGB_)
			imageRectificationUsingOpenCV_->rectify(ir_input_image, rgb_input_image, ir_output_image, rgb_output_image);
		else
			imageRectificationUsingOpenCV_->rectify(rgb_input_image, ir_input_image, rgb_output_image, ir_output_image);
	}
	else
		rectifyImagePairUsingDepth(ir_input_image, rgb_input_image, ir_output_image, rgb_output_image);  // using Kinect's depth information
}

void KinectSensor::prepareRectification()
{
	if (useOpenCV_)
	{
		if (useIRtoRGB_)
			imageRectificationUsingOpenCV_.reset(new ImageRectificationUsingOpenCV(imageSize_ir_, K_ir_, distCoeffs_ir_, imageSize_rgb_, K_rgb_, distCoeffs_rgb_, R_, T_));
		else
			imageRectificationUsingOpenCV_.reset(new ImageRectificationUsingOpenCV(imageSize_rgb_, K_rgb_, distCoeffs_rgb_, imageSize_ir_, K_ir_, distCoeffs_ir_, R_, T_));
	}
	else
	{
#if __USE_OPENCV_UNDISTORTION && __USE_OPENCV_REMAP
		imageUndistortionUsingOpenCV_ir_.reset(new ImageUndistortionUsingOpenCV(imageSize_ir_, K_ir_, distCoeffs_ir_));
		imageUndistortionUsingOpenCV_rgb_.reset(new ImageUndistortionUsingOpenCV(imageSize_rgb_, K_rgb_, distCoeffs_rgb_));
#endif

		computeHomogeneousImageCoordinates(imageSize_ir_, K_ir_, distCoeffs_ir_, IC_homo_ir_, IC_homo_undist_ir_);
		computeHomogeneousImageCoordinates(imageSize_rgb_, K_rgb_, distCoeffs_rgb_, IC_homo_rgb_, IC_homo_undist_rgb_);
	}
}

// [ref] undistort_images_using_formula() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_undistortion.cpp
void KinectSensor::computeHomogeneousImageCoordinates(const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs, cv::Mat &IC_homo, cv::Mat &IC_homo_undist)
{
	// [ref] "Joint Depth and Color Camera Calibration with Distortion Correction", D. Herrera C., J. Kannala, & J. Heikkila, TPAMI, 2012

	// homogeneous image coordinates: zero-based coordinates
	cv::Mat(3, imageSize.height * imageSize.width, CV_64FC1, cv::Scalar::all(1)).copyTo(IC_homo);
	//IC_homo = cv::Mat::ones(3, imageSize.height * imageSize.width, CV_64FC1);
	{
#if 0
		// 0 0 0 ...   0 1 1 1 ...   1 ... 639 639 639 ... 639
		// 0 1 2 ... 479 0 1 2 ... 479 ...   0   1   2 ... 479

		cv::Mat arr(1, imageSize.height, CV_64FC1);
		for (int i = 0; i < imageSize.height; ++i)
			arr.at<double>(0, i) = (double)i;

		for (int i = 0; i < imageSize.width; ++i)
		{
			IC_homo(cv::Range(0, 1), cv::Range(i * imageSize.height, (i + 1) * imageSize.height)).setTo(cv::Scalar::all(i));
			arr.copyTo(IC_homo(cv::Range(1, 2), cv::Range(i * imageSize.height, (i + 1) * imageSize.height)));
		}
#else
		// 0 1 2 ... 639 0 1 2 ... 639 ...   0   1   2 ... 639
		// 0 0 0 ...   0 1 1 1 ...   1 ... 479 479 479 ... 479

		cv::Mat arr(1, imageSize.width, CV_64FC1);
		for (int i = 0; i < imageSize.width; ++i)
			arr.at<double>(0, i) = (double)i;

		for (int i = 0; i < imageSize.height; ++i)
		{
			arr.copyTo(IC_homo(cv::Range(0, 1), cv::Range(i * imageSize.width, (i + 1) * imageSize.width)));
			IC_homo(cv::Range(1, 2), cv::Range(i * imageSize.width, (i + 1) * imageSize.width)).setTo(cv::Scalar::all(i));
		}
#endif
	}

	// homogeneous normalized camera coordinates
	const cv::Mat CC_norm(K.inv() * IC_homo);

	// apply distortion
	{
		//const cv::Mat xn(CC_norm(cv::Range(0,1), cv::Range::all()));
		const cv::Mat xn(CC_norm(cv::Range(0,1), cv::Range::all()) / CC_norm(cv::Range(2,3), cv::Range::all()));
		//const cv::Mat yn(CC_norm(cv::Range(1,2), cv::Range::all()));
		const cv::Mat yn(CC_norm(cv::Range(1,2), cv::Range::all()) / CC_norm(cv::Range(2,3), cv::Range::all()));

		const cv::Mat xn2(xn.mul(xn));
		const cv::Mat yn2(yn.mul(yn));
		const cv::Mat xnyn(xn.mul(yn));
		const cv::Mat r2(xn2 + yn2);
		const cv::Mat r4(r2.mul(r2));
		const cv::Mat r6(r4.mul(r2));

		const double &k1 = distCoeffs.at<double>(0);
		const double &k2 = distCoeffs.at<double>(1);
		const double &k3 = distCoeffs.at<double>(2);
		const double &k4 = distCoeffs.at<double>(3);
		const double &k5 = distCoeffs.at<double>(4);

		const cv::Mat xg(2.0 * k3 * xnyn + k4 * (r2 + 2.0 * xn2));
		const cv::Mat yg(k3 * (r2 + 2.0 * yn2) + 2.0 * k4 * xnyn);

		const cv::Mat coeff(1.0 + k1 * r2 + k2 * r4 + k5 * r6);
		cv::Mat xk(3, imageSize.height * imageSize.width, CV_64FC1, cv::Scalar::all(1));
		cv::Mat(coeff.mul(xn) + xg).copyTo(xk(cv::Range(0,1), cv::Range::all()));
		cv::Mat(coeff.mul(yn) + yg).copyTo(xk(cv::Range(1,2), cv::Range::all()));

		IC_homo_undist = K * xk;
	}
}

// [ref] rectify_kinect_images_using_depth() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void KinectSensor::rectifyImagePairUsingDepth(const cv::Mat &ir_input_image, const cv::Mat &rgb_input_image, cv::Mat &ir_output_image, cv::Mat &rgb_output_image) const
{
	// undistort images
	// TODO [check] >> is undistortion required before rectification?
	//	Undistortion process is required before rectification,
	//	since currently image undistortion is not applied during rectification process in rectify_kinect_images_from_IR_to_RGB_using_depth() & rectify_kinect_images_from_RGB_to_IR_using_depth().
	cv::Mat ir_input_image2, rgb_input_image2;
	{
#if __USE_OPENCV_UNDISTORTION
		imageUndistortionUsingOpenCV_ir_->undistort(ir_input_image, ir_input_image2);
		imageUndistortionUsingOpenCV_rgb_->undistort(rgb_input_image, rgb_input_image2);
#else

#if 1
		undistortImagePairUsingFormula<unsigned short>(ir_input_image, ir_input_image2, IC_homo_ir_, IC_homo_undist_ir_);
		undistortImagePairUsingFormula<cv::Vec3b>(rgb_input_image, rgb_input_image2, IC_homo_rgb_, IC_homo_undist_rgb_);
#else
		undistortImagePairUsingFormula<unsigned short>(ir_input_image, ir_input_image2, IC_homo_ir_, IC_homo_undist_ir_);

		cv::Mat rgb_input_gray_image;
		cv::cvtColor(rgb_input_image, rgb_input_gray_image, CV_BGR2GRAY);
		undistortImagePairUsingFormula<unsigned char>(rgb_input_gray_image, rgb_input_image2, IC_homo_rgb_, IC_homo_undist_rgb_);
#endif

#endif  // __USE_OPENCV_UNDISTORTION
	}

	// rectify images
	if (useIRtoRGB_)
		rectifyImagePairFromIRToRGBUsingDepth(
			ir_input_image2, rgb_input_image2, ir_output_image, rgb_output_image,
			imageSize_ir_, imageSize_rgb_,
			K_ir_, K_rgb_, R_, T_,
			IC_homo_ir_
		);
	else
		rectifyImagePairFromRGBToIRUsingDepth(
			rgb_input_image2, ir_input_image2, rgb_output_image, ir_output_image,
			imageSize_rgb_, imageSize_ir_,
			K_rgb_, K_ir_, R_, T_,
			IC_homo_rgb_
		);  // Not yet implemented.
}

// REF [function] >> rectify_kinect_images_from_IR_to_RGB_using_depth() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
// IR (left) to RGB (right).
void KinectSensor::rectifyImagePairFromIRToRGBUsingDepth(
	const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T,
	const cv::Mat &IC_homo_left
) const
{
	// Homogeneous normalized camera coordinates (left).
	const cv::Mat CC_norm_left(K_left.inv() * IC_homo_left);

	// Camera coordinates (left).
	cv::Mat CC_left;
	{
		cv::Mat tmp;
#if 0
		// 0 0 0 ...   0 1 1 1 ...   1 ... 639 639 639 ... 639.
		// 0 1 2 ... 479 0 1 2 ... 479 ...   0   1   2 ... 479.

		((cv::Mat)input_image_left.t()).convertTo(tmp, CV_64FC1, 1.0, 0.0);
#else
		// 0 1 2 ... 639 0 1 2 ... 639 ...   0   1   2 ... 639.
		// 0 0 0 ...   0 1 1 1 ...   1 ... 479 479 479 ... 479.

		input_image_left.convertTo(tmp, CV_64FC1, 1.0, 0.0);
#endif
		cv::repeat(tmp.reshape(1, 1), 3, 1, CC_left);
		CC_left = CC_left.mul(CC_norm_left);
	}

	// Camera coordinates (right).
	cv::Mat CC_right;
#if 0
	cv::repeat(T, 1, imageSize_left.width*imageSize_left.height, CC_right);
	CC_right = R.t() * (CC_left - CC_right);
#else
	cv::repeat(T, 1, imageSize_left.width*imageSize_left.height, CC_right);
	CC_right = R * CC_left + CC_right;
#endif

	// Homogeneous normalized camera coordinates (right).
	cv::Mat CC_norm_right;
	cv::repeat(CC_right(cv::Range(2, 3), cv::Range::all()), 3, 1, CC_norm_right);
	CC_norm_right = CC_right / CC_norm_right;

	// Homogeneous image coordinates (right).
	const cv::Mat IC_homo_right(K_right * CC_norm_right);  // Zero-based coordinates.

	// The left image is mapped onto the right image.
	cv::Mat(input_image_right.size(), input_image_left.type(), cv::Scalar::all(0)).copyTo(output_image_left);
	//output_image_left = cv::Mat::zeros(input_image_right.size(), input_image_left.type());
#pragma omp parallel
//#pragma omp parallel shared(a, b, c, d, loop_count, chunk) private(i)
//#pragma omp parallel shared(a, b, c, d, loop_count, i, chunk)
	{
#pragma omp for
//#pragma omp for schedule(dynamic, 100) nowait
		for (int idx = 0; idx < imageSize_left.height*imageSize_left.width; ++idx)
		{
			const int &cc = (int)cvRound(IC_homo_right.at<double>(0,idx));
			const int &rr = (int)cvRound(IC_homo_right.at<double>(1,idx));
			if (0 <= cc && cc < imageSize_right.width && 0 <= rr && rr < imageSize_right.height)
				output_image_left.at<unsigned short>(rr, cc) = (unsigned short)cvRound(CC_left.at<double>(2, idx));
		}
	}

	input_image_right.copyTo(output_image_right);
}

// REF [function] >> rectify_kinect_images_from_RGB_to_IR_using_depth() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
// RGB (left) to IR (right).
void KinectSensor::rectifyImagePairFromRGBToIRUsingDepth(
	const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T,
	const cv::Mat &IC_homo_left
) const
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace swl
