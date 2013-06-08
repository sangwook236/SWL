//#include "stdafx.h"
#include "KinectSensor.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>


#define __USE_REMAP 1

namespace {
namespace local {

// [ref] load_kinect_sensor_parameters_from_IR_to_RGB() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void load_kinect_sensor_parameters_from_IR_to_RGB(
	cv::Mat &K_ir, cv::Mat &distCoeffs_ir, cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb,
	cv::Mat &R_ir_to_rgb, cv::Mat &T_ir_to_rgb
)
{
	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	// Caution:
	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab in OpenCV,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// IR (left) to RGB (right)
#if 1
	// the 5th distortion parameter, kc(5) is activated.

	const double fc_ir[] = { 5.865281297534211e+02, 5.866623900166177e+02 };  // [pixel]
	const double cc_ir[] = { 3.371860463542209e+02, 2.485298169373497e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.227084070414958e-01, 5.027511830344261e-01, -2.562850607972214e-03, 6.916249031489476e-03, -5.507709925923052e-01 };  // 5x1 vector
	const double kc_ir[] = { -1.227084070414958e-01, 5.027511830344261e-01, -2.562850607972214e-03, 6.916249031489476e-03, -5.507709925923052e-01, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_rgb[] = { 5.248648751941851e+02, 5.268281060449414e+02 };  // [pixel]
	const double cc_rgb[] = { 3.267484107269922e+02, 2.618261807606497e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.796770514235670e-01, -1.112507253647945e+00, 9.265501548915561e-04, 2.428229310663184e-03, 1.744019737212440e+00 };  // 5x1 vector
	const double kc_rgb[] = { 2.796770514235670e-01, -1.112507253647945e+00, 9.265501548915561e-04, 2.428229310663184e-03, 1.744019737212440e+00, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { -1.936270295074452e-03, 1.331596538715070e-02, 3.404073398703758e-03 };
	const double transVec[] = { 2.515260082139980e+01, 4.059127243511693e+00, -5.588303932036697e+00 };  // [mm]
#else
	// the 5th distortion parameter, kc(5) is deactivated.

	const double fc_ir[] = { 5.864902565580264e+02, 5.867305900503998e+02 };  // [pixel]
	const double cc_ir[] = { 3.376088045224677e+02, 2.480083390372575e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.123867977947529e-01, 3.552017514491446e-01, -2.823972305243438e-03, 7.246763414437084e-03, 0.0 };  // 5x1 vector
	const double kc_ir[] = { -1.123867977947529e-01, 3.552017514491446e-01, -2.823972305243438e-03, 7.246763414437084e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_rgb[] = { 5.256215953836251e+02, 5.278165866956751e+02 };  // [pixel]
	const double cc_rgb[] = { 3.260532981578608e+02, 2.630788286947369e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.394862387380747e-01, -5.840355691714197e-01, 2.567740590187774e-03, 2.044179978023951e-03, 0.0 };  // 5x1 vector
	const double kc_rgb[] = { 2.394862387380747e-01, -5.840355691714197e-01, 2.567740590187774e-03, 2.044179978023951e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { 1.121432126402549e-03, 1.535221550916760e-02, 3.701648572107407e-03 };
	const double transVec[] = { 2.512732389978993e+01, 3.724869927389498e+00, -4.534758982979088e+00 };  // [mm]
#endif

	//
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_ir);
	K_ir.at<double>(0, 0) = fc_ir[0];
	K_ir.at<double>(0, 1) = alpha_c_ir * fc_ir[0];
	K_ir.at<double>(0, 2) = cc_ir[0];
	K_ir.at<double>(1, 1) = fc_ir[1];
	K_ir.at<double>(1, 2) = cc_ir[1];
	K_ir.at<double>(2, 2) = 1.0;
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_rgb);
	K_rgb.at<double>(0, 0) = fc_rgb[0];
	K_rgb.at<double>(0, 1) = alpha_c_rgb * fc_rgb[0];
	K_rgb.at<double>(0, 2) = cc_rgb[0];
	K_rgb.at<double>(1, 1) = fc_rgb[1];
	K_rgb.at<double>(1, 2) = cc_rgb[1];
	K_rgb.at<double>(2, 2) = 1.0;

	cv::Mat(8, 1, CV_64FC1, (void *)kc_ir).copyTo(distCoeffs_ir);
	cv::Mat(8, 1, CV_64FC1, (void *)kc_rgb).copyTo(distCoeffs_rgb);

    cv::Rodrigues(cv::Mat(3, 1, CV_64FC1, (void *)rotVec), R_ir_to_rgb);
	cv::Mat(3, 1, CV_64FC1, (void *)transVec).copyTo(T_ir_to_rgb);
}

// [ref] load_kinect_sensor_parameters_from_RGB_to_IR() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void load_kinect_sensor_parameters_from_RGB_to_IR(
	cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb, cv::Mat &K_ir, cv::Mat &distCoeffs_ir,
	cv::Mat &R_rgb_to_ir, cv::Mat &T_rgb_to_ir
)
{
	// [ref]
	//	Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	// Caution:
	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab in OpenCV,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// RGB (left) to IR (right)
#if 1
	// the 5th distortion parameter, kc(5) is activated.

	const double fc_rgb[] = { 5.248648079874888e+02, 5.268280486062615e+02 };  // [pixel]
	const double cc_rgb[] = { 3.267487100838014e+02, 2.618261169946102e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.796764337988712e-01, -1.112497355183840e+00, 9.264749543097661e-04, 2.428507887293728e-03, 1.743975665436613e+00 };  // 5x1 vector
	const double kc_rgb[] = { 2.796764337988712e-01, -1.112497355183840e+00, 9.264749543097661e-04, 2.428507887293728e-03, 1.743975665436613e+00, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_ir[] = { 5.865282023957649e+02, 5.866624209441105e+02 };  // [pixel]
	const double cc_ir[] = { 3.371875014947813e+02, 2.485295493095561e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.227176734054719e-01, 5.028746725848668e-01, -2.563029340202278e-03, 6.916996280663117e-03, -5.512162545452755e-01 };  // 5x1 vector
	const double kc_ir[] = { -1.227176734054719e-01, 5.028746725848668e-01, -2.563029340202278e-03, 6.916996280663117e-03, -5.512162545452755e-01, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { 1.935939237060295e-03, -1.331788958930441e-02, -3.404128236480992e-03 };
	const double transVec[] = { -2.515262012891160e+01, -4.059118899373607e+00, 5.588237589014362e+00 };  // [mm]
#else
	// the 5th distortion parameter, kc(5) is deactivated.

	const double fc_rgb[] = { 5.256217798767822e+02, 5.278167798992870e+02 };  // [pixel]
	const double cc_rgb[] = { 3.260534767468189e+02, 2.630800669346188e+02 };  // [pixel]
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.394861400525463e-01, -5.840298777969020e-01, 2.568959896208732e-03, 2.044336479083819e-03, 0.0 };  // 5x1 vector
	const double kc_rgb[] = { 2.394861400525463e-01, -5.840298777969020e-01, 2.568959896208732e-03, 2.044336479083819e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double fc_ir[] = { 5.864904832545356e+02, 5.867308191567271e+02 };  // [pixel]
	const double cc_ir[] = { 3.376079004969836e+02, 2.480098376453992e+02 };  // [pixel]
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.123902857373373e-01, 3.552211727724343e-01, -2.823183218548772e-03, 7.246270574438420e-03, 0.0 };  // 5x1 vector
	const double kc_ir[] = { -1.123902857373373e-01, 3.552211727724343e-01, -2.823183218548772e-03, 7.246270574438420e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector

	const double rotVec[] = { -1.121214964017936e-03, -1.535031632771925e-02, -3.701579055761772e-03 };
	const double transVec[] = { -2.512730902761022e+01, -3.724884753207001e+00, 4.534776794502955e+00 };  // [mm]
#endif

	//
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_rgb);
	K_rgb.at<double>(0, 0) = fc_rgb[0];
	K_rgb.at<double>(0, 1) = alpha_c_rgb * fc_rgb[0];
	K_rgb.at<double>(0, 2) = cc_rgb[0];
	K_rgb.at<double>(1, 1) = fc_rgb[1];
	K_rgb.at<double>(1, 2) = cc_rgb[1];
	K_rgb.at<double>(2, 2) = 1.0;
	cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)).copyTo(K_ir);
	K_ir.at<double>(0, 0) = fc_ir[0];
	K_ir.at<double>(0, 1) = alpha_c_ir * fc_ir[0];
	K_ir.at<double>(0, 2) = cc_ir[0];
	K_ir.at<double>(1, 1) = fc_ir[1];
	K_ir.at<double>(1, 2) = cc_ir[1];
	K_ir.at<double>(2, 2) = 1.0;

	cv::Mat(8, 1, CV_64FC1, (void *)kc_rgb).copyTo(distCoeffs_rgb);
	cv::Mat(8, 1, CV_64FC1, (void *)kc_ir).copyTo(distCoeffs_ir);

    cv::Rodrigues(cv::Mat(3, 1, CV_64FC1, (void *)rotVec), R_rgb_to_ir);
	cv::Mat(3, 1, CV_64FC1, (void *)transVec).copyTo(T_rgb_to_ir);
}

}  // namespace local
}  // unnamed namespace

namespace swl {

KinectSensor::KinectSensor(const cv::Size &imageSize_ir, const cv::Size &imageSize_rgb)
: useIRtoRGB_(true), useOpenCV_(false), imageSize_ir_(imageSize_ir), imageSize_rgb_(imageSize_rgb)
{
}

KinectSensor::~KinectSensor()
{
}

void KinectSensor::loadCameraParameters()
{
	// load the camera parameters of a Kinect sensor
	if (useIRtoRGB_)
		local::load_kinect_sensor_parameters_from_IR_to_RGB(K_ir_, distCoeffs_ir_, K_rgb_, distCoeffs_rgb_, R_, T_);
	else
		local::load_kinect_sensor_parameters_from_RGB_to_IR(K_rgb_, distCoeffs_rgb_, K_ir_, distCoeffs_ir_, R_, T_);

	//
	prepareRectification();
}

void KinectSensor::rectifyImagePair(const cv::Mat &ir_input_image, const cv::Mat &rgb_input_image, cv::Mat &ir_output_image, cv::Mat &rgb_output_image) const
{
	if (useOpenCV_)
		rectifyImagePairUsingOpenCV(ir_input_image, rgb_input_image, ir_output_image, rgb_output_image);  // using OpenCV
	else
		rectifyImagePairUsingDepth(ir_input_image, rgb_input_image, ir_output_image, rgb_output_image);  // using Kinect's depth information
}

void KinectSensor::prepareRectification()
{
	if (useOpenCV_)
	{
		cv::Rect validRoi_left, validRoi_right;
		if (useIRtoRGB_)
		{
			cv::stereoRectify(
				K_ir_, distCoeffs_ir_,
				K_rgb_, distCoeffs_rgb_,
				imageSize_ir_, R_, T_, R_left_, R_right_, P_left_, P_right_, Q_,
				cv::CALIB_ZERO_DISPARITY, 1, imageSize_ir_, &validRoi_left, &validRoi_right
			);

			cv::initUndistortRectifyMap(K_ir_, distCoeffs_ir_, R_left_, P_left_, imageSize_ir_, CV_16SC2, rmap_left_[0], rmap_left_[1]);
			cv::initUndistortRectifyMap(K_rgb_, distCoeffs_rgb_, R_right_, P_right_, imageSize_rgb_, CV_16SC2, rmap_right_[0], rmap_right_[1]);
		}
		else
		{
			cv::stereoRectify(
				K_rgb_, distCoeffs_rgb_,
				K_ir_, distCoeffs_ir_,
				imageSize_rgb_, R_, T_, R_left_, R_right_, P_left_, P_right_, Q_,
				cv::CALIB_ZERO_DISPARITY, 1, imageSize_rgb_, &validRoi_left, &validRoi_right
			);

			cv::initUndistortRectifyMap(K_rgb_, distCoeffs_rgb_, R_left_, P_left_, imageSize_rgb_, CV_16SC2, rmap_left_[0], rmap_left_[1]);
			cv::initUndistortRectifyMap(K_ir_, distCoeffs_ir_, R_right_, P_right_, imageSize_ir_, CV_16SC2, rmap_right_[0], rmap_right_[1]);
		}

		// OpenCV can handle left-right or up-down camera arrangements
		//const bool isVerticalStereo = std::fabs(P_right_.at<double>(1, 3)) > std::fabs(P_right_.at<double>(0, 3));
	}
	else
	{
#if __USE_REMAP
		cv::initUndistortRectifyMap(
			K_ir_, distCoeffs_ir_, cv::Mat(),
			cv::getOptimalNewCameraMatrix(K_ir_, distCoeffs_ir_, imageSize_ir_, 1, imageSize_ir_, 0),
			imageSize_ir_, CV_16SC2, rmap_ir_[0], rmap_ir_[1]
		);
		cv::initUndistortRectifyMap(
			K_rgb_, distCoeffs_rgb_, cv::Mat(),
			cv::getOptimalNewCameraMatrix(K_rgb_, distCoeffs_rgb_, imageSize_rgb_, 1, imageSize_rgb_, 0),
			imageSize_rgb_, CV_16SC2, rmap_rgb_[0], rmap_rgb_[1]
		);
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

// [ref] rectify_kinect_images_using_opencv() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void KinectSensor::rectifyImagePairUsingOpenCV(const cv::Mat &ir_input_image, const cv::Mat &rgb_input_image, cv::Mat &ir_output_image, cv::Mat &rgb_output_image) const
{
	if (useIRtoRGB_)
		rectifyImagesPairFromLeftToRightUsingOpenCV(ir_input_image, rgb_input_image, ir_output_image, rgb_output_image);
	else
		rectifyImagesPairFromLeftToRightUsingOpenCV(rgb_input_image, ir_input_image, rgb_output_image, ir_output_image);
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
#if 0
		undistortImagePairUsingOpenCV(ir_input_image, ir_input_image2, imageSize_ir_, K_ir_, distCoeffs_ir_, rmap_ir_);
		undistortImagePairUsingOpenCV(rgb_input_image, rgb_input_image2, imageSize_rgb_, K_rgb_, distCoeffs_rgb_, rmap_rgb_);
#elif 1
		undistortImagePairUsingFormula<unsigned short>(ir_input_image, ir_input_image2, IC_homo_ir_, IC_homo_undist_ir_);
		undistortImagePairUsingFormula<cv::Vec3b>(rgb_input_image, rgb_input_image2, IC_homo_rgb_, IC_homo_undist_rgb_);
#else
		undistortImagePairUsingFormula<unsigned short>(ir_input_image, ir_input_image2, IC_homo_ir_, IC_homo_undist_ir_);

		cv::Mat rgb_input_gray_image;
		cv::cvtColor(rgb_input_image, rgb_input_gray_image, CV_BGR2GRAY);
		undistortImagePairUsingFormula<unsigned char>(rgb_input_gray_image, rgb_input_image2, IC_homo_rgb_, IC_homo_undist_rgb_);
#endif
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
		);  // not yet implemented
}

// [ref] rectify_images_using_opencv() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void KinectSensor::rectifyImagesPairFromLeftToRightUsingOpenCV(const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right) const
{
	cv::remap(input_image_left, output_image_left, rmap_left_[0], rmap_left_[1], CV_INTER_LINEAR);
	cv::remap(input_image_right, output_image_right, rmap_right_[0], rmap_right_[1], CV_INTER_LINEAR);
}

// [ref] undistort_images_using_opencv() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_undistortion.cpp
void KinectSensor::undistortImagePairUsingOpenCV(const cv::Mat &input_image, cv::Mat &output_image, const cv::Size &imageSize, const cv::Mat &K, const cv::Mat &distCoeffs, const cv::Mat rmap[2]) const
{
#if __USE_REMAP
	cv::remap(input_image, output_image, rmap[0], rmap[1], cv::INTER_LINEAR);
#else
	cv::undistort(input_image, output_image, K, distCoeffs, K);
#endif
}

// [ref] rectify_kinect_images_from_IR_to_RGB_using_depth() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
// IR (left) to RGB (right)
void KinectSensor::rectifyImagePairFromIRToRGBUsingDepth(
	const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T,
	const cv::Mat &IC_homo_left
) const
{
	// homogeneous normalized camera coordinates (left)
	const cv::Mat CC_norm_left(K_left.inv() * IC_homo_left);

	// camera coordinates (left)
	cv::Mat CC_left;
	{
		cv::Mat tmp;
#if 0
		// 0 0 0 ...   0 1 1 1 ...   1 ... 639 639 639 ... 639
		// 0 1 2 ... 479 0 1 2 ... 479 ...   0   1   2 ... 479

		((cv::Mat)input_image_left.t()).convertTo(tmp, CV_64FC1, 1.0, 0.0);
#else
		// 0 1 2 ... 639 0 1 2 ... 639 ...   0   1   2 ... 639
		// 0 0 0 ...   0 1 1 1 ...   1 ... 479 479 479 ... 479

		input_image_left.convertTo(tmp, CV_64FC1, 1.0, 0.0);
#endif
		cv::repeat(tmp.reshape(1, 1), 3, 1, CC_left);
		CC_left = CC_left.mul(CC_norm_left);
	}

	// camera coordinates (right)
	cv::Mat CC_right;
#if 0
	cv::repeat(T, 1, imageSize_left.width*imageSize_left.height, CC_right);
	CC_right = R.t() * (CC_left - CC_right);
#else
	cv::repeat(T, 1, imageSize_left.width*imageSize_left.height, CC_right);
	CC_right = R * CC_left + CC_right;
#endif

	// homogeneous normalized camera coordinates (right)
	cv::Mat CC_norm_right;
	cv::repeat(CC_right(cv::Range(2, 3), cv::Range::all()), 3, 1, CC_norm_right);
	CC_norm_right = CC_right / CC_norm_right;

	// homogeneous image coordinates (right)
	const cv::Mat IC_homo_right(K_right * CC_norm_right);  // zero-based coordinates

	// the left image is mapped onto the right image.
	cv::Mat(input_image_right.size(), input_image_left.type(), cv::Scalar::all(0)).copyTo(output_image_left);
	for (int idx = 0; idx < imageSize_left.height*imageSize_left.width; ++idx)
	{
		const int &cc = (int)cvRound(IC_homo_right.at<double>(0,idx));
		const int &rr = (int)cvRound(IC_homo_right.at<double>(1,idx));
		if (0 <= cc && cc < imageSize_right.width && 0 <= rr && rr < imageSize_right.height)
			output_image_left.at<unsigned short>(rr, cc) = (unsigned short)cvRound(CC_left.at<double>(2, idx));
	}

	input_image_right.copyTo(output_image_right);
}

// [ref] rectify_kinect_images_from_RGB_to_IR_using_depth() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
// RGB (left) to IR (right)
void KinectSensor::rectifyImagePairFromRGBToIRUsingDepth(
	const cv::Mat &input_image_left, const cv::Mat &input_image_right, cv::Mat &output_image_left, cv::Mat &output_image_right,
	const cv::Size &imageSize_left, const cv::Size &imageSize_right,
	const cv::Mat &K_left, const cv::Mat &K_right, const cv::Mat &R, const cv::Mat &T,
	const cv::Mat &IC_homo_left
) const
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace swl
