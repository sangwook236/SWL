//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include "swl/machine_vision/KinectSensor.h"
#include "gslic_lib/FastImgSeg.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <ctime>


#define __USE_RECTIFIED_IMAGE 1

namespace {
namespace local {

// [ref] canny() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_edge_detection.cpp
void canny(const cv::Mat &gray, cv::Mat &edge)
{
#if 0
	// down-scale and up-scale the image to filter out the noise
	cv::Mat blurred;
	cv::pyrDown(gray, blurred);
	cv::pyrUp(blurred, edge);
#else
	cv::blur(gray, edge, cv::Size(3, 3));
#endif

	// run the edge detector on grayscale
	const int lowerEdgeThreshold = 5, upperEdgeThreshold = 50;
	const bool useL2 = true;
	cv::Canny(edge, edge, lowerEdgeThreshold, upperEdgeThreshold, 3, useL2);
}

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

// [ref] gSLIC.cpp
void create_superpixel_by_gSLIC(const cv::Mat &input_image, cv::Mat &superpixel_mask, const SEGMETHOD seg_method, const double seg_weight, const int num_segments);
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary);

// [ref] EfficientGraphBasedImageSegmentation.cpp
void segment_image_using_efficient_graph_based_image_segmentation_algorithm(
	const cv::Mat &rgb_input_image, const cv::Mat &depth_input_image, const cv::Mat &depth_guided_mask,
	const float sigma, const float k, const int min_size,
	const float lambda1, const float lambda2, const float lambda3, const float fx_rgb, const float fy_rgb,
	int &num_ccs, cv::Mat &output_image
);
void segment_image_using_efficient_graph_based_image_segmentation_algorithm(const cv::Mat &rgb_input_image, const float sigma, const float k, const int min_size, int &num_ccs, cv::Mat &output_image);

// [ref] Util.cpp
cv::Rect get_bounding_rect(const cv::Mat &img);
void compute_phase_distribution_from_neighborhood(const cv::Mat &depth_map, const int radius);
void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, std::vector<cv::Point> &snake_contour);

void construct_depth_guided_mask_using_superpixel(
	const cv::Mat &rgb_input_image, const cv::Mat &depth_boundary_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_mask,
	const int num_segments, const SEGMETHOD seg_method, const double seg_weight
)
{
	cv::Mat rgb_superpixel_mask;
	cv::Mat filtered_superpixel_mask(rgb_input_image.size(), CV_8UC1, cv::Scalar::all(255)), filtered_superpixel_indexes(rgb_input_image.size(), CV_32SC1, cv::Scalar::all(0)); 
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;

	// PPP [] >>
	//	1. run superpixel.

	// superpixel mask consists of segment indexes.
	create_superpixel_by_gSLIC(rgb_input_image, rgb_superpixel_mask, seg_method, seg_weight, num_segments);

#if 0
	// show superpixel mask.
	cv::minMaxLoc(rgb_superpixel_mask, &minVal, &maxVal);
	rgb_superpixel_mask.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);

	cv::imshow("superpixels by gSLIC - mask", tmp_image);
#endif

#if 0
	// show superpixel boundary.
	cv::Mat rgb_superpixel_boundary;
	swl::create_superpixel_boundary(rgb_superpixel_mask, rgb_superpixel_boundary);

	rgb_input_image.copyTo(tmp_image);
	tmp_image.setTo(cv::Scalar(0, 0, 255), rgb_superpixel_boundary);

	cv::imshow("superpixels by gSLIC - boundary", tmp_image);
#endif

	// PPP [] >>
	//	2. depth info로부터 관심 영역의 boundary를 얻음.
	//		Depth histogram을 이용해 depth region을 분할 => 물체의 경계에 의해서가 아니라 depth range에 의해서 영역이 결정. 전체적으로 연결된 몇 개의 큰 blob이 생성됨.
	//		Depth image의 edge 정보로부터 boundary 추출 => 다른 두 물체가 맞닿아 있는 경우, depth image의 boundary info로부터 접촉면을 식별하기 어려움.

	// FIXME [enhance] >> too slow. speed up.
	{
		// PPP [] >>
		//	3. Depth boundary와 겹치는 superpixel의 index를 얻어옴.
		//		Depth boundary를 mask로 사용하면 쉽게 index를 추출할 수 있음.

		//filtered_superpixel_indexes.setTo(cv::Scalar::all(0));
		rgb_superpixel_mask.copyTo(filtered_superpixel_indexes, depth_boundary_image);
		cv::MatIterator_<int> itBegin = filtered_superpixel_indexes.begin<int>(), itEnd = filtered_superpixel_indexes.end<int>();
		std::sort(itBegin, itEnd);
		cv::MatIterator_<int> itEndNew = std::unique(itBegin, itEnd);
		//std::size_t count = 0;
		//for (cv::MatIterator_<int> it = itBegin; it != itEndNew; ++it, ++count)
		//	std::cout << *it << std::endl;

		// PPP [] >>
		//	4. 추출된 superpixel index들에 해당하는 superpixel 영역을 0, 그외 영역을 1로 지정.

		//filtered_superpixel_mask.setTo(cv::Scalar::all(255));
		for (cv::MatIterator_<int> it = itBegin; it != itEndNew; ++it)
			// FIXME [check] >> why is 0 contained in index list?
			if (*it)
				filtered_superpixel_mask.setTo(cv::Scalar::all(0), rgb_superpixel_mask == *it);

#if 1
		// show filtered superpixel index mask.
		cv::imshow("mask of superpixels on depth boundary", filtered_superpixel_mask);
#endif
	}

	// construct depth guided mask.
	depth_guided_mask.setTo(cv::Scalar::all(127));  // depth boundary region.
	depth_guided_mask.setTo(cv::Scalar::all(255), depth_validity_mask & filtered_superpixel_mask);  // valid depth region (foreground).
	depth_guided_mask.setTo(cv::Scalar::all(0), ~depth_validity_mask & filtered_superpixel_mask);  // invalid depth region (background).
}

void construct_depth_guided_mask_using_morphological_operation_of_depth_boundary(const cv::Mat &rgb_input_image, const cv::Mat &depth_boundary_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_mask)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	cv::Mat dilated_depth_boundary_image;
	cv::dilate(depth_boundary_image, dilated_depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if 1
		// show dilated depth boundary mask.
		cv::imshow("dilated depth boundary mask", dilated_depth_boundary_image);
#endif

	// construct depth guided mask.
	depth_guided_mask.setTo(cv::Scalar::all(127));  // depth boundary region.
	depth_guided_mask.setTo(cv::Scalar::all(255), depth_validity_mask & ~dilated_depth_boundary_image);  // valid depth region (foreground).
	depth_guided_mask.setTo(cv::Scalar::all(0), ~depth_validity_mask & ~dilated_depth_boundary_image);  // invalid depth region (background).
}

void run_grabcut_using_depth_guided_mask(const cv::Mat &rgb_input_image, const cv::Mat &depth_guided_mask)
{
	// PPP [] >>
	//	5. 추출된 superpixel index들로부터 foreground & background 영역을 추출.
	//		선택된 depth range로부터 얻은 영역을 1로, 그외 영역을 0으로 지정한 후, 추출된 superpixel index와 bit-and operation.
	//		1을 가지는 영역의 boundary를 GrabCut의 foreground seed로 사용.
	//		선택된 depth range로부터 얻은 영역을 0로, 그외 영역을 1으로 지정한 후, 추출된 superpixel index와 bit-and operation.
	//		1을 가지는 영역의 boundary를 GrabCut의 background seed로 사용.

	cv::Mat grabCut_mask(rgb_input_image.size(), CV_8UC1);
	cv::Mat grabCut_bgModel, grabCut_fgModel;

#if 1
	// GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
	//grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_BGD));
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_FGD));
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_FGD), 255 == depth_guided_mask);  // foreground.
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_BGD), 0 == depth_guided_mask);  // background.

	cv::grabCut(rgb_input_image, grabCut_mask, cv::Rect(), grabCut_bgModel, grabCut_fgModel, 1, cv::GC_INIT_WITH_MASK);
#else
	// FIXME [enhance] >> too slow. speed up.
	const cv::Rect grabCut_rect(swl::get_bounding_rect(depth_guided_mask > 0));

	cv::grabCut(rgb_input_image, grabCut_mask, grabCut_rect, grabCut_bgModel, grabCut_fgModel, 1, cv::GC_INIT_WITH_RECT);
#endif

	cv::Mat tmp_image;
#if 0
	// show foreground & background masks.
	//cv::imshow("foreground mask", 255 == depth_guided_mask);  // foreground.
	//cv::imshow("background mask", 0 == depth_guided_mask);  // background.

	// show GrabCut mask.
	grabCut_mask.convertTo(tmp_image, CV_8UC1, 255.0 / cv::GC_PR_FGD, 0.0);
	//cv::rectangle(tmp_image, grabCut_rect, cv::Scalar::all(255), 2);
	cv::imshow("GrabCut mask", tmp_image);
#endif

	cv::grabCut(rgb_input_image, grabCut_mask, cv::Rect(), grabCut_bgModel, grabCut_fgModel, 1, cv::GC_EVAL);

#if 1
	rgb_input_image.copyTo(tmp_image, cv::Mat(grabCut_mask & 1));
	cv::imshow("GrabCut result", tmp_image);
#endif
}

void run_grabcut_using_structure_tensor_mask(const cv::Mat &rgb_input_image, const cv::Mat &structure_tensor_mask)
{
	cv::Mat grabCut_mask(rgb_input_image.size(), CV_8UC1);
	cv::Mat grabCut_bgModel, grabCut_fgModel;

#if 1
	// GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
	//grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_BGD));
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_FGD));
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_FGD), 255 == structure_tensor_mask);  // foreground.
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_BGD), 0 == structure_tensor_mask);  // background.

	cv::grabCut(rgb_input_image, grabCut_mask, cv::Rect(), grabCut_bgModel, grabCut_fgModel, 1, cv::GC_INIT_WITH_MASK);
#else
	// FIXME [enhance] >> too slow. speed up.
	const cv::Rect grabCut_rect(swl::get_bounding_rect(structure_tensor_mask > 0));

	cv::grabCut(rgb_input_image, grabCut_mask, grabCut_rect, grabCut_bgModel, grabCut_fgModel, 1, cv::GC_INIT_WITH_RECT);
#endif

	cv::Mat tmp_image;
#if 0
	// show foreground & background masks.
	//cv::imshow("foreground mask", 255 == structure_tensor_mask);  // foreground.
	//cv::imshow("background mask", 0 == structure_tensor_mask);  // background.

	// show GrabCut mask.
	grabCut_mask.convertTo(tmp_image, CV_8UC1, 255.0 / cv::GC_PR_FGD, 0.0);
	//cv::rectangle(tmp_image, grabCut_rect, cv::Scalar::all(255), 2);
	cv::imshow("GrabCut mask", tmp_image);
#endif

	cv::grabCut(rgb_input_image, grabCut_mask, cv::Rect(), grabCut_bgModel, grabCut_fgModel, 1, cv::GC_EVAL);

#if 1
	rgb_input_image.copyTo(tmp_image, cv::Mat(grabCut_mask & 1));
	cv::imshow("GrabCut result", tmp_image);
#endif
}

void run_efficient_graph_based_image_segmentation(const cv::Mat &rgb_input_image, const cv::Mat &depth_input_image, const cv::Mat &depth_guided_mask, const double fx_rgb, const double fy_rgb)
{
	const float sigma = 0.5f;
	const float k = 500.0f;
	const int min_size = 50;

	int num_ccs = 0;
	cv::Mat output_image;
#if 0
	//const float lambda1 = 1.0f, lambda2 = 1.0f, lambda3 = 1.0f;
	//const float lambda1 = 0.1f, lambda2 = 0.1f, lambda3 = 0.1f;
	//const float lambda1 = 0.5f, lambda2 = 0.5f, lambda3 = 0.5f;
	const float lambda1 = 0.01f, lambda2 = 0.0f, lambda3 = 0.0f;
	//const float lambda1 = 0.0f, lambda2 = 0.0f, lambda3 = 0.0f;

	segment_image_using_efficient_graph_based_image_segmentation_algorithm(rgb_input_image, depth_input_image, depth_guided_mask, sigma, k, min_size, lambda1, lambda2, lambda3, (float)fx_rgb, (float)fy_rgb, num_ccs, output_image);

#if 1
	std::cout << "got " << num_ccs << " components" << std::endl;
	cv::imshow("result of depth-guided efficient graph based image segmentation algorithm", output_image);
#endif
#else
	segment_image_using_efficient_graph_based_image_segmentation_algorithm(rgb_input_image, sigma, k, min_size, num_ccs, output_image);

#if 1
	std::cout << "got " << num_ccs << " components" << std::endl;
	cv::imshow("result of the orignal efficient graph based image segmentation algorithm", output_image);

	//cv::Mat tmp_image(output_image.size(), output_image.type(), cv::Scalar::all(0));
	//output_image.copyTo(tmp_image, 127 == depth_guided_mask);
	//cv::imshow("boundry region of the result of the original efficient graph based image segmentation algorithm", tmp_image);
#endif
#endif
}

void segment_image_based_on_depth_guided_map()
{
	const std::size_t num_images = 4;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130530T103805.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130531T023152.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130531T023346.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130531T023359.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130530T103805.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130531T023152.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130531T023346.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130531T023359.png");

	const bool use_depth_range_filtering = false;
	std::vector<cv::Range> depth_range_list;
	{
		depth_range_list.reserve(num_images);
#if 0
		depth_range_list.push_back(cv::Range(500, 3420));
		depth_range_list.push_back(cv::Range(500, 3120));
		depth_range_list.push_back(cv::Range(500, 1700));
		depth_range_list.push_back(cv::Range(500, 1000));
#else
		const int min_depth = 100, max_depth = 3000;
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
#endif
	}

	//
	boost::scoped_ptr<swl::KinectSensor> kinect;
	double fx_rgb = 0.0, fy_rgb = 0.0;
	{
		const bool useIRtoRGB = true;
		cv::Mat K_ir, K_rgb;
		cv::Mat distCoeffs_ir, distCoeffs_rgb;
		cv::Mat R, T;

		// load the camera parameters of a Kinect sensor.
		if (useIRtoRGB)
			local::load_kinect_sensor_parameters_from_IR_to_RGB(K_ir, distCoeffs_ir, K_rgb, distCoeffs_rgb, R, T);
		else
			local::load_kinect_sensor_parameters_from_RGB_to_IR(K_rgb, distCoeffs_rgb, K_ir, distCoeffs_ir, R, T);

		fx_rgb = K_rgb.at<double>(0, 0);
		fy_rgb = K_rgb.at<double>(1, 1);

		kinect.reset(new swl::KinectSensor(useIRtoRGB, imageSize_ir, K_ir, distCoeffs_ir, imageSize_rgb, K_rgb, distCoeffs_rgb, R, T));
		kinect->initialize();
	}

	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	//
	cv::Mat rectified_rgb_image, rectified_depth_image;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_boundary_image, depth_guided_mask(imageSize_rgb, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		const cv::Mat rgb_input_image(cv::imread(rgb_input_file_list[i], CV_LOAD_IMAGE_COLOR));
		if (rgb_input_image.empty())
		{
			std::cout << "fail to load image file: " << rgb_input_file_list[i] << std::endl;
			continue;
		}
		const cv::Mat depth_input_image(cv::imread(depth_input_file_list[i], CV_LOAD_IMAGE_UNCHANGED));
		if (depth_input_image.empty())
		{
			std::cout << "fail to load image file: " << depth_input_file_list[i] << std::endl;
			continue;
		}

		const int64 start = cv::getTickCount();

		// rectify Kinect images.
		{
			kinect->rectifyImagePair(depth_input_image, rgb_input_image, rectified_depth_image, rectified_rgb_image);

#if 1
			// show rectified images
			cv::imshow("rectified RGB image", rectified_rgb_image);

			cv::minMaxLoc(rectified_depth_image, &minVal, &maxVal);
			rectified_depth_image.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("rectified depth image", tmp_image);
#endif

#if 0
			std::ostringstream strm1, strm2;
			strm1 << "../data/kinect_segmentation/rectified_image_depth_" << i << ".png";
			cv::imwrite(strm1.str(), rectified_depth_image);
			strm2 << "../data/kinect_segmentation/rectified_image_rgb_" << i << ".png";
			cv::imwrite(strm2.str(), rectified_rgb_image);
#endif
		}

		// make depth validity mask.
		{
#if __USE_RECTIFIED_IMAGE
			if (use_depth_range_filtering)
				cv::inRange(rectified_depth_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
			else
				cv::Mat(rectified_depth_image > 0).copyTo(depth_validity_mask);
#else
			if (use_depth_range_filtering)
				cv::inRange(depth_input_image, cv::Scalar::all(valid_depth_range.start), cv::Scalar::all(valid_depth_range.end), depth_validity_mask);
			else
				cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);
#endif

			cv::erode(depth_validity_mask, depth_validity_mask, selement3, cv::Point(-1, -1), 3);
			cv::dilate(depth_validity_mask, depth_validity_mask, selement3, cv::Point(-1, -1), 3);

#if 1
			// show depth validity mask.
			cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
			std::ostringstream strm;
			strm << "../data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
#endif
		}

		// construct valid depth image.
		{
			valid_depth_image.setTo(cv::Scalar::all(0));
#if __USE_RECTIFIED_IMAGE
			rectified_depth_image.copyTo(valid_depth_image, depth_validity_mask);
#else
			depth_input_image.copyTo(valid_depth_image, depth_validity_mask);
#endif

#if 0
			std::ostringstream strm;
			strm << "../data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
#endif
		}

		// extract boundary from depth image by edge detector.
		{
			cv::minMaxLoc(valid_depth_image, &minVal, &maxVal);
			valid_depth_image.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			//const double low = 1.0, high = 255.0;
			//const double alpha = (high - low) / (depth_range_list[i].end - depth_range_list[i].start), beta = low - alpha * depth_range_list[i].start;
			//valid_depth_image.convertTo(tmp_image, CV_8UC1, alpha, beta);
				
			local::canny(tmp_image, depth_boundary_image);

#if 1
			// show depth boundary image.
			cv::imshow("depth boundary by Canny", depth_boundary_image);
#endif
		}

#if 0
		// construct depth guided mask using superpixel.
		{
			const int num_segments = 2500;
			const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC
			const double seg_weight = 0.3;

			//cv::dilate(depth_boundary_image, depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if __USE_RECTIFIED_IMAGE
			swl::construct_depth_guided_mask_using_superpixel(rectified_rgb_image, depth_boundary_image, depth_validity_mask, depth_guided_mask, num_segments, seg_method, seg_weight);
#else
			swl::construct_depth_guided_mask_using_superpixel(rgb_input_image, depth_boundary_image, depth_validity_mask, depth_guided_mask, num_segments, seg_method, seg_weight);
#endif
		}
#elif 1
		// construct depth guided mask using morphological operation of depth boundary.
		{
#if __USE_RECTIFIED_IMAGE
			swl::construct_depth_guided_mask_using_morphological_operation_of_depth_boundary(rectified_rgb_image, depth_boundary_image, depth_validity_mask, depth_guided_mask);
#else
			swl::construct_depth_guided_mask_using_morphological_operation_of_depth_boundary(rgb_input_image, depth_boundary_image, depth_validity_mask, depth_guided_mask);
#endif
		}
#endif

#if 1
		// show depth guided mask.
		cv::imshow("depth guided mask", depth_guided_mask);
#endif

#if 0
		std::ostringstream strm;
		cv::cvtColor(depth_guided_mask, tmp_image, CV_GRAY2BGR);
		strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
		cv::imwrite(strm.str(), tmp_image);
#endif

#if 0
		// segment image by GrabCut algorithm.
		{
#if __USE_RECTIFIED_IMAGE
			swl::run_grabcut_using_depth_guided_mask(rectified_rgb_image, depth_guided_mask);
#else
			swl::run_grabcut_using_depth_guided_mask(rgb_input_image, depth_guided_mask);
#endif
		}
#elif 1
		// segment image by efficient graph-based image segmentation algorithm.
		{
#if __USE_RECTIFIED_IMAGE
			swl::run_efficient_graph_based_image_segmentation(rectified_rgb_image, valid_depth_image, depth_guided_mask, fx_rgb, fy_rgb);
#else
			swl::run_efficient_graph_based_image_segmentation(rgb_input_image, valid_depth_image, depth_guided_mask, fx_rgb, fy_rgb);
#endif
		}
#endif

		const int64 elapsed = cv::getTickCount() - start;
		const double freq = cv::getTickFrequency();
		const double etime = elapsed * 1000.0 / freq;
		const double fps = freq / elapsed;
		std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

void segment_foreground_based_on_depth_guided_map()
{
	const std::size_t num_images = 6;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162552.png");

	const bool use_depth_range_filtering = false;
	std::vector<cv::Range> depth_range_list;
	{
		depth_range_list.reserve(num_images);
		const int min_depth = 100, max_depth = 4000;
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
	}

	//
	boost::scoped_ptr<swl::KinectSensor> kinect;
	double fx_rgb = 0.0, fy_rgb = 0.0;
	{
		const bool useIRtoRGB = true;
		cv::Mat K_ir, K_rgb;
		cv::Mat distCoeffs_ir, distCoeffs_rgb;
		cv::Mat R, T;

		// load the camera parameters of a Kinect sensor.
		if (useIRtoRGB)
			local::load_kinect_sensor_parameters_from_IR_to_RGB(K_ir, distCoeffs_ir, K_rgb, distCoeffs_rgb, R, T);
		else
			local::load_kinect_sensor_parameters_from_RGB_to_IR(K_rgb, distCoeffs_rgb, K_ir, distCoeffs_ir, R, T);

		fx_rgb = K_rgb.at<double>(0, 0);
		fy_rgb = K_rgb.at<double>(1, 1);

		kinect.reset(new swl::KinectSensor(useIRtoRGB, imageSize_ir, K_ir, distCoeffs_ir, imageSize_rgb, K_rgb, distCoeffs_rgb, R, T));
		kinect->initialize();
	}

	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	//
	cv::Mat rectified_rgb_image, rectified_depth_image;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_boundary_image, depth_guided_mask(imageSize_rgb, CV_8UC1), depth_changing_image(imageSize_rgb, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		const cv::Mat rgb_input_image(cv::imread(rgb_input_file_list[i], CV_LOAD_IMAGE_COLOR));
		if (rgb_input_image.empty())
		{
			std::cout << "fail to load image file: " << rgb_input_file_list[i] << std::endl;
			continue;
		}
		const cv::Mat depth_input_image(cv::imread(depth_input_file_list[i], CV_LOAD_IMAGE_UNCHANGED));
		if (depth_input_image.empty())
		{
			std::cout << "fail to load image file: " << depth_input_file_list[i] << std::endl;
			continue;
		}

		const int64 start = cv::getTickCount();

		// rectify Kinect images.
		{
			kinect->rectifyImagePair(depth_input_image, rgb_input_image, rectified_depth_image, rectified_rgb_image);

#if 1
			// show rectified images
			cv::imshow("rectified RGB image", rectified_rgb_image);

			cv::minMaxLoc(rectified_depth_image, &minVal, &maxVal);
			rectified_depth_image.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("rectified depth image", tmp_image);
#endif

#if 0
			std::ostringstream strm1, strm2;
			strm1 << "../data/kinect_segmentation/rectified_image_depth_" << i << ".png";
			cv::imwrite(strm1.str(), rectified_depth_image);
			strm2 << "../data/kinect_segmentation/rectified_image_rgb_" << i << ".png";
			cv::imwrite(strm2.str(), rectified_rgb_image);
#endif
		}

		// make depth validity mask.
		{
#if __USE_RECTIFIED_IMAGE
			if (use_depth_range_filtering)
				cv::inRange(rectified_depth_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
			else
				cv::Mat(rectified_depth_image > 0).copyTo(depth_validity_mask);
#else
			if (use_depth_range_filtering)
				cv::inRange(depth_input_image, cv::Scalar::all(valid_depth_range.start), cv::Scalar::all(valid_depth_range.end), depth_validity_mask);
			else
				cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);
#endif

			cv::erode(depth_validity_mask, depth_validity_mask, selement3, cv::Point(-1, -1), 3);
			cv::dilate(depth_validity_mask, depth_validity_mask, selement3, cv::Point(-1, -1), 3);

#if 1
			// show depth validity mask.
			cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
			std::ostringstream strm;
			strm << "../data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
#endif
		}

		// construct valid depth image.
		{
			valid_depth_image.setTo(cv::Scalar::all(0));
#if __USE_RECTIFIED_IMAGE
			rectified_depth_image.copyTo(valid_depth_image, depth_validity_mask);
#else
			depth_input_image.copyTo(valid_depth_image, depth_validity_mask);
#endif

#if 0
			std::ostringstream strm;
			strm << "../data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
#endif
		}

		// extract boundary from depth image by edge detector.
		{
			cv::minMaxLoc(valid_depth_image, &minVal, &maxVal);
			valid_depth_image.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			//const double low = 1.0, high = 255.0;
			//const double alpha = (high - low) / (depth_range_list[i].end - depth_range_list[i].start), beta = low - alpha * depth_range_list[i].start;
			//valid_depth_image.convertTo(tmp_image, CV_8UC1, alpha, beta);
				
			local::canny(tmp_image, depth_boundary_image);

#if 1
			// show depth boundary image.
			cv::imshow("depth boundary by Canny", depth_boundary_image);
#endif
		}

		// compute phase distribution from neighborhood
		{
			const int radius = 2;
			depth_changing_image.setTo(cv::Scalar::all(0));  // CV_8UC1

			// TODO [implement] >>
			compute_phase_distribution_from_neighborhood(valid_depth_image, radius);

#if 1
			// show depth changing image.
			cv::imshow("depth changing image", depth_changing_image);
#endif
		}

#if 0
		// construct depth guided mask using superpixel.
		{
			const int num_segments = 2500;
			const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC
			const double seg_weight = 0.3;

			//cv::dilate(depth_boundary_image, depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if __USE_RECTIFIED_IMAGE
			swl::construct_depth_guided_mask_using_superpixel(rectified_rgb_image, depth_boundary_image, depth_validity_mask, depth_guided_mask, num_segments, seg_method, seg_weight);
#else
			swl::construct_depth_guided_mask_using_superpixel(rgb_input_image, depth_boundary_image, depth_validity_mask, depth_guided_mask, num_segments, seg_method, seg_weight);
#endif
		}
#elif 1
		// construct depth guided mask using morphological operation of depth boundary.
		{
#if __USE_RECTIFIED_IMAGE
			swl::construct_depth_guided_mask_using_morphological_operation_of_depth_boundary(rectified_rgb_image, depth_boundary_image, depth_validity_mask, depth_guided_mask);
#else
			swl::construct_depth_guided_mask_using_morphological_operation_of_depth_boundary(rgb_input_image, depth_boundary_image, depth_validity_mask, depth_guided_mask);
#endif
		}
#endif

#if 1
		// show depth guided mask.
		cv::imshow("depth guided mask", depth_guided_mask);
#endif

#if 0
		std::ostringstream strm;
		cv::cvtColor(depth_guided_mask, tmp_image, CV_GRAY2BGR);
		strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
		cv::imwrite(strm.str(), tmp_image);
#endif

#if 0
		// segment image by GrabCut algorithm.
		{
#if __USE_RECTIFIED_IMAGE
			swl::run_grabcut_using_depth_guided_mask(rectified_rgb_image, depth_guided_mask);
#else
			swl::run_grabcut_using_depth_guided_mask(rgb_input_image, depth_guided_mask);
#endif
		}
#else
		// segment image by efficient graph-based image segmentation algorithm.
		{
#if __USE_RECTIFIED_IMAGE
			swl::run_efficient_graph_based_image_segmentation(rectified_rgb_image, valid_depth_image, depth_guided_mask, fx_rgb, fy_rgb);
#else
			swl::run_efficient_graph_based_image_segmentation(rgb_input_image, valid_depth_image, depth_guided_mask, fx_rgb, fy_rgb);
#endif
		}
#endif

		const int64 elapsed = cv::getTickCount() - start;
		const double freq = cv::getTickFrequency();
		const double etime = elapsed * 1000.0 / freq;
		const double fps = freq / elapsed;
		std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

struct IncreaseHierarchyOp
{
    IncreaseHierarchyOp(const int offset)
    : offset_(offset)
    {}

    cv::Vec4i operator()(const cv::Vec4i &rhs) const
    {
        return cv::Vec4i(rhs[0] == -1 ? -1 : (rhs[0] + offset_), rhs[1] == -1 ? -1 : (rhs[1] + offset_), rhs[2] == -1 ? -1 : (rhs[2] + offset_), rhs[3] == -1 ? -1 : (rhs[3] + offset_));
    }

private:
    const int offset_;
};

void segment_foreground_based_on_structure_tensor()
{
	const std::size_t num_images = 6;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480), imageSize_mask(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list, structure_tesor_mask_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	structure_tesor_mask_file_list.reserve(num_images);
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130614T162552.png");
	structure_tesor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tesor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tesor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tesor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tesor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tesor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

	const bool use_depth_range_filtering = false;
	std::vector<cv::Range> depth_range_list;
	{
		depth_range_list.reserve(num_images);
		const int min_depth = 100, max_depth = 4000;
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
	}

	//
	boost::scoped_ptr<swl::KinectSensor> kinect;
	double fx_rgb = 0.0, fy_rgb = 0.0;
	{
		const bool useIRtoRGB = true;
		cv::Mat K_ir, K_rgb;
		cv::Mat distCoeffs_ir, distCoeffs_rgb;
		cv::Mat R, T;

		// load the camera parameters of a Kinect sensor.
		if (useIRtoRGB)
			local::load_kinect_sensor_parameters_from_IR_to_RGB(K_ir, distCoeffs_ir, K_rgb, distCoeffs_rgb, R, T);
		else
			local::load_kinect_sensor_parameters_from_RGB_to_IR(K_rgb, distCoeffs_rgb, K_ir, distCoeffs_ir, R, T);

		fx_rgb = K_rgb.at<double>(0, 0);
		fy_rgb = K_rgb.at<double>(1, 1);

		kinect.reset(new swl::KinectSensor(useIRtoRGB, imageSize_ir, K_ir, distCoeffs_ir, imageSize_rgb, K_rgb, distCoeffs_rgb, R, T));
		kinect->initialize();
	}

	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	//
	cv::Mat rectified_rgb_image, rectified_depth_image;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_imag, contour_image;
	const bool use_color_processed_structure_tensor_mask = false;
	cv::Mat processed_structure_tensor_mask(imageSize_mask, use_color_processed_structure_tensor_mask ? CV_8UC3 : CV_8UC1);
	cv::Mat grabCut_mask(imageSize_mask, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		const cv::Mat rgb_input_image(cv::imread(rgb_input_file_list[i], CV_LOAD_IMAGE_COLOR));
		if (rgb_input_image.empty())
		{
			std::cout << "fail to load image file: " << rgb_input_file_list[i] << std::endl;
			continue;
		}
		const cv::Mat depth_input_image(cv::imread(depth_input_file_list[i], CV_LOAD_IMAGE_UNCHANGED));
		if (depth_input_image.empty())
		{
			std::cout << "fail to load image file: " << depth_input_file_list[i] << std::endl;
			continue;
		}
		cv::Mat structure_tensor_mask(cv::imread(structure_tesor_mask_file_list[i], CV_LOAD_IMAGE_UNCHANGED));
		if (structure_tensor_mask.empty())
		{
			std::cout << "fail to load mask file: " << structure_tesor_mask_file_list[i] << std::endl;
			continue;
		}

		const int64 start = cv::getTickCount();

		// rectify Kinect images.
		{
			kinect->rectifyImagePair(depth_input_image, rgb_input_image, rectified_depth_image, rectified_rgb_image);

#if 1
			// show rectified images
			cv::imshow("rectified RGB image", rectified_rgb_image);

			//cv::minMaxLoc(rectified_depth_image, &minVal, &maxVal);
			//rectified_depth_image.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
			//cv::imshow("rectified depth image", tmp_image);
#endif

#if 0
			std::ostringstream strm1, strm2;
			strm1 << "../data/kinect_segmentation/rectified_image_rgb_" << i << ".png";
			cv::imwrite(strm1.str(), rectified_rgb_image);
			strm2 << "../data/kinect_segmentation/rectified_image_depth_" << i << ".png";
			cv::imwrite(strm2.str(), rectified_depth_image);
#endif
		}

		// pre-process structure tensor mask.
		{
			structure_tensor_mask = structure_tensor_mask > 0.05;  // CV_8UC1

			cv::dilate(structure_tensor_mask, structure_tensor_mask, selement3, cv::Point(-1, -1), 3);
			cv::erode(structure_tensor_mask, structure_tensor_mask, selement3, cv::Point(-1, -1), 3);

			//cv::erode(structure_tensor_mask, structure_tensor_mask, selement3, cv::Point(-1, -1), 3);
			//cv::dilate(structure_tensor_mask, structure_tensor_mask, selement3, cv::Point(-1, -1), 3);

#if 1
			cv::imshow("structure tensor mask", structure_tensor_mask);
#endif

#if 1
			std::ostringstream strm;
			strm << "../data/kinect_segmentation/structure_tensor_mask_" << i << ".png";
			cv::imwrite(strm.str(), structure_tensor_mask);
#endif
		}

		// find contours.
		const double MIN_CONTOUR_AREA = 200.0;
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		{
			structure_tensor_mask.copyTo(contour_image);

			std::vector<std::vector<cv::Point> > contours2;
			cv::findContours(contour_image, contours2, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

#if 0
			// comment this out if you do not want approximation
			for (std::vector<std::vector<cv::Point> >::iterator it = contours2.begin(); it != contours2.end(); ++it)
			{
				//if (it->empty()) continue;

				std::vector<cv::Point> approxCurve;
				cv::approxPolyDP(cv::Mat(*it), approxCurve, 3.0, true);
				contours.push_back(approxCurve);
			}
#else
			std::copy(contours2.begin(), contours2.end(), std::back_inserter(contours));
#endif

#if 0
			// find all contours.

			processed_structure_tensor_mask.setTo(cv::Scalar::all(0));

			// iterate through all the top-level contours,
			// draw each connected component with its own random color
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
			{
				if (cv::contourArea(cv::Mat(contours[idx])) < MIN_CONTOUR_AREA) continue;

				if (use_color_processed_structure_tensor_mask)
					//cv::drawContours(processed_structure_tensor_mask, contours, idx, cv::Scalar(std::rand() & 255, std::rand() & 255, std::rand() & 255), CV_FILLED, 8, hierarchy, 0, cv::Point());
					cv::drawContours(processed_structure_tensor_mask, contours, idx, cv::Scalar(std::rand() & 255, std::rand() & 255, std::rand() & 255), CV_FILLED, 8, cv::noArray(), 0, cv::Point());
				else
					//cv::drawContours(processed_structure_tensor_mask, contours, idx, cv::Scalar::all(255), CV_FILLED, 8, hierarchy, 0, cv::Point());
					cv::drawContours(processed_structure_tensor_mask, contours, idx, cv::Scalar::all(255), CV_FILLED, 8, cv::noArray(), 0, cv::Point());
			}
#elif 1
			// find a contour with max. area.

			std::vector<std::vector<cv::Point> > pointSets;
			std::vector<cv::Vec4i> hierarchy0;
			if (!hierarchy.empty())
				std::transform(hierarchy.begin(), hierarchy.end(), std::back_inserter(hierarchy0), IncreaseHierarchyOp(pointSets.size()));

			for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
				if (!it->empty()) pointSets.push_back(*it);

			if (!pointSets.empty())
			{
#if 0
				cv::drawContours(img, pointSets, -1, CV_RGB(255, 0, 0), 1, 8, hierarchy0, maxLevel, cv::Point());
#elif 0
				const size_t num = pointSets.size();
				for (size_t k = 0; k < num; ++k)
				{
					if (cv::contourArea(cv::Mat(pointSets[k])) < MIN_AREA) continue;
					const int r = rand() % 256, g = rand() % 256, b = rand() % 256;
					cv::drawContours(img, pointSets, k, CV_RGB(r, g, b), 1, 8, hierarchy0, maxLevel, cv::Point());
				}
#else
				double maxArea = 0.0;
				size_t maxAreaIdx = -1, idx = 0;
				for (std::vector<std::vector<cv::Point> >::iterator it = pointSets.begin(); it != pointSets.end(); ++it, ++idx)
				{
					const double area = cv::contourArea(cv::Mat(*it));
					if (area > maxArea)
					{
						maxArea = area;
						maxAreaIdx = idx;
					}
				}

				processed_structure_tensor_mask.setTo(cv::Scalar::all(0));
				if ((size_t)-1 != maxAreaIdx)
					//cv::drawContours(processed_structure_tensor_mask, pointSets, maxAreaIdx, cv::Scalar::all(255), CV_FILLED, 8, hierarchy0, 0, cv::Point());
					//cv::drawContours(processed_structure_tensor_mask, pointSets, maxAreaIdx, cv::Scalar::all(255), CV_FILLED, 8, hierarchy0, maxLevel, cv::Point());
					cv::drawContours(processed_structure_tensor_mask, pointSets, maxAreaIdx, cv::Scalar::all(255), CV_FILLED, 8, cv::noArray(), 0, cv::Point());
#endif
			}
#endif

			// post-process structure tensor mask.
			cv::morphologyEx(processed_structure_tensor_mask, processed_structure_tensor_mask, cv::MORPH_CLOSE, selement5, cv::Point(-1, -1), 3);
			//cv::imshow("post-processed structure tensor mask 1", processed_structure_tensor_mask);

			cv::morphologyEx(processed_structure_tensor_mask, processed_structure_tensor_mask, cv::MORPH_OPEN, selement5, cv::Point(-1, -1), 3);
			//cv::imshow("post-processed structure tensor mask 2", processed_structure_tensor_mask);

			//cv::erode(processed_structure_tensor_mask, processed_structure_tensor_mask, selement3, cv::Point(-1, -1), 1);
			cv::dilate(processed_structure_tensor_mask, processed_structure_tensor_mask, selement3, cv::Point(-1, -1), 1);
			//cv::imshow("post-processed structure tensor mask 3", processed_structure_tensor_mask);
		}

#if 1
		// show post-processed structure tensor mask.
		{
			cv::imshow("post-processed structure tensor mask", processed_structure_tensor_mask);

			tmp_image = cv::Mat::zeros(rectified_rgb_image.size(), rectified_rgb_image.type());
			rectified_rgb_image.copyTo(tmp_image, processed_structure_tensor_mask);
			cv::imshow("RGB image extracted by post-processed structure tensor mask", tmp_image);

#if 1
			std::ostringstream strm1, strm2;
			strm1 << "../data/kinect_segmentation/processed_structure_tensor_mask_" << i << ".png";
			cv::imwrite(strm1.str(), processed_structure_tensor_mask);
			strm2 << "../data/kinect_segmentation/masked_rgb_image_" << i << ".png";
			cv::imwrite(strm2.str(), tmp_image);
#endif
		}
#endif

#if 0
		// segment foreground using Snake.
		{
			//const std::size_t NUM_SNAKE_POINTS = 50;
			const std::size_t NUM_SNAKE_POINTS = 0;

			cv::Mat gray_image;
#if __USE_RECTIFIED_IMAGE
			cv::cvtColor(rectified_rgb_image, gray_image, CV_BGR2GRAY);
#else
			cv::cvtColor(rgb_input_image, gray_image, CV_BGR2GRAY);
#endif

			std::vector<std::vector<cv::Point> > snake_contours;
			snake_contours.reserve(contours.size());
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
			{
				if (cv::contourArea(cv::Mat(contours[idx])) < MIN_CONTOUR_AREA) continue;

				std::vector<cv::Point> snake_contour;
				swl::fit_contour_by_snake(gray_image, contours[idx], NUM_SNAKE_POINTS, snake_contour);
				snake_contours.push_back(snake_contour);
			}

			// show results of fitting using Snake.
#if __USE_RECTIFIED_IMAGE
			rectified_rgb_image.copyTo(tmp_image);
#else
			rgb_input_image.copyTo(tmp_image);
#endif

			//cv::drawContours(tmp_image, snake_contours, -1, CV_RGB(255, 0, 0), CV_FILLED, 8, cv::noArray(), 0, cv::Point());
			//cv::drawContours(tmp_image, snake_contours, -1, CV_RGB(255, 0, 0), 1, 8, cv::noArray(), 0, cv::Point());
			int idx = 0;
			for (std::vector<std::vector<cv::Point> >::const_iterator cit = snake_contours.begin(); cit != snake_contours.end(); ++cit, ++idx)
			{
				if (cit->empty() || cv::contourArea(cv::Mat(*cit)) < MIN_CONTOUR_AREA) continue;

				const cv::Scalar color(std::rand() & 255, std::rand() & 255, std::rand() & 255);
				cv::drawContours(tmp_image, snake_contours, idx, color, CV_FILLED, 8, cv::noArray(), 0, cv::Point());
			}

			cv::imshow("results of fitting using Snake", tmp_image);
		}
#elif 0
		// segment image by GrabCut algorithm.
		{
			// GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD

			cv::Mat eroded_mask, dilated_mask;
			cv::erode(processed_structure_tensor_mask, eroded_mask, selement3, cv::Point(-1, -1), 1);
			cv::dilate(processed_structure_tensor_mask, dilated_mask, selement5, cv::Point(-1, -1), 5);

			grabCut_mask.setTo(cv::Scalar::all(cv::GC_BGD));
			grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_BGD), dilated_mask);
			//grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_FGD), dilated_mask);
			//grabCut_mask.setTo(cv::Scalar::all(cv::GC_FGD), eroded_mask);
			grabCut_mask.setTo(cv::Scalar::all(cv::GC_FGD), processed_structure_tensor_mask);

#if 1
			// show GrabCut masks.
			grabCut_mask.convertTo(tmp_image, CV_8UC1, 255.0 / cv::GC_PR_FGD, 0.0);
			cv::imshow("GrabCut mask", tmp_image);
			eroded_mask.convertTo(tmp_image, CV_8UC1, 255.0 / cv::GC_PR_FGD, 0.0);
			cv::imshow("eroded GrabCut mask", tmp_image);
#endif

#if __USE_RECTIFIED_IMAGE
			swl::run_grabcut_using_structure_tensor_mask(rectified_rgb_image, grabCut_mask);
#else
			swl::run_grabcut_using_structure_tensor_mask(rgb_input_image, grabCut_mask);
#endif
		}
#endif

		const int64 elapsed = cv::getTickCount() - start;
		const double freq = cv::getTickFrequency();
		const double etime = elapsed * 1000.0 / freq;
		const double fps = freq / elapsed;
		std::cout << std::setprecision(4) << "elapsed time: " << etime <<  ", FPS: " << fps << std::endl;

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace swl

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		//swl::segment_image_based_on_depth_guided_map();
		//swl::segment_foreground_based_on_depth_guided_map();
		swl::segment_foreground_based_on_structure_tensor();

#if 0
		// for testing.
		{
			const unsigned short data[] = {
				1200, 300, 1000, 1500, 600,
				910, 1120, 500, 720, 2000,
				2700, 210, 1000, 1620, 1000,
				1510, 690, 1330, 1230, 470,
				350, 1170, 900, 820, 1380,
			};
			cv::Mat mat(5, 5, CV_16UC1, (void *)data);
			std::cout << "input mat = " << mat << std::endl;
			cv::Mat mat2(mat.size(), CV_8UC1, cv::Scalar::all(0));  // CV_8UC1
			const int radius = 2;
			swl::compute_phase_distribution_from_neighborhood(mat, radius);
		}
#endif
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught: " << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;
		retval = EXIT_FAILURE;
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
