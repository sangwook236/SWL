#include "swl/machine_vision/KinectSensor.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/math/constants/constants.hpp>
#include <algorithm>
#include <cassert>


namespace {
namespace local {

// REF [file] >> load_kinect_sensor_parameters_from_IR_to_RGB() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void load_kinect_sensor_parameters_from_IR_to_RGB(
	cv::Mat &K_ir, cv::Mat &distCoeffs_ir, cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb,
	cv::Mat &R_ir_to_rgb, cv::Mat &T_ir_to_rgb
)
{
	// REF [site] >> Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	// NOTICE [caution] >>
	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab in OpenCV,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// IR (left) to RGB (right).
#if 1
	// The 5th distortion parameter, kc(5) is activated.

	const double fc_ir[] = { 5.865281297534211e+02, 5.866623900166177e+02 };  // [pixel].
	const double cc_ir[] = { 3.371860463542209e+02, 2.485298169373497e+02 };  // [pixel].
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.227084070414958e-01, 5.027511830344261e-01, -2.562850607972214e-03, 6.916249031489476e-03, -5.507709925923052e-01 };  // 5x1 vector.
	const double kc_ir[] = { -1.227084070414958e-01, 5.027511830344261e-01, -2.562850607972214e-03, 6.916249031489476e-03, -5.507709925923052e-01, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double fc_rgb[] = { 5.248648751941851e+02, 5.268281060449414e+02 };  // [pixel].
	const double cc_rgb[] = { 3.267484107269922e+02, 2.618261807606497e+02 };  // [pixel].
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.796770514235670e-01, -1.112507253647945e+00, 9.265501548915561e-04, 2.428229310663184e-03, 1.744019737212440e+00 };  // 5x1 vector.
	const double kc_rgb[] = { 2.796770514235670e-01, -1.112507253647945e+00, 9.265501548915561e-04, 2.428229310663184e-03, 1.744019737212440e+00, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double rotVec[] = { -1.936270295074452e-03, 1.331596538715070e-02, 3.404073398703758e-03 };
	const double transVec[] = { 2.515260082139980e+01, 4.059127243511693e+00, -5.588303932036697e+00 };  // [mm].
#else
	// The 5th distortion parameter, kc(5) is deactivated.

	const double fc_ir[] = { 5.864902565580264e+02, 5.867305900503998e+02 };  // [pixel].
	const double cc_ir[] = { 3.376088045224677e+02, 2.480083390372575e+02 };  // [pixel].
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.123867977947529e-01, 3.552017514491446e-01, -2.823972305243438e-03, 7.246763414437084e-03, 0.0 };  // 5x1 vector.
	const double kc_ir[] = { -1.123867977947529e-01, 3.552017514491446e-01, -2.823972305243438e-03, 7.246763414437084e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double fc_rgb[] = { 5.256215953836251e+02, 5.278165866956751e+02 };  // [pixel].
	const double cc_rgb[] = { 3.260532981578608e+02, 2.630788286947369e+02 };  // [pixel].
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.394862387380747e-01, -5.840355691714197e-01, 2.567740590187774e-03, 2.044179978023951e-03, 0.0 };  // 5x1 vector.
	const double kc_rgb[] = { 2.394862387380747e-01, -5.840355691714197e-01, 2.567740590187774e-03, 2.044179978023951e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double rotVec[] = { 1.121432126402549e-03, 1.535221550916760e-02, 3.701648572107407e-03 };
	const double transVec[] = { 2.512732389978993e+01, 3.724869927389498e+00, -4.534758982979088e+00 };  // [mm].
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

// REF [file] >> load_kinect_sensor_parameters_from_RGB_to_IR() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_image_rectification.cpp
void load_kinect_sensor_parameters_from_RGB_to_IR(
	cv::Mat &K_rgb, cv::Mat &distCoeffs_rgb, cv::Mat &K_ir, cv::Mat &distCoeffs_ir,
	cv::Mat &R_rgb_to_ir, cv::Mat &T_rgb_to_ir
)
{
	// REF [site] >> Camera Calibration Toolbox for Matlab: http://www.vision.caltech.edu/bouguetj/calib_doc/
	//	http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

	// NOTICE [caution] >>
	//	In order to use the calibration results from Camera Calibration Toolbox for Matlab in OpenCV,
	//	a parameter for radial distrtortion, kc(5) has to be active, est_dist(5) = 1.

	// RGB (left) to IR (right).
#if 1
	// The 5th distortion parameter, kc(5) is activated.

	const double fc_rgb[] = { 5.248648079874888e+02, 5.268280486062615e+02 };  // [pixel].
	const double cc_rgb[] = { 3.267487100838014e+02, 2.618261169946102e+02 };  // [pixel].
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.796764337988712e-01, -1.112497355183840e+00, 9.264749543097661e-04, 2.428507887293728e-03, 1.743975665436613e+00 };  // 5x1 vector.
	const double kc_rgb[] = { 2.796764337988712e-01, -1.112497355183840e+00, 9.264749543097661e-04, 2.428507887293728e-03, 1.743975665436613e+00, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double fc_ir[] = { 5.865282023957649e+02, 5.866624209441105e+02 };  // [pixel].
	const double cc_ir[] = { 3.371875014947813e+02, 2.485295493095561e+02 };  // [pixel].
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.227176734054719e-01, 5.028746725848668e-01, -2.563029340202278e-03, 6.916996280663117e-03, -5.512162545452755e-01 };  // 5x1 vector.
	const double kc_ir[] = { -1.227176734054719e-01, 5.028746725848668e-01, -2.563029340202278e-03, 6.916996280663117e-03, -5.512162545452755e-01, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double rotVec[] = { 1.935939237060295e-03, -1.331788958930441e-02, -3.404128236480992e-03 };
	const double transVec[] = { -2.515262012891160e+01, -4.059118899373607e+00, 5.588237589014362e+00 };  // [mm].
#else
	// The 5th distortion parameter, kc(5) is deactivated.

	const double fc_rgb[] = { 5.256217798767822e+02, 5.278167798992870e+02 };  // [pixel].
	const double cc_rgb[] = { 3.260534767468189e+02, 2.630800669346188e+02 };  // [pixel].
	const double alpha_c_rgb = 0.0;
	//const double kc_rgb[] = { 2.394861400525463e-01, -5.840298777969020e-01, 2.568959896208732e-03, 2.044336479083819e-03, 0.0 };  // 5x1 vector.
	const double kc_rgb[] = { 2.394861400525463e-01, -5.840298777969020e-01, 2.568959896208732e-03, 2.044336479083819e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double fc_ir[] = { 5.864904832545356e+02, 5.867308191567271e+02 };  // [pixel].
	const double cc_ir[] = { 3.376079004969836e+02, 2.480098376453992e+02 };  // [pixel].
	const double alpha_c_ir = 0.0;
	//const double kc_ir[] = { -1.123902857373373e-01, 3.552211727724343e-01, -2.823183218548772e-03, 7.246270574438420e-03, 0.0 };  // 5x1 vector.
	const double kc_ir[] = { -1.123902857373373e-01, 3.552211727724343e-01, -2.823183218548772e-03, 7.246270574438420e-03, 0.0, 0.0, 0.0, 0.0 };  // 8x1 vector.

	const double rotVec[] = { -1.121214964017936e-03, -1.535031632771925e-02, -3.701579055761772e-03 };
	const double transVec[] = { -2.512730902761022e+01, -3.724884753207001e+00, 4.534776794502955e+00 };  // [mm].
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

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * Parameters:
 * 		im    Binary image with range = [0,1].
 * 		iter  0=even, 1=odd.
 */
void thinningZhangSuenIteration(cv::Mat& img, int iter)
{
	CV_Assert(1 == img.channels());
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	uchar *nw, *no, *ne;  // North (pAbove).
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;  // South (pBelow).

	uchar *pDst;

	// Initialize row pointers.
	uchar *pAbove = NULL;
	uchar *pCurr  = img.ptr<uchar>(0);
	uchar *pBelow = img.ptr<uchar>(1);

	int x, y;
	for (y = 1; y < img.rows - 1; ++y)
	{
		// Shift the rows up by one.
		pAbove = pCurr;
		pCurr  = pBelow;
		pBelow = img.ptr<uchar>(y+1);

		pDst = marker.ptr<uchar>(y);

		// Initialize col pointers.
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x)
		{
			// Shift col pointers left by one (scan left to right).
			nw = no;
			no = ne;
			ne = &(pAbove[x+1]);
			we = me;
			me = ea;
			ea = &(pCurr[x+1]);
			sw = so;
			so = se;
			se = &(pBelow[x+1]);

			const int A  = (0 == *no && 1 == *ne) + (0 == *ne && 1 == *ea) + 
				(0 == *ea && 1 == *se) + (0 == *se && 1 == *so) + 
				(0 == *so && 1 == *sw) + (0 == *sw && 1 == *we) +
				(0 == *we && 1 == *nw) + (0 == *nw && 1 == *no);
			const int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			const int m1 = 0 == iter ? (*no * *ea * *so) : (*no * *ea * *we);
			const int m2 = 0 == iter ? (*ea * *so * *we) : (*no * *so * *we);

			if (1 == A && (B >= 2 && B <= 6) && 0 == m1 && 0 == m2)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}

/**
* Perform one thinning iteration.
* Normally you wouldn't call this function directly from your code.
*
* @param  im    Binary image with range = 0-1.
* @param  iter  0=even, 1=odd.
*/
void thinningGuoHallIteration(cv::Mat &im, const int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

	for (int i = 1; i < im.rows; ++i)
	{
		for (int j = 1; j < im.cols; ++j)
		{
			const uchar &p2 = im.at<uchar>(i-1, j);
			const uchar &p3 = im.at<uchar>(i-1, j+1);
			const uchar &p4 = im.at<uchar>(i, j+1);
			const uchar &p5 = im.at<uchar>(i+1, j+1);
			const uchar &p6 = im.at<uchar>(i+1, j);
			const uchar &p7 = im.at<uchar>(i+1, j-1);
			const uchar &p8 = im.at<uchar>(i, j-1); 
			const uchar &p9 = im.at<uchar>(i-1, j-1);

			const int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
			const int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
			const int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
			const int N  = N1 < N2 ? N1 : N2;
			const int m  = 0 == iter ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

			if (1 == C && (N >= 2 && N <= 3) && 0 == m)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}

}  // namespace local
}  // unnamed namespace

namespace swl {
	
cv::Rect get_bounding_rect(const cv::Mat &img)
{
	std::vector<cv::Point> pts;
	pts.reserve(img.rows * img.cols);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			if (!img.at<unsigned char>(i, j))
				pts.push_back(cv::Point(j, i));

	return cv::boundingRect(pts);
}

void compute_phase_distribution_from_neighborhood(const cv::Mat &depth_map, const int radius, cv::Mat &depth_changing_mask)
{
	const int width = depth_map.cols;
	const int height = depth_map.rows;

	unsigned short dep0, dep;
	int lowerx, upperx, lowery, uppery, lowery2, uppery2;
	int i, j;

	const int num_pixels = (2*radius + 1) * (2*radius + 1) - 1;
	int num;

	// FIXME [enhance] >> Speed up.
	//cv::Mat phase(height, width, CV_32FC1, cv::Scalar::all(0)), mag(height, width, CV_32FC1, cv::Scalar::all(0));
	float *phases = new float [num_pixels], *mags = new float [num_pixels];
	float *ptr_phase, *ptr_mag;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			dep0 = depth_map.at<unsigned short>(y, x);
			if (0 == dep0) continue;  // Invalid depth.

			memset(phases, 0, num_pixels * sizeof(float));
			memset(mags, 0, num_pixels * sizeof(float));
			ptr_phase = phases;
			ptr_mag = mags;

			num = 0;
			for (int r = 1; r <= radius; ++r)
			{
				lowerx = std::max(0, x-r);
				upperx = std::min(x+r, width-1);
				lowery = std::max(0, y-r);
				uppery = std::min(y+r, height-1);
				lowery2 = std::max(0, y-r+1);
				uppery2 = std::min(y+r-1, height-1);

				if (y - r == lowery)
				{
					// Upper horizontal pixels (rightward).
					for (i = lowerx; i <= upperx; ++i)
					{
						dep = depth_map.at<unsigned short>(lowery, i);
						// FIXME [check] >> Does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // Invalid depth.
						if (0 == dep || dep0 == dep) continue;  // Invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - lowery), float(i - x)) : std::atan2(float(lowery - y), float(x - i));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(lowery - y), float(i - x)) : std::atan2(float(y - lowery), float(x - i));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
				if (x + r == upperx)
				{
					// Right vertical pixels (downward).
					for (j = lowery2; j <= uppery2; ++j)
					{
						dep = depth_map.at<unsigned short>(j, upperx);
						// FIXME [check] >> Does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // Invalid depth.
						if (0 == dep || dep0 == dep) continue;  // Invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - j), float(upperx - x)) : std::atan2(float(j - y), float(x - upperx));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(j - y), float(upperx - x)) : std::atan2(float(y - j), float(x - upperx));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
				if (y + r == uppery)
				{
					// Lower horizontal pixels (leftward).
					for (i = upperx; i >= lowerx; --i)
					{
						dep = depth_map.at<unsigned short>(uppery, i);
						// FIXME [check] >> Does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // Invalid depth.
						if (0 == dep || dep0 == dep) continue;  // Invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - uppery), float(i - x)) : std::atan2(float(uppery - y), float(x - i));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(uppery - y), float(i - x)) : std::atan2(float(y - uppery), float(x - i));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
				if (x - r == lowerx)
				{
					// left vertical pixels (upward)
					for (j = uppery2; j >= lowery2; --j)
					{
						dep = depth_map.at<unsigned short>(j, lowerx);
						// FIXME [check] >> Does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // Invalid depth.
						if (0 == dep || dep0 == dep) continue;  // Invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - j), float(lowerx - x)) : std::atan2(float(j - y), float(x - lowerx));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(j - y), float(lowerx - x)) : std::atan2(float(y - j), float(x - lowerx));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
			}

#if 0
			// For checking.
			std::cout << "Phases (" << y << ", " << x << ") = " << std::endl;
			for (int ii = 0; ii < num_pixels; ++ii)
				std::cout << (phases[ii] * 180.0f / boost::math::constants::pi<float>()) << ", ";
			std::cout << std::endl;
			std::cout << "Magnitude (" << y << ", " << x << ") = " << std::endl;
			for (int ii = 0; ii < num_pixels; ++ii)
				std::cout << mags[ii] << ", ";
			std::cout << std::endl;
			std::cout << "Num (" << y << ", " << x << ") = " << num << std::endl;
#endif
		}
	}

	// FIXME [implemented] >>
	//depth_changing_mask

	delete [] phases;
	delete [] mags;
}

#if 0
// REF [file] >> snake() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void snake(IplImage *srcImage, IplImage *grayImage)
{
	const int NUMBER_OF_SNAKE_POINTS = 50;
	const int threshold = 90;

	float alpha = 3.0f;
	float beta = 5.0f;
	float gamma = 2.0f;
	const int use_gradient = 1;
	const CvSize win = cvSize(21, 21);
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	IplImage *img = cvCloneImage(grayImage);

	{
		IplImage *tmp_img = cvCloneImage(grayImage);

		// Make a average filtering.
		cvSmooth(tmp_img, img, CV_BLUR, 31, 15);
		//iplBlur(tmp_img, img, 31, 31, 15, 15);  // Don't use IPL.

		// Do a threshold.
		cvThreshold(img, tmp_img, threshold, 255, CV_THRESH_BINARY);
		//iplThreshold(img, tmp_img, threshold);  // Don't use IPL.

		// Expand the thresholded image of ones - smoothing the edge.
		// Move start position of snake out since there are no balloon force.
		cvDilate(tmp_img, img, NULL, 3);

		cvReleaseImage(&tmp_img);
	}

	// Find the contours.
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *contour = NULL;
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// Run through the found coutours.
	std::vector<CvPoint> points(NUMBER_OF_SNAKE_POINTS, cvPoint(0, 0));
	while (contour)
	{
		if (NUMBER_OF_SNAKE_POINTS <= contour->total)
		{
			//memset(points, 0, NUMBER_OF_SNAKE_POINTS * sizeof(CvPoint));

			cvSmooth(grayImage, img, CV_BLUR, 7, 3);
			//iplBlur(grayImage, img, 7, 7, 3, 3);  // Don't use IPL.

#if 0
			std::vecto<CvPoint> pts(contour->total);
			cvCvtSeqToArray(contour, &pts[0], CV_WHOLE_SEQ);  // Copy the contour to an array.

			// number of jumps between the desired points (downsample only!).
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				points[i].x = pts[int(i * stride)].x;
				points[i].y = pts[int(i * stride)].y;
			}
#else
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				CvPoint *pt = CV_GET_SEQ_ELEM(CvPoint, contour, i * stride);
				points[i].x = pt->x;
				points[i].y = pt->y;
			}
#endif

			// Iterate snake.
			cvSnakeImage(img, &points[0], NUMBER_OF_SNAKE_POINTS, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);

			// Draw snake on image.
			CvPoint *points_ptr = (CvPoint *)&points[0];
			cvPolyLine(srcImage, (CvPoint **)points_ptr, &NUMBER_OF_SNAKE_POINTS, 1, 1, CV_RGB(255, 0, 0), 3, 8, 0);
		}

		// Get next contours.
		contour = contour->h_next;
	}

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
}

// REF [file] >> fit_contour_by_snake() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, const float alpha, const float beta, const float gamma, const bool use_gradient, const CvSize &win, std::vector<cv::Point> &snake_contour)
{
	snake_contour.clear();
	if (contour.empty()) return;

/*
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> hierarchy;
	{
		const int threshold = 90;

		cv::Mat binary_img;

		// Make a average filtering.
		cv::blur(gray_img, binary_img, cv::Size(31, 15));

		// Thresholding
		cv::threshold(binary_img, binary_img, threshold, 255, cv::THRESH_BINARY);

		// Expand the thressholded image of ones - smoothing the edge.
		// Move start position of snake out since there are no ballon force.
		{
			const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
			cv::dilate(binary_img, binary_img, selement, cv::Point(-1, -1), 3);
		}

		cv::findContours(binary_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
*/

	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	// Run through the found coutours.
	const size_t &numPts = contour.size();
	const size_t numSnakePts = 0 == numSnakePoints ? numPts : numSnakePoints;
	if (numPts >= numSnakePts)
	{
		std::vector<CvPoint> points(numSnakePts, cvPoint(0, 0));

		cv::Mat blurred_img;
		cv::blur(gray_img, blurred_img, cv::Size(7, 3));

		const int stride = int(numPts / numSnakePts);
		for (size_t i = 0; i < numSnakePts; ++i)
		{
			const cv::Point &pt = contour[i * stride];
			points[i] = cvPoint(pt.x, pt.y);
		}

		// Iterate snake.
#if defined(__GNUC__)
        IplImage blurred_img_ipl = (IplImage)blurred_img;
		cvSnakeImage(&blurred_img_ipl, &points[0], numSnakePts, (float *)&alpha, (float *)&beta, (float *)&gamma, CV_VALUE, win, term_criteria, use_gradient ? 1 : 0);
#else
		cvSnakeImage(&(IplImage)blurred_img, &points[0], numSnakePts, (float *)&alpha, (float *)&beta, (float *)&gamma, CV_VALUE, win, term_criteria, use_gradient ? 1 : 0);
#endif

		snake_contour.assign(points.begin(), points.end());
	}
}
#endif

// REF [file] >> zhang_suen_thinning_algorithm() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_skeletonization_and_thinning.cpp
void zhang_suen_thinning_algorithm(const cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	dst /= 255;  // Convert to binary image.

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do
	{
		local::thinningZhangSuenIteration(dst, 0);
		local::thinningZhangSuenIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}

// REF [file] >> guo_hall_thinning_algorithm() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_skeletonization_and_thinning.cpp
void guo_hall_thinning_algorithm(cv::Mat &im)
{
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	do
	{
		local::thinningGuoHallIteration(im, 0);
		local::thinningGuoHallIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;
}

// REF [file] >> simple_convex_hull() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
bool simple_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int pixVal, std::vector<cv::Point> &convexHull)
{
	const cv::Mat &roi_img = roi.width == 0 || roi.height == 0 ? img : img(roi);

	std::vector<cv::Point> points;
	points.reserve(roi_img.rows * roi_img.cols);
	for (int r = 0; r < roi_img.rows; ++r)
		for (int c = 0; c < roi_img.cols; ++c)
		{
			if (roi_img.at<unsigned char>(r, c) == pixVal)
				points.push_back(cv::Point(roi.x + c, roi.y + r));
		}
	if (points.empty()) return false;

	cv::convexHull(cv::Mat(points), convexHull, false);
	if (convexHull.empty()) return false;

#if 1
    // Comment this out if you do not want approximation.
	cv::approxPolyDP(convexHull, convexHull, 3.0, true);
#endif

	return true;
}

void smooth_image(const cv::Mat &in, cv::Mat &out)
{
#if 0
	// METHOD #1: down-scale and up-scale the image to filter out the noise.

	{
		cv::Mat tmp;
		cv::pyrDown(in, tmp);
		cv::pyrUp(tmp, out);
	}
#elif 0
	// METHOD #2: Gaussian filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int kernelSize = 3;
		const double sigma = 0;
		cv::GaussianBlur(in, out, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
	}
#elif 1
	// METHOD #3: box filtering.

	{
		// FIXME [adjust] >> Adjust parameters.
		const int ddepth = -1;  // The output image depth. -1 to use src.depth().
		const int kernelSize = 5;
		const bool normalize = true;
		cv::boxFilter(in, out, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
		//cv::blur(in, out, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
	}
#elif 0
	// METHOD #4: bilateral filtering.

	{
		// FIXME [adjust] >> Adjust parameters.
		const int diameter = -1;  // Diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
		const double sigmaColor = 3.0;  // For range filter.
		const double sigmaSpace = 50.0;  // For space filter.
		cv::bilateralFilter(in, out, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
	}
#else
	// METHOD #5: no filtering.

	out = in;
#endif
}

// REF [file] >> canny() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_edge_detection.cpp
void canny(const cv::Mat &gray, const int lowerEdgeThreshold, const int upperEdgeThreshold, const bool useL2, cv::Mat &edge)
{
	// Smoothing.
	smooth_image(gray, edge);

	// Run the edge detector on grayscale.
	cv::Canny(edge, edge, lowerEdgeThreshold, upperEdgeThreshold, 3, useL2);
}

bool load_kinect_images(const std::string &rgb_input_filename, const std::string &depth_input_filename, const bool useRectifiedImages, cv::Mat &rgb_input_image, cv::Mat &depth_input_image, double *fx_rgb, double *fy_rgb)
{
	const cv::Mat rgb_input_image2(cv::imread(rgb_input_filename, CV_LOAD_IMAGE_COLOR));
	if (rgb_input_image2.empty())
	{
		std::cout << "fail to load image file: " << rgb_input_filename << std::endl;
		return false;
	}
	const cv::Mat depth_input_image2(cv::imread(depth_input_filename, CV_LOAD_IMAGE_UNCHANGED));  // CV_16UC1
	if (depth_input_image2.empty())
	{
		std::cout << "fail to load image file: " << depth_input_filename << std::endl;
		return false;
	}

	// Rectify Kinect images.
	if (useRectifiedImages)
	{
		boost::scoped_ptr<swl::KinectSensor> kinect;
		{
			const bool useIRtoRGB = true;
			cv::Mat K_ir, K_rgb;
			cv::Mat distCoeffs_ir, distCoeffs_rgb;
			cv::Mat R, T;

			// Load the camera parameters of a Kinect sensor.
			if (useIRtoRGB)
				local::load_kinect_sensor_parameters_from_IR_to_RGB(K_ir, distCoeffs_ir, K_rgb, distCoeffs_rgb, R, T);
			else
				local::load_kinect_sensor_parameters_from_RGB_to_IR(K_rgb, distCoeffs_rgb, K_ir, distCoeffs_ir, R, T);

			if (fx_rgb) *fx_rgb = K_rgb.at<double>(0, 0);
			if (fy_rgb) *fy_rgb = K_rgb.at<double>(1, 1);

			kinect.reset(new swl::KinectSensor(useIRtoRGB, depth_input_image2.size(), K_ir, distCoeffs_ir, rgb_input_image2.size(), K_rgb, distCoeffs_rgb, R, T));
			kinect->initialize();
		}

		kinect->rectifyImagePair(depth_input_image2, rgb_input_image2, depth_input_image, rgb_input_image);
	}
	else
	{
		rgb_input_image = rgb_input_image2;
		depth_input_image = depth_input_image2;

		// FIXME [correct] >>
		if (fx_rgb) *fx_rgb = 1.0;
		if (fy_rgb) *fy_rgb = 1.0;
	}

#if 1
	{
		// Show input images.
		cv::imshow("Input RGB image", rgb_input_image);

		cv::Mat tmp_image;
		double minVal, maxVal;
		cv::minMaxLoc(depth_input_image, &minVal, &maxVal);
		depth_input_image.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
		cv::imshow("Input depth image", tmp_image);
	}
#endif

#if 0
	{
		std::ostringstream strm1, strm2;
		strm1 << "./data/kinect_segmentation/rectified_image_depth_" << i << ".png";
		cv::imwrite(strm1.str(), rectified_depth_image);
		strm2 << "./data/kinect_segmentation/rectified_image_rgb_" << i << ".png";
		cv::imwrite(strm2.str(), rectified_rgb_image);
	}
#endif

	return true;
}

bool load_structure_tensor_mask(const std::string &filename, cv::Mat &structure_tensor_mask)
{
	structure_tensor_mask = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	if (structure_tensor_mask.empty())
	{
		std::cout << "Structure tensor mask file not found: " << filename << std::endl;
		return false;
	}

	return true;
}

void construct_valid_depth_image(const cv::Mat &depth_input_image, cv::Mat &depth_validity_mask, cv::Mat &valid_depth_image)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	// Make depth validity mask.
	{
		cv::erode(depth_validity_mask, depth_validity_mask, selement3, cv::Point(-1, -1), 3);
		cv::dilate(depth_validity_mask, depth_validity_mask, selement3, cv::Point(-1, -1), 3);
	}

	// Construct valid depth image.
	{
		valid_depth_image.setTo(cv::Scalar::all(0));
		depth_input_image.copyTo(valid_depth_image, depth_validity_mask);
	}
}

// REF [file] >> normalize_histogram() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void normalize_histogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	//cv::Mat tmp(hist);
	//tmp.convertTo(hist, -1, factor / sums[0], 0.0);
	hist *= factor / sums[0];
#endif
}

// REF [file] >> structure_tensor_2d() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_structure_tensor.cpp
void structure_tensor_2d(const cv::Mat &img, const double deriv_sigma, const double blur_sigma, cv::Mat &eval1, cv::Mat &eval2, cv::Mat &evec1, cv::Mat &evec2)
{
	const double sigma2 = deriv_sigma * deriv_sigma;
	const double _2sigma2 = 2.0 * sigma2;
	const double sigma3 = sigma2 * deriv_sigma;
	const double den = std::sqrt(2.0 * boost::math::constants::pi<double>()) * sigma3;

	const int deriv_kernel_size = 2 * (int)std::ceil(deriv_sigma) + 1;
	cv::Mat kernelX(1, deriv_kernel_size, CV_64FC1), kernelY(deriv_kernel_size, 1, CV_64FC1);

	// Construct derivative kernels.
	for (int i = 0, k = -deriv_kernel_size/2; k <= deriv_kernel_size/2; ++i, ++k)
	{
		const double val = k * std::exp(-k*k / _2sigma2) / den;
		kernelX.at<double>(0, i) = val;
		kernelY.at<double>(i, 0) = val;
	}

	// Compute x- & y-gradients.
	cv::Mat Ix, Iy;
	cv::filter2D(img, Ix, -1, kernelX, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
	cv::filter2D(img, Iy, -1, kernelY, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);

	// Solve eigensystem.

	const cv::Mat Ix2 = Ix.mul(Ix);  // Ix^2 = Ix * Ix.
	const cv::Mat Iy2 = Iy.mul(Iy);  // Iy^2 = Iy * Iy.
	const cv::Mat IxIy = Ix.mul(Iy);  // Ix * Iy.

#if 1
	// TODO [add] >> If Gaussian blur is required, blurring is applied to Ix2, Iy2, & IxIy.
	const int blur_kernel_size = 2 * (int)std::ceil(blur_sigma) + 1;
	cv::GaussianBlur(Ix2, Ix2, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_DEFAULT);
	cv::GaussianBlur(Iy2, Iy2, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_DEFAULT);
	cv::GaussianBlur(IxIy, IxIy, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_DEFAULT);
#endif

	// Structure tensor at point (i, j), S = [ Ix2(i, j) IxIy(i, j) ; IxIy(i, j) Iy2(i, j) ].
	const cv::Mat detS = Ix2.mul(Iy2) - IxIy.mul(IxIy);
	const cv::Mat S11_plus_S22 = Ix2 + Iy2;
#if 0
	cv::Mat sqrtDiscriminant(img.size(), CV_64FC1);
	cv::sqrt(S11_plus_S22.mul(S11_plus_S22) - 4.0 * detS, sqrtDiscriminant);
#else
	cv::Mat sqrtDiscriminant(S11_plus_S22.mul(S11_plus_S22) - 4.0 * detS);

	const double tol = 1.0e-10;
	const int count1 = cv::countNonZero(sqrtDiscriminant < 0.0);
	if (count1 > 0)
	{
		std::cout << "non-zero count = " << count1 << std::endl;

		const int count2 = cv::countNonZero(sqrtDiscriminant < -tol);
		if (count2 > 0)
		{
#if defined(DEBUG) || defined(_DEBUG)
			for (int i = 0; i < img.rows; ++i)
				for (int j = 0; j < img.cols; ++j)
					if (sqrtDiscriminant.at<double>(i, j) < 0.0)
						std::cout << i << ", " << j << " = " << sqrtDiscriminant.at<double>(i, j) << std::endl;
#endif

			std::cerr << "complex eigenvalues exist" << std::endl;
			return;
		}
		else
			sqrtDiscriminant.setTo(0.0, sqrtDiscriminant < 0.0);
	}

	cv::sqrt(sqrtDiscriminant, sqrtDiscriminant);
#endif

	// Eigenvalues.
	eval1 = (S11_plus_S22 + sqrtDiscriminant) * 0.5;
	eval2 = (S11_plus_S22 - sqrtDiscriminant) * 0.5;
	// Eigenvectors.
	evec1 = cv::Mat::zeros(img.size(), CV_64FC2);
	evec2 = cv::Mat::zeros(img.size(), CV_64FC2);

	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (std::fabs(eval1.at<double>(i, j)) < std::fabs(eval2.at<double>(i, j)))
				std::swap(eval1.at<double>(i, j), eval2.at<double>(i, j));

			const double a = Ix2.at<double>(i, j);
			const double b = IxIy.at<double>(i, j);
			const double lambda1 = eval1.at<double>(i, j);
			const double lambda2 = eval2.at<double>(i, j);
			evec1.at<cv::Vec2d>(i, j) = cv::Vec2d(-b, a - lambda1);
			evec2.at<cv::Vec2d>(i, j) = cv::Vec2d(-b, a - lambda2);
		}
}

// REF [function] >> compute_valid_region_using_coherence() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_structure_tensor.cpp
void compute_valid_region_using_coherence(const cv::Mat &eval1, const cv::Mat &eval2, const cv::Mat &valid_eval_region_mask, const cv::Mat &constant_region_mask, cv::Mat &valid_region)
{
	// Coherence = 1 when the gradient is totally aligned, and coherence = 0 (lambda1 = lambda2) when it has no predominant direction.
	cv::Mat coherence((eval1 - eval2) / (eval1 + eval2));  // if eigenvalue2 > 0.
	coherence = coherence.mul(coherence);

	double minVal, maxVal;
	cv::minMaxLoc(coherence, &minVal, &maxVal);
	std::cout << "coherence: min = " << minVal << ", max = " << maxVal << std::endl;

#if 0
	const double threshold = 0.5;
	valid_region = coherence <= threshold;
#elif 0
	const double threshold = 0.9;
	valid_region = coherence >= threshold;
#else
	const double threshold1 = 0.2, threshold2 = 0.8;
	valid_region = threshold1 <= coherence & coherence <= threshold2;
#endif

	valid_region.setTo(cv::Scalar::all(0), constant_region_mask);
}

// REF [function] >> compute_valid_region_using_ev_ratio() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_structure_tensor.cpp
void compute_valid_region_using_ev_ratio(const cv::Mat &eval1, const cv::Mat &eval2, const cv::Mat &valid_eval_region_mask, const cv::Mat &constant_region_mask, cv::Mat &valid_region)
{
	cv::Mat eval_ratio(valid_eval_region_mask.size(), CV_8UC1, cv::Scalar::all(0));
	cv::Mat(eval1 / eval2).copyTo(eval_ratio, valid_eval_region_mask);

	double minVal, maxVal;
	cv::minMaxLoc(eval_ratio, &minVal, &maxVal);
	std::cout << "ev ratio: min = " << minVal << ", max = " << maxVal << std::endl;

#if 0
	const double threshold = 0.5;
	valid_region = cv::abs(eval_ratio - 1.0f) <= threshold;  // If lambda1 = lambda2, the gradient in the window has no predominant direction.
#else
	const double threshold1 = 1.0, threshold2 = 5.0;
	valid_region = threshold1 <= eval_ratio & eval_ratio <= threshold2;
#endif

	valid_region.setTo(cv::Scalar::all(0), constant_region_mask);
}

void construct_depth_variation_mask_using_structure_tensor(const cv::Mat &depth_image, cv::Mat &depth_variation_mask)
{
	cv::Mat img_double;
	double minVal, maxVal;

	cv::minMaxLoc(depth_image, &minVal, &maxVal);
	depth_image.convertTo(img_double, CV_64FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

	const double deriv_sigma = 3.0;
	const double blur_sigma = 2.0;
	cv::Mat eval1, eval2, evec1, evec2;
	structure_tensor_2d(img_double, deriv_sigma, blur_sigma, eval1, eval2, evec1, evec2);

	// Post-processing.
	eval1 = cv::abs(eval1);
	eval2 = cv::abs(eval2);

#if 0
	cv::minMaxLoc(eval1, &minVal, &maxVal);
	std::cout << "Max eigenvalue: " << minVal << ", " << maxVal << std::endl;
	cv::minMaxLoc(eval2, &minVal, &maxVal);
	std::cout << "Min eigenvalue: " << minVal << ", " << maxVal << std::endl;
#endif

	const double tol = 1.0e-10;
	const cv::Mat valid_eval_region_mask(eval2 >= tol);
	const cv::Mat constant_region_mask(eval1 < tol & eval2 < tol);  // If lambda1 = lambda2 = 0, the image within the window is constant.

	cv::Mat valid_region;
#if 1
	// METHOD #1: using coherence.
	//	REF [site] >> http://en.wikipedia.org/wiki/Structure_tensor
	compute_valid_region_using_coherence(eval1, eval2, valid_eval_region_mask, constant_region_mask, depth_variation_mask);
#else
	// METHOD #2: using the ratio of eigenvales.
	compute_valid_region_using_ev_ratio(eval1, eval2, valid_eval_region_mask, constant_region_mask, depth_variation_mask);
#endif
}

void construct_depth_variation_mask_using_depth_changing(const cv::Mat &depth_image, cv::Mat &depth_variation_mask)
{
	// Compute phase distribution from neighborhood.
	const int radius = 2;
	compute_phase_distribution_from_neighborhood(depth_image, radius, depth_variation_mask);
}

}  // namespace swl
