//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include "DepthGuidedMap.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <stdexcept>


namespace swl {

// [ref] Util.cpp
cv::Rect get_bounding_rect(const cv::Mat &img);
void compute_phase_distribution_from_neighborhood(const cv::Mat &depth_map, const int radius, cv::Mat &depth_variation_mask);
void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, const float alpha, const float beta, const float gamma, const bool use_gradient, const CvSize &win, std::vector<cv::Point> &snake_contour);

void zhang_suen_thinning_algorithm(const cv::Mat &src, cv::Mat &dst);
void guo_hall_thinning_algorithm(cv::Mat &im);

bool load_kinect_images(const std::string &rgb_input_filename, const std::string &depth_input_filename, const bool useRectifiedImages, cv::Mat &rgb_input_image, cv::Mat &depth_input_image, double *fx_rgb, double *fy_rgb);
bool load_structure_tensor_mask(const std::string &filename, cv::Mat &depth_variation_mask);
void construct_valid_depth_image(const cv::Mat &depth_input_image, cv::Mat &depth_validity_mask, cv::Mat &valid_depth_image);

void construct_depth_variation_mask_using_structure_tensor(const cv::Mat &depth_image, cv::Mat &depth_variation_mask);
void construct_depth_variation_mask_using_depth_changing(const cv::Mat &depth_image, cv::Mat &depth_variation_mask);

// [ref] SegmentationUsingGrabCut.cpp
void run_grabcut_using_depth_guided_mask(const cv::Mat &rgb_image, const cv::Mat &depth_guided_map);

// [ref] SegmentationUsingGraphCut.cpp
void run_interactive_graph_cuts_segmentation(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_guided_map);
void run_efficient_graph_based_image_segmentation(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_guided_map, const double fx_rgb, const double fy_rgb);

void preprocess_image(cv::Mat &image)
{
#if 0
	// METHOD #1: down-scale and up-scale the image to filter out the noise.

	cv::Mat tmp_image;
	cv::pyrDown(image, tmp_image);
	cv::pyrUp(tmp_image, image);
#elif 0
	// METHOD #2: Gaussian filtering.

	{
		// FIXME [adjust] >> adjust parameters.
		const int kernelSize = 3;
		const double sigma = 0;
		cv::GaussianBlur(image, image, cv::Size(kernelSize, kernelSize), sigma, sigma);
	}
#elif 0
	// METHOD #3: box filtering.

	{
		cv::Mat tmp_image = image;
		// FIXME [adjust] >> adjust parameters.
		const int d = -1;
		const int kernelSize = 3;
		const bool normalize = true;
		cv::boxFilter(tmp_image, image, d, cv::Size(kernelSize, kernelSize), cv::Point(-1,-1), normalize, cv::BORDER_DEFAULT);
	}
#elif 0
	// METHOD #4: bilateral filtering.

	{
		cv::Mat tmp_image = image;
		// FIXME [adjust] >> adjust parameters.
		const int d = -1;
		const double sigmaColor = 3.0;
		const double sigmaSpace = 50.0;
		cv::bilateralFilter(tmp_image, image, d, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
	}
#endif
}

void extract_foreground_based_on_depth_guided_map()
{
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480), imageSize_mask(640, 480);

#if 0
	const std::size_t num_images = 4;
	const bool useDepthVariation = false;
	const bool useRectifiedImages = true;
	const bool useDepthRangeFiltering = false;

	std::vector<std::string> rgb_input_file_list, depth_input_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130530T103805.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130531T023152.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130531T023346.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130531T023359.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130530T103805.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130531T023152.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130531T023346.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130531T023359.png");

	std::vector<cv::Range> depth_range_list;
	depth_range_list.reserve(num_images);
	{
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
#elif 0
	// for Kinect ver1.

	const std::size_t num_images = 6;
	const bool useDepthVariation = true;
	const bool useRectifiedImages = true;
	const bool useDepthRangeFiltering = false;

	std::vector<std::string> rgb_input_file_list, depth_input_file_list, structure_tensor_mask_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	structure_tensor_mask_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162552.png");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

	std::vector<cv::Range> depth_range_list;
	depth_range_list.reserve(num_images);
	{
		const int min_depth = 100, max_depth = 4000;
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
	}
#elif 1
	// for Kinect ver2.

	const std::size_t num_images = 5;
	const bool useDepthVariation = true;
	const bool useRectifiedImages = false;
	const bool useDepthRangeFiltering = false;

	std::vector<std::string> rgb_input_file_list, depth_input_file_list, structure_tensor_mask_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	structure_tensor_mask_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect2_rgba_20130725T211659.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect2_rgba_20130725T211705.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect2_rgba_20130725T211713.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect2_rgba_20130725T211839.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect2_rgba_20130725T211842.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_20130725T211659.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_20130725T211705.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_20130725T211713.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_20130725T211839.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_20130725T211842.png");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_ev_ratio_20130725T211659.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_ev_ratio_20130725T211705.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_ev_ratio_20130725T211713.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_ev_ratio_20130725T211839.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect2_depth_transformed_ev_ratio_20130725T211842.tif");

	std::vector<cv::Range> depth_range_list;
	depth_range_list.reserve(num_images);
	{
		const int min_depth = 100, max_depth = 4000;
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
		depth_range_list.push_back(cv::Range(min_depth, max_depth));
	}
#endif

	//
	cv::Mat rgb_input_image, depth_input_image, depth_variation_mask;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	cv::Mat foreground_mask(imageSize_mask, CV_8UC1), background_mask(imageSize_mask, CV_8UC1), filtered_depth_variation_mask;
	double fx_rgb, fy_rgb;
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// [1] load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, &fx_rgb, &fy_rgb))
			continue;

		// [2] pre-process input images (optional).
		preprocess_image(rgb_input_image);

		// [3] construct depth variation mask (optional).
		if (useDepthVariation)
		{
#if 1
			// METHOD #1: load structure tensor mask.
			if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], depth_variation_mask))
				continue;
#elif 0
			// METHOD #2: compute structure tensor mask.
			construct_depth_variation_mask_using_structure_tensor(depth_input_image, depth_variation_mask);
#elif 0
			// METHOD #3: compute depth changing mask.
			construct_depth_variation_mask_using_depth_changing(depth_input_image, depth_variation_mask);
#endif

#if 0
			// show depth variation mask.
			cv::imshow("depth variation mask", depth_variation_mask);
#endif

#if 0
			{
				std::ostringstream strm;
				strm << "./data/kinect_segmentation/depth_variation_mask_" << i << ".png";
				cv::imwrite(strm.str(), depth_variation_mask);
			}
#endif
		}

		const int64 startTime = cv::getTickCount();

		// [4] construct valid depth image.
		if (useDepthRangeFiltering)
			cv::inRange(depth_input_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
		else
			cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);

		construct_valid_depth_image(depth_input_image, depth_validity_mask, valid_depth_image);

#if 0
		// show depth validity mask.
		cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
		}
#endif

		// [5] construct depth-guided map.
#if 0
		// METHOD #1: construct depth-guided map using superpixel.
		construct_depth_guided_map_using_superpixel(rgb_input_image, valid_depth_image, depth_validity_mask, depth_guided_map);
#elif 0
		// METHOD #2: construct depth-guided map using edge detection & morphological operation.
		construct_depth_guided_map_using_edge_detection_and_morphological_operation(valid_depth_image, depth_validity_mask, depth_guided_map);
#elif 1
		// METHOD #3: construct depth-guided map using depth variation.
		if (depth_variation_mask.empty())
		{
			std::cerr << "depth variation mask doesn't exist" << std::endl;
			continue;
		}
		else
			construct_depth_guided_map_using_depth_variation(depth_variation_mask, depth_input_image, depth_guided_map, filtered_depth_variation_mask);
#endif

#if 1
		// show depth-guided map.
		{
			cv::minMaxLoc(depth_guided_map, &minVal, &maxVal);
			depth_guided_map.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			cv::imshow("depth-guided map", tmp_image);
		}
#endif

#if 0
		// show filtered depth variation mask.
		cv::imshow("filtered depth variation mask", filtered_depth_variation_mask);
#endif

#if 1
		{
			std::ostringstream strm;

			//cv::minMaxLoc(depth_guided_map, &minVal, &maxVal);
			//depth_guided_map.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);
			//cv::cvtColor(tmp_image, tmp_image, CV_GRAY2BGR);
			cv::cvtColor(depth_guided_map, tmp_image, CV_GRAY2BGR);

			strm << "./data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
		}
#endif
#if 0
		// wirte trimap.
		{
			std::ostringstream strm;
			cv::Mat trimap(depth_guided_map.size(), CV_8UC1, cv::Scalar::all(128));
			trimap.setTo(cv::Scalar::all(0), SWL_BGD == depth_guided_map | SWL_PR_BGD == depth_guided_map);
			trimap.setTo(cv::Scalar::all(255), SWL_FGD == depth_guided_map);
			strm << "./data/kinect_segmentation/trimap_" << i << ".png";
			cv::imwrite(strm.str(), trimap);
		}
#endif
#if 0
		// wirte scribble.
		{
			std::ostringstream strm;
			cv::Mat scribble = rgb_input_image.clone();
			scribble.setTo(cv::Scalar::all(0), SWL_PR_BGD == depth_guided_map);
			scribble.setTo(cv::Scalar::all(255), SWL_FGD == depth_guided_map);
			strm << "./data/kinect_segmentation/scribble_" << i << ".png";
			cv::imwrite(strm.str(), scribble);
		}
#endif

		// [6] extract foreground.
#if 0
		// METHOD #1: segment foreground using Snake.
		{
			std::vector<std::vector<cv::Point> > contours;
			cv::findContours(cv::Mat(SWL_PR_FGD == depth_guided_map), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point());
			//std::vector<cv::Vec4i> hierarchy;
			//cv::findContours(cv::Mat(SWL_PR_FGD == depth_guided_map), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

			//const std::size_t NUM_SNAKE_POINTS = 50;
			const std::size_t NUM_SNAKE_POINTS = 0;
			const float alpha = 5.0f;  // weight(s) of continuity energy, single float or array of length floats, one for each contour point.
			const float beta = 5.0f;  // weight(s) of curvature energy, single float or array of length floats, one for each contour point.
			const float gamma = 5.0f;  // weight(s) of image energy, single float or array of length floats, one for each contour point.
			const bool use_gradient = true;  // gradient flag; if true, the function calculates the gradient magnitude for every image pixel and consideres it as the energy field, otherwise the input image itself is considered.
			const CvSize win = cvSize(21, 21);  // size of neighborhood of every point used to search the minimum, both win.width and win.height must be odd.
			const double MIN_CONTOUR_AREA = 100;

			cv::Mat gray_image;
			cv::cvtColor(rgb_input_image, gray_image, CV_BGR2GRAY);

			std::vector<std::vector<cv::Point> > snake_contours;
			snake_contours.reserve(contours.size());
			std::cout << "start snake ..." << std::endl;
			for (std::vector<std::vector<cv::Point> >::const_iterator cit = contours.begin(); cit != contours.end(); ++cit)
			{
				if (cit->empty() || cv::contourArea(cv::Mat(*cit)) < MIN_CONTOUR_AREA)
					snake_contours.push_back(std::vector<cv::Point>());
				else
				{
					std::vector<cv::Point> snake_contour;
					{
						swl::fit_contour_by_snake(gray_image, *cit, NUM_SNAKE_POINTS, alpha, beta, gamma, use_gradient, win, snake_contour);
					}
					snake_contours.push_back(snake_contour);
				}
			}
			std::cout << "end snake ..." << std::endl;

			// show results of fitting using Snake.
			//rgb_input_image.copyTo(tmp_image);

			//cv::drawContours(tmp_image, snake_contours, -1, CV_RGB(255, 0, 0), CV_FILLED, 8, cv::noArray(), 0, cv::Point());
			//cv::drawContours(tmp_image, snake_contours, -1, CV_RGB(255, 0, 0), 1, 8, cv::noArray(), 0, cv::Point());
			int idx = 0;
			for (std::vector<std::vector<cv::Point> >::const_iterator cit = snake_contours.begin(); cit != snake_contours.end(); ++cit, ++idx)
			{
				if (cit->empty() || cv::contourArea(cv::Mat(*cit)) < MIN_CONTOUR_AREA) continue;

				//rgb_input_image.copyTo(tmp_image);
				cv::cvtColor(gray_image, tmp_image, CV_GRAY2BGR);

				std::cout << "contour id: " << (idx + 1) << " / " << snake_contours.size() << std::endl;

				//const cv::Scalar color1(std::rand() & 255, std::rand() & 255, std::rand() & 255);
				const cv::Scalar color1(0, 255, 0);
				//cv::drawContours(tmp_image, contours, idx, color1, CV_FILLED, 8, cv::noArray(), 0, cv::Point());
				cv::drawContours(tmp_image, contours, idx, color1, 2, 8, cv::noArray(), 0, cv::Point());

				//const cv::Scalar color2(std::rand() & 255, std::rand() & 255, std::rand() & 255);
				const cv::Scalar color2(0, 0, 255);
				//cv::drawContours(tmp_image, snake_contours, idx, color2, CV_FILLED, 8, cv::noArray(), 0, cv::Point());
				cv::drawContours(tmp_image, snake_contours, idx, color2, 2, 8, cv::noArray(), 0, cv::Point());

				cv::imshow("results of fitting using Snake", tmp_image);

				cv::waitKey(0);
			}

			//cv::imshow("results of fitting using Snake", tmp_image);
		}
#elif 0
		// METHOD #2: segment image by interactive graph-cuts segmentation algorithm.
		run_interactive_graph_cuts_segmentation(rgb_input_image, valid_depth_image, depth_guided_map);
#elif 0
		// METHOD #3: segment image by efficient graph-based image segmentation algorithm.
		run_efficient_graph_based_image_segmentation(rgb_input_image, valid_depth_image, depth_guided_map, fx_rgb, fy_rgb);
#elif 0
		// METHOD #4: segment image by GrabCut algorithm.
		run_grabcut_using_depth_guided_mask(rgb_input_image, depth_guided_map);
#elif 0
		// METHOD #5: segment image by matting.

		// FIXME [implement] >>
#endif

		cv::Mat edge;
		{
			// edge detection on grayscale.
			cv::Mat gray;
			cv::cvtColor(rgb_input_image, gray, CV_BGR2GRAY);

			const int lowerEdgeThreshold = 30, upperEdgeThreshold = 50;
			const bool useL2 = true;  // if true, use L2 norm. otherwise, use L1 norm (faster).
			const int apertureSize = 3;  // aperture size for the Sobel() operator.
			cv::Canny(gray, edge, lowerEdgeThreshold, upperEdgeThreshold, apertureSize, useL2);
		}

/*
		{
			//const int distanceType = CV_DIST_C;  // C/Inf metric.
			//const int distanceType = CV_DIST_L1;  // L1 metric.
			const int distanceType = CV_DIST_L2;  // L2 metric.
			//const int maskSize = CV_DIST_MASK_3;
			//const int maskSize = CV_DIST_MASK_5;
			const int maskSize = CV_DIST_MASK_PRECISE;
			//const int labelType = cv::DIST_LABEL_CCOMP;
			const int labelType = cv::DIST_LABEL_PIXEL;

			cv::Mat dist32f, labels;
			cv::distanceTransform(cv::Scalar::all(255) - edge, dist32f, labels, distanceType, maskSize, labelType);

			{
				cv::minMaxLoc(dist32f, &minVal, &maxVal);
				dist32f.convertTo(dist32f, CV_32FC1, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

				cv::imshow("distance transform - dt", dist32f);
			}

			cv::Mat edge_labels(labels.size(), labels.type(), cv::Scalar::all(0));
			labels.copyTo(edge_labels, edge);

			cv::Mat depth_labels(labels.size(), labels.type(), cv::Scalar::all(0));
			labels.copyTo(depth_labels, filtered_depth_variation_mask > 0);

			cv::minMaxLoc(depth_labels, &minVal, &maxVal);

			cv::Mat out_img = rgb_input_image.clone();
			//const int num_labels = cv::countNonZero(edge);
			cv::Point minLoc, maxLoc;
			for (int i = (int)minVal; i <= (int)maxVal; ++i)
			{
				if (cv::countNonZero(i == depth_labels) > 0)
				{
					cv::minMaxLoc(i == edge_labels, NULL, NULL, &minLoc, &maxLoc);
					cv::circle(out_img, maxLoc, 1, CV_RGB(0, 255, 0), CV_FILLED, CV_AA, 0);
				}
			}

			{
				cv::imshow("distance transform - result", out_img);
			}
		}
*/
		const int64 elapsed = cv::getTickCount() - startTime;
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
#if 1
	const std::size_t num_images = 4;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130530T103805.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130531T023152.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130531T023346.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130531T023359.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130530T103805.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130531T023152.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130531T023346.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130531T023359.png");

	const bool useDepthRangeFiltering = false;
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
#elif 0
	const std::size_t num_images = 6;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162552.png");

	const bool useDepthRangeFiltering = false;
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
#endif

	//
	const bool useRectifiedImages = true;
	cv::Mat rgb_input_image, depth_input_image;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	double fx_rgb, fy_rgb;
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// [1] load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, &fx_rgb, &fy_rgb))
			continue;

		// [2] pre-process input images (optional).
		preprocess_image(rgb_input_image);

		// [3] construct depth variation mask (optional).
		// do nothing

		const int64 startTime = cv::getTickCount();

		// [4] construct valid depth image.
		if (useDepthRangeFiltering)
			cv::inRange(depth_input_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
		else
			cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);

		construct_valid_depth_image(depth_input_image, depth_validity_mask, valid_depth_image);

#if 1
		// show depth validity mask.
		cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
		}
#endif

		// [5] construct depth-guided map.
#if 0
		// construct depth-guided map using superpixel.
		construct_depth_guided_map_using_superpixel(rgb_input_image, valid_depth_image, depth_validity_mask, depth_guided_map);
#elif 1
		// construct depth-guided map using edge detection & morphological operation.
		construct_depth_guided_map_using_edge_detection_and_morphological_operation(valid_depth_image, depth_validity_mask, depth_guided_map);
#endif

#if 1
		// show depth-guided map.
		{
			cv::minMaxLoc(depth_guided_map, &minVal, &maxVal);
			depth_guided_map.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			cv::imshow("depth-guided map", tmp_image);
#endif

#if 0
			std::ostringstream strm;
			cv::cvtColor(depth_guided_map, tmp_image, CV_GRAY2BGR);
			strm << "./data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
#endif
		}

		// [6] extract foreground.
#if 1
		// segment image by GrabCut algorithm.
		run_grabcut_using_depth_guided_mask(rgb_input_image, depth_guided_map);
#elif 0
		// segment image by efficient graph-based image segmentation algorithm.
		run_efficient_graph_based_image_segmentation(rgb_input_image, valid_depth_image, depth_guided_map, fx_rgb, fy_rgb);
#endif

		const int64 elapsed = cv::getTickCount() - startTime;
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

void segment_foreground_based_on_structure_tensor()
{
	const std::size_t num_images = 6;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480), imageSize_mask(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list, structure_tensor_mask_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	structure_tensor_mask_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162552.png");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

	const bool useDepthRangeFiltering = false;
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
	const bool useRectifiedImages = true;
	cv::Mat rgb_input_image, depth_input_image, depth_variation_mask;
	cv::Mat valid_depth_image, depth_validity_mask(imageSize_rgb, CV_8UC1);
	cv::Mat depth_guided_map(imageSize_rgb, CV_8UC1), grabCut_mask(imageSize_mask, CV_8UC1), filtered_depth_variation_mask;
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, NULL, NULL))
			continue;

		// [2] pre-process input images (optional).
		// do nothing

		// [3] construct depth variation mask (optional).
#if 0
		// METHOD #1: load structure tensor mask.
		if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], depth_variation_mask))
			continue;
#elif 1
		// METHOD #2: compute structure tensor mask.
		construct_depth_variation_mask_using_structure_tensor(depth_input_image, depth_variation_mask);
#elif 0
		// METHOD #3: compute depth changing mask.
		construct_depth_variation_mask_using_depth_changing(depth_input_image, depth_variation_mask);
#endif

#if 1
		// show depth variation mask.
		cv::imshow("depth variation mask", depth_variation_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_variation_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_variation_mask);
		}
#endif

		const int64 startTime = cv::getTickCount();

		// [4] construct valid depth image.
		if (useDepthRangeFiltering)
			cv::inRange(depth_input_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
		else
			cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);

		construct_valid_depth_image(depth_input_image, depth_validity_mask, valid_depth_image);

#if 1
		// show depth validity mask.
		cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
		}
#endif

		// [5] construct depth-guided map.
		construct_depth_guided_map_using_depth_variation(depth_variation_mask, depth_input_image, depth_guided_map, filtered_depth_variation_mask);

#if 1
		// show depth-guided map.
		{
			cv::minMaxLoc(depth_guided_map, &minVal, &maxVal);
			depth_guided_map.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			cv::imshow("depth-guided map", tmp_image);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			cv::cvtColor(depth_guided_map, tmp_image, CV_GRAY2BGR);
			strm << "./data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
		}
#endif

#if 1
		// show filtered depth variation mask.
		cv::imshow("filtered depth variation mask", filtered_depth_variation_mask);
#endif

		// [6] extract foreground.
#if 0
		// segment foreground using Snake.
		{
			//const std::size_t NUM_SNAKE_POINTS = 50;
			const std::size_t NUM_SNAKE_POINTS = 0;
			const float alpha = 3.0f;  // weight(s) of continuity energy, single float or array of length floats, one for each contour point.
			const float beta = 5.0f;  // weight(s) of curvature energy, single float or array of length floats, one for each contour point.
			const float gamma = 2.0f;  // weight(s) of image energy, single float or array of length floats, one for each contour point.
			const bool use_gradient = true;  // gradient flag; if true, the function calculates the gradient magnitude for every image pixel and consideres it as the energy field, otherwise the input image itself is considered.
			const CvSize win = cvSize(21, 21);  // size of neighborhood of every point used to search the minimum, both win.width and win.height must be odd.

			cv::Mat gray_image;
			cv::cvtColor(rgb_input_image, gray_image, CV_BGR2GRAY);

			std::vector<std::vector<cv::Point> > snake_contours;
			snake_contours.reserve(contours.size());
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
			{
				if (cv::contourArea(cv::Mat(contours[idx])) < MIN_CONTOUR_AREA) continue;

				std::vector<cv::Point> snake_contour;
				swl::fit_contour_by_snake(gray_image, contours[idx], NUM_SNAKE_POINTS, alpha, beta, gamma, use_gradient, win, snake_contour);
				snake_contours.push_back(snake_contour);
			}

			// show results of fitting using Snake.
			rgb_input_image.copyTo(tmp_image);

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
		run_grabcut_using_depth_guided_mask(rgb_input_image, depth_guided_map);
#endif

		const int64 elapsed = cv::getTickCount() - startTime;
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

void segment_foreground_using_single_layered_graphical_model()
{
	const std::size_t num_images = 6;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480), imageSize_mask(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list, structure_tensor_mask_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	structure_tensor_mask_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162552.png");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tensor_mask_file_list.push_back("./data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

	const bool useDepthRangeFiltering = false;
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
	const bool useRectifiedImages = true;
	cv::Mat rgb_input_image, depth_input_image, depth_variation_mask;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	cv::Mat foreground_mask(imageSize_mask, CV_8UC1), background_mask(imageSize_mask, CV_8UC1), filtered_depth_variation_mask;
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// [1] load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, NULL, NULL))
			continue;

		// [2] pre-process input images (optional).
		preprocess_image(rgb_input_image);

		// [3] construct depth variation mask (optional).
#if 0
		// METHOD #1: load structure tensor mask.
		if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], depth_variation_mask))
			continue;
#elif 1
		// METHOD #2: compute structure tensor mask.
		construct_depth_variation_mask_using_structure_tensor(depth_input_image, depth_variation_mask);
#elif 0
		// METHOD #3: compute depth changing mask.
		construct_depth_variation_mask_using_depth_changing(depth_input_image, depth_variation_mask);
#endif

#if 1
		// show depth variation mask.
		cv::imshow("depth variation mask", depth_variation_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_variation_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_variation_mask);
		}
#endif

		const int64 startTime = cv::getTickCount();

		// [4] construct valid depth image.
		if (useDepthRangeFiltering)
			cv::inRange(depth_input_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
		else
			cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);

		construct_valid_depth_image(depth_input_image, depth_validity_mask, valid_depth_image);

#if 1
		// show depth validity mask.
		cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
		}
#endif

		// [5] construct depth-guided map.
		construct_depth_guided_map_using_depth_variation(depth_variation_mask, depth_input_image, depth_guided_map, filtered_depth_variation_mask);

#if 1
		// show depth-guided map.
		{
			cv::minMaxLoc(depth_guided_map, &minVal, &maxVal);
			depth_guided_map.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			cv::imshow("depth-guided map", tmp_image);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			cv::cvtColor(depth_guided_map, tmp_image, CV_GRAY2BGR);
			strm << "./data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
		}
#endif

#if 1
		// show filtered depth variation mask.
		cv::imshow("filtered depth variation mask", filtered_depth_variation_mask);
#endif

		// [6] extract foreground.
		// segment image by interactive graph-cuts segmentation algorithm.
		run_interactive_graph_cuts_segmentation(rgb_input_image, valid_depth_image, depth_guided_map);

		const int64 elapsed = cv::getTickCount() - startTime;
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

void segment_foreground_using_two_layered_graphical_model()
{
	const std::size_t num_images = 6;
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

	std::vector<std::string> rgb_input_file_list, depth_input_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162309.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162314.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162348.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162459.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162525.png");
	rgb_input_file_list.push_back("./data/kinect_segmentation/kinect_rgba_20130614T162552.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162309.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162314.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162348.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162459.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162525.png");
	depth_input_file_list.push_back("./data/kinect_segmentation/kinect_depth_20130614T162552.png");

	const bool useDepthRangeFiltering = false;
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
	const bool useRectifiedImages = true;
	cv::Mat rgb_input_image, depth_input_image, depth_variation_mask;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// [1] load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, NULL, NULL))
			continue;

		// [2] pre-process input images (optional).
		preprocess_image(rgb_input_image);

		// [3] construct depth variation mask (optional).
#if 0
		// METHOD #1: load structure tensor mask.
		if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], depth_variation_mask))
			continue;
#elif 1
		// METHOD #2: compute structure tensor mask.
		construct_depth_variation_mask_using_structure_tensor(depth_input_image, depth_variation_mask);
#elif 0
		// METHOD #3: compute depth changing mask.
		construct_depth_variation_mask_using_depth_changing(depth_input_image, depth_variation_mask);
#endif

#if 1
		// show depth variation mask.
		cv::imshow("depth variation mask", depth_variation_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_variation_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_variation_mask);
		}
#endif

		const int64 startTime = cv::getTickCount();

		// [4] construct valid depth image.
		if (useDepthRangeFiltering)
			cv::inRange(depth_input_image, cv::Scalar::all(depth_range_list[i].start), cv::Scalar::all(depth_range_list[i].end), depth_validity_mask);
		else
			cv::Mat(depth_input_image > 0).copyTo(depth_validity_mask);

		construct_valid_depth_image(depth_input_image, depth_validity_mask, valid_depth_image);

#if 1
		// show depth validity mask.
		cv::imshow("depth validity mask", depth_validity_mask);
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/depth_validity_mask_" << i << ".png";
			cv::imwrite(strm.str(), depth_validity_mask);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "./data/kinect_segmentation/valid_depth_image_" << i << ".png";
			cv::imwrite(strm.str(), valid_depth_image);
		}
#endif

		// [5] construct depth-guided map.

		// FIXME [implement] >>

#if 1
		// show depth-guided map.
		{
			cv::minMaxLoc(depth_guided_map, &minVal, &maxVal);
			depth_guided_map.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

			cv::imshow("depth-guided map", tmp_image);
#endif

#if 0
			std::ostringstream strm;
			cv::cvtColor(depth_guided_map, tmp_image, CV_GRAY2BGR);
			strm << "./data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
#endif
		}

		// [6] extract foreground.

		// FIXME [implement] >>

		const int64 elapsed = cv::getTickCount() - startTime;
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

/*
		//swl::segment_foreground_based_on_depth_guided_map();
		//swl::segment_foreground_based_on_structure_tensor();

        // FIXME [implement] >> not completed
        //swl::segment_foreground_using_single_layered_graphical_model();
		//swl::segment_foreground_using_two_layered_graphical_model();
*/

		// integrated version.
		swl::extract_foreground_based_on_depth_guided_map();

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
			swl::compute_phase_distribution_from_neighborhood(mat, radius, depth_variation_mask);
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
