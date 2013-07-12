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
void compute_phase_distribution_from_neighborhood(const cv::Mat &depth_map, const int radius, cv::Mat &depth_changing_mask);
void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, std::vector<cv::Point> &snake_contour);

void zhang_suen_thinning_algorithm(const cv::Mat &src, cv::Mat &dst);
void guo_hall_thinning_algorithm(cv::Mat &im);

bool load_kinect_images(const std::string &rgb_input_filename, const std::string &depth_input_filename, const bool useRectifiedImages, cv::Mat &rgb_input_image, cv::Mat &depth_input_image, double *fx_rgb, double *fy_rgb);
bool load_structure_tensor_mask(const std::string &filename, cv::Mat &structure_tensor_mask);
void construct_valid_depth_image(const bool useDepthRangeFiltering, const double minRange, const double maxRange, const cv::Mat &depth_input_image, cv::Mat &valid_depth_image, cv::Mat &depth_validity_mask);

// [ref] SegmentationUsingGrabCut.cpp
void run_grabcut_using_depth_guided_mask(const cv::Mat &rgb_image, const cv::Mat &depth_guided_map);

// [ref] SegmentationUsingGraphCut.cpp
void run_interactive_graph_cuts_segmentation(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_guided_map);
void run_efficient_graph_based_image_segmentation(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_guided_map, const double fx_rgb, const double fy_rgb);

void extract_foreground_based_on_depth_guided_map()
{
	const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480), imageSize_mask(640, 480);

#if 0
	const std::size_t num_images = 4;
	const bool useStructureTensor = false;

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
#elif 1
	const std::size_t num_images = 6;
	const bool useStructureTensor = true;

	std::vector<std::string> rgb_input_file_list, depth_input_file_list, structure_tensor_mask_file_list;
	rgb_input_file_list.reserve(num_images);
	depth_input_file_list.reserve(num_images);
	structure_tensor_mask_file_list.reserve(num_images);
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
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

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
	cv::Mat rgb_input_image, depth_input_image, structure_tensor_mask;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	cv::Mat foreground_mask(imageSize_mask, CV_8UC1), background_mask(imageSize_mask, CV_8UC1);
	double fx_rgb, fy_rgb;
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, &fx_rgb, &fy_rgb))
			continue;

		// pre-process input image.

#if 1
		// METHOD #1: load structure tensor mask.

		if (useStructureTensor)
		{
			if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], structure_tensor_mask))
				continue;
		}
#elif 0
		// METHOD #2: compute structure tensor mask.

		// FIXME [implement] >>
#elif 0
		// METHOD #3: compute depth changing mask.

		// compute phase distribution from neighborhood
		{
			const int radius = 2;
			cv::Mat depth_changing_mask(imageSize_rgb, CV_8UC1, cv::Scalar::all(0))

			// FIXME [implement] >>
			compute_phase_distribution_from_neighborhood(depth_image, radius, depth_changing_mask);

#if 1
			// show depth changing mask.
			cv::imshow("depth changing mask", depth_changing_mask);
#endif
		}
#endif

		const int64 startTime = cv::getTickCount();

		// construct valid depth image.
		construct_valid_depth_image(useDepthRangeFiltering, depth_range_list[i].start, depth_range_list[i].end, depth_input_image, valid_depth_image, depth_validity_mask);

		// construct depth-guided map.
#if 0
		// METHOD #1: construct depth-guided map using superpixel.
		construct_depth_guided_map_using_superpixel(rgb_input_image, valid_depth_image, depth_validity_mask, depth_guided_map);
#elif 0
		// METHOD #2: construct depth-guided map using edge detection & morphological operation.
		construct_depth_guided_map_using_edge_detection_and_morphological_operation(valid_depth_image, depth_validity_mask, depth_guided_map);
#elif 1
		// METHOD #3: construct depth-guided map using structure tensor.
		if (structure_tensor_mask.empty())
		{
			std::cerr << "structure tensor mask doesn't exist" << std::endl;
			continue;
		}
		else
			construct_depth_guided_map_using_structure_tensor(structure_tensor_mask, depth_guided_map);
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
		{
			std::ostringstream strm;
			cv::cvtColor(depth_guided_map, tmp_image, CV_GRAY2BGR);
			strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
		}
#endif

		// extract foreground.
#if 0
		// METHOD #1: segment foreground using Snake.
		{
			//const std::size_t NUM_SNAKE_POINTS = 50;
			const std::size_t NUM_SNAKE_POINTS = 0;

			cv::Mat gray_image;
			cv::cvtColor(rgb_input_image, gray_image, CV_BGR2GRAY);

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
		// METHOD #2: segment image by GrabCut algorithm.
		run_grabcut_using_depth_guided_mask(rgb_input_image, depth_guided_map);
#elif 1
		// METHOD #3: segment image by interactive graph-cuts segmentation algorithm.
		run_interactive_graph_cuts_segmentation(rgb_input_image, valid_depth_image, depth_guided_map);
#elif 0
		// METHOD #4: segment image by efficient graph-based image segmentation algorithm.
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

void segment_foreground_based_on_depth_guided_map()
{
#if 1
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
		// load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, &fx_rgb, &fy_rgb))
			continue;

		const int64 startTime = cv::getTickCount();

		// construct valid depth image.
		construct_valid_depth_image(useDepthRangeFiltering, depth_range_list[i].start, depth_range_list[i].end, depth_input_image, valid_depth_image, depth_validity_mask);

		// construct depth-guided map.
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
			strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
#endif
		}

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
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

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
	cv::Mat rgb_input_image, depth_input_image, structure_tensor_mask;
	cv::Mat valid_depth_image, depth_validity_mask(imageSize_rgb, CV_8UC1);
	cv::Mat depth_guided_map(imageSize_rgb, CV_8UC1), grabCut_mask(imageSize_mask, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, NULL, NULL))
			continue;
#if 1
		// METHOD #1: load structure tensor mask.

		if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], structure_tensor_mask))
			continue;
#elif 0
		// METHOD #2: compute structure tensor mask.

		// FIXME [implement] >>
#elif 0
		// METHOD #3: compute depth changing mask.

		// compute phase distribution from neighborhood
		{
			const int radius = 2;
			cv::Mat depth_changing_mask(imageSize_rgb, CV_8UC1, cv::Scalar::all(0))

			// FIXME [implement] >>
			compute_phase_distribution_from_neighborhood(depth_image, radius, depth_changing_mask);

#if 1
			// show depth changing mask.
			cv::imshow("depth changing mask", depth_changing_mask);
#endif
		}
#endif

		const int64 startTime = cv::getTickCount();

		// construct valid depth image.
		construct_valid_depth_image(useDepthRangeFiltering, depth_range_list[i].start, depth_range_list[i].end, depth_input_image, valid_depth_image, depth_validity_mask);

#if 1
		// show structure tensor mask.
		{
			cv::imshow("structure tensor mask", structure_tensor_mask);
		}
#endif

#if 0
		{
			std::ostringstream strm;
			strm << "../data/kinect_segmentation/structure_tensor_mask_" << i << ".png";
			cv::imwrite(strm.str(), structure_tensor_mask);
		}
#endif

		// construct depth-guided map.
		construct_depth_guided_map_using_structure_tensor(structure_tensor_mask, depth_guided_map);

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
			strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
		}
#endif

#if 0
		// segment foreground using Snake.
		{
			//const std::size_t NUM_SNAKE_POINTS = 50;
			const std::size_t NUM_SNAKE_POINTS = 0;

			cv::Mat gray_image;
			cv::cvtColor(rgb_input_image, gray_image, CV_BGR2GRAY);

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
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162309.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162314.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162348.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162459.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162525.tif");
	structure_tensor_mask_file_list.push_back("../data/kinect_segmentation/kinect_depth_rectified_valid_ev_ratio_20130614T162552.tif");

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
	cv::Mat rgb_input_image, depth_input_image, structure_tensor_mask;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	cv::Mat foreground_mask(imageSize_mask, CV_8UC1), background_mask(imageSize_mask, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, NULL, NULL))
			continue;
#if 1
		// METHOD #1: load structure tensor mask.

		if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], structure_tensor_mask))
			continue;
#elif 0
		// METHOD #2: compute structure tensor mask.

		// FIXME [implement] >>
#elif 0
		// METHOD #3: compute depth changing mask.

		// compute phase distribution from neighborhood
		{
			const int radius = 2;
			cv::Mat depth_changing_mask(imageSize_rgb, CV_8UC1, cv::Scalar::all(0))

			// FIXME [implement] >>
			compute_phase_distribution_from_neighborhood(depth_image, radius, depth_changing_mask);

#if 1
			// show depth changing mask.
			cv::imshow("depth changing mask", depth_changing_mask);
#endif
		}
#endif

		const int64 startTime = cv::getTickCount();

		// construct valid depth image.
		construct_valid_depth_image(useDepthRangeFiltering, depth_range_list[i].start, depth_range_list[i].end, depth_input_image, valid_depth_image, depth_validity_mask);

		// construct depth-guided map.
		construct_depth_guided_map_using_structure_tensor(structure_tensor_mask, depth_guided_map);

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
			strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
		}
#endif

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
	cv::Mat rgb_input_image, depth_input_image;
	cv::Mat depth_validity_mask(imageSize_rgb, CV_8UC1), valid_depth_image, depth_guided_map(imageSize_rgb, CV_8UC1);
	double minVal = 0.0, maxVal = 0.0;
	cv::Mat tmp_image;
	for (std::size_t i = 0; i < num_images; ++i)
	{
		// load images.
		if (!load_kinect_images(rgb_input_file_list[i], depth_input_file_list[i], useRectifiedImages, rgb_input_image, depth_input_image, NULL, NULL))
			continue;
#if 0
		// METHOD #1: load structure tensor mask.

		if (!load_structure_tensor_mask(structure_tensor_mask_file_list[i], structure_tensor_mask))
			continue;
#elif 0
		// METHOD #2: compute structure tensor mask.

		// FIXME [implement] >>
#elif 0
		// METHOD #3: compute depth changing mask.

		// compute phase distribution from neighborhood
		{
			const int radius = 2;
			cv::Mat depth_changing_mask(imageSize_rgb, CV_8UC1, cv::Scalar::all(0))

			// FIXME [implement] >>
			compute_phase_distribution_from_neighborhood(depth_image, radius, depth_changing_mask);

#if 1
			// show depth changing mask.
			cv::imshow("depth changing mask", depth_changing_mask);
#endif
		}
#endif

		const int64 startTime = cv::getTickCount();

		// construct valid depth image.
		construct_valid_depth_image(useDepthRangeFiltering, depth_range_list[i].start, depth_range_list[i].end, depth_input_image, valid_depth_image, depth_validity_mask);

		// construct depth-guided map.

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
			strm << "../data/kinect_segmentation/depth_guided_mask_" << i << ".png";
			cv::imwrite(strm.str(), tmp_image);
#endif
		}

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

		//swl::segment_foreground_based_on_depth_guided_map();
		//swl::segment_foreground_based_on_structure_tensor();

        // FIXME [implement] >> not completed
        //swl::segment_foreground_using_single_layered_graphical_model();
		//swl::segment_foreground_using_two_layered_graphical_model();

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
			swl::compute_phase_distribution_from_neighborhood(mat, radius, depth_changing_mask);
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
