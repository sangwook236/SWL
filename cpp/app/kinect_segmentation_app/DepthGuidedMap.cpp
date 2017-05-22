//#include "stdafx.h"
#include "DepthGuidedMap.h"
//#include "nms.hpp"
#include "swl/machine_vision/NonMaximumSuppression.h"
#include "gslic_lib/FastImgSeg.h"
#include <boost/smart_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <iterator>


#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#define __USE_gSLIC 1
#else
#undef __USE_gSLIC
#endif

namespace {
namespace local {

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

}  // namespace local
}  // unnamed namespace

namespace swl {

#if defined(__USE_gSLIC)
// REF [file] >> gSLIC.cpp
void create_superpixel_by_gSLIC(const cv::Mat &input_image, cv::Mat &superpixel_mask, const SEGMETHOD seg_method, const double seg_weight, const int num_segments);
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary);
#else
void create_superpixel_by_gSLIC(const cv::Mat &input_image, cv::Mat &superpixel_mask, const SEGMETHOD seg_method, const double seg_weight, const int num_segments)
{
    throw std::runtime_error("gSLIC not supported");
}
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary)
{
    throw std::runtime_error("gSLIC not supported");
}
#endif

// REF [file] >> Util.cpp
bool simple_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int pixVal, std::vector<cv::Point> &convexHull);
void canny(const cv::Mat &gray, const int lowerEdgeThreshold, const int upperEdgeThreshold, const bool useL2, cv::Mat &edge);
void zhang_suen_thinning_algorithm(const cv::Mat &src, cv::Mat &dst);
void guo_hall_thinning_algorithm(cv::Mat &im);

void construct_depth_guided_map_using_superpixel(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	double minVal = 0.0, maxVal = 0.0;
	cv::Mat depth_boundary_image, tmp_image;

	// Extract boundary from depth image by edge detector.
	{
		cv::minMaxLoc(depth_image, &minVal, &maxVal);
		depth_image.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

		const int lowerEdgeThreshold = 5, upperEdgeThreshold = 50;
		const bool useL2 = true;
		canny(tmp_image, lowerEdgeThreshold, upperEdgeThreshold, useL2, depth_boundary_image);

		//cv::dilate(depth_boundary_image, depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if 1
		// Show depth boundary image.
		cv::imshow("Depth boundary by Canny", depth_boundary_image);
#endif
	}

	cv::Mat rgb_superpixel_mask;
	cv::Mat filtered_superpixel_mask(rgb_image.size(), CV_8UC1, cv::Scalar::all(255)), filtered_superpixel_indexes(rgb_image.size(), CV_32SC1, cv::Scalar::all(0));

	// PPP [] >>
	//	1. Run superpixel.

	// Superpixel mask consists of segment indexes.
	const int num_segments = 2500;
	const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC.
	const double seg_weight = 0.3;
	create_superpixel_by_gSLIC(rgb_image, rgb_superpixel_mask, seg_method, seg_weight, num_segments);

#if 0
	// Show superpixel mask.
	{
		cv::minMaxLoc(rgb_superpixel_mask, &minVal, &maxVal);
		rgb_superpixel_mask.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("Superpixels by gSLIC - mask", tmp_image);
	}
#endif

#if 0
	// Show superpixel boundary.
	{
		cv::Mat rgb_superpixel_boundary;
		create_superpixel_boundary(rgb_superpixel_mask, rgb_superpixel_boundary);

		rgb_image.copyTo(tmp_image);
		tmp_image.setTo(cv::Scalar(0, 0, 255), rgb_superpixel_boundary);

		cv::imshow("Superpixels by gSLIC - boundary", tmp_image);
	}
#endif

	// PPP [] >>
	//	2. Depth info로부터 관심 영역의 boundary를 얻음.
	//		Depth histogram을 이용해 depth region을 분할 => 물체의 경계에 의해서가 아니라 depth range에 의해서 영역이 결정. 전체적으로 연결된 몇 개의 큰 blob이 생성됨.
	//		Depth image의 edge 정보로부터 boundary 추출 => 다른 두 물체가 맞닿아 있는 경우, depth image의 boundary info로부터 접촉면을 식별하기 어려움.

	// FIXME [enhance] >> Too slow. speed up.
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

#if 0
		// Show filtered superpixel index mask.
		cv::imshow("Mask of superpixels on depth boundary", filtered_superpixel_mask);
#endif
	}

	// Construct depth-guided map.
	depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
	depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), depth_validity_mask & filtered_superpixel_mask);  // Valid depth region (foreground).
	depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), ~depth_validity_mask & filtered_superpixel_mask);  // Invalid depth region (background).
}

void construct_depth_guided_map_using_edge_detection_and_morphological_operation(const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	double minVal = 0.0, maxVal = 0.0;
	cv::Mat depth_boundary_image, tmp_image;

	// Extract boundary from depth image by edge detector.
	{
		cv::minMaxLoc(depth_image, &minVal, &maxVal);
		depth_image.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

		const int lowerEdgeThreshold = 5, upperEdgeThreshold = 50;
		const bool useL2 = true;
		canny(tmp_image, lowerEdgeThreshold, upperEdgeThreshold, useL2, depth_boundary_image);

		cv::dilate(depth_boundary_image, depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if 1
		// Show depth boundary mask.
		cv::imshow("Depth boundary mask", depth_boundary_image);
#endif
	}

	// Construct depth-guided map.
	depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
	depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), depth_validity_mask & ~depth_boundary_image);  // Valid depth region (foreground).
	depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), ~depth_validity_mask & ~depth_boundary_image);  // Invalid depth region (background).
}

void construct_depth_guided_map_using_depth_variation(const cv::Mat &depth_variation_mask, const cv::Mat &depth_input_image, cv::Mat &depth_guided_map, cv::Mat &filtered_depth_variation_mask)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	// Pre-process depth variation mask.
	cv::Mat truncated_depth_variation_mask;
	{
		truncated_depth_variation_mask = depth_variation_mask > 0.05;  // CV_8UC1.
		//truncated_depth_variation_mask = 0.05 < depth_variation_mask & depth_variation_mask < 0.5;  // CV_8UC1.

		cv::dilate(truncated_depth_variation_mask, truncated_depth_variation_mask, selement3, cv::Point(-1, -1), 3);
		cv::erode(truncated_depth_variation_mask, truncated_depth_variation_mask, selement3, cv::Point(-1, -1), 3);

		//cv::erode(truncated_depth_variation_mask, truncated_depth_variation_mask, selement3, cv::Point(-1, -1), 3);
		//cv::dilate(truncated_depth_variation_mask, truncated_depth_variation_mask, selement3, cv::Point(-1, -1), 3);
	}

#if 0
	// Show truncated depth variation mask.
	cv::imshow("Truncated depth variation mask", truncated_depth_variation_mask);
#endif

	const bool use_color_processed_depth_variation_mask = false;
	cv::Mat processed_depth_variation_mask(depth_variation_mask.size(), use_color_processed_depth_variation_mask ? CV_8UC3 : CV_8UC1);
	const double MIN_CONTOUR_AREA = 200.0;
	cv::Mat contour_image, foreground_mask(depth_variation_mask.size(), CV_8UC1), background_mask(depth_variation_mask.size(), CV_8UC1);

	// Find contours.
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	{
		truncated_depth_variation_mask.copyTo(contour_image);

		std::vector<std::vector<cv::Point> > contours2;
		cv::findContours(contour_image, contours2, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

#if 0
		// Comment this out if you do not want approximation.
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
		// Find all contours.

		processed_depth_variation_mask.setTo(cv::Scalar::all(0));

		// Iterate through all the top-level contours,
		// draw each connected component with its own random color.
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			if (cv::contourArea(cv::Mat(contours[idx])) < MIN_CONTOUR_AREA) continue;

			if (use_color_processed_depth_variation_mask)
				//cv::drawContours(processed_depth_variation_mask, contours, idx, cv::Scalar(std::rand() & 255, std::rand() & 255, std::rand() & 255), cv::FILLED, cv::LINE_8, hierarchy, 0, cv::Point());
				cv::drawContours(processed_depth_variation_mask, contours, idx, cv::Scalar(std::rand() & 255, std::rand() & 255, std::rand() & 255), cv::FILLED, cv::LINE_8, cv::noArray(), 0, cv::Point());
			else
				//cv::drawContours(processed_depth_variation_mask, contours, idx, cv::Scalar::all(255), cv::FILLED, cv::LINE_8, hierarchy, 0, cv::Point());
				cv::drawContours(processed_depth_variation_mask, contours, idx, cv::Scalar::all(255), cv::FILLED, cv::LINE_8, cv::noArray(), 0, cv::Point());
		}
#elif 1
		// Find a contour with max area.

		std::vector<std::vector<cv::Point> > pointSets;
		std::vector<cv::Vec4i> hierarchy0;
		if (!hierarchy.empty())
			std::transform(hierarchy.begin(), hierarchy.end(), std::back_inserter(hierarchy0), local::IncreaseHierarchyOp(pointSets.size()));

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

			processed_depth_variation_mask.setTo(cv::Scalar::all(0));
			if ((size_t)-1 != maxAreaIdx)
				//cv::drawContours(processed_depth_variation_mask, pointSets, maxAreaIdx, cv::Scalar::all(255), CV_FILLED, 8, hierarchy0, 0, cv::Point());
				//cv::drawContours(processed_depth_variation_mask, pointSets, maxAreaIdx, cv::Scalar::all(255), CV_FILLED, 8, hierarchy0, maxLevel, cv::Point());
				cv::drawContours(processed_depth_variation_mask, pointSets, maxAreaIdx, cv::Scalar::all(255), CV_FILLED, 8, cv::noArray(), 0, cv::Point());
#endif
		}
#endif

		// Post-process depth variation mask.
		{
			cv::morphologyEx(processed_depth_variation_mask, processed_depth_variation_mask, cv::MORPH_CLOSE, selement3, cv::Point(-1, -1), 3);
			//cv::imshow("Post-processed depth variation mask 1", processed_depth_variation_mask);

			cv::morphologyEx(processed_depth_variation_mask, processed_depth_variation_mask, cv::MORPH_OPEN, selement3, cv::Point(-1, -1), 3);
			//cv::imshow("Post-processed depth variation mask 2", processed_depth_variation_mask);

			//cv::erode(processed_depth_variation_mask, processed_depth_variation_mask, selement3, cv::Point(-1, -1), 1);
			//cv::dilate(processed_depth_variation_mask, processed_depth_variation_mask, selement3, cv::Point(-1, -1), 1);

#if 0
			// Show post-processed depth variation mask.
			cv::imshow("Post-processed depth variation mask", processed_depth_variation_mask);
#endif
		}

#if 0
		filtered_depth_variation_mask = cv::Mat::zeros(truncated_depth_variation_mask.size(), truncated_depth_variation_mask.type());
		truncated_depth_variation_mask.copyTo(filtered_depth_variation_mask, processed_depth_variation_mask);
#else
		filtered_depth_variation_mask = processed_depth_variation_mask;
#endif

		{
			cv::Mat tmp(depth_input_image.size(), depth_input_image.type(), cv::Scalar::all(0));
			depth_input_image.copyTo(tmp, filtered_depth_variation_mask);

			const cv::Scalar mean, stdDev;
			cv::meanStdDev(tmp, mean, stdDev, filtered_depth_variation_mask > 0);

			const double sigmaRatio = 0.5;
			tmp.setTo(cv::Scalar::all(0), mean[0] - sigmaRatio * stdDev[0] < tmp & tmp > mean[0] + sigmaRatio * stdDev[0]);

			filtered_depth_variation_mask = tmp > 0;

			//cv::morphologyEx(filtered_depth_variation_mask, filtered_depth_variation_mask, cv::MORPH_CLOSE, selement3, cv::Point(-1, -1), 3);
			//cv::morphologyEx(filtered_depth_variation_mask, filtered_depth_variation_mask, cv::MORPH_OPEN, selement3, cv::Point(-1, -1), 3);
		}

		// Create foreground & background masks.
#if 0
		// METHOD #1: using dilation & erosion.

		cv::erode(filtered_depth_variation_mask, foreground_mask, selement5, cv::Point(-1, -1), 3);
		cv::dilate(filtered_depth_variation_mask, background_mask, selement5, cv::Point(-1, -1), 3);
		foreground_mask = foreground_mask > 0;
		background_mask = 0 == background_mask;

		// Construct depth-guided map.
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
		depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), foreground_mask);  // Valid depth region (foreground).
		depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), background_mask);  // Invalid depth region (background).
#elif 1
		// METHOD #2: using distance transform for foreground and dilation for background.

		cv::Mat dist32f;
		//const int distanceType = cv::DIST_C;  // C/Inf metric.
		const int distanceType = cv::DIST_L1;  // L1 metric.
		//const int distanceType = cv::DIST_L2;  // L2 metric.
		//const int maskSize = cv::DIST_MASK_3;
		const int maskSize = cv::DIST_MASK_5;
		//const int maskSize = cv::DIST_MASK_PRECISE;
		cv::distanceTransform(filtered_depth_variation_mask, dist32f, distanceType, maskSize);

#if 0
		{
			double minVal, maxVal;
			cv::minMaxLoc(dist32f, &minVal, &maxVal);
			cv::Mat tmp_image;
			dist32f.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("Distance transform of foreground mask", tmp_image);
		}
#endif

#if 1
		foreground_mask = dist32f >= 5.0f;
		cv::erode(foreground_mask, foreground_mask, selement3, cv::Point(-1, -1), 3);
#elif 0
		// FIXME [fix] >> Not correctly working.
		NonMaximumSuppression::computeNonMaximumSuppression(dist32f, foreground_mask);
		//NonMaximumSuppression::findMountainChain(dist32f, foreground_mask);
#elif 0
		const int winSize = 10;  // The size of the window.
		NonMaximumSuppression::computeNonMaximumSuppression(dist32f, winSize, foreground_mask);
#endif

		cv::dilate(filtered_depth_variation_mask, background_mask, selement5, cv::Point(-1, -1), 7);
		background_mask = 0 == background_mask;

		cv::Mat background_info_mask;
		cv::erode(background_mask, background_info_mask, selement5, cv::Point(-1, -1), 5);
		background_mask.setTo(cv::Scalar::all(0), background_info_mask);

		// Construct depth-guided map.
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
		depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), foreground_mask);  // Valid depth region (foreground).
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_BGD), background_mask);  // Invalid depth region (background).
		depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), background_info_mask);  // Invalid depth region (background).
#elif 0
		// METHOD #3: using distance transform for foreground and convex hull for background.

		cv::Mat dist32f;
		const int distanceType = cv::_DIST_C;  // C/Inf metric.
		//const int distanceType = cv::DIST_L1;  // L1 metric.
		//const int distanceType = cv::DIST_L2;  // L2 metric.
		//const int maskSize = cv::DIST_MASK_3;
		//const int maskSize = cv::DIST_MASK_5;
		const int maskSize = cv::DIST_MASK_PRECISE;
		cv::distanceTransform(filtered_depth_variation_mask, dist32f, distanceType, maskSize);

		{
			double minVal, maxVal;
			cv::minMaxLoc(dist32f, &minVal, &maxVal);
			cv::Mat tmp_image;
			dist32f.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
			cv::imshow("Distance transform of foreground mask", tmp_image);
		}

		foreground_mask = dist32f >= 5.0f;

		std::vector<cv::Point> convexHull;
		simple_convex_hull(filtered_depth_variation_mask, cv::Rect(), 255, convexHull);

		background_mask = cv::Mat::ones(background_mask.size(), background_mask.type()) * 255;
		std::vector<std::vector<cv::Point> > contours;
		contours.push_back(convexHull);
		cv::drawContours(background_mask, contours, 0, cv::Scalar(0), cv::FILLED, cv::LINE_8);

		cv::Mat background_info_mask;
		cv::erode(background_mask, background_mask, selement5, cv::Point(-1, -1), 3);
		cv::erode(background_mask, background_info_mask, selement5, cv::Point(-1, -1), 5);
		background_mask.setTo(cv::Scalar::all(0), background_info_mask);

		// Construct depth-guided map.
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
		depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), foreground_mask);  // Valid depth region (foreground).
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_BGD), background_mask);  // Invalid depth region (background).
		depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), background_info_mask);  // Invalid depth region (background).
#elif 0
		// METHOD #4: using thinning for foreground and dilation for background.

		cv::Mat bw;
		cv::threshold(filtered_depth_variation_mask, bw, 50, 255, CV_THRESH_BINARY);
		zhang_suen_thinning_algorithm(bw, foreground_mask);
		//foreground_mask = bw.clone();
		//guo_hall_thinning_algorithm(foreground_mask);

		cv::dilate(foreground_mask, foreground_mask, selement5, cv::Point(-1, -1), 3);

		//cv::imshow("Thinning of foreground mask", foreground_mask);

		cv::dilate(filtered_depth_variation_mask, background_mask, selement5, cv::Point(-1, -1), 5);
		background_mask = 0 == background_mask;

		// Construct depth-guided map.
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
		depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), foreground_mask);  // Valid depth region (foreground).
		depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), background_mask);  // Invalid depth region (background).
#elif 0
		// METHOD #5: using thinning for foreground and convex hull for background.

		cv::Mat bw;
		cv::threshold(filtered_depth_variation_mask, bw, 10, 255, CV_THRESH_BINARY);
		zhang_suen_thinning_algorithm(bw, foreground_mask);
		//foreground_mask = bw.clone();
		//guo_hall_thinning_algorithm(foreground_mask);

		//cv::dilate(foreground_mask, foreground_mask, selement5, cv::Point(-1, -1), 3);

		cv::imshow("Thinning of foreground mask", foreground_mask);

		std::vector<cv::Point> convexHull;
		simple_convex_hull(filtered_depth_variation_mask, cv::Rect(), 255, convexHull);

		background_mask = cv::Mat::ones(background_mask.size(), background_mask.type()) * 255;
		std::vector<std::vector<cv::Point> > contours;
		contours.push_back(convexHull);
		cv::drawContours(background_mask, contours, 0, cv::Scalar(0), cv::FILLED, cv::LINE_8);

		// Construct depth-guided map.
		depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // Depth boundary region.
		depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), foreground_mask);  // Valid depth region (foreground).
		depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), background_mask);  // Invalid depth region (background).
#endif
	}

#if 0
	// Show foreground & background masks.
	cv::imshow("Foreground mask", foreground_mask);
	cv::imshow("Background mask", background_mask);
#endif
}

}  // namespace swl
