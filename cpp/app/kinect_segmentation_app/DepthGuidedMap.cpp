//#include "stdafx.h"
#include "DepthGuidedMap.h"
#include "gslic_lib/FastImgSeg.h"
#include <boost/smart_ptr.hpp>
#include <boost/timer/timer.hpp>


#if defined(_WIN32) || defined(WIN32)
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
// [ref] gSLIC.cpp
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

// [ref] Util.cpp
bool simple_convex_hull(const cv::Mat &img, const cv::Rect &roi, const int pixVal, std::vector<cv::Point> &convexHull);
void canny(const cv::Mat &gray, cv::Mat &edge);

void construct_depth_guided_map_using_superpixel(const cv::Mat &rgb_image, const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	double minVal = 0.0, maxVal = 0.0;
	cv::Mat depth_boundary_image, tmp_image;

	// extract boundary from depth image by edge detector.
	{
		cv::minMaxLoc(depth_image, &minVal, &maxVal);
		depth_image.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

		canny(tmp_image, depth_boundary_image);

		//cv::dilate(depth_boundary_image, depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if 1
		// show depth boundary image.
		cv::imshow("depth boundary by Canny", depth_boundary_image);
#endif
	}

	cv::Mat rgb_superpixel_mask;
	cv::Mat filtered_superpixel_mask(rgb_image.size(), CV_8UC1, cv::Scalar::all(255)), filtered_superpixel_indexes(rgb_image.size(), CV_32SC1, cv::Scalar::all(0));

	// PPP [] >>
	//	1. run superpixel.

	// superpixel mask consists of segment indexes.
	const int num_segments = 2500;
	const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC
	const double seg_weight = 0.3;
	create_superpixel_by_gSLIC(rgb_image, rgb_superpixel_mask, seg_method, seg_weight, num_segments);

#if 0
	// show superpixel mask.
	{
		cv::minMaxLoc(rgb_superpixel_mask, &minVal, &maxVal);
		rgb_superpixel_mask.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("superpixels by gSLIC - mask", tmp_image);
	}
#endif

#if 0
	// show superpixel boundary.
	{
		cv::Mat rgb_superpixel_boundary;
		swl::create_superpixel_boundary(rgb_superpixel_mask, rgb_superpixel_boundary);

		rgb_image.copyTo(tmp_image);
		tmp_image.setTo(cv::Scalar(0, 0, 255), rgb_superpixel_boundary);

		cv::imshow("superpixels by gSLIC - boundary", tmp_image);
	}
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

#if 0
		// show filtered superpixel index mask.
		cv::imshow("mask of superpixels on depth boundary", filtered_superpixel_mask);
#endif
	}

	// construct depth-guided map.
	depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // depth boundary region.
	depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), depth_validity_mask & filtered_superpixel_mask);  // valid depth region (foreground).
	depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), ~depth_validity_mask & filtered_superpixel_mask);  // invalid depth region (background).
}

void construct_depth_guided_map_using_morphological_operation_of_depth_boundary(const cv::Mat &depth_image, const cv::Mat &depth_validity_mask, cv::Mat &depth_guided_map)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	double minVal = 0.0, maxVal = 0.0;
	cv::Mat depth_boundary_image, tmp_image;

	// extract boundary from depth image by edge detector.
	{
		cv::minMaxLoc(depth_image, &minVal, &maxVal);
		depth_image.convertTo(tmp_image, CV_8UC1, 255.0 / maxVal, 0.0);

		canny(tmp_image, depth_boundary_image);

		cv::dilate(depth_boundary_image, depth_boundary_image, selement3, cv::Point(-1, -1), 3);

#if 1
		// show depth boundary mask.
		cv::imshow("depth boundary mask", depth_boundary_image);
#endif
	}

	// construct depth-guided map.
	depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // depth boundary region.
	depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), depth_validity_mask & ~depth_boundary_image);  // valid depth region (foreground).
	depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), ~depth_validity_mask & ~depth_boundary_image);  // invalid depth region (background).
}

void construct_depth_guided_map_using_structure_tensor(const cv::Mat &structure_tensor_mask, cv::Mat &depth_guided_map)
{
	const cv::Mat &selement3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	const cv::Mat &selement5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	const cv::Mat &selement7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

	// pre-process structure tensor mask.
	cv::Mat structure_tensor_mask2;
	{
		structure_tensor_mask2 = structure_tensor_mask > 0.05;  // CV_8UC1

		cv::dilate(structure_tensor_mask2, structure_tensor_mask2, selement3, cv::Point(-1, -1), 3);
		cv::erode(structure_tensor_mask2, structure_tensor_mask2, selement3, cv::Point(-1, -1), 3);

		//cv::erode(structure_tensor_mask2, structure_tensor_mask2, selement3, cv::Point(-1, -1), 3);
		//cv::dilate(structure_tensor_mask2, structure_tensor_mask2, selement3, cv::Point(-1, -1), 3);
	}

	const bool use_color_processed_structure_tensor_mask = false;
	cv::Mat processed_structure_tensor_mask(structure_tensor_mask.size(), use_color_processed_structure_tensor_mask ? CV_8UC3 : CV_8UC1);
	const double MIN_CONTOUR_AREA = 200.0;
	cv::Mat contour_image, foreground_mask(structure_tensor_mask.size(), CV_8UC1), background_mask(structure_tensor_mask.size(), CV_8UC1);
	cv::Mat tmp_image;
	double minVal, maxVal;

	// find contours.
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	{
		structure_tensor_mask2.copyTo(contour_image);

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
		//cv::dilate(processed_structure_tensor_mask, processed_structure_tensor_mask, selement3, cv::Point(-1, -1), 1);

#if 0
		// show post-processed structure tensor mask.
		{
			cv::imshow("post-processed structure tensor mask", processed_structure_tensor_mask);
		}
#endif

		// create foreground & background masks.
#if 0
		// METHOD #1: using dilation & erosion.

		tmp_image = cv::Mat::zeros(structure_tensor_mask2.size(), structure_tensor_mask2.type());
		structure_tensor_mask2.copyTo(tmp_image, processed_structure_tensor_mask);
		cv::erode(tmp_image, foreground_mask, selement5, cv::Point(-1, -1), 3);
		cv::dilate(tmp_image, background_mask, selement5, cv::Point(-1, -1), 3);
		foreground_mask = foreground_mask > 0;
		background_mask = 0 == background_mask;
#elif 1
		// METHOD #2: using distance transform for foreground and convex hull for background.

		tmp_image = cv::Mat::zeros(structure_tensor_mask2.size(), structure_tensor_mask2.type());
		structure_tensor_mask.copyTo(tmp_image, processed_structure_tensor_mask);

		cv::Mat dist32f;
		const int distanceType = CV_DIST_C;  // C/Inf metric
		//const int distanceType = CV_DIST_L1;  // L1 metric
		//const int distanceType = CV_DIST_L2;  // L2 metric
		//const int maskSize = CV_DIST_MASK_3;
		//const int maskSize = CV_DIST_MASK_5;
		const int maskSize = CV_DIST_MASK_PRECISE;
		cv::distanceTransform(tmp_image, dist32f, distanceType, maskSize);
		foreground_mask = dist32f >= 7.5f;

		std::vector<cv::Point> convexHull;
		simple_convex_hull(tmp_image, cv::Rect(), 255, convexHull);

		background_mask = cv::Mat::ones(background_mask.size(), background_mask.type()) * 255;
		std::vector<std::vector<cv::Point> > contours;
		contours.push_back(convexHull);
		cv::drawContours(background_mask, contours, 0, cv::Scalar(0), CV_FILLED, 8);

		cv::erode(background_mask, background_mask, selement5, cv::Point(-1, -1), 3);
		cv::Mat bg_mask;
		cv::erode(background_mask, bg_mask, selement5, cv::Point(-1, -1), 5);
		background_mask.setTo(cv::Scalar::all(0), bg_mask);

		cv::minMaxLoc(dist32f, &minVal, &maxVal);
		dist32f.convertTo(tmp_image, CV_32FC1, 1.0 / maxVal, 0.0);
		cv::imshow("distance transform of foreground mask", tmp_image);
#elif 0
		// METHOD #3: using thinning for foreground and convex hull for background.

		tmp_image = cv::Mat::zeros(structure_tensor_mask.size(), structure_tensor_mask.type());
		structure_tensor_mask.copyTo(tmp_image, processed_structure_tensor_mask);

		cv::Mat bw;
		cv::threshold(tmp_image, bw, 10, 255, CV_THRESH_BINARY);
		zhang_suen_thinning_algorithm(bw, foreground_mask);
		//guo_hall_thinning_algorithm(bw, foreground_mask);

		std::vector<cv::Point> convexHull;
		simple_convex_hull(tmp_image, cv::Rect(), 255, convexHull);

		background_mask = cv::Mat::ones(background_mask.size(), background_mask.type()) * 255;
		std::vector<std::vector<cv::Point> > contours;
		contours.push_back(convexHull);
		cv::drawContours(background_mask, contours, 0, cv::Scalar(0), CV_FILLED, 8);

		cv::imshow("thinning of foreground mask", foreground_mask);
#endif
	}

#if 0
	// show foreground & background masks.
	{
		cv::imshow("foreground mask", foreground_mask);
		cv::imshow("background mask", background_mask);
	}
#endif

	// construct depth-guided map.
	depth_guided_map.setTo(cv::Scalar::all(SWL_PR_FGD));  // depth boundary region.
	depth_guided_map.setTo(cv::Scalar::all(SWL_FGD), foreground_mask);  // valid depth region (foreground).
	depth_guided_map.setTo(cv::Scalar::all(SWL_BGD), background_mask);  // invalid depth region (background).
}

}  // namespace swl
