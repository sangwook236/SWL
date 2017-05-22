#include "swl/Config.h"
#include "swl/machine_vision/BoundaryExtraction.h"
#include <set>
#include <vector>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// Boundary Extraction.

/*virtual*/ IBoundaryExtraction::~IBoundaryExtraction()
{}

//--------------------------------------------------------------------------
// Naive Boundary Extraction.

NaiveBoundaryExtraction::NaiveBoundaryExtraction(const bool use8connectivity /*= true*/)
: use8connectivity_(use8connectivity)
{}

/*virtual*/ void NaiveBoundaryExtraction::extractBoundary(const cv::Mat &label, cv::Mat &boundary) const /*override*/
{
	std::vector<cv::Point> neighbors;
	if (use8connectivity_)
	{
		// 8-connectivity.
		neighbors.reserve(8);
		neighbors.push_back(cv::Point(1, 0));
		neighbors.push_back(cv::Point(1, -1));
		neighbors.push_back(cv::Point(0, -1));
		neighbors.push_back(cv::Point(-1, -1));
		neighbors.push_back(cv::Point(-1, 0));
		neighbors.push_back(cv::Point(-1, 1));
		neighbors.push_back(cv::Point(0, 1));
		neighbors.push_back(cv::Point(1, 1));
	}
	else
	{
		// 4-connectivity.
		neighbors.reserve(4);
		neighbors.push_back(cv::Point(1, 0));
		neighbors.push_back(cv::Point(0, -1));
		neighbors.push_back(cv::Point(-1, 0));
		neighbors.push_back(cv::Point(0, 1));
	}

	const cv::Rect rct(0, 0, label.cols, label.rows);
	for (int r = 0; r < label.rows; ++r)
	{
		for (int c = 0; c < label.cols; ++c)
		{
			std::set<unsigned short> neighborLabels;
			for (const auto &neighbor : neighbors)
			{
				const cv::Point pt(c + neighbor.x,  r + neighbor.y);
				if (rct.contains(pt))
					neighborLabels.insert(label.at<unsigned short>(pt.y, pt.x));
			}
			if (neighborLabels.size() > 1)
				boundary.at<unsigned short>(r, c) = label.at<unsigned short>(r, c);
		}
	}
}

//--------------------------------------------------------------------------
// Contour Boundary Extraction.

ContourBoundaryExtraction::ContourBoundaryExtraction()
{}

/*virtual*/ void ContourBoundaryExtraction::extractBoundary(const cv::Mat &label, cv::Mat &boundary) const /*override*/
{
	double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(label, &minVal, &maxVal);
	const unsigned short minLabel = (unsigned short)std::floor(minVal + 0.5), maxLabel = (unsigned short)std::floor(maxVal + 0.5);

#if 1
	cv::Mat binary(cv::Mat::zeros(label.size(), CV_8UC1));
	for (unsigned short lbl = minLabel; lbl <= maxLabel; ++lbl)
	{
		binary.setTo(cv::Scalar::all(0));
		binary.setTo(cv::Scalar::all(255), label == lbl);

		if (0 == cv::countNonZero(binary)) continue;

		// Find contours.
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(binary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		//cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		cv::drawContours(boundary, contours, -1, cv::Scalar::all(lbl), 1, cv::LINE_8, hierarchy, INT_MAX, cv::Point());
	}
#else
	cv::Mat binary(cv::Mat::zeros(label.size(), CV_8UC1));
	for (unsigned short lbl = minLabel; lbl <= maxLabel; ++lbl)
		binary.setTo(cv::Scalar::all(lbl), label == lbl);

	// Find contours.
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	//cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	int idx = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
		cv::drawContours(boundary, contours, idx, cv::Scalar::all(idx + 1), 1, cv::LINE_8, hierarchy, INT_MAX, cv::Point());
#endif
}
	
}  // namespace swl
