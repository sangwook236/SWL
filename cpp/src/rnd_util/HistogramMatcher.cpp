#include "swl/rnd_util/HistogramMatcher.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace swl {

//-----------------------------------------------------------------------------
//

/*static*/ size_t HistogramMatcher::match(const std::vector<histogram_type> &refHistograms, const cv::MatND &hist, double &minDist)
{
	std::vector<double> dists;
	dists.reserve(refHistograms.size());
	for (std::vector<histogram_type>::const_iterator it = refHistograms.begin(); it != refHistograms.end(); ++it)
		// Correlation: cv::HISTCMP_CORREL.
		// Chi-square statistic: cv::HISTCMP_CHISQR.
		// Intersection: cv::HISTCMP_INTERSECT.
		// Bhattacharyya distance: cv::HISTCMP_BHATTACHARYYA.
		dists.push_back(cv::compareHist(hist, *it, cv::HISTCMP_BHATTACHARYYA));

	std::vector<double>::iterator itMin = std::min_element(dists.begin(), dists.end());
	minDist = *itMin;
	return (size_t)std::distance(dists.begin(), itMin);
}

}  // namespace swl
