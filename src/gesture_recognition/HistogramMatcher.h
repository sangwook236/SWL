#if !defined(__SWL_GESTURE_RECOGNITION__HISTOGRAM_MATCHER__H_)
#define __SWL_GESTURE_RECOGNITION__HISTOGRAM_MATCHER__H_ 1


#include <vector>

namespace cv {

class Mat;
typedef Mat MatND;

}


namespace swl {

//-----------------------------------------------------------------------------
//

struct HistogramMatcher
{
	static size_t match(const std::vector<const cv::MatND> &refHistograms, const cv::MatND &hist, double &minDist);
};

}  // namespace swl


#endif  // __SWL_GESTURE_RECOGNITION__HISTOGRAM_MATCHER__H_
