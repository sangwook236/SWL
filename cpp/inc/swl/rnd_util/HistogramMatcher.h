#if !defined(__SWL_RND_UTIL__HISTOGRAM_MATCHER__H_)
#define __SWL_RND_UTIL__HISTOGRAM_MATCHER__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <vector>

namespace cv {

class Mat;
typedef Mat MatND;

}


namespace swl {

//-----------------------------------------------------------------------------
//

struct SWL_RND_UTIL_API HistogramMatcher
{
#if defined(__GNUC__)
    typedef cv::MatND histogram_type;
#else
    typedef const cv::MatND histogram_type;
#endif

	static std::size_t match(const std::vector<histogram_type> &refHistograms, const cv::MatND &hist, double &minDist);
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HISTOGRAM_MATCHER__H_
