#if !defined(__SWL_RND_UTIL__HISTOGRAM_UTIL__H_)
#define __SWL_RND_UTIL__HISTOGRAM_UTIL__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <vector>

namespace cv {

class Mat;
typedef Mat MatND;

}


namespace swl {

//-----------------------------------------------------------------------------
//

struct SWL_RND_UTIL_API HistogramUtil
{
	static void normalizeHistogram(cv::MatND &hist, const double factor);
	static void drawHistogram1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg);
	static void drawHistogram2D(const cv::MatND &hist, const int horzBinCount, const int vertBinCount, const double maxVal, const int horzBinSize, const int vertBinSize, cv::Mat &histImg);
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HISTOGRAM_UTIL__H_
