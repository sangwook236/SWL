#if !defined(__SWL_RND_UTIL__HISTOGRAM_ACCUMULATOR__H_)
#define __SWL_RND_UTIL__HISTOGRAM_ACCUMULATOR__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <boost/circular_buffer.hpp>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------
//

class SWL_RND_UTIL_API HistogramAccumulator
{
public:
	//typedef HistogramAccumulator base_type;
    typedef cv::MatND histogram_type;

public:
	HistogramAccumulator(const size_t histogramNum);
	HistogramAccumulator(const std::vector<float> &weights);

private:
	HistogramAccumulator(const HistogramAccumulator &rhs);
	HistogramAccumulator & operator=(const HistogramAccumulator &rhs);

public:
	void addHistogram(const cv::MatND &hist)  {  histograms_.push_back(hist);  }
	void clearAllHistograms()  {  histograms_.clear();  }

	size_t getHistogramSize() const  {  return histograms_.size();  }
	bool isFull() const  {  return histograms_.full();  }

	cv::MatND createAccumulatedHistogram() const;
	cv::MatND createTemporalHistogram() const;
	cv::MatND createTemporalHistogram(const std::vector<histogram_type> &refHistograms, const double histDistThreshold) const;

private:
	const size_t histogramNum_;
	const std::vector<float> weights_;

	boost::circular_buffer<histogram_type> histograms_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HISTOGRAM_ACCUMULATOR__H_
