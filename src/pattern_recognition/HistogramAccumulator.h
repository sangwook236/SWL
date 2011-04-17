#if !defined(__SWL_PATTERN_RECOGNITION__HISTOGRAM_ACCUMULATOR__H_)
#define __SWL_PATTERN_RECOGNITION__HISTOGRAM_ACCUMULATOR__H_ 1


#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <boost/circular_buffer.hpp>
#include <vector>


namespace swl {

//-----------------------------------------------------------------------------
//

class HistogramAccumulator
{
public:
	//typedef HistogramAccumulator base_type;

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
	cv::MatND createTemporalHistogram(const std::vector<const cv::MatND> &refHistograms, const double histDistThreshold) const;

private:
	const size_t histogramNum_;
	const std::vector<float> weights_;

	boost::circular_buffer<const cv::MatND> histograms_;
};

}  // namespace swl


#endif  // __SWL_PATTERN_RECOGNITION__HISTOGRAM_ACCUMULATOR__H_
