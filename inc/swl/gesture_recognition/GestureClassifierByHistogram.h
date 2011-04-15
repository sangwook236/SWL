#if !defined(__SWL_GESTURE_RECOGNITION__GESTURE_CLASSIFIER_BY_HISTOGRAM__H_)
#define __SWL_GESTURE_RECOGNITION__GESTURE_CLASSIFIER_BY_HISTOGRAM__H_ 1


#include "swl/gesture_recognition/IGestureClassifier.h"
#include <boost/circular_buffer.hpp>
#include <boost/smart_ptr.hpp>
#include <vector>

namespace cv {

class Mat;
typedef Mat MatND;

}

namespace swl {

class HistogramAccumulator;

//-----------------------------------------------------------------------------
//

class GestureClassifierByHistogram: public IGestureClassifier
{
public:
	struct Params
	{
		size_t ACCUMULATED_HISTOGRAM_NUM_FOR_SHORT_TIME_GESTURE;
		size_t ACCUMULATED_HISTOGRAM_NUM_FOR_LONG_TIME_GESTURE;
		size_t ACCUMULATED_HISTOGRAM_NUM_FOR_THIRD_CLASS_GESTURE;
		size_t MAX_MATCHED_HISTOGRAM_NUM;

		double histDistThresholdForShortTimeGesture;
		//double histDistThresholdForShortTimeGesture_LeftMove;
		//double histDistThresholdForShortTimeGesture_Others;
		double histDistThresholdForLongTimeGesture;
		double histDistThresholdForThirdClassGesture;

		double histDistThresholdForGestureIdPattern;

		size_t matchedIndexCountThresholdForShortTimeGesture;
		size_t matchedIndexCountThresholdForLongTimeGesture;
		size_t matchedIndexCountThresholdForThirdClassGesture;

		bool doesApplyMagnitudeFiltering;
		double magnitudeFilteringMinThresholdRatio;
		double magnitudeFilteringMaxThresholdRatio;
		bool doesApplyTimeWeighting;
		bool doesApplyMagnitudeWeighting;
	};

public:
	typedef IGestureClassifier base_type;

public:
	GestureClassifierByHistogram(const Params &params);
	explicit GestureClassifierByHistogram(const GestureClassifierByHistogram &rhs);
	~GestureClassifierByHistogram();

	GestureClassifierByHistogram & operator=(const GestureClassifierByHistogram &rhs);

public:
	/*virtual*/ bool analyzeOpticalFlow(const cv::Rect &roi, const cv::Mat &flow, const cv::Mat *flow2 = NULL);
	/*virtual*/ bool classifyGesture();
	/*virtual*/ GestureType::Type getGestureType() const  {  return gestureId_;  }

	void clearShortTimeGestureHistory();
	void clearLongTimeGestureHistory();
	void clearThirdClassGestureHistory();
	void clearTimeSeriesGestureHistory();

	// for display
	void initWindows() const;
	void destroyWindows() const;

private:
	bool classifyShortTimeGesture();
	bool classifyLongTimeGesture();
	bool classifyThirdClassGesture();

	GestureType::Type classifyShortTimeGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const bool useGestureIdPattern) const;
	GestureType::Type classifyLongTimeGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const;
	GestureType::Type classifyThirdClassGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const;
	GestureType::Type classifyTimeSeriesGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const;

	void createReferenceFullPhaseHistograms();
	void createReferenceHistogramsForShortTimeGesture();
	void createReferenceHistogramsForLongTimeGesture();
	void createReferenceHistogramsForThirdClassGesture();
	void createGestureIdPatternHistogramsForShortTimeGesture();

	size_t matchHistogramByGestureIdPattern(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const std::vector<const cv::MatND> &gestureIdPatternHistograms, const double histDistThreshold) const;
	size_t matchHistogramByFrequency(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const size_t countThreshold) const;

	void drawAccumulatedPhaseHistogram(const cv::MatND &accumulatedHist, const std::string &windowName) const;
	void drawMatchedIdPatternHistogram(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const std::string &windowName) const;
	void drawMatchedReferenceHistogram(const std::vector<const cv::MatND> &refHistograms, const size_t matchedIdx, const std::string &windowName) const;

	void drawTemporalPhaseHistogram(const cv::MatND &temporalHist, const std::string &windowName) const;

private:
	Params params_;

	std::vector<const cv::MatND> refFullPhaseHistograms_;

	std::vector<const cv::MatND> refHistogramsForShortTimeGesture_;
	std::vector<const cv::MatND> refHistogramsForLongTimeGesture_;
	std::vector<const cv::MatND> refHistogramsForThirdClassGesture_;

	std::vector<const cv::MatND> gestureIdPatternHistogramsForShortTimeGesture_;

	boost::shared_ptr<HistogramAccumulator> histogramAccumulatorForShortTimeGesture_;
	boost::shared_ptr<HistogramAccumulator> histogramAccumulatorForLongTimeGesture_;
	boost::shared_ptr<HistogramAccumulator> histogramAccumulatorForThirdClassGesture_;

	boost::circular_buffer<size_t> matchedHistogramIndexes1ForShortTimeGesture_, matchedHistogramIndexes2ForShortTimeGesture_;
	boost::circular_buffer<size_t> matchedHistogramIndexesForLongTimeGesture_;
	boost::circular_buffer<size_t> matchedHistogramIndexesForThirdClassGesture_;

	GestureType::Type gestureId_;
};

}  // namespace swl


#endif  // __SWL_GESTURE_RECOGNITION__GESTURE_CLASSIFIER_BY_HISTOGRAM__H_
