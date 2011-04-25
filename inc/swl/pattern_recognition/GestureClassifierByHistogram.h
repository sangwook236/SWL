#if !defined(__SWL_PATTERN_RECOGNITION__GESTURE_CLASSIFIER_BY_HISTOGRAM__H_)
#define __SWL_PATTERN_RECOGNITION__GESTURE_CLASSIFIER_BY_HISTOGRAM__H_ 1


#include "swl/pattern_recognition/ExportPatternRecognition.h"
#include "swl/pattern_recognition/IGestureClassifier.h"
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

class SWL_PATTERN_RECOGNITION_API GestureClassifierByHistogram: public IGestureClassifier
{
public:
	struct Params
	{
		size_t accumulatedHistogramNumForClass1Gesture;
		size_t accumulatedHistogramNumForClass2Gesture;
		size_t accumulatedHistogramNumForClass3Gesture;
		size_t maxMatchedHistogramNum;

		double histDistThresholdForClass1Gesture;
		//double histDistThresholdForClass1Gesture_LeftMove;
		//double histDistThresholdForClass1Gesture_Others;
		double histDistThresholdForClass2Gesture;
		double histDistThresholdForClass3Gesture;

		double histDistThresholdForGestureIdPattern;

		size_t matchedIndexCountThresholdForClass1Gesture;
		size_t matchedIndexCountThresholdForClass2Gesture;
		size_t matchedIndexCountThresholdForClass3Gesture;

		bool doesApplyMagnitudeFiltering;
		double magnitudeFilteringMinThresholdRatio;
		double magnitudeFilteringMaxThresholdRatio;
		bool doesApplyTimeWeighting;
		bool doesApplyMagnitudeWeighting;
	};

	enum GestureClassType
	{
		GCT_CLASS_ALL = 0x00,
		GCT_CLASS_1 = 0x01,
		GCT_CLASS_2 = 0x02,
		GCT_CLASS_3 = 0x04,
		GCT_CLASS_TIME_SERIES = 0x08
	};

public:
	typedef IGestureClassifier base_type;

public:
	GestureClassifierByHistogram(const Params &params);
	explicit GestureClassifierByHistogram(const GestureClassifierByHistogram &rhs);
	~GestureClassifierByHistogram();

	GestureClassifierByHistogram & operator=(const GestureClassifierByHistogram &rhs);

public:
	/*virtual*/ bool analyzeOrientation(const int gestureClassToApply, const cv::Mat &orientation);

	/*virtual*/ bool classifyGesture();
	/*virtual*/ GestureType::Type getGestureType() const  {  return gestureId_;  }

	/*virtual*/ void clearGestureHistory(const int gestureClassToApply);

private:
	bool classifyClass1Gesture();
	bool classifyClass2Gesture();
	bool classifyClass3Gesture();

	GestureType::Type classifyClass1Gesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const bool useGestureIdPattern) const;
	GestureType::Type classifyClass2Gesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const;
	GestureType::Type classifyClass3Gesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const;
	GestureType::Type classifyTimeSeriesGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const;

	void clearClass1GestureHistory();
	void clearClass2GestureHistory();
	void clearClass3GestureHistory();
	void clearTimeSeriesGestureHistory();

	void createReferenceFullPhaseHistograms();
	void createReferenceHistogramsForClass1Gesture();
	void createReferenceHistogramsForClass2Gesture();
	void createReferenceHistogramsForClass3Gesture();
	void createGestureIdPatternHistogramsForClass1Gesture();

	size_t matchHistogramByGestureIdPattern(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const std::vector<const cv::MatND> &gestureIdPatternHistograms, const double histDistThreshold) const;
	size_t matchHistogramByFrequency(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const size_t countThreshold) const;

	void drawAccumulatedPhaseHistogram(const cv::MatND &accumulatedHist, const std::string &windowName) const;
	void drawMatchedIdPatternHistogram(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const std::string &windowName) const;
	void drawMatchedReferenceHistogram(const std::vector<const cv::MatND> &refHistograms, const size_t matchedIdx, const std::string &windowName) const;

	void drawTemporalOrientationHistogram(const cv::MatND &temporalHist, const std::string &windowName) const;

	// for display
	void initWindows() const;
	void destroyWindows() const;

private:
	Params params_;

	std::vector<const cv::MatND> refFullPhaseHistograms_;

	std::vector<const cv::MatND> refHistogramsForClass1Gesture_;
	std::vector<const cv::MatND> refHistogramsForClass2Gesture_;
	std::vector<const cv::MatND> refHistogramsForClass3Gesture_;

	std::vector<const cv::MatND> gestureIdPatternHistogramsForClass1Gesture_;
	//std::vector<const cv::MatND> gestureIdPatternHistogramsForClass2Gesture_;
	//std::vector<const cv::MatND> gestureIdPatternHistogramsForClass3Gesture_;

	boost::shared_ptr<HistogramAccumulator> histogramAccumulatorForClass1Gesture_;
	boost::shared_ptr<HistogramAccumulator> histogramAccumulatorForClass2Gesture_;
	boost::shared_ptr<HistogramAccumulator> histogramAccumulatorForClass3Gesture_;

	boost::circular_buffer<size_t> matchedHistogramIndexes1ForClass1Gesture_, matchedHistogramIndexes2ForClass1Gesture_;
	boost::circular_buffer<size_t> matchedHistogramIndexesForClass2Gesture_;
	boost::circular_buffer<size_t> matchedHistogramIndexesForClass3Gesture_;

	GestureType::Type gestureId_;
};

}  // namespace swl


#endif  // __SWL_PATTERN_RECOGNITION__GESTURE_CLASSIFIER_BY_HISTOGRAM__H_
