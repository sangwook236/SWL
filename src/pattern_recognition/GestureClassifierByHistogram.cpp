#include "swl/pattern_recognition/GestureClassifierByHistogram.h"
#include "HistogramGenerator.h"
#include "swl/rnd_util/HistogramAccumulator.h"
#include "swl/rnd_util/HistogramMatcher.h"
#include "swl/rnd_util/HistogramUtil.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>


namespace swl {

#define __VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_ 1


namespace {
namespace local {

// histograms' parameters
const int histDims = 1;

const int phaseHistBins = 360;
const int phaseHistSize[] = { phaseHistBins };
// phase varies from 0 to 359
const float phaseHistRange1[] = { 0, phaseHistBins };
const float *phaseHistRanges[] = { phaseHistRange1 };
// we compute the histogram from the 0-th channel
const int phaseHistChannels[] = { 0 };
const int phaseHistBinWidth = 1, phaseHistMaxHeight = 100;

const int indexHistBins = ReferenceFullPhaseHistogramGenerator::REF_HISTOGRAM_NUM;
const int indexHistSize[] = { indexHistBins };
const float indexHistRange1[] = { 0, indexHistBins };
const float *indexHistRanges[] = { indexHistRange1 };
// we compute the histogram from the 0-th channel
const int indexHistChannels[] = { 0 };
const int indexHistBinWidth = 5, indexHistMaxHeight = 100;

const int phaseHorzScale = 10, phaseVertScale = 2;

//
const double refFullPhaseHistogramSigma = 8.0;
const double class1GestureRefHistogramSigma = 8.0;
const double class2GestureRefHistogramSigma = 16.0;
const double class3GestureRefHistogramSigma = 20.0;
const size_t refHistogramBinNum = phaseHistBins;
const double refHistogramNormalizationFactor = 5000.0;
const double gesturePatternHistogramSigma = 1.0;
const size_t gesturePatternHistogramBinNum = indexHistBins;
const double gesturePatternHistogramNormalizationFactor = 10.0; //(double)params_.maxMatchedHistogramNum;

std::vector<float> getHistogramTimeWeight(const size_t histogramNum)
{
	std::vector<float> histogramTimeWeight(histogramNum, 0.0f);

	float sum = 0.0f;
	for (size_t i = 0; i < histogramNum; ++i)
	{
		const float weight = std::exp(-(float)i / (float)histogramNum);
		histogramTimeWeight[i] = weight * weight;
		sum += weight;
	}
	for (size_t i = 0; i < histogramNum; ++i)
		histogramTimeWeight[i] /= sum;

	return histogramTimeWeight;
}

struct MaxFrequencyComparator
{
	bool operator()(const std::pair<size_t, size_t> &rhs, const std::pair<size_t, size_t> &lhs) const
	{
		return rhs.second < lhs.second;
	}
};

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
const std::string windowNameClass1Gesture1("gesture recognition - (STG) actual histogram");
const std::string windowNameClass1Gesture2("gesture recognition - (STG) matched histogram");
const std::string windowNameClass1Gesture3("gesture recognition - (STG) matched id histogram");
const std::string windowNameClass2Gesture1("gesture recognition - (LTG) actual histogram");
const std::string windowNameClass2Gesture2("gesture recognition - (LTG) matched histogram");
const std::string windowNameClass2Gesture3("gesture recognition - (LTG) matched id histogram");
const std::string windowNameClass3Gesture1("gesture recognition - (TCG) actual histogram");
const std::string windowNameClass3Gesture2("gesture recognition - (TCG) matched histogram");
const std::string windowNameClass3Gesture3("gesture recognition - (TCG) matched id histogram");
const std::string windowNameForTemporalOrientationHistogram("gesture recognition - temporal histogram");
const std::string windowNameForHorizontalOrientationHistogram("gesture recognition - horizontal histogram");
const std::string windowNameForVerticalOrientationHistogram("gesture recognition - vertical histogram");
#endif

}  // namespace local
}  // unnamed namespace

//-----------------------------------------------------------------------------
//

GestureClassifierByHistogram::GestureClassifierByHistogram(const Params &params)
: base_type(),
  params_(params),
  refFullPhaseHistograms_(),
  refHistogramsForClass1Gesture_(), refHistogramsForClass2Gesture_(), refHistogramsForClass3Gesture_(),
  gestureIdPatternHistogramsForClass1Gesture_(),
  histogramAccumulatorForClass1Gesture_(params_.doesApplyTimeWeighting ? new HistogramAccumulator(local::getHistogramTimeWeight(params_.accumulatedHistogramNumForClass1Gesture)) : new HistogramAccumulator(params_.accumulatedHistogramNumForClass1Gesture)),
  histogramAccumulatorForClass2Gesture_(params_.doesApplyTimeWeighting ? new HistogramAccumulator(local::getHistogramTimeWeight(params_.accumulatedHistogramNumForClass2Gesture)) : new HistogramAccumulator(params_.accumulatedHistogramNumForClass2Gesture)),
  histogramAccumulatorForClass3Gesture_(params_.doesApplyTimeWeighting ? new HistogramAccumulator(local::getHistogramTimeWeight(params_.accumulatedHistogramNumForClass3Gesture)) : new HistogramAccumulator(params_.accumulatedHistogramNumForClass3Gesture)),
  matchedHistogramIndexes1ForClass1Gesture_(params_.maxMatchedHistogramNum), matchedHistogramIndexes2ForClass1Gesture_(params_.maxMatchedHistogramNum), matchedHistogramIndexesForClass2Gesture_(params_.maxMatchedHistogramNum), matchedHistogramIndexesForClass3Gesture_(params_.maxMatchedHistogramNum),
  gestureId_(GestureType::GT_UNDEFINED)
{
	createReferenceFullPhaseHistograms();
	createReferenceHistogramsForClass1Gesture();
	createReferenceHistogramsForClass2Gesture();
	createReferenceHistogramsForClass3Gesture();
	createGestureIdPatternHistogramsForClass1Gesture();
}

GestureClassifierByHistogram::GestureClassifierByHistogram(const GestureClassifierByHistogram &rhs)
: base_type(),
  params_(rhs.params_),
  refFullPhaseHistograms_(rhs.refFullPhaseHistograms_),
  refHistogramsForClass1Gesture_(rhs.refHistogramsForClass1Gesture_), refHistogramsForClass2Gesture_(rhs.refHistogramsForClass2Gesture_), refHistogramsForClass3Gesture_(rhs.refHistogramsForClass3Gesture_),
  gestureIdPatternHistogramsForClass1Gesture_(rhs.gestureIdPatternHistogramsForClass1Gesture_),
  histogramAccumulatorForClass1Gesture_(rhs.histogramAccumulatorForClass1Gesture_), histogramAccumulatorForClass2Gesture_(rhs.histogramAccumulatorForClass2Gesture_), histogramAccumulatorForClass3Gesture_(rhs.histogramAccumulatorForClass3Gesture_),
  matchedHistogramIndexes1ForClass1Gesture_(rhs.matchedHistogramIndexes1ForClass1Gesture_),  matchedHistogramIndexes2ForClass1Gesture_(rhs.matchedHistogramIndexes2ForClass1Gesture_), matchedHistogramIndexesForClass2Gesture_(rhs.matchedHistogramIndexesForClass2Gesture_), matchedHistogramIndexesForClass3Gesture_(rhs.matchedHistogramIndexesForClass3Gesture_),
  gestureId_(rhs.gestureId_)
{
#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	initWindows();
#endif
}

GestureClassifierByHistogram::~GestureClassifierByHistogram()
{
#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	destroyWindows();
#endif
}

GestureClassifierByHistogram & GestureClassifierByHistogram::operator=(const GestureClassifierByHistogram &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;

	params_ = rhs.params_;
	refFullPhaseHistograms_.assign(rhs.refFullPhaseHistograms_.begin(), rhs.refFullPhaseHistograms_.end());
	refHistogramsForClass1Gesture_.assign(rhs.refHistogramsForClass1Gesture_.begin(), rhs.refHistogramsForClass1Gesture_.end());
	refHistogramsForClass2Gesture_.assign(rhs.refHistogramsForClass2Gesture_.begin(), rhs.refHistogramsForClass2Gesture_.end());
	refHistogramsForClass3Gesture_.assign(rhs.refHistogramsForClass3Gesture_.begin(), rhs.refHistogramsForClass3Gesture_.end());
	gestureIdPatternHistogramsForClass1Gesture_.assign(rhs.gestureIdPatternHistogramsForClass1Gesture_.begin(), rhs.gestureIdPatternHistogramsForClass1Gesture_.end());
	histogramAccumulatorForClass1Gesture_ = rhs.histogramAccumulatorForClass1Gesture_;
	histogramAccumulatorForClass2Gesture_ = rhs.histogramAccumulatorForClass2Gesture_;
	histogramAccumulatorForClass3Gesture_ = rhs.histogramAccumulatorForClass3Gesture_;
	matchedHistogramIndexes1ForClass1Gesture_.assign(rhs.matchedHistogramIndexes1ForClass1Gesture_.begin(), rhs.matchedHistogramIndexes1ForClass1Gesture_.end());
	matchedHistogramIndexes2ForClass1Gesture_.assign(rhs.matchedHistogramIndexes2ForClass1Gesture_.begin(), rhs.matchedHistogramIndexes2ForClass1Gesture_.end());
	matchedHistogramIndexesForClass2Gesture_.assign(rhs.matchedHistogramIndexesForClass2Gesture_.begin(), rhs.matchedHistogramIndexesForClass2Gesture_.end());
	matchedHistogramIndexesForClass3Gesture_.assign(rhs.matchedHistogramIndexesForClass3Gesture_.begin(), rhs.matchedHistogramIndexesForClass3Gesture_.end());
	gestureId_ = rhs.gestureId_;

	return *this;
}

bool GestureClassifierByHistogram::analyzeOrientation(const int gestureClassToApply, const cv::Mat &orientation)
{
	// calculate phase histogram
	cv::MatND hist;
	cv::calcHist(&orientation, 1, local::phaseHistChannels, cv::Mat(), hist, local::histDims, local::phaseHistSize, local::phaseHistRanges, true, false);

	//
	if (GCT_CLASS_ALL == gestureClassToApply || (GCT_CLASS_1 & gestureClassToApply) == GCT_CLASS_1) histogramAccumulatorForClass1Gesture_->addHistogram(hist);
	if (GCT_CLASS_ALL == gestureClassToApply || (GCT_CLASS_2 & gestureClassToApply) == GCT_CLASS_2) histogramAccumulatorForClass2Gesture_->addHistogram(hist);
	if (GCT_CLASS_ALL == gestureClassToApply || (GCT_CLASS_3 & gestureClassToApply) == GCT_CLASS_3) histogramAccumulatorForClass3Gesture_->addHistogram(hist);
	// TODO [check] >>
	//if (GCT_CLASS_TIME_SERIES == gestureClassToApply || (GCT_CLASS_TIME_SERIES & gestureClassToApply) == GCT_CLASS_TIME_SERIES) histogramAccumulatorForClass3Gesture_->addHistogram(hist);

	return true;
}

/*virtual*/ bool GestureClassifierByHistogram::classifyGesture()
{
	gestureId_ = GestureType::GT_UNDEFINED;

	// classify class 2 gesture
	if (histogramAccumulatorForClass2Gesture_->isFull() && classifyClass2Gesture()) return true;

	// classify class 1 gesture
	if (histogramAccumulatorForClass1Gesture_->isFull() && classifyClass1Gesture()) return true;

	// TODO [check] >>
	// classify class 3 gesture
	if (histogramAccumulatorForClass3Gesture_->isFull() && classifyClass3Gesture()) return true;

	return false;
}

void GestureClassifierByHistogram::clearGestureHistory(const int gestureClassToApply)
{
	if (GCT_CLASS_ALL == gestureClassToApply || (GCT_CLASS_1 & gestureClassToApply) == GCT_CLASS_1) clearClass1GestureHistory();
	if (GCT_CLASS_ALL == gestureClassToApply || (GCT_CLASS_2 & gestureClassToApply) == GCT_CLASS_2) clearClass2GestureHistory();
	if (GCT_CLASS_ALL == gestureClassToApply || (GCT_CLASS_3 & gestureClassToApply) == GCT_CLASS_3) clearClass3GestureHistory();
	if (GCT_CLASS_TIME_SERIES == gestureClassToApply || (GCT_CLASS_TIME_SERIES & gestureClassToApply) == GCT_CLASS_TIME_SERIES) clearTimeSeriesGestureHistory();
}

bool GestureClassifierByHistogram::classifyClass1Gesture()
{
	// create accumulated phase histograms
	cv::MatND &accumulatedHist = histogramAccumulatorForClass1Gesture_->createAccumulatedHistogram();
	// normalize histogram
	HistogramUtil::normalizeHistogram(accumulatedHist, local::refHistogramNormalizationFactor);

	{
		// FIXME [restore] >> have to decide which one is used
		//cv::MatND &temporalHist = histogramAccumulatorForClass1Gesture_->createTemporalHistogram();
		cv::MatND &temporalHist = histogramAccumulatorForClass1Gesture_->createTemporalHistogram(refFullPhaseHistograms_, params_.histDistThresholdForClass1Gesture);
		// normalize histogram
		HistogramUtil::normalizeHistogram(temporalHist, local::refHistogramNormalizationFactor);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		// draw temporal orientation histogram
		drawTemporalOrientationHistogram(temporalHist, local::windowNameForTemporalOrientationHistogram);
#endif
	}

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	// draw accumulated phase histogram
	drawAccumulatedPhaseHistogram(accumulatedHist, local::windowNameClass1Gesture1);
#endif

	// match histogram
	double minHistDist1 = std::numeric_limits<double>::max();
	const size_t &matchedIdx1 = refFullPhaseHistograms_.empty() ? -1 : HistogramMatcher::match(refFullPhaseHistograms_, accumulatedHist, minHistDist1);

// FIXME [restore] >>
#if 1
	if (minHistDist1 < params_.histDistThresholdForClass1Gesture)
#else
	if ((((0 <= matchedIdx && matchedIdx <= 4) || (31 <= matchedIdx && matchedIdx <= 35)) && minHistDist < local::histDistThresholdForClass1Gesture_LeftMove) ||
		(minHistDist < local::histDistThresholdForClass1Gesture_Others))
#endif
	{
		matchedHistogramIndexes1ForClass1Gesture_.push_back(matchedIdx1);

		// classify class 1 gesture
		gestureId_ = classifyClass1Gesture(matchedHistogramIndexes1ForClass1Gesture_, true);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexes1ForClass1Gesture_, local::windowNameClass1Gesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refFullPhaseHistograms_, matchedIdx1, local::windowNameClass1Gesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexes1ForClass1Gesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameClass1Gesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	// FIXME [modify] >> the above & this codes should be integrated
	double minHistDist2 = std::numeric_limits<double>::max();
	const size_t &matchedIdx2 = refHistogramsForClass1Gesture_.empty() ? -1 : HistogramMatcher::match(refHistogramsForClass1Gesture_, accumulatedHist, minHistDist2);

	if (minHistDist2 < params_.histDistThresholdForClass1Gesture)
	{
		matchedHistogramIndexes2ForClass1Gesture_.push_back(matchedIdx2);

		// classify class 1 gesture
		gestureId_ = classifyClass1Gesture(matchedHistogramIndexes2ForClass1Gesture_, false);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexes2ForClass1Gesture_, local::windowNameClass1Gesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refHistogramsForClass1Gesture_, matchedIdx2, local::windowNameClass1Gesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexes2ForClass1Gesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameClass1Gesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	return false;
}

bool GestureClassifierByHistogram::classifyClass2Gesture()
{
	// accumulate phase histograms
	cv::MatND &accumulatedHist = histogramAccumulatorForClass2Gesture_->createAccumulatedHistogram();
	// normalize histogram
	HistogramUtil::normalizeHistogram(accumulatedHist, local::refHistogramNormalizationFactor);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	// draw accumulated phase histogram
	drawAccumulatedPhaseHistogram(accumulatedHist, local::windowNameClass2Gesture1);
#endif

	// match histogram
	double minHistDist = std::numeric_limits<double>::max();
	const size_t &matchedIdx = refHistogramsForClass2Gesture_.empty() ? -1 : HistogramMatcher::match(refHistogramsForClass2Gesture_, accumulatedHist, minHistDist);

	if (minHistDist < params_.histDistThresholdForClass2Gesture)
	{
		matchedHistogramIndexesForClass2Gesture_.push_back(matchedIdx);

		// classify class 2 gesture
		gestureId_ = classifyClass2Gesture(matchedHistogramIndexesForClass2Gesture_);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexesForClass2Gesture_, local::windowNameClass2Gesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refHistogramsForClass2Gesture_, matchedIdx, local::windowNameClass2Gesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexesForClass2Gesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameClass2Gesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	return false;
}

bool GestureClassifierByHistogram::classifyClass3Gesture()
{
	// accumulate phase histograms
	cv::MatND &accumulatedHist = histogramAccumulatorForClass3Gesture_->createAccumulatedHistogram();
	// normalize histogram
	HistogramUtil::normalizeHistogram(accumulatedHist, local::refHistogramNormalizationFactor);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	// draw accumulated phase histogram
	drawAccumulatedPhaseHistogram(accumulatedHist, local::windowNameClass3Gesture1);
#endif

	// match histogram
	double minHistDist = std::numeric_limits<double>::max();
	const size_t &matchedIdx = refHistogramsForClass3Gesture_.empty() ? -1 : HistogramMatcher::match(refHistogramsForClass3Gesture_, accumulatedHist, minHistDist);

	if (minHistDist < params_.histDistThresholdForClass3Gesture)
	{
		matchedHistogramIndexesForClass3Gesture_.push_back(matchedIdx);

		// classify class 3 gesture
		gestureId_ = classifyClass3Gesture(matchedHistogramIndexesForClass3Gesture_);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexesForClass3Gesture_, local::windowNameClass3Gesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refHistogramsForClass3Gesture_, matchedIdx, local::windowNameClass3Gesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexesForClass3Gesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameClass3Gesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	return false;
}

GestureType::Type GestureClassifierByHistogram::classifyClass1Gesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const bool useGestureIdPattern) const
{
	// match histogram by gesture ID pattern
	if (useGestureIdPattern)
	{
		const size_t &matchedIdx = matchHistogramByGestureIdPattern(matchedHistogramIndexes, gestureIdPatternHistogramsForClass1Gesture_, params_.histDistThresholdForGestureIdPattern);
		switch (matchedIdx)
		{
		case 1:
			return GestureType::GT_LEFT_MOVE;
		case 2:
			return GestureType::GT_RIGHT_MOVE;
		case 3:
			return GestureType::GT_UP_MOVE;
		case 4:
			return GestureType::GT_DOWN_MOVE;
		}
	}
	// match histogram by frequency
	else
	{
		// TODO [adjust] >> design parameter
		const size_t countThreshold(matchedHistogramIndexes.size() / 2);
		//const size_t countThreshold(params_.matchedIndexCountThresholdForClass1Gesture);

		const size_t &matchedIdx = matchHistogramByFrequency(matchedHistogramIndexes, countThreshold);
		switch (matchedIdx)
		{
		case 0:
			return GestureType::GT_HAND_OPEN;
		//case :
		//	return GestureType::GT_HAND_CLOSE;
		}
	}

	return GestureType::GT_UNDEFINED;
}

GestureType::Type GestureClassifierByHistogram::classifyClass2Gesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const
{
#if 0
	const size_t &matchedIdx = matchHistogramByGestureIdPattern(matchedHistogramIndexes, gestureIdPatternHistogramsForClass2Gesture_, params_.histDistThresholdForGestureIdPattern);
	switch (matchedIdx)
	{
	case :
		return GestureType::GT_HORIZONTAL_FLIP;
	case :
		return GestureType::GT_VERTICAL_FLIP;
	// FIXME [implement] >>
/*
	case :
		return GestureType::GT_JAMJAM;
	case :
		return GestureType::GT_SHAKE;
	case :
		return GestureType::GT_LEFT_90_TURN;
	case :
		return GestureType::GT_RIGHT_90_TURN;
*/
	}
#else
	// TODO [adjust] >> design parameter
	const size_t countThreshold(matchedHistogramIndexes.size() / 2);
	//const size_t countThreshold(params_.matchedIndexCountThresholdForClass2Gesture);

	const size_t &matchedIdx = matchHistogramByFrequency(matchedHistogramIndexes, countThreshold);
	switch (matchedIdx)
	{
	case 0:
		return GestureType::GT_HORIZONTAL_FLIP;
	case 1:
		return GestureType::GT_VERTICAL_FLIP;
	// FIXME [implement] >>
/*
	case :
		return GestureType::GT_JAMJAM;
	case :
		return GestureType::GT_SHAKE;
	case :
		return GestureType::GT_LEFT_90_TURN;
	case :
		return GestureType::GT_RIGHT_90_TURN;
*/
	}
#endif

	return GestureType::GT_UNDEFINED;
}

GestureType::Type GestureClassifierByHistogram::classifyClass3Gesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const
{
#if 0
	const size_t &matchedIdx = matchHistogramByGestureIdPattern(matchedHistogramIndexes, gestureIdPatternHistogramsForThirdClassGesture_, params_.histDistThresholdForGestureIdPattern);
	switch (matchedIdx)
	{
	case :
		return GestureType::GT_INFINITY;
	case :
		return GestureType::GT_TRIANGLE;
	}
#else
	// TODO [adjust] >> design parameter
	const size_t countThreshold(matchedHistogramIndexes.size() / 2);
	//const size_t countThreshold(params_.matchedIndexCountThresholdForClass3Gesture);

	const size_t &matchedIdx = matchHistogramByFrequency(matchedHistogramIndexes, countThreshold);
	switch (matchedIdx)
	{
/*
	case :
		return GestureType::GT_LEFT_FAST_MOVE;
	case :
		return GestureType::GT_RIGHT_FAST_MOVE;
*/
	case 0:
		return GestureType::GT_INFINITY;
	case 1:
		return GestureType::GT_TRIANGLE;
	}
#endif

	return GestureType::GT_UNDEFINED;
}

GestureType::Type GestureClassifierByHistogram::classifyTimeSeriesGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");
/*
	switch ()
	{
	case :
		return GestureType::GT_CW;
	case :
		return GestureType::GT_CCW;
	case :
		return GestureType::GT_INFINITY;
	case :
		return GestureType::GT_TRIANGLE;
	}

	return GestureType::GT_UNDEFINED;
*/
}

void GestureClassifierByHistogram::clearClass1GestureHistory()
{
	histogramAccumulatorForClass1Gesture_->clearAllHistograms();
	matchedHistogramIndexes1ForClass1Gesture_.clear();
	matchedHistogramIndexes2ForClass1Gesture_.clear();
}

void GestureClassifierByHistogram::clearClass2GestureHistory()
{
	histogramAccumulatorForClass2Gesture_->clearAllHistograms();
	matchedHistogramIndexesForClass2Gesture_.clear();
}

void GestureClassifierByHistogram::clearClass3GestureHistory()
{
	histogramAccumulatorForClass3Gesture_->clearAllHistograms();
	matchedHistogramIndexesForClass3Gesture_.clear();
}

void GestureClassifierByHistogram::clearTimeSeriesGestureHistory()
{
	// FIXME [implement] >>
}

void GestureClassifierByHistogram::createReferenceFullPhaseHistograms()
{
	// create reference histograms
	ReferenceFullPhaseHistogramGenerator refHistogramGenerator(local::refFullPhaseHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refFullPhaseHistograms_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistograms_.begin(); it != refHistograms_.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
		HistogramUtil::drawHistogram1D(*it, local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, histImg);

		cv::imshow(local::windowNameClass1Gesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createReferenceHistogramsForClass1Gesture()
{
	// create reference histograms
	ReferenceHistogramGeneratorForClass1Gesture refHistogramGenerator(local::class1GestureRefHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refHistogramsForClass1Gesture_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistogramsForClass1Gesture_.begin(); it != refHistogramsForClass1Gesture_.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
		HistogramUtil::drawHistogram1D(*it, local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, histImg);

		cv::imshow(local::windowNameClass1Gesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createReferenceHistogramsForClass2Gesture()
{
	// create reference histograms
	ReferenceHistogramGeneratorForClass2Gesture refHistogramGenerator(local::class2GestureRefHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refHistogramsForClass2Gesture_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistogramsForClass2Gesture_.begin(); it != refHistogramsForClass2Gesture_.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
		HistogramUtil::drawHistogram1D(*it, local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, histImg);

		cv::imshow(local::windowNameClass1Gesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createReferenceHistogramsForClass3Gesture()
{
	// create reference histograms
	ReferenceHistogramGeneratorForClass3Gesture refHistogramGenerator(local::class3GestureRefHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refHistogramsForClass3Gesture_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistogramsForClass3Gesture_.begin(); it != refHistogramsForClass3Gesture_.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
		HistogramUtil::drawHistogram1D(*it, local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, histImg);

		cv::imshow(local::windowNameClass1Gesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createGestureIdPatternHistogramsForClass1Gesture()
{
	// create gesture pattern histograms
	GestureIdPatternHistogramGeneratorForClass1Gesture gestureIdPatternHistogramGenerator(local::gesturePatternHistogramSigma);
	gestureIdPatternHistogramGenerator.createHistograms(local::gesturePatternHistogramBinNum, local::gesturePatternHistogramNormalizationFactor);
	const std::vector<cv::MatND> &gesturePatternHistograms = gestureIdPatternHistogramGenerator.getHistograms();

	gestureIdPatternHistogramsForClass1Gesture_.assign(gesturePatternHistograms.begin(), gesturePatternHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw gesture pattern histograms
	for (std::vector<cv::MatND>::const_iterator it = gesturePatternHistograms.begin(); it != gesturePatternHistograms.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = local::gesturePatternHistogramNormalizationFactor;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(local::indexHistMaxHeight, local::gesturePatternHistogramBinNum*local::indexHistBinWidth, CV_8UC3));
		HistogramUtil::drawHistogram1D(*it, local::gesturePatternHistogramBinNum, maxVal, local::indexHistBinWidth, local::indexHistMaxHeight, histImg);

		cv::imshow(local::windowNameClass1Gesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

size_t GestureClassifierByHistogram::matchHistogramByGestureIdPattern(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const std::vector<const cv::MatND> &gestureIdPatternHistograms, const double histDistThreshold) const
{
	// create matched ID histogram
#if 0
	cv::MatND hist;
	cv::calcHist(
		&cv::Mat(std::vector<unsigned char>(matchedHistogramIndexes.begin(), matchedHistogramIndexes.end())),
		1, phaseHistChannels, cv::Mat(), hist, histDims, phaseHistSize, phaseHistRanges, true, false
	);
#else
	cv::MatND hist = cv::MatND::zeros(local::gesturePatternHistogramBinNum, 1, CV_32F);
	float *binPtr = (float *)hist.data;
	for (boost::circular_buffer<size_t>::const_iterator it = matchedHistogramIndexes.begin(); it != matchedHistogramIndexes.end(); ++it)
		if (*it != -1) ++(binPtr[*it]);
#endif

	// match histogram
	double minHistDist = std::numeric_limits<double>::max();
	const size_t &matchedIdx = gestureIdPatternHistograms.empty() ? -1 : HistogramMatcher::match(gestureIdPatternHistograms, hist, minHistDist);

	// FIXME [delete] >>
	//std::cout << "\t\t\t*** " << minHistDist << std::endl;

	return minHistDist < histDistThreshold ? matchedIdx : -1;
}

size_t GestureClassifierByHistogram::matchHistogramByFrequency(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const size_t countThreshold) const
{
	std::map<const size_t, size_t> freqs;
	for (boost::circular_buffer<size_t>::const_iterator it = matchedHistogramIndexes.begin(); it != matchedHistogramIndexes.end(); ++it)
	{
		if (freqs.find(*it) == freqs.end()) freqs[*it] = 1;
		else ++freqs[*it];
	}

	std::map<const size_t, size_t>::const_iterator itMaxFreq = std::max_element(freqs.begin(), freqs.end(), local::MaxFrequencyComparator());
	return (freqs.end() != itMaxFreq && itMaxFreq->second > countThreshold) ? itMaxFreq->first : -1;
}

void GestureClassifierByHistogram::drawAccumulatedPhaseHistogram(const cv::MatND &accumulatedHist, const std::string &windowName) const
{
#if 0
	double maxVal = 0.0;
	cv::minMaxLoc(accumulatedHist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

	cv::Mat histImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
	HistogramUtil::drawHistogram1D(accumulatedHist, local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, histImg);

	cv::imshow(windowName, histImg);
}

void GestureClassifierByHistogram::drawMatchedIdPatternHistogram(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const std::string &windowName) const
{
	// calculate matched index histogram
	cv::MatND hist;
	cv::calcHist(&cv::Mat(std::vector<unsigned char>(matchedHistogramIndexes.begin(), matchedHistogramIndexes.end())), 1, local::indexHistChannels, cv::Mat(), hist, local::histDims, local::indexHistSize, local::indexHistRanges, true, false);

	// normalize histogram
	//HistogramUtil::normalizeHistogram(hist, params_.maxMatchedHistogramNum);

	// draw matched index histogram
	cv::Mat histImg(cv::Mat::zeros(local::indexHistMaxHeight, local::indexHistBins*local::indexHistBinWidth, CV_8UC3));
	HistogramUtil::drawHistogram1D(hist, local::indexHistBins, params_.maxMatchedHistogramNum, local::indexHistBinWidth, local::indexHistMaxHeight, histImg);
				
	std::ostringstream sstream;
	sstream << "count: " << matchedHistogramIndexes.size();
	cv::putText(histImg, sstream.str(), cv::Point(10, 15), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 255), 1, 8, false);
				
	cv::imshow(windowName, histImg);
}

void GestureClassifierByHistogram::drawMatchedReferenceHistogram(const std::vector<const cv::MatND> &refHistograms, const size_t matchedIdx, const std::string &windowName) const
{
#if 0
	double maxVal = 0.0;
	cv::minMaxLoc(accumulatedHist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

	cv::Mat refHistImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
	HistogramUtil::drawHistogram1D(refHistograms[matchedIdx], local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, refHistImg);

	std::ostringstream sstream;
	sstream << "idx: " << matchedIdx;
	cv::putText(refHistImg, sstream.str(), cv::Point(10, 15), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 255), 1, 8, false);

	cv::imshow(windowName, refHistImg);
}

void GestureClassifierByHistogram::drawTemporalOrientationHistogram(const cv::MatND &temporalHist, const std::string &windowName) const
{
#if 1
	double maxVal = 0.0;
	cv::minMaxLoc(temporalHist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = local::refHistogramNormalizationFactor * 0.05;
#endif

	cv::Mat histImg(cv::Mat::zeros(temporalHist.rows*local::phaseVertScale, temporalHist.cols*local::phaseHorzScale, CV_8UC3));
	HistogramUtil::drawHistogram2D(temporalHist, temporalHist.cols, temporalHist.rows, maxVal, local::phaseHorzScale, local::phaseVertScale, histImg);

	cv::imshow(windowName, histImg);
}

void GestureClassifierByHistogram::initWindows() const
{
	cv::namedWindow(local::windowNameClass1Gesture1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass1Gesture2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass1Gesture3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass2Gesture1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass2Gesture2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass2Gesture3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass3Gesture1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass3Gesture2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameClass3Gesture3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameForTemporalOrientationHistogram, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameForHorizontalOrientationHistogram, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameForVerticalOrientationHistogram, cv::WINDOW_AUTOSIZE);
}

void GestureClassifierByHistogram::destroyWindows() const
{
	cv::destroyWindow(local::windowNameClass1Gesture1);
	cv::destroyWindow(local::windowNameClass1Gesture2);
	cv::destroyWindow(local::windowNameClass1Gesture3);
	cv::destroyWindow(local::windowNameClass2Gesture1);
	cv::destroyWindow(local::windowNameClass2Gesture2);
	cv::destroyWindow(local::windowNameClass2Gesture3);
	cv::destroyWindow(local::windowNameClass3Gesture1);
	cv::destroyWindow(local::windowNameClass3Gesture2);
	cv::destroyWindow(local::windowNameClass3Gesture3);
	cv::destroyWindow(local::windowNameForTemporalOrientationHistogram);
	cv::destroyWindow(local::windowNameForHorizontalOrientationHistogram);
	cv::destroyWindow(local::windowNameForVerticalOrientationHistogram);
}

}  // namespace swl
