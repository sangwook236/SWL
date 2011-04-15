#include "swl/gesture_recognition/GestureClassifierByHistogram.h"
#include "HistogramAccumulator.h"
#include "HistogramMatcher.h"
#include "HistogramGenerator.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>


namespace swl {

//#define __VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_ 1


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

const int magHistBins = 30;
const int magHistSize[] = { magHistBins };
// magnitude varies from 1 to 30
const float magHistRange1[] = { 1, magHistBins + 1 };
const float *magHistRanges[] = { magHistRange1 };
// we compute the histogram from the 0-th channel
const int magHistChannels[] = { 0 };
const int magHistBinWidth = 5, magHistMaxHeight = 100;

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
const double shortTimeGestureRefHistogramSigma = 8.0;
const double longTimeGestureRefHistogramSigma = 16.0;
const double thirdClassGestureRefHistogramSigma = 20.0;
const size_t refHistogramBinNum = phaseHistBins;
const double refHistogramNormalizationFactor = 5000.0;
const double gesturePatternHistogramSigma = 1.0;
const size_t gesturePatternHistogramBinNum = indexHistBins;
const double gesturePatternHistogramNormalizationFactor = 10.0; //(double)params_.MAX_MATCHED_HISTOGRAM_NUM;

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
const std::string windowName1("gesture recognition - magnitude histogram");
const std::string windowNameShortTimeGesture1("gesture recognition - (STG) actual histogram");
const std::string windowNameShortTimeGesture2("gesture recognition - (STG) matched histogram");
const std::string windowNameShortTimeGesture3("gesture recognition - (STG) matched id histogram");
const std::string windowNameLongTimeGesture1("gesture recognition - (LTG) actual histogram");
const std::string windowNameLongTimeGesture2("gesture recognition - (LTG) matched histogram");
const std::string windowNameLongTimeGesture3("gesture recognition - (LTG) matched id histogram");
const std::string windowNameThirdClassGesture1("gesture recognition - (TCG) actual histogram");
const std::string windowNameThirdClassGesture2("gesture recognition - (TCG) matched histogram");
const std::string windowNameThirdClassGesture3("gesture recognition - (TCG) matched id histogram");
const std::string windowNameForTemporalPhaseHistogram("gesture recognition - temporal histogram");
#endif

}  // namespace local
}  // unnamed namespace

//-----------------------------------------------------------------------------
//

GestureClassifierByHistogram::GestureClassifierByHistogram(const Params &params)
: base_type(),
  params_(params),
  refFullPhaseHistograms_(),
  refHistogramsForShortTimeGesture_(), refHistogramsForLongTimeGesture_(), refHistogramsForThirdClassGesture_(),
  gestureIdPatternHistogramsForShortTimeGesture_(),
  histogramAccumulatorForShortTimeGesture_(params_.doesApplyTimeWeighting ? new HistogramAccumulator(local::getHistogramTimeWeight(params_.ACCUMULATED_HISTOGRAM_NUM_FOR_SHORT_TIME_GESTURE)) : new HistogramAccumulator(params_.ACCUMULATED_HISTOGRAM_NUM_FOR_SHORT_TIME_GESTURE)),
  histogramAccumulatorForLongTimeGesture_(params_.doesApplyTimeWeighting ? new HistogramAccumulator(local::getHistogramTimeWeight(params_.ACCUMULATED_HISTOGRAM_NUM_FOR_LONG_TIME_GESTURE)) : new HistogramAccumulator(params_.ACCUMULATED_HISTOGRAM_NUM_FOR_LONG_TIME_GESTURE)),
  histogramAccumulatorForThirdClassGesture_(params_.doesApplyTimeWeighting ? new HistogramAccumulator(local::getHistogramTimeWeight(params_.ACCUMULATED_HISTOGRAM_NUM_FOR_THIRD_CLASS_GESTURE)) : new HistogramAccumulator(params_.ACCUMULATED_HISTOGRAM_NUM_FOR_THIRD_CLASS_GESTURE)),
  matchedHistogramIndexes1ForShortTimeGesture_(params_.MAX_MATCHED_HISTOGRAM_NUM), matchedHistogramIndexes2ForShortTimeGesture_(params_.MAX_MATCHED_HISTOGRAM_NUM), matchedHistogramIndexesForLongTimeGesture_(params_.MAX_MATCHED_HISTOGRAM_NUM), matchedHistogramIndexesForThirdClassGesture_(params_.MAX_MATCHED_HISTOGRAM_NUM),
  gestureId_(GestureType::GT_UNDEFINED)
{
	createReferenceFullPhaseHistograms();
	createReferenceHistogramsForShortTimeGesture();
	createReferenceHistogramsForLongTimeGesture();
	createReferenceHistogramsForThirdClassGesture();
	createGestureIdPatternHistogramsForShortTimeGesture();
}

GestureClassifierByHistogram::GestureClassifierByHistogram(const GestureClassifierByHistogram &rhs)
: base_type(),
  params_(rhs.params_),
  refFullPhaseHistograms_(rhs.refFullPhaseHistograms_),
  refHistogramsForShortTimeGesture_(rhs.refHistogramsForShortTimeGesture_), refHistogramsForLongTimeGesture_(rhs.refHistogramsForLongTimeGesture_), refHistogramsForThirdClassGesture_(rhs.refHistogramsForThirdClassGesture_),
  gestureIdPatternHistogramsForShortTimeGesture_(rhs.gestureIdPatternHistogramsForShortTimeGesture_),
  histogramAccumulatorForShortTimeGesture_(rhs.histogramAccumulatorForShortTimeGesture_), histogramAccumulatorForLongTimeGesture_(rhs.histogramAccumulatorForLongTimeGesture_), histogramAccumulatorForThirdClassGesture_(rhs.histogramAccumulatorForThirdClassGesture_),
  matchedHistogramIndexes1ForShortTimeGesture_(rhs.matchedHistogramIndexes1ForShortTimeGesture_),  matchedHistogramIndexes2ForShortTimeGesture_(rhs.matchedHistogramIndexes2ForShortTimeGesture_), matchedHistogramIndexesForLongTimeGesture_(rhs.matchedHistogramIndexesForLongTimeGesture_), matchedHistogramIndexesForThirdClassGesture_(rhs.matchedHistogramIndexesForThirdClassGesture_),
  gestureId_(rhs.gestureId_)
{
}

GestureClassifierByHistogram::~GestureClassifierByHistogram()
{
}

GestureClassifierByHistogram & GestureClassifierByHistogram::operator=(const GestureClassifierByHistogram &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;

	params_ = rhs.params_;
	refFullPhaseHistograms_.assign(rhs.refFullPhaseHistograms_.begin(), rhs.refFullPhaseHistograms_.end());
	refHistogramsForShortTimeGesture_.assign(rhs.refHistogramsForShortTimeGesture_.begin(), rhs.refHistogramsForShortTimeGesture_.end());
	refHistogramsForLongTimeGesture_.assign(rhs.refHistogramsForLongTimeGesture_.begin(), rhs.refHistogramsForLongTimeGesture_.end());
	refHistogramsForThirdClassGesture_.assign(rhs.refHistogramsForThirdClassGesture_.begin(), rhs.refHistogramsForThirdClassGesture_.end());
	gestureIdPatternHistogramsForShortTimeGesture_.assign(rhs.gestureIdPatternHistogramsForShortTimeGesture_.begin(), rhs.gestureIdPatternHistogramsForShortTimeGesture_.end());
	histogramAccumulatorForShortTimeGesture_ = rhs.histogramAccumulatorForShortTimeGesture_;
	histogramAccumulatorForLongTimeGesture_ = rhs.histogramAccumulatorForLongTimeGesture_;
	histogramAccumulatorForThirdClassGesture_ = rhs.histogramAccumulatorForThirdClassGesture_;
	matchedHistogramIndexes1ForShortTimeGesture_.assign(rhs.matchedHistogramIndexes1ForShortTimeGesture_.begin(), rhs.matchedHistogramIndexes1ForShortTimeGesture_.end());
	matchedHistogramIndexes2ForShortTimeGesture_.assign(rhs.matchedHistogramIndexes2ForShortTimeGesture_.begin(), rhs.matchedHistogramIndexes2ForShortTimeGesture_.end());
	matchedHistogramIndexesForLongTimeGesture_.assign(rhs.matchedHistogramIndexesForLongTimeGesture_.begin(), rhs.matchedHistogramIndexesForLongTimeGesture_.end());
	matchedHistogramIndexesForThirdClassGesture_.assign(rhs.matchedHistogramIndexesForThirdClassGesture_.begin(), rhs.matchedHistogramIndexesForThirdClassGesture_.end());
	gestureId_ = rhs.gestureId_;

	return *this;
}

void GestureClassifierByHistogram::initWindows() const
{
#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	cv::namedWindow(local::windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameShortTimeGesture1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameShortTimeGesture2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameShortTimeGesture3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameLongTimeGesture1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameLongTimeGesture2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameLongTimeGesture3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameThirdClassGesture1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameThirdClassGesture2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameThirdClassGesture3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(local::windowNameForTemporalPhaseHistogram, cv::WINDOW_AUTOSIZE);
#endif
}

void GestureClassifierByHistogram::destroyWindows() const
{
#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	cv::destroyWindow(local::windowName1);
	cv::destroyWindow(local::windowNameShortTimeGesture1);
	cv::destroyWindow(local::windowNameShortTimeGesture2);
	cv::destroyWindow(local::windowNameShortTimeGesture3);
	cv::destroyWindow(local::windowNameLongTimeGesture1);
	cv::destroyWindow(local::windowNameLongTimeGesture2);
	cv::destroyWindow(local::windowNameLongTimeGesture3);
	cv::destroyWindow(local::windowNameThirdClassGesture1);
	cv::destroyWindow(local::windowNameThirdClassGesture2);
	cv::destroyWindow(local::windowNameThirdClassGesture3);
	cv::destroyWindow(local::windowNameForTemporalPhaseHistogram);
#endif
}

/*virtual*/ bool GestureClassifierByHistogram::analyzeOpticalFlow(const cv::Rect & /*roi*/, const cv::Mat &flow, const cv::Mat *flow2 /*= NULL*/)
{
	cv::MatND hist;

	{
		std::vector<cv::Mat> flows;
		cv::split(flow, flows);

		cv::Mat flow_phase, flow_mag;
		cv::phase(flows[0], flows[1], flow_phase, true);  // return type: CV_32F
		cv::magnitude(flows[0], flows[1], flow_mag);  // return type: CV_32F

		// filter by magnitude
		if (params_.doesApplyMagnitudeFiltering)
		{
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(flow_mag, &minVal, &maxVal, NULL, NULL);
			const double mag_min_threshold = minVal + (maxVal - minVal) * params_.magnitudeFilteringMinThresholdRatio;
			const double mag_max_threshold = minVal + (maxVal - minVal) * params_.magnitudeFilteringMaxThresholdRatio;

			// TODO [check] >> magic number, -1 is correct ?
			flow_phase.setTo(cv::Scalar::all(-1), flow_mag < mag_min_threshold);
			flow_phase.setTo(cv::Scalar::all(-1), flow_mag > mag_max_threshold);

			flow_mag.setTo(cv::Scalar::all(0), flow_mag < mag_min_threshold);
			flow_mag.setTo(cv::Scalar::all(0), flow_mag > mag_max_threshold);
		}

		// calculate phase histogram
		cv::calcHist(&flow_phase, 1, local::phaseHistChannels, cv::Mat(), hist, local::histDims, local::phaseHistSize, local::phaseHistRanges, true, false);
		histogramAccumulatorForShortTimeGesture_->addHistogram(hist);
		histogramAccumulatorForLongTimeGesture_->addHistogram(hist);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		// FIXME [delete] >> draw magnitude histogram
		{
			// calculate magnitude histogram
			cv::calcHist(&flow_mag, 1, local::magHistChannels, cv::Mat(), hist, local::histDims, local::magHistSize, local::magHistRanges, true, false);
			// normalize histogram
			HistogramUtil::normalizeHistogram(hist, local::refHistogramNormalizationFactor);

			// draw magnitude histogram
			cv::Mat histImg(cv::Mat::zeros(local::magHistMaxHeight, local::magHistBins*local::magHistBinWidth, CV_8UC3));
			const double maxVal = local::refHistogramNormalizationFactor;
			HistogramUtil::drawHistogram1D(hist, local::magHistBins, maxVal, local::magHistBinWidth, local::magHistMaxHeight, histImg);

			cv::imshow(local::windowName1, histImg);
		}
#endif
	}

	if (flow2 && !flow2->empty())
	{
		std::vector<cv::Mat> flows;
		cv::split(*flow2, flows);

		cv::Mat flow_phase, flow_mag;
		cv::phase(flows[0], flows[1], flow_phase, true);  // return type: CV_32F
		cv::magnitude(flows[0], flows[1], flow_mag);  // return type: CV_32F

		// filter by magnitude
		if (params_.doesApplyMagnitudeFiltering)
		{
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(flow_mag, &minVal, &maxVal, NULL, NULL);
			const double mag_min_threshold = minVal + (maxVal - minVal) * params_.magnitudeFilteringMinThresholdRatio;
			const double mag_max_threshold = minVal + (maxVal - minVal) * params_.magnitudeFilteringMaxThresholdRatio;

			// TODO [check] >> magic number, -1 is correct ?
			flow_phase.setTo(cv::Scalar::all(-1), flow_mag < mag_min_threshold);
			flow_phase.setTo(cv::Scalar::all(-1), flow_mag > mag_max_threshold);

			flow_mag.setTo(cv::Scalar::all(0), flow_mag < mag_min_threshold);
			flow_mag.setTo(cv::Scalar::all(0), flow_mag > mag_max_threshold);
		}

		// calculate phase histogram
		cv::calcHist(&flow_phase, 1, local::phaseHistChannels, cv::Mat(), hist, local::histDims, local::phaseHistSize, local::phaseHistRanges, true, false);
		histogramAccumulatorForThirdClassGesture_->addHistogram(hist);
	}
	else
	{
		// TODO [check] >>
		//clearThirdClassGestureHistory();
	}

	return true;
}

/*virtual*/ bool GestureClassifierByHistogram::classifyGesture()
{
	gestureId_ = GestureType::GT_UNDEFINED;

	// classify long-time gesture
	//if (histogramAccumulatorForLongTimeGesture_->isFull() && classifyLongTimeGesture()) return true;

	// classify short-time gesture
	if (histogramAccumulatorForShortTimeGesture_->isFull() && classifyShortTimeGesture()) return true;

	// TODO [check] >>
	// classify 3rd-class gesture
	if (histogramAccumulatorForThirdClassGesture_->isFull() && classifyThirdClassGesture()) return true;

	return false;
}

void GestureClassifierByHistogram::clearShortTimeGestureHistory()
{
	histogramAccumulatorForShortTimeGesture_->clearAllHistograms();
	matchedHistogramIndexes1ForShortTimeGesture_.clear();
	matchedHistogramIndexes2ForShortTimeGesture_.clear();
}

void GestureClassifierByHistogram::clearLongTimeGestureHistory()
{
	histogramAccumulatorForLongTimeGesture_->clearAllHistograms();
	matchedHistogramIndexesForLongTimeGesture_.clear();
}

void GestureClassifierByHistogram::clearThirdClassGestureHistory()
{
	histogramAccumulatorForThirdClassGesture_->clearAllHistograms();
	matchedHistogramIndexesForThirdClassGesture_.clear();
}

void GestureClassifierByHistogram::clearTimeSeriesGestureHistory()
{
	// FIXME [implement] >>
}

bool GestureClassifierByHistogram::classifyShortTimeGesture()
{
	// create accumulated phase histograms
	cv::MatND &accumulatedHist = histogramAccumulatorForShortTimeGesture_->createAccumulatedHistogram();
	// normalize histogram
	HistogramUtil::normalizeHistogram(accumulatedHist, local::refHistogramNormalizationFactor);

	// FIXME [restore] >> have to decide which one is used
	{
		//cv::MatND &temporalHist = histogramAccumulatorForShortTimeGesture_->createTemporalHistogram();
		cv::MatND &temporalHist = histogramAccumulatorForShortTimeGesture_->createTemporalHistogram(refFullPhaseHistograms_, params_.histDistThresholdForShortTimeGesture);
		// normalize histogram
		HistogramUtil::normalizeHistogram(temporalHist, local::refHistogramNormalizationFactor);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		// draw temporal phase histogram
		drawTemporalPhaseHistogram(temporalHist, local::windowNameForTemporalPhaseHistogram);
#endif
	}

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	// draw accumulated phase histogram
	drawAccumulatedPhaseHistogram(accumulatedHist, local::windowNameShortTimeGesture1);
#endif

	// match histogram
	double minHistDist1 = std::numeric_limits<double>::max();
	const size_t &matchedIdx1 = refFullPhaseHistograms_.empty() ? -1 : HistogramMatcher::match(refFullPhaseHistograms_, accumulatedHist, minHistDist1);

// FIXME [restore] >>
#if 1
	if (minHistDist1 < params_.histDistThresholdForShortTimeGesture)
#else
	if ((((0 <= matchedIdx && matchedIdx <= 4) || (31 <= matchedIdx && matchedIdx <= 35)) && minHistDist < local::histDistThresholdForShortTimeGesture_LeftMove) ||
		(minHistDist < local::histDistThresholdForShortTimeGesture_Others))
#endif
	{
		matchedHistogramIndexes1ForShortTimeGesture_.push_back(matchedIdx1);

		// classify short-time gesture
		gestureId_ = classifyShortTimeGesture(matchedHistogramIndexes1ForShortTimeGesture_, true);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexes1ForShortTimeGesture_, local::windowNameShortTimeGesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refFullPhaseHistograms_, matchedIdx1, local::windowNameShortTimeGesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexes1ForShortTimeGesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameShortTimeGesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	// FIXME [modify] >> the above & this codes should be integrated
	double minHistDist2 = std::numeric_limits<double>::max();
	const size_t &matchedIdx2 = refHistogramsForShortTimeGesture_.empty() ? -1 : HistogramMatcher::match(refHistogramsForShortTimeGesture_, accumulatedHist, minHistDist2);

	if (minHistDist2 < params_.histDistThresholdForShortTimeGesture)
	{
		matchedHistogramIndexes2ForShortTimeGesture_.push_back(matchedIdx2);

		// classify short-time gesture
		gestureId_ = classifyShortTimeGesture(matchedHistogramIndexes2ForShortTimeGesture_, false);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexes2ForShortTimeGesture_, local::windowNameShortTimeGesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refHistogramsForShortTimeGesture_, matchedIdx2, local::windowNameShortTimeGesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexes2ForShortTimeGesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameShortTimeGesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	return false;
}

bool GestureClassifierByHistogram::classifyLongTimeGesture()
{
	// accumulate phase histograms
	cv::MatND &accumulatedHist = histogramAccumulatorForLongTimeGesture_->createAccumulatedHistogram();
	// normalize histogram
	HistogramUtil::normalizeHistogram(accumulatedHist, local::refHistogramNormalizationFactor);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	// draw accumulated phase histogram
	drawAccumulatedPhaseHistogram(accumulatedHist, local::windowNameLongTimeGesture1);
#endif

	// match histogram
	double minHistDist = std::numeric_limits<double>::max();
	const size_t &matchedIdx = refHistogramsForLongTimeGesture_.empty() ? -1 : HistogramMatcher::match(refHistogramsForLongTimeGesture_, accumulatedHist, minHistDist);

	if (minHistDist < params_.histDistThresholdForLongTimeGesture)
	{
		matchedHistogramIndexesForLongTimeGesture_.push_back(matchedIdx);

		// classify long-time gesture
		gestureId_ = classifyLongTimeGesture(matchedHistogramIndexesForLongTimeGesture_);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexesForLongTimeGesture_, local::windowNameLongTimeGesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refHistogramsForLongTimeGesture_, matchedIdx, local::windowNameLongTimeGesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexesForLongTimeGesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameLongTimeGesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	return false;
}

bool GestureClassifierByHistogram::classifyThirdClassGesture()
{
	// accumulate phase histograms
	cv::MatND &accumulatedHist = histogramAccumulatorForThirdClassGesture_->createAccumulatedHistogram();
	// normalize histogram
	HistogramUtil::normalizeHistogram(accumulatedHist, local::refHistogramNormalizationFactor);

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
	// draw accumulated phase histogram
	drawAccumulatedPhaseHistogram(accumulatedHist, local::windowNameThirdClassGesture1);
#endif

	// match histogram
	double minHistDist = std::numeric_limits<double>::max();
	const size_t &matchedIdx = refHistogramsForThirdClassGesture_.empty() ? -1 : HistogramMatcher::match(refHistogramsForThirdClassGesture_, accumulatedHist, minHistDist);

	if (minHistDist < params_.histDistThresholdForThirdClassGesture)
	{
		matchedHistogramIndexesForThirdClassGesture_.push_back(matchedIdx);

		// classify 3rd-class gesture
		gestureId_ = classifyThirdClassGesture(matchedHistogramIndexesForThirdClassGesture_);

		// FIXME [delete] >>
		//std::cout << matchedIdx << ", " << minHistDist << std::endl;

#if defined(__VISUALIZE_HISTOGRAMS_IN_GESTURE_CLASSIFIER_BY_HISTOGRAM_)
		if (GestureType::GT_UNDEFINED != gestureId_)
		{
			// FIXME [modify] >> delete ???
			// draw matched index histogram
			drawMatchedIdPatternHistogram(matchedHistogramIndexesForThirdClassGesture_, local::windowNameThirdClassGesture3);

			// draw matched reference histogram
			drawMatchedReferenceHistogram(refHistogramsForThirdClassGesture_, matchedIdx, local::windowNameThirdClassGesture2);
		}
#endif

		//return GestureType::GT_UNDEFINED != gestureId_;
		if (GestureType::GT_UNDEFINED != gestureId_) return true;
	}
	else
	{
		matchedHistogramIndexesForThirdClassGesture_.push_back(-1);

		// FIXME [delete] >>
		//std::cout << "-----------, " << minHistDist << std::endl;

		//cv::imshow(windowNameThirdClassGesture2, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));

		//return false;
	}

	return false;
}

GestureType::Type GestureClassifierByHistogram::classifyShortTimeGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const bool useGestureIdPattern) const
{
	// match histogram by gesture ID pattern
	if (useGestureIdPattern)
	{
		const size_t &matchedIdx = matchHistogramByGestureIdPattern(matchedHistogramIndexes, gestureIdPatternHistogramsForShortTimeGesture_, params_.histDistThresholdForGestureIdPattern);
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
		//const size_t countThreshold(params_.matchedIndexCountThresholdForShortTimeGesture);

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

GestureType::Type GestureClassifierByHistogram::classifyLongTimeGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const
{
#if 0
	const size_t &matchedIdx = matchHistogramByGestureIdPattern(matchedHistogramIndexes, gestureIdPatternHistogramsForLongTimeGesture_, params_.histDistThresholdForGestureIdPattern);
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
	//const size_t countThreshold(params_.matchedIndexCountThresholdForLongTimeGesture);

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

GestureType::Type GestureClassifierByHistogram::classifyThirdClassGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes) const
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
	//const size_t countThreshold(params_.matchedIndexCountThresholdForThirdClassGesture);

	const size_t &matchedIdx = matchHistogramByFrequency(matchedHistogramIndexes, countThreshold);
	switch (matchedIdx)
	{
/*
	case :
		return GestureType::GT_LEFT_FAST_MOVE;
	case :
		return GestureType::GT_RIGHT_FAST_MOVE;
	}
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

		cv::imshow(local::windowNameShortTimeGesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createReferenceHistogramsForShortTimeGesture()
{
	// create reference histograms
	ReferenceHistogramGeneratorForShortTimeGesture refHistogramGenerator(local::shortTimeGestureRefHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refHistogramsForShortTimeGesture_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistogramsForShortTimeGesture_.begin(); it != refHistogramsForShortTimeGesture_.end(); ++it)
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

		cv::imshow(local::windowNameShortTimeGesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createReferenceHistogramsForLongTimeGesture()
{
	// create reference histograms
	ReferenceHistogramGeneratorForLongTimeGesture refHistogramGenerator(local::longTimeGestureRefHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refHistogramsForLongTimeGesture_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistogramsForLongTimeGesture_.begin(); it != refHistogramsForLongTimeGesture_.end(); ++it)
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

		cv::imshow(local::windowNameShortTimeGesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createReferenceHistogramsForThirdClassGesture()
{
	// create reference histograms
	ReferenceHistogramGeneratorForThirdClassGesture refHistogramGenerator(local::thirdClassGestureRefHistogramSigma);
	refHistogramGenerator.createHistograms(local::phaseHistBins, local::refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refHistogramsForThirdClassGesture_.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistogramsForThirdClassGesture_.begin(); it != refHistogramsForThirdClassGesture_.end(); ++it)
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

		cv::imshow(local::windowNameShortTimeGesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

void GestureClassifierByHistogram::createGestureIdPatternHistogramsForShortTimeGesture()
{
	// create gesture pattern histograms
	GestureIdPatternHistogramGeneratorForShortTimeGesture gestureIdPatternHistogramGenerator(local::gesturePatternHistogramSigma);
	gestureIdPatternHistogramGenerator.createHistograms(local::gesturePatternHistogramBinNum, local::gesturePatternHistogramNormalizationFactor);
	const std::vector<cv::MatND> &gesturePatternHistograms = gestureIdPatternHistogramGenerator.getHistograms();

	gestureIdPatternHistogramsForShortTimeGesture_.assign(gesturePatternHistograms.begin(), gesturePatternHistograms.end());

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

		cv::imshow(local::windowNameShortTimeGesture2, histImg);
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
	//HistogramUtil::normalizeHistogram(hist, params_.MAX_MATCHED_HISTOGRAM_NUM);

	// draw matched index histogram
	cv::Mat histImg(cv::Mat::zeros(local::indexHistMaxHeight, local::indexHistBins*local::indexHistBinWidth, CV_8UC3));
	HistogramUtil::drawHistogram1D(hist, local::indexHistBins, params_.MAX_MATCHED_HISTOGRAM_NUM, local::indexHistBinWidth, local::indexHistMaxHeight, histImg);
				
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

void GestureClassifierByHistogram::drawTemporalPhaseHistogram(const cv::MatND &temporalHist, const std::string &windowName) const
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

}  // namespace swl
