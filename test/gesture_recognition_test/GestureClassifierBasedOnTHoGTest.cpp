//#include "stdafx.h"
#include "HistogramGenerator.h"  // FIXME [solve] >> these files are duplicates of ${SWL_ROOT}/src/swl_pattern_recognition/HistogramGenerator.h & .cpp
#include "swl/pattern_recognition/MotionSegmenter.h"
#include "swl/rnd_util/HistogramAccumulator.h"
#include "swl/rnd_util/HistogramUtil.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <ctime>


namespace {
namespace local {

void filterMotionRegionByCCL(cv::Mat &detected_motion, const size_t minMotionAreaThreshold, const size_t maxMotionAreaThreshold, std::vector<std::vector<cv::Point> > &selectedContours)
{
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(detected_motion, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (!contours.empty())
	{
		for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
		{
			const double &area = std::fabs(cv::contourArea(cv::Mat(*it)));
#if 1
			if (minMotionAreaThreshold < area && area < maxMotionAreaThreshold)  // reject too large & small motion regions
				selectedContours.push_back(*it);
#else
			if (area > maxMotionAreaThreshold)  // reject too large motion regions
			{
				selectedContours.clear();
				break;
			}
			else if (area > minMotionAreaThreshold)  // reject small motion regions
				selectedContours.push_back(*it);
#endif
		}
	}
}

void calcOrientationAndMagnitudeUsingOpticalFlow(const cv::Mat &flow, const bool doesApplyMagnitudeFiltering, const double magnitudeFilteringMinThresholdRatio, const double magnitudeFilteringMaxThresholdRatio, cv::Mat &orientation, cv::Mat &magnitude)
{
	std::vector<cv::Mat> flows;
	cv::split(flow, flows);

	cv::phase(flows[0], flows[1], orientation, true);  // return type: CV_32F
	cv::magnitude(flows[0], flows[1], magnitude);  // return type: CV_32F

	// filter by magnitude
	if (doesApplyMagnitudeFiltering)
	{
		double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(magnitude, &minVal, &maxVal, NULL, NULL);
		const double mag_min_threshold = minVal + (maxVal - minVal) * magnitudeFilteringMinThresholdRatio;
		const double mag_max_threshold = minVal + (maxVal - minVal) * magnitudeFilteringMaxThresholdRatio;

		// TODO [check] >> magic number, -1 is correct ?
#if 0
		orientation.setTo(cv::Scalar::all(-1), magnitude < mag_min_threshold);
		orientation.setTo(cv::Scalar::all(-1), magnitude > mag_max_threshold);
#else
		orientation.setTo(cv::Scalar::all(-1), magnitude < mag_min_threshold | magnitude > mag_max_threshold);
#endif

#if 0
		magnitude.setTo(cv::Scalar::all(0), magnitude < mag_min_threshold);
		magnitude.setTo(cv::Scalar::all(0), magnitude > mag_max_threshold);
#else
		magnitude.setTo(cv::Scalar::all(0), magnitude < mag_min_threshold | magnitude > mag_max_threshold);
#endif
	}
}

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

//
const size_t accumulatedOrientationHistogramNum = 10;
const size_t accumulatedHistogramNumForClass2Gesture = 15;
const size_t accumulatedHistogramNumForClass3Gesture = 30;
const size_t maxMatchedHistogramNum = 10;

const double histDistThresholdForTemporalOrientationHistogram = 0.5;
const double histDistThresholdForClass2Gesture = 0.5;
const double histDistThresholdForClass3Gesture = 0.5;

const double histDistThresholdForGestureIdPattern = 0.8;

const size_t matchedIndexCountThresholdForClass1Gesture = maxMatchedHistogramNum / 2;  // currently not used
const size_t matchedIndexCountThresholdForClass2Gesture = maxMatchedHistogramNum / 2;  // currently not used
const size_t matchedIndexCountThresholdForClass3Gesture = maxMatchedHistogramNum / 2;  // currently not used

const bool doesApplyMagnitudeFiltering = true;
const double magnitudeFilteringMinThresholdRatio = 0.3;
const double magnitudeFilteringMaxThresholdRatio = 1.0;
const bool doesApplyTimeWeighting = true;
const bool doesApplyMagnitudeWeighting = false;  // FIXME [implement] >> not yet supported

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

const int phaseHorzScale = 10, phaseVertScale = 1;

//
const double refFullPhaseHistogramSigma = 8.0;
const double class1GestureRefHistogramSigma = 8.0;
const double class2GestureRefHistogramSigma = 16.0;
const double class3GestureRefHistogramSigma = 20.0;
const size_t refHistogramBinNum = phaseHistBins;
const double refHistogramNormalizationFactor = 5000.0;
const double gesturePatternHistogramSigma = 1.0;
const size_t gesturePatternHistogramBinNum = swl::ReferenceFullPhaseHistogramGenerator::REF_HISTOGRAM_NUM;
const double gesturePatternHistogramNormalizationFactor = 10.0; //(double)maxMatchedHistogramNum;

//
std::vector<swl::HistogramAccumulator::histogram_type> refFullPhaseHistograms;
boost::shared_ptr<swl::HistogramAccumulator> orientationHistogramAccumulator(doesApplyTimeWeighting ? new swl::HistogramAccumulator(getHistogramTimeWeight(accumulatedOrientationHistogramNum)) : new swl::HistogramAccumulator(accumulatedOrientationHistogramNum));

void accumulateOrientationHistogram(const cv::Mat &orientation, std::ostream *stream)
{
	// calculate phase histogram
	cv::MatND hist;
	cv::calcHist(&orientation, 1, phaseHistChannels, cv::Mat(), hist, histDims, phaseHistSize, phaseHistRanges, true, false);

	//
	orientationHistogramAccumulator->addHistogram(hist);

	// save HoG
	if (NULL != stream)
	{
		swl::HistogramUtil::normalizeHistogram(hist, std::floor(refHistogramNormalizationFactor / accumulatedOrientationHistogramNum + 0.5));
		for (int row = 0; row < hist.rows; ++row)
			*stream << hist.at<float>(row, 0) << " ";
		*stream << std::endl;
	}
}

void clearOrientationHistogramHistory()
{
	orientationHistogramAccumulator->clearAllHistograms();
}

void drawTemporalOrientationHistogram(const cv::MatND &temporalHist, const std::string &windowName)
{
#if 1
	double maxVal = 0.0;
	cv::minMaxLoc(temporalHist, NULL, &maxVal, NULL, NULL);
#else
	const double maxVal = refHistogramNormalizationFactor * 0.05;
#endif

	cv::Mat histImg(cv::Mat::zeros(temporalHist.rows*phaseVertScale, temporalHist.cols*phaseHorzScale, CV_8UC3));
	swl::HistogramUtil::drawHistogram2D(temporalHist, temporalHist.cols, temporalHist.rows, maxVal, phaseHorzScale, phaseVertScale, histImg);

	cv::imshow(windowName, histImg);
}

void createReferenceFullPhaseHistograms()
{
	// create reference histograms
	swl::ReferenceFullPhaseHistogramGenerator refHistogramGenerator(refFullPhaseHistogramSigma);
	refHistogramGenerator.createHistograms(phaseHistBins, refHistogramNormalizationFactor);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

	refFullPhaseHistograms.assign(refHistograms.begin(), refHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistograms_.begin(); it != refHistograms_.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = refHistogramNormalizationFactor * 0.05;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(local::phaseHistMaxHeight, local::phaseHistBins*local::phaseHistBinWidth, CV_8UC3));
		HistogramUtil::drawHistogram1D(*it, local::phaseHistBins, maxVal, local::phaseHistBinWidth, local::phaseHistMaxHeight, histImg);

		cv::imshow(local::windowNameClass1Gesture2, histImg);
		cv::waitKey(0);
	}
#endif
}

bool computeTemporalOrientationHistogram(cv::MatND &temporalOrientationHist)
{
	// classify gesture
	if (orientationHistogramAccumulator->isFull())
	{
		// create accumulated phase histograms
		cv::MatND accumulatedHist(orientationHistogramAccumulator->createAccumulatedHistogram());
		// normalize histogram
		swl::HistogramUtil::normalizeHistogram(accumulatedHist, refHistogramNormalizationFactor);

		// FIXME [restore] >> have to decide which one is used
#if 1
		temporalOrientationHist = orientationHistogramAccumulator->createTemporalHistogram();
#else
		temporalOrientationHist = orientationHistogramAccumulator->createTemporalHistogram(refFullPhaseHistograms, histDistThresholdForTemporalOrientationHistogram);
#endif
		// normalize histogram
		swl::HistogramUtil::normalizeHistogram(temporalOrientationHist, refHistogramNormalizationFactor);

		return true;
	}

	return false;
}

bool classifyGesture(const cv::MatND &temporalOrientationHist)
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");

	return false;
}

}  // namespace local
}  // unnamed namespace

// temporal HoG (THoG) or temporal orientation histogram (TOH)
void recognizeGestureBasedOnTHoG(cv::VideoCapture &capture, std::ostream *streamTHoG, std::ostream *streamHoG)
{
	const int IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480;
	const bool IMAGE_DOWNSIZING = true;

	const double MHI_TIME_DURATION = 0.5;
	//const size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 1000 : 2000, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);
	const size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 100 : 200, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);

	//
	local::createReferenceFullPhaseHistograms();

	//
	const std::string windowName1("gesture recognition by THoG - Motion");
	const std::string windowName2("gesture recognition by THoG - THoG");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, img, tmp_img, blurred;
	cv::Mat processed_mhi, component_label_map;
	cv::Mat contour_mask;
	cv::Mat prev_img_for_flow, curr_img_for_flow, flow, flow_phase, flow_mag;
	cv::MatND temporalOrientationHist;
	for (;;)
	{
		const double timestamp = (double)std::clock() / CLOCKS_PER_SEC;  // get current time in seconds

		if (IMAGE_DOWNSIZING)
		{
			capture >> frame2;
			if (frame2.empty())
			{
				std::cout << "a frame not found ..." << std::endl;
				break;
				//continue;
			}

			//cv::resize(frame2, frame, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else
		{
			capture >> frame;
			if (frame.empty())
			{
				std::cout << "a frame not found ..." << std::endl;
				break;
				//continue;
			}
		}

		cv::cvtColor(frame, gray, CV_BGR2GRAY);

		//if (blurred.empty()) blurred = gray.clone();

		// smoothing
#if 0
		// down-scale and up-scale the image to filter out the noise
		cv::pyrDown(gray, blurred);
		cv::pyrUp(blurred, gray);
#elif 0
		blurred = gray;
		cv::boxFilter(blurred, gray, blurred.type(), cv::Size(5, 5));
#endif

		cv::cvtColor(gray, img, CV_GRAY2BGR);

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32FC1);
			if (processed_mhi.empty())
				processed_mhi.create(mhi.size(), mhi.type());
			if (component_label_map.empty())
				component_label_map.create(mhi.size(), CV_8UC1);
			if (contour_mask.empty())
				contour_mask.create(mhi.size(), CV_8UC1);

			std::vector<cv::Rect> component_rects;
			processed_mhi.setTo(cv::Scalar::all(0));
			component_label_map.setTo(cv::Scalar::all(0));
			swl::MotionSegmenter::segmentUsingMHI(timestamp, MHI_TIME_DURATION, prevgray, gray, mhi, processed_mhi, component_label_map, component_rects);

			//img.setTo(cv::Scalar(255, 255, 255), component_label_map);

			//
			std::vector<std::vector<cv::Point> > selectedContours;
			local::filterMotionRegionByCCL(component_label_map, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD, selectedContours);

			contour_mask.setTo(cv::Scalar::all(0));
			if (!selectedContours.empty())
			{
				size_t idx = 0;
				for (std::vector<std::vector<cv::Point> >::iterator it = selectedContours.begin(); it != selectedContours.end(); ++it, ++idx)
				{
					cv::drawContours(contour_mask, selectedContours, idx, cv::Scalar::all(255), CV_FILLED, 8);

					cv::drawContours(img, selectedContours, idx, CV_RGB(0, 0, 255), 2, 8);  // for display
				}
			}

			//
			if (!selectedContours.empty())
			{
				prevgray.copyTo(prev_img_for_flow, contour_mask);
				gray.copyTo(curr_img_for_flow, contour_mask);

				// calculate optical flow
				cv::calcOpticalFlowFarneback(prev_img_for_flow, curr_img_for_flow, flow, 0.5, 3, 15, 3, 5, 1.1, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
				//cv::calcOpticalFlowFarneback(prev_img_for_flow, curr_img_for_flow, flow, 0.5, 7, 15, 3, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
				//cv::calcOpticalFlowFarneback(prev_img_for_flow, curr_img_for_flow, flow, 0.25, 7, 15, 3, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);

				//local::calcOrientationUsingOpticalFlow(flow, local::doesApplyMagnitudeFiltering, local::magnitudeFilteringMinThresholdRatio, local::magnitudeFilteringMaxThresholdRatio, flow_phase);
				local::calcOrientationAndMagnitudeUsingOpticalFlow(flow, local::doesApplyMagnitudeFiltering, local::magnitudeFilteringMinThresholdRatio, local::magnitudeFilteringMaxThresholdRatio, flow_phase, flow_mag);

				local::accumulateOrientationHistogram(flow_phase, streamHoG);
			}
			else
			{
				std::cout << "motion not detected !!!" << std::endl;
				local::clearOrientationHistogramHistory();
			}

			if (local::computeTemporalOrientationHistogram(temporalOrientationHist))
			{
				// draw THoG
				local::drawTemporalOrientationHistogram(temporalOrientationHist, windowName2);

				//if (local::classifyGesture(temporalOrientationHist))  // not yet implemented
				{
					// FIXME [implement] >>
				}

				// save THoG
				if (NULL != streamTHoG)
				{
					for (int col = 0; col < temporalOrientationHist.cols; ++col)
						for (int row = 0; row < temporalOrientationHist.rows; ++row)
							*streamTHoG << temporalOrientationHist.at<float>(row, col) << " ";
					*streamTHoG << std::endl;
				}
			}

			//
			cv::imshow(windowName1, img);
		}

		const int ch = cv::waitKey(1);
		if ('q' == ch || 27 == ch)  // 'q' or ESC
			break;

		std::swap(prevgray, gray);
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}
