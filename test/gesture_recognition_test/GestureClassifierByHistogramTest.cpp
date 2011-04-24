//#define __USE_IR_SENSOR 1

#include "stdafx.h"
#include "swl/pattern_recognition/GestureClassifierByHistogram.h"
#include "swl/pattern_recognition/MotionSegmenter.h"
#if defined(__USE_IR_SENSOR)
#include "VideoInput/videoInput.h"
#endif
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/smart_ptr.hpp>
#include <ctime>
#include <stdio.h>
#include <iostream>

void gestureRecognitionByHistogram()
{
	const int imageWidth = 640, imageHeight = 480;
	//const int imageWidth = 320, imageHeight = 240;

#if defined(__USE_IR_SENSOR)
	videoInput VI;
	
	// prints out a list of available devices and returns num of devices found
	const int numDevices = VI.listDevices();	
	const int device1 = 0;  // this could be any deviceID that shows up in listDevices
		
	// if you want to capture at a different frame rate (default is 30) 
	// specify it here, you are not guaranteed to get this fps though.
	VI.setIdealFramerate(device1, 30);	
	
	// setup the first device - there are a number of options:
	VI.setupDevice(device1, 720, 480, VI_COMPOSITE);  // or setup device with video size and connection type
		
	// as requested width and height can not always be accomodated make sure to check the size once the device is setup
	const int width = VI.getWidth(device1);
	const int height = VI.getHeight(device1);
	const int size = VI.getSize(device1);
#else
	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "fail to open vision sensor" << std::endl;
		return;
	}

	//capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

	//const double &propFrameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	//const double &propFrameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
#endif

	bool isPowered = false;

	const std::string windowName("gesture recognition by histogram");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	// TODO [adjust] >> design parameter of gesture classifier
	swl::GestureClassifierByHistogram::Params params;
	{
		params.accumulatedHistogramNumForClass1Gesture = 3;
		params.accumulatedHistogramNumForClass2Gesture = 15;
		params.accumulatedHistogramNumForClass3Gesture = 30;
		params.maxMatchedHistogramNum = 10;

		params.histDistThresholdForClass1Gesture = 0.38;
		//params.histDistThresholdForClass1Gesture_LeftMove = 0.4;
		//params.histDistThresholdForClass1Gesture_Others = params.histDistThresholdForClass1Gesture;
		params.histDistThresholdForClass2Gesture = 0.5;
		params.histDistThresholdForClass3Gesture = 0.5;

		params.histDistThresholdForGestureIdPattern = 0.8;

		params.matchedIndexCountThresholdForClass1Gesture = params.maxMatchedHistogramNum / 2;  // currently not used
		params.matchedIndexCountThresholdForClass2Gesture = params.maxMatchedHistogramNum / 2;  // currently not used
		params.matchedIndexCountThresholdForClass3Gesture = params.maxMatchedHistogramNum / 2;  // currently not used

		params.doesApplyMagnitudeFiltering = true;
		params.magnitudeFilteringMinThresholdRatio = 0.3;
		params.magnitudeFilteringMaxThresholdRatio = 1.0;
		params.doesApplyTimeWeighting = true;
		params.doesApplyMagnitudeWeighting = false;  // FIXME [implement] >> not yet supported
	}

	// gesture classifier
	boost::shared_ptr<swl::IGestureClassifier> gestureClassifier(new swl::GestureClassifierByHistogram(params));
	dynamic_cast<swl::GestureClassifierByHistogram *>(gestureClassifier.get())->initWindows();

	const double MHI_TIME_DURATION = 1.0;

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, img, tmp_img, blurred;
	// FIXME [check] >> for fast gesture analysis
	cv::Mat prevgray2, gray2, mhi2;
	for (;;)
	{
		const double timestamp = (double)std::clock() / CLOCKS_PER_SEC;  // get current time in seconds

#if defined(__USE_IR_SENSOR)
		if (frame.empty())
			frame = cv::Mat::zeros(cv::Size(720, 480), CV_8UC3);

		// to get the data from the device first check if the data is new
		if (VI.isFrameNew(device1))
			VI.getPixels(device1, (unsigned char *)frame.data, false, true); //fills pixels as a BGR (for openCV) unsigned char array - flipping
#else
#if 1
		capture >> frame;
#else
		capture >> frame2;

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif
#endif

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

		// FIXME [check] >> for fast gesture analysis
#if 0
		cv::pyrDown(gray, blurred);
		cv::pyrDown(blurred, gray2);
#elif 1
		cv::pyrDown(gray, gray2);
#elif 0
		cv::resize(gray, gray2, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
#elif 0
		cv::resize(gray, gray2, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
#endif

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32FC1);

			cv::Mat processed_mhi, component_label_map;
			std::vector<cv::Rect> component_rects;
			swl::MotionSegmenter::segmentUsingMHI(timestamp, MHI_TIME_DURATION, prevgray, gray, mhi, processed_mhi, component_label_map, component_rects);

			// FIXME [check] >> for fast gesture analysis
			if (mhi2.empty())
				mhi2.create(gray2.rows, gray2.cols, CV_32FC1);

			cv::Mat processed_mhi2, component_label_map2;
			std::vector<cv::Rect> component_rects2;
			swl::MotionSegmenter::segmentUsingMHI(timestamp, MHI_TIME_DURATION, prevgray2, gray2, mhi2, processed_mhi2, component_label_map2, component_rects2);

			//
			{
				double minVal = 0.0, maxVal = 0.0;
				cv::minMaxLoc(processed_mhi, &minVal, &maxVal);
				minVal = maxVal - 1.5 * MHI_TIME_DURATION;

				const double scale = (255.0 - 1.0) / (maxVal - minVal);
				const double offset = 1.0 - scale * minVal;
				processed_mhi.convertTo(tmp_img, CV_8UC1, scale, offset);
			
				// TODO [decide] >> want to use it ?
				tmp_img.setTo(cv::Scalar(0), component_label_map == 0);

				cv::cvtColor(tmp_img, img, CV_GRAY2BGR);
				img.setTo(CV_RGB(0, 0, 255), processed_mhi >= (timestamp - 1.0e-20));  // last silhouette
			}

			// TODO [check] >> unexpected result
			//	in general, it happens when no motion exists.
			//	but it happens that the component areas obtained by MHI disappear in motion, especially when changing motion direction
			if (component_rects.empty())
				continue;

			cv::Rect selected_rect;
			{
				size_t k = 1;
				double min_dist = std::numeric_limits<double>::max();
				const double center_x = gray.cols * 0.5, center_y = gray.rows * 0.5;
				for (std::vector<cv::Rect>::const_iterator it = component_rects.begin(); it != component_rects.end(); ++it, ++k)
				{
					// reject very small components
					if (it->area() < 50 || it->width + it->height < 50)
						continue;

					// check for the case of little motion
					const size_t count = (size_t)cv::countNonZero((component_label_map == k)(*it));
					if (count < it->width * it->height * 0.05)
						continue;

					cv::rectangle(img, it->tl(), it->br(), CV_RGB(63, 0, 0), 2, 8, 0);

					const double x = it->x + it->width * 0.5, y = it->y + it->height * 0.5;
					const double dist = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
					if (dist < min_dist)
					{
						min_dist = dist;
						selected_rect = *it;
					}
				}
			}

			// FIXME [check] >> for fast gesture analysis
			cv::Rect selected_rect2;
			{
				size_t k = 1;
				double min_dist = std::numeric_limits<double>::max();
				const double center_x = gray2.cols * 0.5, center_y = gray2.rows * 0.5;
				for (std::vector<cv::Rect>::const_iterator it = component_rects2.begin(); it != component_rects2.end(); ++it, ++k)
				{
					// reject very small components
					if (it->area() < 15 || it->width + it->height < 15)
						continue;

					// check for the case of little motion
					const size_t count = (size_t)cv::countNonZero((component_label_map2 == k)(*it));
					if (count < it->width * it->height * 0.05)
						continue;

					const double x = it->x + it->width * 0.5, y = it->y + it->height * 0.5;
					const double dist = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
					if (dist < min_dist)
					{
						min_dist = dist;
						selected_rect2 = *it;
					}
				}
			}

			swl::GestureClassifierByHistogram *gestureClassifierByHistogram = dynamic_cast<swl::GestureClassifierByHistogram *>(gestureClassifier.get());

			if (selected_rect.area() > 0 &&
				(selected_rect.area() <= gray.rows * gray.cols / 2))  // reject too large area
				//selected_rect.area() <= 1.5 * average_area)  // reject too much area variation
			{
				cv::rectangle(img, selected_rect.tl(), selected_rect.br(), CV_RGB(255, 0, 0), 2, 8, 0);

#if 1
				// calculate optical flow
				cv::Mat flow;
				// FIXME [change] >> change parameters for large motion
				//cv::calcOpticalFlowFarneback(prevgray(selected_rect), gray(selected_rect), flow, 0.5, 3, 15, 3, 5, 1.1, 0);
				cv::calcOpticalFlowFarneback(prevgray(selected_rect), gray(selected_rect), flow, 0.5, 7, 15, 3, 7, 1.5, 0);
				//cv::calcOpticalFlowFarneback(prevgray(selected_rect), gray(selected_rect), flow, 0.25, 7, 15, 3, 7, 1.5, 0);

				// FIXME [check] >> for fast gesture analysis
				cv::Mat flow2;
				if (selected_rect2.area() > 0)
				{
					// FIXME [change] >> change parameters for large motion
					cv::calcOpticalFlowFarneback(prevgray2(selected_rect2), gray2(selected_rect2), flow2, 0.5, 5, 15, 3, 5, 1.1, 0);
					//cv::calcOpticalFlowFarneback(prevgray2(selected_rect2), gray2(selected_rect2), flow2, 0.25, 5, 15, 3, 7, 1.5, 0);
				}

				gestureClassifier->analyzeOpticalFlow(selected_rect, flow, &flow);
				// for fast gesture analysis
				//gestureClassifier->analyzeOpticalFlow(selected_rect, flow, &flow2);
				//gestureClassifier->analyzeOpticalFlow(selected_rect2, flow2, &flow);  // run-time error
#else
				// calculate motion gradient orientation and valid orientation mask
				const double MAX_TIME_DELTA = 0.5;
				const double MIN_TIME_DELTA = 0.05;
				const int motion_gradient_aperture_size = 3;
				cv::Mat motion_orientation_mask;  // valid orientation mask
				cv::Mat motion_orientation;  // orientation
				cv::calcMotionGradient(processed_mhi, motion_orientation_mask, motion_orientation, MIN_TIME_DELTA, MAX_TIME_DELTA, motion_gradient_aperture_size);

				gestureClassifier->analyzeOrientation(selected_rect, motion_orientation);
#endif
			}
			else
			{
				//std::cout << timestamp << ": ************************************************" << std::endl;

				gestureClassifierByHistogram->clearClass1GestureHistory();
				gestureClassifierByHistogram->clearClass2GestureHistory();
				gestureClassifierByHistogram->clearClass3GestureHistory();
				gestureClassifierByHistogram->clearTimeSeriesGestureHistory();
			}

			// classify gesture
			gestureClassifier->classifyGesture();
			const swl::GestureType::Type &gestureId = gestureClassifier->getGestureType();
			switch (gestureId)
			{
			case swl::GestureType::GT_LEFT_MOVE:
				gestureClassifierByHistogram->clearClass1GestureHistory();
				// TODO [check] >>
				//gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_RIGHT_MOVE:
				gestureClassifierByHistogram->clearClass1GestureHistory();
				// TODO [check] >>
				//gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_UP_MOVE:
				gestureClassifierByHistogram->clearClass1GestureHistory();
				// TODO [check] >>
				//gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_DOWN_MOVE:
				gestureClassifierByHistogram->clearClass1GestureHistory();
				// TODO [check] >>
				//gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_LEFT_FAST_MOVE:
				// TODO [check] >> not yet applied
				//gestureClassifierByHistogram->clearClass1GestureHistory();
				gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_RIGHT_FAST_MOVE:
				// TODO [check] >> not yet applied
				//gestureClassifierByHistogram->clearClass1GestureHistory();
				gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_HORIZONTAL_FLIP:
				gestureClassifierByHistogram->clearClass2GestureHistory();
				break;
			case swl::GestureType::GT_VERTICAL_FLIP:
				gestureClassifierByHistogram->clearClass2GestureHistory();
				break;
			case swl::GestureType::GT_JAMJAM:
				// TODO [check] >> not yet applied
				gestureClassifierByHistogram->clearClass2GestureHistory();
				break;
			case swl::GestureType::GT_SHAKE:
				// TODO [check] >> not yet applied
				gestureClassifierByHistogram->clearClass2GestureHistory();
				break;
			case swl::GestureType::GT_LEFT_90_TURN:
				// TODO [check] >> not yet applied
				gestureClassifierByHistogram->clearClass2GestureHistory();
				break;
			case swl::GestureType::GT_RIGHT_90_TURN:
				// TODO [check] >> not yet applied
				gestureClassifierByHistogram->clearClass2GestureHistory();
				break;
			case swl::GestureType::GT_CW:
				// TODO [check] >> not yet applied
				gestureClassifierByHistogram->clearTimeSeriesGestureHistory();
				break;
			case swl::GestureType::GT_CCW:
				// TODO [check] >> not yet applied
				gestureClassifierByHistogram->clearTimeSeriesGestureHistory();
				break;
			case swl::GestureType::GT_INFINITY:
				// TODO [check] >>
				//gestureClassifierByHistogram->clearTimeSeriesGestureHistory();
				gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_TRIANGLE:
				// TODO [check] >>
				//gestureClassifierByHistogram->clearTimeSeriesGestureHistory();
				gestureClassifierByHistogram->clearClass3GestureHistory();
				break;
			case swl::GestureType::GT_HAND_OPEN:
				if (!isPowered)
				{
					isPowered = true;
				}
				gestureClassifierByHistogram->clearClass1GestureHistory();
				break;
			case swl::GestureType::GT_HAND_CLOSE:
				// TODO [check] >> not yet applied
				if (isPowered)
				{
					isPowered = false;
				}
				gestureClassifierByHistogram->clearClass1GestureHistory();
				break;
			default:
				break;
			}
			
			cv::putText(img, swl::GestureType::getGestureName(gestureId), cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX, 1.0, CV_RGB(255, 0, 0), 1, 8, false);
			cv::imshow(windowName, img);
		}

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prevgray, gray);
		std::swap(prevgray2, gray2);
	}

	dynamic_cast<swl::GestureClassifierByHistogram *>(gestureClassifier.get())->destroyWindows();
	cv::destroyWindow(windowName);
}
