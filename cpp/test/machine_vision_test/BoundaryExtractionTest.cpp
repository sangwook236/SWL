//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/machine_vision/BoundaryExtraction.h"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <string>
#include <list>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace {
namespace local {

void generate_test_label(cv::Mat &label, cv::Mat &boundary)
{
	const int WIDTH = 300, HEIGHT = 200;
	label = cv::Mat::zeros(HEIGHT, WIDTH, CV_16UC1);
	boundary = cv::Mat::zeros(HEIGHT, WIDTH, CV_16UC1);

	unsigned short label_id = 1;
	cv::rectangle(label, cv::Rect(60, 60, 50, 50), cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::rectangle(boundary, cv::Rect(60, 60, 50, 50), cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::rectangle(label, cv::Rect(40, 40, 40, 40), cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::rectangle(boundary, cv::Rect(40, 40, 40, 40), cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::rectangle(label, cv::Rect(50, 50, 20, 20), cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::rectangle(boundary, cv::Rect(50, 50, 20, 20), cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::circle(label, cv::Point(120, 100), 20, cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::circle(boundary, cv::Point(120, 100), 20, cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::circle(label, cv::Point(160, 70), 40, cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::circle(boundary, cv::Point(160, 70), 40, cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::circle(label, cv::Point(160, 70), 20, cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::circle(boundary, cv::Point(160, 70), 20, cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::circle(label, cv::Point(160, 70), 10, cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::circle(boundary, cv::Point(160, 70), 10, cv::Scalar::all(label_id), 1, cv::LINE_8);

	++label_id;
	cv::rectangle(label, cv::Rect(170, 100, 50, 30), cv::Scalar::all(label_id), cv::FILLED, cv::LINE_8);
	cv::rectangle(boundary, cv::Rect(170, 100, 50, 30), cv::Scalar::all(label_id), 1, cv::LINE_8);
}

}  // namespace local
}  // unnamed namespace

void boundary_extraction()
{
	cv::Mat label, boundary_true;
	local::generate_test_label(label, boundary_true);

	// Create a boundary extractor.
	//swl::IBoundaryExtraction &extractor = swl::NaiveBoundaryExtraction(true);
	swl::IBoundaryExtraction &extractor = swl::ContourBoundaryExtraction();

	// Extract boundaries.
	cv::Mat boundary(cv::Mat::zeros(label.size(), label.type()));
	{
		boost::timer::auto_cpu_timer timer;
		extractor.extractBoundary(label, boundary);
	}

#if 1
	// Compute boundary weight.
	cv::Mat boundaryWeight_filtered(cv::Mat::zeros(boundary.size(), CV_16UC1));
	{
		// Distance transform.
		cv::Mat dist;
		{
			cv::Mat binary(cv::Mat::ones(boundary.size(), CV_8UC1));
			binary.setTo(cv::Scalar::all(0), boundary > 0);
			cv::distanceTransform(binary, dist, cv::DIST_L2, cv::DIST_MASK_3);
		}

		// Gaussian weighting.
		const double sigma2 = 1.0;  // sigma^2.
		cv::Mat boundaryWeight_float;
		cv::multiply(dist, dist, dist);
		cv::exp(-dist / (2.0 * sigma2), boundaryWeight_float);

		// NOTICE [info] >> Cannot save images of 32-bit (signed/unsigned) integer or float.

		//double minVal = 0.0, maxVal = 0.0;
		//cv::minMaxLoc(boundaryWeight_float, &minVal, &maxVal);
		cv::Mat boundaryWeight_int;
		boundaryWeight_float.convertTo(boundaryWeight_int, boundaryWeight_filtered.type(), std::numeric_limits<unsigned short>::max(), 0.0);

		//boundaryWeight_filtered = boundaryWeight_int;  // Do not filter out.
		boundaryWeight_int.copyTo(boundaryWeight_filtered, boundary > 0 | 0 == label);  // On boundaries or outside of objects.
	}
#endif

	// Output the result.
#if 1
	cv::imshow("Boundary extraction - Label", label);
	cv::imshow("Boundary extraction - True boundary", boundary_true);
	cv::imshow("Boundary extraction - Extracted boundary", boundary);
	cv::imshow("Boundary extraction - Boundary weight", boundaryWeight_filtered);
#else
	cv::imwrite("./data/machine_vision/label.png", label);
	cv::imwrite("./data/machine_vision/label_true_boundary.png", boundary_true);
	cv::imwrite("./data/machine_vision/label_extracted_boundary.png", boundary);
	cv::imwrite("./data/machine_vision/label_boundary_weight.png", boundaryWeight_filtered);
#endif
	cv::waitKey(0);

	cv::destroyAllWindows();
}
