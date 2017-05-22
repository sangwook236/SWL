//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/machine_vision/BoundaryExtraction.h"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <string>
#include <list>
#include <memory>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

namespace {
namespace local {

void generate_test_label(cv::Mat &label, cv::Mat &boundary)
{
	const int WIDTH = 300, HEIGHT = 300;
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
	//std::unique_ptr<swl::IBoundaryExtraction> extractor(new swl::NaiveBoundaryExtraction(true));
	std::unique_ptr<swl::IBoundaryExtraction> extractor(new swl::ContourBoundaryExtraction());

	// Extract boundaries.
	cv::Mat boundary(cv::Mat::zeros(label.size(), label.type()));
	{
		boost::timer::auto_cpu_timer timer;
		extractor->extractBoundary(label, boundary);
	}

	// Show the result.
	cv::imshow("Boundary extraction - Label", label);
	cv::imshow("Boundary extraction - True boundary", boundary_true);
	cv::imshow("Boundary extraction - Extracted boundary", boundary);

	//cv::imwrite("./data/machine_vision/label.png", label);
	//cv::imwrite("./data/machine_vision/label_true_boundary.png", boundary_true);
	//cv::imwrite("./data/machine_vision/label_extracted_boundary.png", boundary);

	cv::waitKey(0);

	cv::destroyAllWindows();
}
