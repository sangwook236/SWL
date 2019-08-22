//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/DynamicTimeWarping.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// [ref] normalize_histogram() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void normalize_histogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	//cv::Mat tmp(hist);
	//tmp.convertTo(hist, -1, factor / sums[0], 0.0);
	hist *= factor / sums[0];
#endif
}

double compare_histogram(const cv::Mat &histo1, const cv::Mat &histo2)
{
	const double eps = 1.0e-20;

	// TODO [adjust] >> how to deal with if histo1 or histo1 is a zero histogram?

	const cv::Scalar sums1(cv::sum(histo1));
	const cv::Scalar sums2(cv::sum(histo2));

#if 0
	// for correlation.

	if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) >= eps) return -1.0;
	else if (std::fabs(sums1[0]) >= eps && std::fabs(sums2[0]) < eps) return -1.0;
	else if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) < eps) return 1.0;

	return cv::compareHist(histo1, histo2, CV_COMP_CORREL);
#elif 0
	// for histogram intersection.

	if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) >= eps) return 0.0;
	else if (std::fabs(sums1[0]) >= eps && std::fabs(sums2[0]) < eps) return 0.0;
	else if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) < eps) return 1.0;

	return cv::compareHist(histo1, histo2, CV_COMP_INTERSECT);
#elif 0
	// for chi-square distance.

	if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) >= eps) return std::numeric_limits<double>::max();
	else if (std::fabs(sums1[0]) >= eps && std::fabs(sums2[0]) < eps) return std::numeric_limits<double>::max();
	else if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) < eps) return 0.0;

	return cv::compareHist(histo1, histo2, CV_COMP_CHISQR);
#elif 1
	// for Bhattacharyya distance.

	if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) >= eps) return 1.0;
	//if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) >= eps) return std::numeric_limits<double>::max();
	else if (std::fabs(sums1[0]) >= eps && std::fabs(sums2[0]) < eps) return 1.0;
	//else if (std::fabs(sums1[0]) >= eps && std::fabs(sums2[0]) < eps) return std::numeric_limits<double>::max();
	else if (std::fabs(sums1[0]) < eps && std::fabs(sums2[0]) < eps) return 0.0;

	return cv::compareHist(histo1, histo2, cv::HISTCMP_BHATTACHARYYA);
#endif
}

void THoG_example()
{
	std::vector<std::string> filename_list;
	filename_list.push_back("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog2_1deg_segmented/M_1_1.HoG");
	filename_list.push_back("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog2_1deg_segmented/M_2_1.HoG");
	filename_list.push_back("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog2_1deg_segmented/M_4_1.HoG");
	filename_list.push_back("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog2_1deg_segmented/M_7_1.HoG");

	std::vector<cv::Mat> THoG_list;
	THoG_list.reserve(filename_list.size());
	for (std::size_t i = 0; i < filename_list.size(); ++i)
	{
		// read HoG.
		std::vector<std::vector<float> > data;
		{
#if defined(__GNUC__)
			std::ifstream strm(filename_list[i].c_str());
#else
			std::ifstream strm(filename_list[i]);
#endif

			std::string str;
			std::vector<float> record;
			while (strm)
			{
				if (!std::getline(strm, str)) break;

				record.clear();

				std::istringstream sstrm(str);
				while (sstrm)
				{
					if (!std::getline(sstrm, str, ',')) break;
					record.push_back((float)strtod(str.c_str(), NULL));
				}

				data.push_back(record);
			}

			if (!strm.eof())
			{
				std::cerr << "Fooey!" << std::endl;
			}
		}

		//
		//const std::size_t gesture_id = std::size_t(data[0][0]);
		const std::size_t num_features = std::size_t(data[1][0]);
		const std::size_t num_frames = std::size_t(data[1][1]);

		cv::Mat THoG(num_features, num_frames, CV_32FC1);
		for (std::size_t i = 2; i < data.size(); ++i)
			for (std::size_t j = 0; j < data[i].size(); ++j)
				THoG.at<float>(i - 2, j) = data[i][j];

		//normalize_histogram(THoG, 1.0);
		THoG_list.push_back(THoG);
	}

	//
	std::vector<std::vector<std::vector<double> > > result(THoG_list.size(), std::vector<std::vector<double> >(THoG_list.size()));

	//const std::size_t N = 5;
	const std::size_t N = 10;
	const std::size_t maximumWarpingDistance = 5;
	for (std::size_t i = 0; i < THoG_list.size(); ++i)
	{
		boost::timer::auto_cpu_timer timer;

		std::vector<cv::Mat> HoG_list1(THoG_list[i].cols, cv::Mat(THoG_list[i].rows, 1, CV_32FC1));
		for (int k = 0; k < THoG_list[i].cols; ++k)
		{
			THoG_list[i].col(k).copyTo(HoG_list1[k]);

			// TODO [check] >> zero histogram is treated as an uniform distribution.
			const cv::Scalar sums(cv::sum(HoG_list1[k]));
			const double eps = 1.0e-20;
			if (std::fabs(sums[0]) < eps) HoG_list1[k] = cv::Mat::ones(HoG_list1[k].size(), HoG_list1[k].type());

			normalize_histogram(HoG_list1[k], 1.0);
		}

		for (std::size_t j = 0; j < THoG_list.size(); ++j)
		{
			result[i][j].resize(THoG_list[j].cols - int(N), 0.0);

			std::vector<cv::Mat> HoG_list2(N, cv::Mat(THoG_list[j].rows, 1, CV_32FC1));
			for (int k = 0; k < THoG_list[j].cols - int(N); ++k)
			{
				for (std::size_t l = 0; l < N; ++l)
				{
					THoG_list[j].col(k + l).copyTo(HoG_list2[l]);

					// TODO [check] >> zero histogram is treated as an uniform distribution.
					const cv::Scalar sums(cv::sum(HoG_list2[l]));
					const double eps = 1.0e-20;
					if (std::fabs(sums[0]) < eps) HoG_list2[l] = cv::Mat::ones(HoG_list2[l].size(), HoG_list2[l].type());

					normalize_histogram(HoG_list2[l], 1.0);
				}

				const double dist = swl::computeFastDynamicTimeWarping(HoG_list1, HoG_list2, maximumWarpingDistance, compare_histogram);

				result[i][j][k] = dist;
			}
		}
	}

	//
	std::cout << "dynamic time warping (DTW) test for THoG ..." << std::endl;

#if 1
	const std::string resultant_filename("./data/THoG_DTW_result.txt");
#if defined(__GNUC__)
	std::ofstream stream(resultant_filename.c_str(), std::ios::out | std::ios::trunc);
#else
	std::ofstream stream(resultant_filename, std::ios::out | std::ios::trunc);
#endif
	if (!stream.is_open())
	{
		std::cerr << "file not found: " << resultant_filename << std::endl;
		return;
	}
#else
	std::ostream stream = std::cout;
#endif

	for (std::size_t i = 0; i < result.size(); ++i)
		for (std::size_t j = 0; j < result[i].size(); ++j)
		{
			for (std::size_t k = 0; k < result[i][j].size(); ++k)
				stream << result[i][j][k] << ", ";
			stream << std::endl;
		}
}

}  // namespace local
}  // unnamed namespace

void dynamic_time_warping()
{
	local::THoG_example();
}
