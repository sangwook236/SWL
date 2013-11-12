//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>


namespace {
namespace local {
	
}  // namespace local
}  // unnamed namespace

namespace swl {

void gestureRecognitionByHistogram(cv::VideoCapture &capture);
//void recognizeGestureBasedOnTHoG(cv::VideoCapture &capture, const bool IGNORE_NO_MOION, const bool IMAGE_DOWNSIZING, const double MHI_TIME_DURATION, const std::size_t MIN_MOTION_AREA_THRESHOLD, const std::size_t MAX_MOTION_AREA_THRESHOLD, std::ostream *streamTHoG, std::ostream *streamHoG);
void recognizeGestureBasedOnTHoG(cv::VideoCapture &capture, const bool IGNORE_NO_MOION, const bool IMAGE_DOWNSIZING, const double MHI_TIME_DURATION, const std::size_t MIN_MOTION_AREA_THRESHOLD, const std::size_t MAX_MOTION_AREA_THRESHOLD, std::ostream *streamTHoG, std::ostream *streamHoG, std::ostream *streamNoMotion = NULL);

bool extractTHoG(cv::VideoCapture &capture, const std::string &avi_filename, const std::string &output_directory_path, const bool IGNORE_NO_MOION, const bool IMAGE_DOWNSIZING, const double MHI_TIME_DURATION, const std::size_t MIN_MOTION_AREA_THRESHOLD, const std::size_t MAX_MOTION_AREA_THRESHOLD)
{

	const std::string::size_type pos = avi_filename.find_last_of('.');
	const std::string thog_filename(avi_filename.substr(0, pos) + ".THoG");
	const std::string hog_filename(avi_filename.substr(0, pos) + ".HoG");

	std::ofstream streamTHoG(output_directory_path + '/' + thog_filename, std::ios::out);
	std::ofstream streamHoG(output_directory_path + '/' + hog_filename, std::ios::out);
	if (!streamTHoG.is_open())
	{
		std::cout << "a THoG file, '" << (output_directory_path + '/' + thog_filename) << "' not opened" << std::endl;
		return false;
	}
	if (!streamHoG.is_open())
	{
		std::cout << "an HoG file, '" << (output_directory_path + '/' + hog_filename) << "' not opened" << std::endl;
		return false;
	}

#if 0
	const std::string no_motion_filename(avi_filename.substr(0, pos) + "_no_motion.txt");
	std::ofstream streamNoMotion(output_directory_path + '/' + no_motion_filename, std::ios::out);
	if (!streamNoMotion.is_open())
	{
		std::cout << "a no-motion file, '" << (output_directory_path + '/' + no_motion_filename) << "' not opened" << std::endl;
		return false;
	}
#else
	std::ofstream streamNoMotion;
#endif

	try
	{
		//gestureRecognitionByHistogram(capture);

		// temporal HoG (THoG) or temporal orientation histogram (TOH).
		//recognizeGestureBasedOnTHoG(capture, IGNORE_NO_MOION, IMAGE_DOWNSIZING, MHI_TIME_DURATION, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD, (streamTHoG.is_open() ? &streamTHoG : NULL), (streamHoG.is_open() ? &streamHoG : NULL));
		recognizeGestureBasedOnTHoG(capture, IGNORE_NO_MOION, IMAGE_DOWNSIZING, MHI_TIME_DURATION, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD, (streamTHoG.is_open() ? &streamTHoG : NULL), (streamHoG.is_open() ? &streamHoG : NULL), (streamNoMotion.is_open() ? &streamNoMotion : NULL));
	}
	catch (const cv::Exception &)
	{
		return false;
	}

	return true;
}

}  // namespace swl

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
#if 0
#	if 0
		const int camId = -1;
		cv::VideoCapture capture(camId);
		if (!capture.isOpened())
		{
			std::cout << "a vision sensor not found" << std::endl;
			retval = EXIT_FAILURE;
		}

		//capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
		//capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

		//const double &propFrameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
		//const double &propFrameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
#	else
		const std::string avi_filename("./machine_vision_data/opencv/tree.avi");
		//const std::string avi_filename("./machine_vision_data/opencv/s01_g01_1_ccw_normal.avi");
	
		//const int imageWidth = 640, imageHeight = 480;

		cv::VideoCapture capture(avi_filename);
		if (!capture.isOpened())
		{
			std::cout << "a video file not found" << std::endl;
			retval = EXIT_FAILURE;
		}
#	endif

		if (EXIT_SUCCESS == retval)
		{
			//swl::gestureRecognitionByHistogram(capture);

			// temporal HoG (THoG) or temporal orientation histogram (TOH).
			const int IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480;
			const bool IMAGE_DOWNSIZING = true;

			const double MHI_TIME_DURATION = 0.5;  // [sec].
			//const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 1000 : 2000, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);
			const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 100 : 200, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);

			const bool IGNORE_NO_MOION = true;
			swl::recognizeGestureBasedOnTHoG(capture, IGNORE_NO_MOION, IMAGE_DOWNSIZING, MHI_TIME_DURATION, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD, NULL, NULL);
		}
#elif 0
		// for AIM's gesture dataset.

		const std::string input_directory_path("F:/AIM_gesture_dataset/s01_sangwook_lee_20120719_per_gesture_avi_640x480_30fps_3000kbps");
		const std::string output_directory_path(input_directory_path + "_thog");
		const std::string avi_filename_list("file_list_s01.txt");

		std::vector<std::string> avi_filenames;
		std::ifstream stream(input_directory_path + '/' + avi_filename_list, std::ios::in);
		if (stream.is_open())
		{
			while (!stream.eof())
			{
				std::string filename;
				std::getline(stream, filename);
				if (!filename.empty())
					avi_filenames.push_back(filename);
			}
			stream.close();
		}
		else
		{
			std::cout << "a list file, '" << (input_directory_path + '/' + avi_filename_list) << "' not found" << std::endl;
			retval = EXIT_FAILURE;
		}

		//
		for (std::vector<std::string>::iterator it = avi_filenames.begin(); it != avi_filenames.end(); ++it)
		{
			cv::VideoCapture capture(input_directory_path + '/' + *it);
			if (capture.isOpened())
			{
				const int IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480;
				const bool IMAGE_DOWNSIZING = true;

				const double MHI_TIME_DURATION = 0.5;  // [sec].
				//const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 1000 : 2000, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);
				const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 100 : 200, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);

				const bool IGNORE_NO_MOION = false;
				swl::extractTHoG(capture, *it, output_directory_path, IGNORE_NO_MOION, IMAGE_DOWNSIZING, MHI_TIME_DURATION, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD);
			}
			else
			{
				std::cout << "a video file, '" << (input_directory_path + '/' + *it) << "' not found" << std::endl;
				retval = EXIT_FAILURE;
			}
		}
#elif 1
		// for AIM's gesture dataset (segmented).

		const std::string input_directory_path("F:/AIM_gesture_dataset_segmented/s01_sangwook_lee_20120719_per_gesture_mp4_640x480");
		const std::string output_directory_path(input_directory_path + "_thog");
		const std::string avi_filename_list("file_list_s01.txt");

		std::vector<std::string> avi_filenames;
		std::ifstream stream(input_directory_path + '/' + avi_filename_list, std::ios::in);
		if (stream.is_open())
		{
			while (!stream.eof())
			{
				std::string filename;
				std::getline(stream, filename);
				if (!filename.empty())
					avi_filenames.push_back(filename);
			}
			stream.close();
		}
		else
		{
			std::cout << "a list file, '" << (input_directory_path + '/' + avi_filename_list) << "' not found" << std::endl;
			retval = EXIT_FAILURE;
		}

		//
		for (std::vector<std::string>::iterator it = avi_filenames.begin(); it != avi_filenames.end(); ++it)
		{
			cv::VideoCapture capture(input_directory_path + '/' + *it);
			if (capture.isOpened())
			{
				const int IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480;
				const bool IMAGE_DOWNSIZING = true;

				const double MHI_TIME_DURATION = 0.5;  // [sec].
				//const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 1000 : 2000, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);
				const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 100 : 200, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);

				const bool IGNORE_NO_MOION = false;
				swl::extractTHoG(capture, *it, output_directory_path, IGNORE_NO_MOION, IMAGE_DOWNSIZING, MHI_TIME_DURATION, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD);
			}
			else
			{
				std::cout << "a video file, '" << (input_directory_path + '/' + *it) << "' not found" << std::endl;
				retval = EXIT_FAILURE;
			}
		}
#elif 0
		// for ChaLearn Gesture Challenge dataset.
		//	[ref] http://gesture.chalearn.org/data

		const int DATASET_START_INDEX = 11, DATASET_END_INDEX = 20;
		for (int idx = DATASET_START_INDEX; idx <= DATASET_END_INDEX; ++idx)
		{
			std::ostringstream sstream;
			sstream << std::setw(2) << std::setfill('0') << idx;
			const std::string input_directory_path("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel" + sstream.str());
			//const std::string input_directory_path("E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/lossy_format/devel-1-20_valid-1-20/devel" + sstream.str());
			const std::string avi_file_prefix = "M_";  // RGB image.
			//const std::string avi_file_prefix = "K_";  // depth image.
			const std::string output_directory_path(input_directory_path + "_thog2");

			std::vector<std::string> avi_filenames;
#if 0
			const std::string avi_filename_list("devel" + sstream.str() + "_train.csv");

			std::ifstream stream(input_directory_path + '/' + avi_filename_list, std::ios::in);
			if (stream.is_open())
			{
				std::size_t line_no = 1;
				while (!stream.eof())
				{
					std::string line;
					std::getline(stream, line);
					if (!line.empty())
					{
						std::ostringstream sstream;
						sstream << avi_file_prefix << line_no << ".avi";
						avi_filenames.push_back(sstream.str());
						++line_no;
					}
				}
				stream.close();
			}
			else
			{
				std::cout << "a list file, '" << (input_directory_path + '/' + avi_filename_list) << "' not found" << std::endl;
				retval = EXIT_FAILURE;
			}
#else
			const int AVI_FILE_START_INDEX = 1, AVI_FILE_END_INDEX = 47;
			for (int file_no = AVI_FILE_START_INDEX; file_no <= AVI_FILE_END_INDEX; ++file_no)
			{
				std::ostringstream sstream;
				sstream << avi_file_prefix << file_no << ".avi";
				avi_filenames.push_back(sstream.str());
			}
#endif

			//
			std::size_t seq_idx = 1;
			for (std::vector<std::string>::iterator it = avi_filenames.begin(); it != avi_filenames.end(); ++it, ++seq_idx)
			{
				std::cout << "seq: " << seq_idx << ", file: " << *it << std::endl;

				cv::VideoCapture capture(input_directory_path + '/' + *it);
				if (capture.isOpened())
				{
					const int IMAGE_WIDTH = 320, IMAGE_HEIGHT = 240;
					const bool IMAGE_DOWNSIZING = false;

					const double MHI_TIME_DURATION = 0.5;  // [sec].
					//const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 1000 : 2000, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);
					const std::size_t MIN_MOTION_AREA_THRESHOLD = IMAGE_DOWNSIZING ? 200 : 400, MAX_MOTION_AREA_THRESHOLD = (IMAGE_WIDTH * IMAGE_HEIGHT) / (IMAGE_DOWNSIZING ? 4 : 2);

					const bool IGNORE_NO_MOION = false;
					swl::extractTHoG(capture, *it, output_directory_path, IGNORE_NO_MOION, IMAGE_DOWNSIZING, MHI_TIME_DURATION, MIN_MOTION_AREA_THRESHOLD, MAX_MOTION_AREA_THRESHOLD);
				}
				else
				{
					std::cout << "a video file, '" << (input_directory_path + '/' + *it) << "' not found" << std::endl;
					retval = EXIT_FAILURE;
				}
			}
		}
#endif
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught: " << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;
		retval = EXIT_FAILURE;
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
