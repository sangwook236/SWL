//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include "KinectSensor.h"
#include "gslic_lib/FastImgSeg.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace swl {

void create_superpixel_by_gSLIC(const cv::Mat &input_image, cv::Mat &superpixel_mask, const SEGMETHOD seg_method, const double seg_weight, const int num_segments);
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary);

}  // namespace swl

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
		const std::size_t num_images = 4;
		const cv::Size imageSize_ir(640, 480), imageSize_rgb(640, 480);

		std::vector<std::string> rgb_input_file_list, depth_input_file_list;
		rgb_input_file_list.reserve(num_images);
		depth_input_file_list.reserve(num_images);
		rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130530T103805.png");
		rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130531T023152.png");
		rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130531T023346.png");
		rgb_input_file_list.push_back("../data/kinect_segmentation/kinect_rgba_20130531T023359.png");
		depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130530T103805.png");
		depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130531T023152.png");
		depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130531T023346.png");
		depth_input_file_list.push_back("../data/kinect_segmentation/kinect_depth_20130531T023359.png");

		//
		swl::KinectSensor kinect(imageSize_ir, imageSize_rgb);
		kinect.loadCameraParameters();

		//
		const int num_segments = 1200;
		const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC
		const double seg_weight = 0.3;

		cv::Mat rgb_superpixel_mask;
		cv::Mat depth_output_image, rgb_output_image;
		for (std::size_t i = 0; i < num_images; ++i)
		{
			// load images.
			const cv::Mat rgb_input_image(cv::imread(rgb_input_file_list[i], CV_LOAD_IMAGE_COLOR));
			if (rgb_input_image.empty())
			{
				std::cout << "fail to load image file: " << rgb_input_file_list[i] << std::endl;
				continue;
			}
			const cv::Mat depth_input_image(cv::imread(depth_input_file_list[i], CV_LOAD_IMAGE_UNCHANGED));
			if (depth_input_image.empty())
			{
				std::cout << "fail to load image file: " << depth_input_file_list[i] << std::endl;
				continue;
			}

			// superpixel mask consists of segment indexes.
			swl::create_superpixel_by_gSLIC(rgb_input_image, rgb_superpixel_mask, seg_method, seg_weight, num_segments);

#if 0
			// show superpixel mask.
			cv::Mat mask;
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(rgb_superpixel_mask, &minVal, &maxVal);
			rgb_superpixel_mask.convertTo(mask, CV_32FC1, 1.0 / maxVal, 0.0);

			cv::imshow("superpixels by gSLIC - mask", mask);
#endif

#if 1
			// show superpixel boundary.
			cv::Mat rgb_superpixel_boundary;
			swl::create_superpixel_boundary(rgb_superpixel_mask, rgb_superpixel_boundary);

			cv::Mat img(rgb_input_image.clone());
			img.setTo(cv::Scalar(0, 0, 255), rgb_superpixel_boundary);

			cv::imshow("superpixels by gSLIC - boundary", img);
#endif

			// rectify Kinect images.
			kinect.rectifyImagePair(depth_input_image, rgb_input_image, depth_output_image, rgb_output_image);

#if 1
			// show rectified images
			cv::Mat depth_output_image2;
			double minVal = 0.0, maxVal = 0.0;
			cv::minMaxLoc(depth_output_image, &minVal, &maxVal);
			depth_output_image.convertTo(depth_output_image2, CV_32FC1, 1.0 / maxVal, 0.0);

			cv::imshow("rectified depth image", depth_output_image2);
			cv::imshow("rectified RGB image", rgb_output_image);
#endif

#if 0
			std::ostringstream strm1, strm2;
			strm1 << "../data/kinect_segmentation/rectified_image_depth_" << i << ".png";
			cv::imwrite(strm1.str(), depth_output_image);
			strm2 << "../data/kinect_segmentation/rectified_image_rgb_" << i << ".png";
			cv::imwrite(strm2.str(), rgb_output_image);
#endif

			const unsigned char key = cv::waitKey(0);
			if (27 == key)
				break;
		}

		cv::destroyAllWindows();
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
