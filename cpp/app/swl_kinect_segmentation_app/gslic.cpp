//#include "stdafx.h"
#include "gslic_lib/FastImgSeg.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

// [ref] create_superpixel_by_gSLIC() in ${CPP_RND_HOME}/test/segmentation/gslic/gslic_main.cpp
void create_superpixel_by_gSLIC(const cv::Mat &input_image, cv::Mat &superpixel_mask, const SEGMETHOD seg_method, const double seg_weight, const int num_segments)
{
	// gSLIC currently only support 4-dimensional image
	unsigned char *imgBuffer = new unsigned char [input_image.cols * input_image.rows * 4];
	memset(imgBuffer, 0, sizeof(unsigned char) * input_image.cols * input_image.rows * 4);

	unsigned char *ptr = imgBuffer;
	for (int i = 0; i < input_image.rows; ++i)
		for (int j = 0; j < input_image.cols; ++j)
		{
			const cv::Vec3b &bgr = input_image.at<cv::Vec3b>(i, j);

			*ptr++ = bgr[0];
			*ptr++ = bgr[1];
			*ptr++ = bgr[2];
			++ptr;
		}

	//
	FastImgSeg gslic;
	gslic.initializeFastSeg(input_image.cols, input_image.rows, num_segments);

	gslic.LoadImg(imgBuffer);
	gslic.DoSegmentation(seg_method, seg_weight);
	//gslic.Tool_GetMarkedImg();  // required for display of segmentation boundary

	delete [] imgBuffer;
	imgBuffer = NULL;

	//
	cv::Mat(input_image.size(), CV_32SC1, (void *)gslic.segMask).copyTo(superpixel_mask);

	//gslic.clearFastSeg();
}

// [ref] create_superpixel_boundary() in ${CPP_RND_HOME}/test/segmentation/gslic/gslic_main.cpp
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary)
{
	superpixel_boundary = cv::Mat::zeros(superpixel_mask.size(), CV_8UC1);

	for (int i = 1; i < superpixel_mask.rows - 1; ++i)
		for (int j = 1; j < superpixel_mask.cols - 1; ++j)
		{
			const int idx = superpixel_mask.at<int>(i, j);
			if (idx != superpixel_mask.at<int>(i, j - 1) || idx != superpixel_mask.at<int>(i, j + 1) ||
				idx != superpixel_mask.at<int>(i - 1, j) || idx != superpixel_mask.at<int>(i + 1, j))
				superpixel_boundary.at<unsigned char>(i, j) = 255;
		}
}

}  // namespace swl
