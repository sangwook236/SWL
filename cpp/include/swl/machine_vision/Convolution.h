#pragma once

#if !defined(__SWL_MACHINE_VISION__CONVOLUTION__H_)
#define __SWL_MACHINE_VISION__CONVOLUTION__H_ 1


#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
// 2D Convolution.

// A DFT-based convolution of two 2D real arrays.
//	Much faster.
//	REF [site] >> https://docs.opencv.org/4.0.0/d2/de8/group__core__array.html

template<typename SrcType, typename DstType = SrcType>
bool convolve2d(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, int borderType = cv::BORDER_DEFAULT, cv::Scalar borderVal = cv::Scalar::all(0))
{
	if (0 == kernel.rows % 2 || 0 == kernel.cols % 2)
		return false;

	// REF [Note] >> The function does actually compute correlation, not the convolution.
	//cv::Mat kernel_flip;
	//cv::flip(kernel, kernel_flip, -1);
	const cv::Mat &kernel_flip = kernel;

	const int border_x = kernel.cols / 2;
	const int border_y = kernel.rows / 2;
	cv::Mat src_ex;
	cv::copyMakeBorder(src, src_ex, border_y, border_y, border_x, border_x, borderType, borderVal);

	//dst = cv::Mat(src.size(), CV_32FC1);
	for (int c = 0; c < src_ex.cols - 2 * border_x && c < src.cols; ++c)
		for (int r = 0; r < src_ex.rows - 2 * border_y && r < src.rows; ++r)
		{
			DstType sum = (DstType)0;
			for (int kc = 0; kc < kernel_flip.cols; ++kc)
				for (int kr = 0; kr < kernel_flip.rows; ++kr)
					sum += kernel_flip.at<SrcType>(kr, kc) * src_ex.at<SrcType>(r + kr, c + kc);
			dst.at<DstType>(r, c) = sum;
			/*
			const cv::Mat src_ex_roi(src_ex, cv::Rect(c, r, kernel_flip.cols, kernel_flip.rows));
			std::vector<SrcType> values;
			values.reserve(kernel_flip.cols * kernel_flip.rows);
			std::transform(kernel_flip.begin<SrcType>(), kernel_flip.end<SrcType>(), src_ex_roi.begin<SrcType>(), std::back_inserter(values), std::multiplies<SrcType>());
			dst.at<DstType>(r, c) = std::accumulate(values.begin(), values.end(), (DstType)0);
			*/
			/*
			const cv::Mat src_ex_roi(src_ex, cv::Rect(c, r, kernel_flip.cols, kernel_flip.rows));
			dst.at<DstType>(r, c) = std::transform_reduce(std::execution::par, kernel_flip.begin<SrcType>(), kernel_flip.end<SrcType>(), src_ex_roi.begin<SrcType>(), (DstType)0, std::plus<DstType>(), std::multiplies<SrcType>());
			*/
		}

	return true;
}

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__CONVOLUTION__H_
