#pragma once

#if !defined(__SWL_MACHINE_VISION__CONVOLUTION__H_)
#define __SWL_MACHINE_VISION__CONVOLUTION__H_ 1


#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//--------------------------------------------------------------------------
// 2D Convolution.

template<typename SrcType, typename DstType = SrcType>
bool convolve2d(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, int borderType = cv::BORDER_CONSTANT, cv::Scalar borderVal = cv::Scalar::all(0))
{
	if (0 == kernel.rows % 2 || 0 == kernel.cols % 2)
		return false;

	const int border_x = kernel.rows / 2;
	const int border_y = kernel.cols / 2;

	// REF [Note] >> The function does actually compute correlation, not the convolution.
	//cv::Mat kernel_flip;
	//cv::flip(kernel, kernel_flip, -1);
	const cv::Mat &kernel_flip = kernel;

	cv::Mat src_ex;
	cv::copyMakeBorder(src, src_ex, border_y, border_y, border_x, border_x, borderType, borderVal);

	//dst = cv::Mat(src.size(), CV_32FC1);
	for (int c = 0, ic = 0; c < src_ex.cols - 2 * border_y && ic < src.cols; ++c, ++ic)
	{
		for (int r = 0, ir = 0; r < src_ex.rows - 2 * border_x && ir < src.rows; ++r, ++ir)
		{
			DstType sum = (DstType)0.0;
			for (int kc = 0; kc < kernel_flip.cols; ++kc)
				for (int kr = 0; kr < kernel_flip.rows; ++kr)
					sum += kernel_flip.at<SrcType>(kr, kc) * src_ex.at<SrcType>(r + kr, c + kc);

			dst.at<DstType>(ir, ic) = sum;
		}
	}

	return true;
}

}  // namespace swl


#endif  // __SWL_MACHINE_VISION__CONVOLUTION__H_
