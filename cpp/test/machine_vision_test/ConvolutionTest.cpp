#include "swl/Config.h"
#include "swl/machine_vision/Convolution.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <iterator>
#include <chrono>
//#include <execution>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// REF [file] >> FilterEngine in ${OPENCV_HOME}/modules/imagproc/src/filter.cpp
// REF [site] >> https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html

// Convolve arbitrary regions in an image with a kernel.
// ROI can be a set of points, a mask image (binary image), or a set of (rectangular) regions.
//	We can use a set of blobs (images).
template<typename SrcType, typename DstType = SrcType>
class ParallelLoopConvolve2D : public cv::ParallelLoopBody
{
public:
	ParallelLoopConvolve2D(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_DEFAULT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: src_ex_(), dst_(dst), kernel_flip_(), points_(points), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernel.rows % 2 || 0 == kernel.cols % 2)
		{
			std::cerr << "Invalid kernel." << std::endl;
			return;
		}

		// REF [Note] >> The function does actually compute correlation, not the convolution.
		//cv::flip(kernel, kernel_flip_, -1);
		kernel_flip_ = kernel;

		const int border_x = kernel.cols / 2;
		const int border_y = kernel.rows / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}

	/*virtual*/ void operator()(const cv::Range &r) const
	{
		for (int i = r.start; i < r.end && i < num_points_; ++i)
		{
			const cv::Point &pt = points_[i];

#if false
			const cv::Mat src_ex_roi(src_ex_, cv::Rect(pt.x, pt.y, kernel_flip_.cols, kernel_flip_.rows));
			std::vector<SrcType> values;
			values.reserve(kernel_flip_.cols * kernel_flip_.rows);
			std::transform(kernel_flip_.begin<SrcType>(), kernel_flip_.end<SrcType>(), src_ex_roi.begin<SrcType>(), std::back_inserter(values), std::multiplies<SrcType>());
			dst_.at<DstType>(pt.y, pt.x) = std::accumulate(values.begin(), values.end(), (DstType)0);
#elif false
			const cv::Mat src_ex_roi(src_ex_, cv::Rect(pt.x, pt.y, kernel_flip_.cols, kernel_flip_.rows));
			dst_.at<DstType>(pt.y, pt.x) = std::transform_reduce(std::execution::par, kernel_flip_.begin<SrcType>(), kernel_flip_.end<SrcType>(), src_ex_roi.begin<SrcType>(), (DstType)0, std::plus<DstType>(), std::multiplies<SrcType>());
#else
			DstType sum = (DstType)0;
			for (int kc = 0; kc < kernel_flip_.cols; ++kc)
				for (int kr = 0; kr < kernel_flip_.rows; ++kr)
					sum += kernel_flip_.at<SrcType>(kr, kc) * src_ex_.at<SrcType>(pt.y + kr, pt.x + kc);
			dst_.at<DstType>(pt.y, pt.x) = sum;
#endif
		}
	}

private:
	cv::Mat src_ex_;
	cv::Mat &dst_;
	cv::Mat kernel_flip_;
	const std::vector<cv::Point> &points_;
	const int borderType_;
	const cv::Scalar &borderVal_;
	const size_t num_points_;
};

void simple_convolution2d_example()
{
#if true
	const cv::Size kernelSize(3, 3);
	cv::Mat kernel(kernelSize, CV_32F, cv::Scalar::all(0));
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(1, 0) = 0.0f;
	kernel.at<float>(2, 0) = 1.0f;
	kernel.at<float>(0, 1) = -2.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(2, 1) = 2.0f;
	kernel.at<float>(0, 2) = -1.0f;
	kernel.at<float>(1, 2) = 0.0f;
	kernel.at<float>(2, 2) = 1.0f;

	cv::Mat src(kernelSize, CV_32F, cv::Scalar::all(0));
	src.at<float>(0, 0) = 1.0f;
	src.at<float>(1, 0) = 4.0f;
	src.at<float>(2, 0) = 7.0f;
	src.at<float>(0, 1) = 2.0f;
	src.at<float>(1, 1) = 5.0f;
	src.at<float>(2, 1) = 8.0f;
	src.at<float>(0, 2) = 3.0f;
	src.at<float>(1, 2) = 6.0f;
	src.at<float>(2, 2) = 9.0f;
#elif false
	const cv::Size kernelSize(5, 3);  // 3 (rows) x 5 (cols).
	cv::Mat kernel(kernelSize, CV_32F, cv::Scalar::all(0));
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(1, 0) = 0.0f;
	kernel.at<float>(2, 0) = 1.0f;
	kernel.at<float>(0, 1) = -2.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(2, 1) = 2.0f;
	kernel.at<float>(0, 2) = -3.0f;
	kernel.at<float>(1, 2) = 0.0f;
	kernel.at<float>(2, 2) = 3.0f;
	kernel.at<float>(0, 3) = -2.0f;
	kernel.at<float>(1, 3) = 0.0f;
	kernel.at<float>(2, 3) = 2.0f;
	kernel.at<float>(0, 4) = -1.0f;
	kernel.at<float>(1, 4) = 0.0f;
	kernel.at<float>(2, 4) = 1.0f;

	const cv::Size imageSize(6, 7);  // 7 (rows) x 6 (cols).
	cv::Mat src(kernelSize, CV_32F, cv::Scalar::all(0));
	for (int r = 0, val = 1; r < src.rows; ++r)
		for (int c = 0; c < src.cols; ++c, ++val)
			src.at<float>(r, c) = (float)val;
#else
	const cv::Size kernelSize(5, 3);  // 3 (rows) x 5 (cols).
	cv::Mat kernel(kernelSize, CV_32F, cv::Scalar::all(1));

	const cv::Size imageSize(6, 7);  // 7 (rows) x 6 (cols).
	cv::Mat src(imageSize, CV_32F, cv::Scalar::all(0));
	for (int r = 0, val = 1; r < src.rows; ++r)
		for (int c = 0; c < src.cols; ++c, ++val)
			src.at<float>(r, c) = (float)val;
#endif

	//--------------------
	{
		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		bool retval = false;
		{
			const auto start = std::chrono::high_resolution_clock::now();
			retval = swl::convolve2d<float, float>(src, dst, kernel, cv::BORDER_CONSTANT);
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		if (retval)
			std::cout << "Convolution Result 1 =\n" << dst << std::endl;
		else std::cerr << "Convolution failed." << std::endl;
	}

	//--------------------
	{
		std::vector<cv::Point> roi_points;
		roi_points.reserve(src.cols * src.rows);
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(c, r));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			const auto start = std::chrono::high_resolution_clock::now();
			// REF [site] >>
			//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
			//	https://laonple.blog.me/220866708835
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_points, cv::BORDER_CONSTANT));
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		std::cout << "Convolution Result 2 =\n" << dst << std::endl;
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0.0;
		const int ddepth = -1;

		cv::Mat dst;
		{
			const auto start = std::chrono::high_resolution_clock::now();
			cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_CONSTANT);
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		std::cout << "Convolution Result 3 =\n" << dst << std::endl;
	}
}

void image_convolution2d_example()
{
	const std::string img_filepath("../data/machine_vision/lena.jpg");
	cv::Mat src(cv::imread(img_filepath, cv::IMREAD_GRAYSCALE));
	if (src.empty())
	{
		std::cout << "Image not found: " << img_filepath << std::endl;
		return;
	}
	src.convertTo(src, CV_32FC1, 1.0f / 255.0f);

	const cv::Size kernelSize(3, 3);
	cv::Mat kernel(kernelSize, CV_32F, cv::Scalar::all(0));
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(1, 0) = 0.0f;
	kernel.at<float>(2, 0) = 1.0f;
	kernel.at<float>(0, 1) = -2.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(2, 1) = 2.0f;
	kernel.at<float>(0, 2) = -1.0f;
	kernel.at<float>(1, 2) = 0.0f;
	kernel.at<float>(2, 2) = 1.0f;

	//--------------------
	{
		cv::Mat dst(src.size(), src.type());
		bool retval = false;
		{
			const auto start = std::chrono::high_resolution_clock::now();
			retval = swl::convolve2d<float, float>(src, dst, kernel);
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		if (retval)
			cv::imshow("Convolution Result 1", dst);
		else std::cerr << "Convolvution failed." << std::endl;
	}

	//--------------------
	{
		std::vector<cv::Point> roi_points;
		roi_points.reserve(src.cols * src.rows);
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(c, r));

		cv::Mat dst(src.size(), src.type());
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			const auto start = std::chrono::high_resolution_clock::now();
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_points));
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		cv::imshow("Convolution Result 2", dst);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0.0;
		const int ddepth = -1;

		cv::Mat dst;
		{
			const auto start = std::chrono::high_resolution_clock::now();
			cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		cv::imshow("Convolution Result 3", dst);
	}

	cv::waitKey(0);
}

void image_roi_convolution2d_example()
{
	const std::string img_filepath("../data/machine_vision/lena.jpg");
	cv::Mat src(cv::imread(img_filepath, cv::IMREAD_GRAYSCALE));
	if (src.empty())
	{
		std::cout << "Image not found: " << img_filepath << std::endl;
		return;
	}
	src.convertTo(src, CV_32FC1, 1.0f / 255.0f);

	const cv::Mat src_roi(src, cv::Rect(100, 50, 200, 300));
	std::vector<cv::Point> roi_points;
	roi_points.reserve(src_roi.cols * src_roi.rows);
	for (int c = 100; c < 300; ++c)
		for (int r = 50; r < 350; ++r)
			roi_points.push_back(cv::Point(c, r));

	std::vector<cv::Point> roi_boundary_points;
	roi_boundary_points.reserve(2 * (src_roi.cols + src_roi.rows));
	for (int c = 100; c < 300; ++c)
	{
		roi_boundary_points.push_back(cv::Point(c, 50));
		roi_boundary_points.push_back(cv::Point(c, 340));
	}
	for (int r = 50; r < 350; ++r)
	{
		roi_boundary_points.push_back(cv::Point(100, r));
		roi_boundary_points.push_back(cv::Point(299, r));
	}

	cv::Size wholeSize;
	cv::Point offsetPt;
	src_roi.locateROI(wholeSize, offsetPt);
	std::cout << "Offset point = " << offsetPt << std::endl;
	std::cout << "Whole size = " << wholeSize << std::endl;

	const cv::Size kernelSize(3, 3);
	cv::Mat kernel(kernelSize, CV_32F, cv::Scalar::all(0));
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(1, 0) = 0.0f;
	kernel.at<float>(2, 0) = 1.0f;
	kernel.at<float>(0, 1) = -2.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(2, 1) = 2.0f;
	kernel.at<float>(0, 2) = -1.0f;
	kernel.at<float>(1, 2) = 0.0f;
	kernel.at<float>(2, 2) = 1.0f;

	//--------------------
	{
		cv::Mat dst_roi(src_roi.size(), src_roi.type());
		bool retval = false;
		{
			const auto start = std::chrono::high_resolution_clock::now();
			retval = swl::convolve2d<float, float>(src_roi, dst_roi, kernel);
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		if (!retval)
		{
			std::cerr << "Convolution ROI failed." << std::endl;
			return;
		}

		//cv::imshow("Convolution ROI Result 1", dst_roi);
		cv::Mat dst;  src.copyTo(dst);
		dst_roi.copyTo(dst(cv::Rect(offsetPt.x, offsetPt.y, dst_roi.cols, dst_roi.rows)));
		cv::imshow("Convolution ROI Result 1", dst);
	}

	//--------------------
	{
		//cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		cv::Mat dst;  src.copyTo(dst);
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			const auto start = std::chrono::high_resolution_clock::now();
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_points));
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		cv::imshow("Convolution ROI Result 2", dst);
	}

	//--------------------
	{
		//cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		cv::Mat dst;  src.copyTo(dst);
		{
			const auto start = std::chrono::high_resolution_clock::now();
			cv::parallel_for_(cv::Range(0, (int)roi_boundary_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_boundary_points));
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
		cv::imshow("Convolution ROI Result 3", dst);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0.0;
		const int ddepth = -1;

		cv::Mat dst_roi;
		{
			const auto start = std::chrono::high_resolution_clock::now();
			cv::filter2D(src_roi, dst_roi, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
			std::cout << "Took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}

		//cv::imshow("Convolution ROI Result 3", dst_roi);
		cv::Mat dst;  src.copyTo(dst);
		dst_roi.copyTo(dst(cv::Rect(offsetPt.x, offsetPt.y, dst_roi.cols, dst_roi.rows)));
		cv::imshow("Convolution ROI Result 4", dst);
	}

	cv::waitKey(0);
}

}  // namespace local
}  // unnamed namespace

void convolution_test()
{
	local::simple_convolution2d_example();

	// Examples of parallel processing based on cv::parallel_for_() & cv::ParallelLoopBody.
	//local::image_convolution2d_example();
	//local::image_roi_convolution2d_example();
}
