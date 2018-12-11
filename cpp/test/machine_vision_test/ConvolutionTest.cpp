#include "swl/Config.h"
#include "swl/machine_vision/Convolution.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <vector>
#include <string>


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
template<typename T>
class ParallelLoopConvolve2D : public cv::ParallelLoopBody
{
public:
	ParallelLoopConvolve2D(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_CONSTANT, const cv::Scalar &borderVal = cv::Scalar::all(0))
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

		const int border_x = kernel.rows / 2;
		const int border_y = kernel.cols / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}

	/*virtual*/ void operator()(const cv::Range &r) const
	{
		for (int i = r.start; i < r.end && i < num_points_; ++i)
		{
			const cv::Point &pt = points_[i];
			T sum = (T)0;
			for (int kc = 0; kc < kernel_flip_.cols; ++kc)
				for (int kr = 0; kr < kernel_flip_.rows; ++kr)
					sum += kernel_flip_.at<T>(kr, kc) * src_ex_.at<T>(pt.x + kr, pt.y + kc);

			dst_.at<T>(pt.x, pt.y) = sum;
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

// Erode arbitrary regions in an image.
// A kernel can be a size or an image.
// ROI can be a set of points, a mask image (binary image), or a set of (rectangular) regions.
//	We can use a set of blobs (images) as ROI.
template<typename T>
class ParallelLoopErode : public cv::ParallelLoopBody
{
public:
	ParallelLoopErode(const cv::Mat &src, cv::Mat &dst, const cv::Size kernelSize, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_CONSTANT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: src_ex_(), dst_(dst), points_(points), kernelSize_(kernelSize), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernelSize_.width % 2 || 0 == kernelSize_.height % 2)
		{
			std::cerr << "Invalid kernel size." << std::endl;
			return;
		}

		const int border_x = kernelSize_.width / 2;
		const int border_y = kernelSize_.height / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}

	/*virtual*/ void operator()(const cv::Range &r) const
	{
		for (int i = r.start; i < r.end && i < num_points_; ++i)
		{
			const cv::Point &pt = points_[i];
			T min = std::numeric_limits<T>::max();
			for (int kc = 0; kc < kernelSize_.height; ++kc)
				for (int kr = 0; kr < kernelSize_.width; ++kr)
					min = std::min(min, src_ex_.at<T>(pt.x + kr, pt.y + kc));

			dst_.at<T>(pt.x, pt.y) = min;
		}
	}

private:
	cv::Mat src_ex_;
	cv::Mat &dst_;
	const cv::Size kernelSize_;
	const std::vector<cv::Point> &points_;
	const int borderType_;
	const cv::Scalar &borderVal_;
	const size_t num_points_;
};

// Dilate arbitrary regions in an image.
// A kernel can be a size or an image.
// ROI can be a set of points, a mask image (binary image), or a set of (rectangular) regions.
//	We can use a set of blobs (images) as ROI.
template<typename T>
class ParallelLoopDilate : public cv::ParallelLoopBody
{
public:
	ParallelLoopDilate(const cv::Mat &src, cv::Mat &dst, const cv::Size kernelSize, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_CONSTANT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: src_ex_(), dst_(dst), points_(points), kernelSize_(kernelSize), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernelSize_.width % 2 || 0 == kernelSize_.height % 2)
		{
			std::cerr << "Invalid kernel size." << std::endl;
			return;
		}

		const int border_x = kernelSize_.width / 2;
		const int border_y = kernelSize_.height / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}

	/*virtual*/ void operator()(const cv::Range &r) const
	{
		for (int i = r.start; i < r.end && i < num_points_; ++i)
		{
			const cv::Point &pt = points_[i];
			T max = std::numeric_limits<T>::min();
			for (int kc = 0; kc < kernelSize_.height; ++kc)
				for (int kr = 0; kr < kernelSize_.width; ++kr)
					max = std::max(max, src_ex_.at<T>(pt.x + kr, pt.y + kc));

			dst_.at<T>(pt.x, pt.y) = max;
		}
	}

private:
	cv::Mat src_ex_;
	cv::Mat &dst_;
	const cv::Size kernelSize_;
	const std::vector<cv::Point> &points_;
	const int borderType_;
	const cv::Scalar &borderVal_;
	const size_t num_points_;
};

void simple_convolution2d_example()
{
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

	//--------------------
	{
		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		bool retval = false;
		{
			boost::timer::auto_cpu_timer timer;
			retval = swl::convolve2d<float, float>(src, dst, kernel);
		}
		if (retval)
			std::cout << "Convolution Result 1 =\n" << dst << std::endl;
		else std::cerr << "Convolution failed." << std::endl;
	}

	//--------------------
	{
		std::vector<cv::Point> roi_points;
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(r, c));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_points));
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
			boost::timer::auto_cpu_timer timer;
			cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
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
			boost::timer::auto_cpu_timer timer;
			retval = swl::convolve2d<float, float>(src, dst, kernel);
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
				roi_points.push_back(cv::Point(r, c));

		cv::Mat dst(src.size(), src.type());
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_points));
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
			boost::timer::auto_cpu_timer timer;
			cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
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
			roi_points.push_back(cv::Point(r, c));

	std::vector<cv::Point> roi_boundary_points;
	roi_boundary_points.reserve(2 * (src_roi.cols + src_roi.rows));
	for (int c = 100; c < 300; ++c)
	{
		roi_boundary_points.push_back(cv::Point(50, c));
		roi_boundary_points.push_back(cv::Point(340, c));
	}
	for (int r = 50; r < 350; ++r)
	{
		roi_boundary_points.push_back(cv::Point(r, 100));
		roi_boundary_points.push_back(cv::Point(r, 299));
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
			boost::timer::auto_cpu_timer timer;
			retval = swl::convolve2d<float, float>(src_roi, dst_roi, kernel);
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
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_points));
		}
		cv::imshow("Convolution ROI Result 2", dst);
	}

	//--------------------
	{
		//cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		cv::Mat dst;  src.copyTo(dst);
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_boundary_points.size()), ParallelLoopConvolve2D<float>(src, dst, kernel, roi_boundary_points));
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
			boost::timer::auto_cpu_timer timer;
			cv::filter2D(src_roi, dst_roi, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
		}

		//cv::imshow("Convolution ROI Result 3", dst_roi);
		cv::Mat dst;  src.copyTo(dst);
		dst_roi.copyTo(dst(cv::Rect(offsetPt.x, offsetPt.y, dst_roi.cols, dst_roi.rows)));
		cv::imshow("Convolution ROI Result 4", dst);
	}

	cv::waitKey(0);
}

void image_erosion_example()
{
	const std::string img_filepath("../data/machine_vision/box_256x256_1.png");
	cv::Mat src(cv::imread(img_filepath, cv::IMREAD_GRAYSCALE));
	if (src.empty())
	{
		std::cout << "Image not found: " << img_filepath << std::endl;
		return;
	}

	const cv::Size kernelSize(3, 3);

	//--------------------
	{
		std::vector<cv::Point> roi_points;
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(r, c));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopErode<uint8_t>(src, dst, kernelSize, roi_points));
		}
		cv::imshow("Erosion Result 1", dst);
	}

	//--------------------
	// NOTE [info] >> Not too-bad implementation.
	{
		std::vector<cv::Point> roi_boundary_points;
		roi_boundary_points.reserve(2 * (128 + 128));
		for (int c = 65; c < 192; ++c)
		{
			roi_boundary_points.push_back(cv::Point(65, c));
			//roi_boundary_points.push_back(cv::Point(191, c));
		}
		for (int r = 65; r < 192; ++r)
		{
			//roi_boundary_points.push_back(cv::Point(r, 65));
			roi_boundary_points.push_back(cv::Point(r, 191));
		}
		for (int i = 65; i < 192; ++i)
		{
			roi_boundary_points.push_back(cv::Point(i, i));
			roi_boundary_points.push_back(cv::Point(i, i));
		}

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			boost::timer::auto_cpu_timer timer;

			// Find pixels which can be operated with boundary points.
			cv::Mat mask(src.size(), src.type(), cv::Scalar::all(0));
			std::for_each(roi_boundary_points.begin(), roi_boundary_points.end(), [&](const cv::Point &pt) { mask.at<uint8_t>(pt.x, pt.y) = 1; });
			const cv::Mat kern(cv::Mat::ones(kernelSize, src.type()));
			cv::filter2D(mask, mask, -1, kern, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
			//cv::imshow("Mask 0", mask > 0);

			// Find intersection pixels.
			cv::bitwise_and(src, mask, mask);
			//cv::imshow("Mask 1", mask > 0);

			// Find extended boundary points.
			std::vector<cv::Point> roi_ext_boundary_points;
			cv::findNonZero(mask, roi_ext_boundary_points);

			// Erode.
			cv::parallel_for_(cv::Range(0, (int)roi_ext_boundary_points.size()), ParallelLoopErode<uint8_t>(src, dst, kernelSize, roi_ext_boundary_points));
		}
		cv::imshow("Erosion Result 2", dst);
		//cv::imshow("Erosion Result 2", src - dst > 0);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0.0;
		const cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, kernelSize, anchor));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			boost::timer::auto_cpu_timer timer;
			//cv::erode(src, dst, cv::Mat(), anchor, delta, cv::BORDER_DEFAULT);
			cv::erode(src, dst, kernel, anchor, delta, cv::BORDER_DEFAULT);
		}
		cv::imshow("Erosion Result 3", dst);
	}

	cv::waitKey(0);
}

void image_dilation_example()
{
	const std::string img_filepath("../data/machine_vision/box_256x256_1.png");
	cv::Mat src(cv::imread(img_filepath, cv::IMREAD_GRAYSCALE));
	if (src.empty())
	{
		std::cout << "Image not found: " << img_filepath << std::endl;
		return;
	}
	
	const cv::Size kernelSize(3, 3);

	//--------------------
	{
		std::vector<cv::Point> roi_points;
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(r, c));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopDilate<uint8_t>(src, dst, kernelSize, roi_points));
		}
		cv::imshow("Dilation Result 1", dst);
	}

	//--------------------
	// NOTE [info] >> Not too-bad implementation.
	{
		std::vector<cv::Point> roi_boundary_points;
		roi_boundary_points.reserve(2 * (128 + 128));
		for (int c = 65; c < 192; ++c)
		{
			roi_boundary_points.push_back(cv::Point(65, c));
			//roi_boundary_points.push_back(cv::Point(191, c));
		}
		for (int r = 65; r < 192; ++r)
		{
			//roi_boundary_points.push_back(cv::Point(r, 65));
			roi_boundary_points.push_back(cv::Point(r, 191));
		}
		for (int i = 65; i < 192; ++i)
		{
			roi_boundary_points.push_back(cv::Point(i, i));
			roi_boundary_points.push_back(cv::Point(i, i));
		}

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			boost::timer::auto_cpu_timer timer;

			// Find pixels which can be operated with boundary points.
			cv::Mat mask(src.size(), src.type(), cv::Scalar::all(0));
			std::for_each(roi_boundary_points.begin(), roi_boundary_points.end(), [&](const cv::Point &pt) { mask.at<uint8_t>(pt.x, pt.y) = 1; });
			const cv::Mat kern(cv::Mat::ones(kernelSize, src.type()));
			cv::filter2D(mask, mask, -1, kern, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
			//cv::imshow("Mask 0", mask > 0);

			// Find difference pixels.
			cv::bitwise_or(src, mask, mask);
			mask -= src;
			//cv::imshow("Mask 1", mask > 0);

			// Find extended boundary points.
			std::vector<cv::Point> roi_ext_boundary_points;
			cv::findNonZero(mask, roi_ext_boundary_points);

			// Dilate.
			cv::parallel_for_(cv::Range(0, (int)roi_ext_boundary_points.size()), ParallelLoopDilate<uint8_t>(src, dst, kernelSize, roi_ext_boundary_points));
		}
		//cv::imshow("Dilation Result 2", dst);
		cv::imshow("Dilation Result 2", src + dst > 0);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0.0;
		const cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, kernelSize, anchor));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			boost::timer::auto_cpu_timer timer;
			//cv::dilate(src, dst, cv::Mat(), anchor, delta, cv::BORDER_DEFAULT);
			cv::dilate(src, dst, kernel, anchor, delta, cv::BORDER_DEFAULT);
		}
		cv::imshow("Dilation Result 3", dst);
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

	local::image_erosion_example();
	//local::image_dilation_example();
}
