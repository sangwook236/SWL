#include "swl/Config.h"
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

// Erode arbitrary regions in an image.
// A kernel can be a size or an image.
// ROI can be a set of points, a mask image (binary image), or a set of (rectangular) regions.
//	We can use a set of blobs (images) as ROI.
template<typename T>
class ParallelLoopErode : public cv::ParallelLoopBody
{
public:
	typedef cv::ParallelLoopBody base_type;

public:
	ParallelLoopErode(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_DEFAULT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: isRectangularKernel_(false), src_ex_(), dst_(dst), kernel_(kernel), points_(points), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernel_.cols % 2 || 0 == kernel_.rows % 2)
		{
			std::cerr << "Invalid kernel size." << std::endl;
			return;
		}

		const int border_x = kernel_.cols / 2;
		const int border_y = kernel_.rows / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}
	ParallelLoopErode(const cv::Mat &src, cv::Mat &dst, const cv::Size &kernelSize, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_DEFAULT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: isRectangularKernel_(true), src_ex_(), dst_(dst), kernel_(cv::Mat::ones(kernelSize, src.type())), points_(points), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernel_.cols % 2 || 0 == kernel_.rows % 2)
		{
			std::cerr << "Invalid kernel size." << std::endl;
			return;
		}

		const int border_x = kernel_.cols / 2;
		const int border_y = kernel_.rows / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}

public:
	/*virtual*/ void operator()(const cv::Range &r) const
	{
		if (isRectangularKernel_)
			for (int i = r.start; i < r.end && i < num_points_; ++i)
			{
				const cv::Point &pt = points_[i];
#if false
				const cv::Mat src_ex_roi(src_ex_, cv::Rect(pt.x, pt.y, kernel_.cols, kernel_.rows));
				dst_.at<T>(pt.y, pt.x) = *std::min_element(src_ex_roi.begin<T>(), src_ex_roi.end<T>());
#else
				T min = std::numeric_limits<T>::max();
				for (int kr = 0; kr < kernel_.rows; ++kr)
					for (int kc = 0; kc < kernel_.cols; ++kc)
						min = std::min(min, src_ex_.at<T>(pt.y + kr, pt.x + kc));
				dst_.at<T>(pt.y, pt.x) = min;
#endif
			}
		else
			for (int i = r.start; i < r.end && i < num_points_; ++i)
			{
				const cv::Point &pt = points_[i];
				T min = std::numeric_limits<T>::max();
				for (int kr = 0; kr < kernel_.rows; ++kr)
					for (int kc = 0; kc < kernel_.cols; ++kc)
						if (kernel_.at<T>(kr, kc) > (T)0)
							min = std::min(min, src_ex_.at<T>(pt.y + kr, pt.x + kc));
				dst_.at<T>(pt.y, pt.x) = min;
			}
	}

private:
	cv::Mat src_ex_;
	cv::Mat &dst_;
	const bool isRectangularKernel_;
	const cv::Mat &kernel_;
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
	ParallelLoopDilate(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_DEFAULT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: isRectangularKernel_(false), src_ex_(), dst_(dst), kernel_(kernel), points_(points), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernel_.cols % 2 || 0 == kernel_.rows % 2)
		{
			std::cerr << "Invalid kernel size." << std::endl;
			return;
		}

		const int border_x = kernel_.cols / 2;
		const int border_y = kernel_.rows / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}
	ParallelLoopDilate(const cv::Mat &src, cv::Mat &dst, const cv::Size &kernelSize, const std::vector<cv::Point> &points, const int borderType = cv::BORDER_DEFAULT, const cv::Scalar &borderVal = cv::Scalar::all(0))
	: isRectangularKernel_(true), src_ex_(), dst_(dst), kernel_(cv::Mat::ones(kernelSize, src.type())), points_(points), borderType_(borderType), borderVal_(borderVal), num_points_(points.size())
	{
		if (0 == kernel_.cols % 2 || 0 == kernel_.rows % 2)
		{
			std::cerr << "Invalid kernel size." << std::endl;
			return;
		}

		const int border_x = kernel_.cols / 2;
		const int border_y = kernel_.rows / 2;
		cv::copyMakeBorder(src, src_ex_, border_y, border_y, border_x, border_x, borderType_, borderVal_);
	}

public:
	/*virtual*/ void operator()(const cv::Range &r) const
	{
		for (int i = r.start; i < r.end && i < num_points_; ++i)
		{
			const cv::Point &pt = points_[i];
#if false
			const cv::Mat src_ex_roi(src_ex_, cv::Rect(pt.x, pt.y, kernel_.cols, kernel_.rows));
			dst_.at<T>(pt.y, pt.x) = *std::max_element(src_ex_roi.begin<T>(), src_ex_roi.end<T>());
#else
			T max = std::numeric_limits<T>::min();
			for (int kr = 0; kr < kernel_.rows; ++kr)
				for (int kc = 0; kc < kernel_.cols; ++kc)
					max = std::max(max, src_ex_.at<T>(pt.y + kr, pt.x + kc));
			dst_.at<T>(pt.y, pt.x) = max;
#endif
		}
		return;

		if (isRectangularKernel_)
			for (int i = r.start; i < r.end && i < num_points_; ++i)
			{
				const cv::Point &pt = points_[i];
#if false
				const cv::Mat src_ex_roi(src_ex_, cv::Rect(pt.x, pt.y, kernel_.cols, kernel_.rows));
				dst_.at<T>(pt.y, pt.x) = *std::max_element(src_ex_roi.begin<T>(), src_ex_roi.end<T>());
#else
				T max = std::numeric_limits<T>::min();
				for (int kr = 0; kr < kernel_.rows; ++kr)
					for (int kc = 0; kc < kernel_.cols; ++kc)
						max = std::max(max, src_ex_.at<T>(pt.y + kr, pt.x + kc));
				dst_.at<T>(pt.y, pt.x) = max;
#endif
			}
		else
			for (int i = r.start; i < r.end && i < num_points_; ++i)
			{
				const cv::Point &pt = points_[i];
				T max = std::numeric_limits<T>::min();
				for (int kr = 0; kr < kernel_.rows; ++kr)
					for (int kc = 0; kc < kernel_.cols; ++kc)
						if (kernel_.at<T>(kr, kc) >(T)0)
							max = std::max(max, src_ex_.at<T>(pt.y + kr, pt.x + kc));
				dst_.at<T>(pt.y, pt.x) = max;
			}
	}

private:
	cv::Mat src_ex_;
	cv::Mat &dst_;
	const bool isRectangularKernel_;
	const cv::Mat &kernel_;
	const std::vector<cv::Point> &points_;
	const int borderType_;
	const cv::Scalar &borderVal_;
	const size_t num_points_;
};

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
		roi_points.reserve(src.cols * src.rows);
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(c, r));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopErode<uint8_t>(src, dst, kernelSize, roi_points));
			//cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopErode<uint8_t>(src, dst, cv::Mat::ones(kernelSize, src.type()), roi_points));
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
			roi_boundary_points.push_back(cv::Point(c, 65));
			//roi_boundary_points.push_back(cv::Point(c, 191));
		}
		for (int r = 65; r < 192; ++r)
		{
			//roi_boundary_points.push_back(cv::Point(65, r));
			roi_boundary_points.push_back(cv::Point(191, r));
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
			//cv::parallel_for_(cv::Range(0, (int)roi_ext_boundary_points.size()), ParallelLoopErode<uint8_t>(src, dst, cv::Mat::ones(kernelSize, src.type()), roi_ext_boundary_points));
		}
		//cv::imshow("Erosion Result 2", dst);
		cv::imshow("Erosion Result 2", src - dst > 0);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const int iterations = 1;
		const cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, kernelSize, anchor));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			boost::timer::auto_cpu_timer timer;
			//cv::erode(src, dst, cv::Mat(), anchor, delta, cv::BORDER_DEFAULT);
			cv::erode(src, dst, kernel, anchor, iterations, cv::BORDER_DEFAULT);
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
		roi_points.reserve(src.cols * src.rows);
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				roi_points.push_back(cv::Point(c, r));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopDilate<uint8_t>(src, dst, kernelSize, roi_points));
			//cv::parallel_for_(cv::Range(0, (int)roi_points.size()), ParallelLoopDilate<uint8_t>(src, dst, cv::Mat::ones(kernelSize, src.type()), roi_points));
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
			roi_boundary_points.push_back(cv::Point(c, 65));
			//roi_boundary_points.push_back(cv::Point(c, 191));
		}
		for (int r = 65; r < 192; ++r)
		{
			//roi_boundary_points.push_back(cv::Point(65, r));
			roi_boundary_points.push_back(cv::Point(191, r));
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
			//cv::parallel_for_(cv::Range(0, (int)roi_ext_boundary_points.size()), ParallelLoopDilate<uint8_t>(src, dst, cv::Mat::ones(kernelSize, src.type()), roi_ext_boundary_points));
		}
		//cv::imshow("Dilation Result 2", dst);
		cv::imshow("Dilation Result 2", src + dst > 0);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const int iterations = 1;
		const cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, kernelSize, anchor));

		cv::Mat dst(src.size(), src.type(), cv::Scalar::all(0));
		{
			boost::timer::auto_cpu_timer timer;
			//cv::dilate(src, dst, cv::Mat(), anchor, iterations, cv::BORDER_DEFAULT);
			cv::dilate(src, dst, kernel, anchor, iterations, cv::BORDER_DEFAULT);
		}
		cv::imshow("Dilation Result 3", dst);
	}

	cv::waitKey(0);
}

}  // namespace local
}  // unnamed namespace

void morphology_test()
{
	// Examples of parallel processing based on cv::parallel_for_() & cv::ParallelLoopBody.
	local::image_erosion_example();
	local::image_dilation_example();
}
