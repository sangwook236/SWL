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
			float sum = 0.0f;
			for (int kc = 0; kc < kernel_flip_.cols; ++kc)
				for (int kr = 0; kr < kernel_flip_.rows; ++kr)
					sum += kernel_flip_.at<float>(kr, kc) * src_ex_.at<float>(pt.x + kr, pt.y + kc);

			dst_.at<float>(pt.x, pt.y) = sum;
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
			float min = std::numeric_limits<float>::max();
			for (int kc = 0; kc < kernelSize_.height; ++kc)
				for (int kr = 0; kr < kernelSize_.width; ++kr)
					min = std::min(min, src_ex_.at<float>(pt.x + kr, pt.y + kc));

			dst_.at<float>(pt.x, pt.y) = min;
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
	const int kernel_size = 3;
	cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);
	kernel.at<float>(0, 0) = -1.0f;
	kernel.at<float>(1, 0) = 0.0f;
	kernel.at<float>(2, 0) = 1.0f;
	kernel.at<float>(0, 1) = -2.0f;
	kernel.at<float>(1, 1) = 0.0f;
	kernel.at<float>(2, 1) = 2.0f;
	kernel.at<float>(0, 2) = -1.0f;
	kernel.at<float>(1, 2) = 0.0f;
	kernel.at<float>(2, 2) = 1.0f;

	cv::Mat src = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);
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
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
	bool retval = false;
	{
		boost::timer::auto_cpu_timer timer;
		retval = swl::convolve2d<float, float>(src, dst, kernel);
	}
	if (retval)
		std::cout << "Result 1 =\n" << dst << std::endl;
	else std::cerr << "Convolvution failed." << std::endl;

	//--------------------
	std::vector<cv::Point> points;
	for (int c = 0; c < src.cols; ++c)
		for (int r = 0; r < src.rows; ++r)
			points.push_back(cv::Point(r, c));

	dst = cv::Mat::zeros(src.size(), src.type());
	// REF [site] >>
	//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
	//	https://laonple.blog.me/220866708835
	{
		boost::timer::auto_cpu_timer timer;
		cv::parallel_for_(cv::Range(0, (int)points.size()), ParallelLoopConvolve2D(src, dst, kernel, points));
	}
	std::cout << "Result 2 =\n" << dst << std::endl;

	//--------------------
	const cv::Point anchor(-1, -1);
	const double delta = 0;
	const int ddepth = -1;

	dst = cv::Mat::zeros(src.size(), src.type());
	{
		boost::timer::auto_cpu_timer timer;
		cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
	}
	std::cout << "Result 3 =\n" << dst << std::endl;
}

void image_convolution2d_example()
{
	const std::string img_filepath("../data/machine_vision/lena.jpg");
	cv::Mat src = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
	if (src.empty())
	{
		std::cout << "Image not found: " << img_filepath << std::endl;
		return;
	}
	src.convertTo(src, CV_32FC1);

	const int kernel_size = 3;
	cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);
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
		cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
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
		std::vector<cv::Point> points;
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				points.push_back(cv::Point(r, c));

		cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)points.size()), ParallelLoopConvolve2D(src, dst, kernel, points));
		}
		cv::imshow("Convolution Result 2", dst);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0;
		const int ddepth = -1;

		cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
		{
			boost::timer::auto_cpu_timer timer;
			cv::filter2D(src, dst, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
		}
		cv::imshow("Convolution Result 3", dst);
	}

	cv::waitKey(0);
}

void image_erosion_example()
{
	const std::string img_filepath("../data/machine_vision/box_256x256_1.png");
	cv::Mat src = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
	if (src.empty())
	{
		std::cout << "Image not found: " << img_filepath << std::endl;
		return;
	}
	src.convertTo(src, CV_32FC1);
	
	const cv::Size kernelSize(3, 3);

	//--------------------
	{
		std::vector<cv::Point> points;
		for (int c = 0; c < src.cols; ++c)
			for (int r = 0; r < src.rows; ++r)
				points.push_back(cv::Point(r, c));

		cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
		// REF [site] >>
		//	https://docs.opencv.org/4.0.0/db/de0/group__core__utils.html
		//	https://laonple.blog.me/220866708835
		{
			boost::timer::auto_cpu_timer timer;
			cv::parallel_for_(cv::Range(0, (int)points.size()), ParallelLoopErode(src, dst, kernelSize, points));
		}
		cv::imshow("Erosion Result 1", dst);
	}

	//--------------------
	{
		const cv::Point anchor(-1, -1);
		const double delta = 0;
		const int ddepth = -1;

		cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
		{
			boost::timer::auto_cpu_timer timer;
			cv::erode(src, dst, cv::Mat(), anchor, delta, cv::BORDER_DEFAULT);
		}
		cv::imshow("Erosion Result 2", dst);
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
	local::image_erosion_example();
}
