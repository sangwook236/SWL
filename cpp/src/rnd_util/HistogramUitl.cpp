#include "swl/rnd_util/HistogramUtil.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>


namespace swl {

//-----------------------------------------------------------------------------
//

/*static*/ void HistogramUtil::normalizeHistogram(cv::MatND &hist, const double factor)
{
#if 0
	// FIXME [modify] >>
	cvNormalizeHist(&(CvHistogram)hist, factor);
#else
	const cv::Scalar sums(cv::sum(hist));

	const double eps = 1.0e-20;
	if (std::fabs(sums[0]) < eps) return;

	cv::Mat tmp(hist);
	tmp.convertTo(hist, -1, factor / sums[0], 0.0);
#endif
}

/*static*/ void HistogramUtil::drawHistogram1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg)
{
#if 0
	for (int i = 0; i < binCount; ++i)
	{
		const float binVal(hist.at<float>(i));
		const int binHeight(cvRound(binVal * maxHeight / maxVal));
		cv::rectangle(
			histImg,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			binVal > maxVal ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255),
			cv::FILLED
		);
	}
#else
	const float *binPtr = (const float *)hist.data;
	for (int i = 0; i < binCount; ++i, ++binPtr)
	{
		const int binHeight(cvRound(*binPtr * maxHeight / maxVal));
		cv::rectangle(
			histImg,
			cv::Point(i*binWidth, maxHeight), cv::Point((i+1)*binWidth - 1, maxHeight - binHeight),
			*binPtr > maxVal ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255),
			cv::FILLED
		);
	}
#endif
}

/*static*/ void HistogramUtil::drawHistogram2D(const cv::MatND &hist, const int horzBinCount, const int vertBinCount, const double maxVal, const int horzBinSize, const int vertBinSize, cv::Mat &histImg)
{
#if 0
	for (int v = 0; v < vertBinCount; ++v)
		for (int h = 0; h < horzBinCount; ++h)
		{
			const float binVal(hist.at<float>(v, h));
			cv::rectangle(
				histImg,
				cv::Point(h*horzBinSize, v*vertBinSize), cv::Point((h+1)*horzBinSize - 1, (v+1)*vertBinSize - 1),
				binVal > maxVal ? CV_RGB(255, 0, 0) : cv::Scalar::all(cvRound(binVal * 255.0 / maxVal)),
				CV_FILLED
			);
		}
#else
	const float *binPtr = (const float *)hist.data;
	for (int v = 0; v < vertBinCount; ++v)
		for (int h = 0; h < horzBinCount; ++h, ++binPtr)
		{
			const int intensity();
			cv::rectangle(
				histImg,
				cv::Point(h*horzBinSize, v*vertBinSize), cv::Point((h+1)*horzBinSize - 1, (v+1)*vertBinSize - 1),
				*binPtr > maxVal ? cv::Scalar(CV_RGB(255, 0, 0)) : cv::Scalar::all(cvRound(*binPtr * 255.0 / maxVal)),
				cv::FILLED
			);
		}
#endif
}

}  // namespace swl
