#include "HistogramGenerator.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/math/distributions/normal.hpp>


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
			binVal > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			CV_FILLED
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
			*binPtr > maxVal ? CV_RGB(255, 0, 0) : CV_RGB(255, 255, 255),
			CV_FILLED
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
				*binPtr > maxVal ? CV_RGB(255, 0, 0) : cv::Scalar::all(cvRound(*binPtr * 255.0 / maxVal)),
				CV_FILLED
			);
		}
#endif
}

//-----------------------------------------------------------------------------
//

HistogramGeneratorBase::~HistogramGeneratorBase()
{
}

//-----------------------------------------------------------------------------
//

void ReferencePhaseHistogramGeneratorBase::createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const
{
	boost::math::normal dist(0.0, sigma);

#if 0
	for (int i = -180; i < 180; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % 360) : (360 + kk);
		hist.at<float>(idx) += (float)boost::math::pdf(dist, i);
	}
#else
	float *binPtr = (float *)hist.data;
	for (int i = -180; i < 180; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % 360) : (360 + kk);
		binPtr[idx] += (float)boost::math::pdf(dist, i);
	}
#endif
}

void ReferencePhaseHistogramGeneratorBase::createUniformHistogram(cv::MatND &hist) const
{
#if 0
	for (int i = 0; i < 360; ++i)
		hist.at<float>(i) += 1.0f / 360.0f;
#else
	float *binPtr = (float *)hist.data;
	for (int i = 0; i < 360; ++i, ++binPtr)
		*binPtr += 1.0f / 360.0f;
#endif
}

//-----------------------------------------------------------------------------
//

/*virtual*/ void ReferenceFullPhaseHistogramGenerator::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// create reference histograms
	histograms_.reserve(REF_HISTOGRAM_NUM);

	// unimodal distribution: for all gestures
	if (REF_UNIMODAL_HISTOGRAM_NUM > 0)
	{
		const size_t ref_unimodal_histogram_bin_width = (size_t)cvRound(360.0 / (REF_UNIMODAL_HISTOGRAM_NUM ? REF_UNIMODAL_HISTOGRAM_NUM : 1));
		for (size_t i = 0; i < REF_UNIMODAL_HISTOGRAM_NUM; ++i)
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(ref_unimodal_histogram_bin_width * i, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}

	// uniform distribution
	if (REF_UNIFORM_HISTOGRAM_NUM > 0)
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
}

//-----------------------------------------------------------------------------
//

/*virtual*/ void ReferenceHistogramGeneratorForClass1Gesture::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// create reference histograms
	histograms_.reserve(REF_HISTOGRAM_NUM);

	// bimodal distribution: for two-hand gesture
	if (REF_BIMODAL_HISTOGRAM_NUM_FOR_TWO_HAND_GESTURE > 0)
	{
		// bimodal distribution: for hand open & hand close
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(0, sigma_, tmp_hist);
			createNormalHistogram(180, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
		// bimodal distribution: for vertical scissors
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(90, sigma_, tmp_hist);
			createNormalHistogram(270, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}

	// uniform distribution
	if (REF_UNIFORM_HISTOGRAM_NUM > 0)
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
}

//-----------------------------------------------------------------------------
//

/*virtual*/ void ReferenceHistogramGeneratorForClass2Gesture::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// create reference histograms
	histograms_.reserve(REF_HISTOGRAM_NUM);

	// bimodal distribution: for horizontal & vertical flip
	if (REF_BIMODAL_HISTOGRAM_NUM > 0)
	{
		const size_t ref_bimodal_histogram_bin_width = (size_t)cvRound(180.0 / (REF_BIMODAL_HISTOGRAM_NUM ? REF_BIMODAL_HISTOGRAM_NUM : 1));
		for (size_t i = 0; i < REF_BIMODAL_HISTOGRAM_NUM; ++i)
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(ref_bimodal_histogram_bin_width * i, sigma_, tmp_hist);
			createNormalHistogram(ref_bimodal_histogram_bin_width * i + 180, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}

	// uniform distribution
	if (REF_UNIFORM_HISTOGRAM_NUM > 0)
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
}

//-----------------------------------------------------------------------------
//

/*virtual*/ void ReferenceHistogramGeneratorForClass3Gesture::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// create reference histograms
	histograms_.reserve(REF_HISTOGRAM_NUM);

	// unimodal distribution: for left & right fast move
	if (REF_UNIMODAL_HISTOGRAM_NUM > 0)
	{
		// FIXME [check] >>
		const size_t ref_unimodal_histogram_bin_width = (size_t)cvRound(360.0 / (REF_UNIMODAL_HISTOGRAM_NUM ? REF_UNIMODAL_HISTOGRAM_NUM : 1));
		for (size_t i = 0; i < REF_UNIMODAL_HISTOGRAM_NUM; ++i)
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(ref_unimodal_histogram_bin_width * i, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}

	// 3-modal distribution
	if (REF_TRIMODAL_HISTOGRAM_NUM > 0)
	{
		// for infinity
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(40, sigma_, tmp_hist);
			createNormalHistogram(140, sigma_, tmp_hist);
			createNormalHistogram(270, sigma_, tmp_hist);
			//createNormalHistogram(270, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
		// for triangle
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(50, sigma_, tmp_hist);
			createNormalHistogram(180, sigma_, tmp_hist);
			createNormalHistogram(310, sigma_, tmp_hist);

			// normalize histogram
			HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}

	// uniform distribution
	if (REF_UNIFORM_HISTOGRAM_NUM > 0)
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
}

//-----------------------------------------------------------------------------
//

void GestureIdPatternHistogramGeneratorBase::createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const
{
	boost::math::normal dist(0.0, sigma);

#if 0
	const int halfrows = hist.rows / 2;
	for (int i = -halfrows; i < halfrows; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % hist.rows) : (hist.rows + kk);
		hist.at<float>(idx) += (float)boost::math::pdf(dist, i);
	}
#else
	const int halfrows = hist.rows / 2;
	float *binPtr = (float *)hist.data;
	for (int i = -halfrows; i < halfrows; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % hist.rows) : (hist.rows + kk);
		binPtr[idx] += (float)boost::math::pdf(dist, i);
	}
#endif
}

void GestureIdPatternHistogramGeneratorBase::createUniformHistogram(cv::MatND &hist) const
{
#if 0
	for (int i = 0; i < hist.rows; ++i)
		hist.at<float>(i) += 1.0f / (float)hist.rows;
#else
	float *binPtr = (float *)hist.data;
	for (int i = 0; i < hist.rows; ++i, ++binPtr)
		*binPtr += 1.0f / (float)hist.rows;
#endif
}

//-----------------------------------------------------------------------------
//

/*virtual*/ void GestureIdPatternHistogramGeneratorForClass1Gesture::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// uniform distribution: for undefined gesture
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for left move & left fast move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(18, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for right move & right fast move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(0, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for up move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(27, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for down move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(9, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
/*
	// unimodal distribution: for hand open & hand close
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(36, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
*/
/*
	// bimodal distribution: for horizontal flip
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(0, sigma_, tmp_hist);
		createNormalHistogram(18, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// bimodal distribution: for vertical flip
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(9, sigma_, tmp_hist);
		createNormalHistogram(27, sigma_, tmp_hist);

		// normalize histogram
		HistogramUtil::normalizeHistogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
*/
}

}  // namespace swl
