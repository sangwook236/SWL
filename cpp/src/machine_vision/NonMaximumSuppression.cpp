#include "swl/Config.h"
#include "swl/machine_vision/NonMaximumSuppression.h"


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace swl {

/*static*/ void NonMaximumSuppression::computeNonMaximumSuppression(const cv::Mat &in_float, cv::Mat &out_uint8)
{
	// Non-maximum suppression.
	out_uint8 = cv::Mat::zeros(in_float.size(), CV_8UC1);

	const int &rows = in_float.rows;
	const int &cols = in_float.cols;
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			const float &pix = in_float.at<float>(r, c);

			if (r - 1 >= 0)  // (r - 1, c).
			{
				const float &pix2 = in_float.at<float>(r - 1, c);
				if (pix <= pix2) continue;
			}
			if (c + 1 > cols)  // (r, c + 1).
			{
				const float &pix2 = in_float.at<float>(r, c + 1);
				if (pix <= pix2) continue;
			}
			if (r + 1 > rows)  // (r + 1, c).
			{
				const float &pix2 = in_float.at<float>(r + 1, c);
				if (pix <= pix2) continue;
			}
			if (c - 1 >= 0)  // (r, c - 1).
			{
				const float &pix2 = in_float.at<float>(r, c - 1);
				if (pix <= pix2) continue;
			}

			if (r - 1 >= 0 && c + 1 < cols)  // (r - 1, c + 1).
			{
				const float &pix2 = in_float.at<float>(r - 1, c + 1);
				if (pix <= pix2) continue;
			}
			if (r + 1 < rows && c + 1 < cols)  // (r + 1, c + 1).
			{
				const float &pix2 = in_float.at<float>(r + 1, c + 1);
				if (pix <= pix2) continue;
			}
			if (r + 1 < rows && c - 1 >= 0)  // (r + 1, c - 1).
			{
				const float &pix2 = in_float.at<float>(r + 1, c - 1);
				if (pix <= pix2) continue;
			}
			if (r - 1 >= 0 && c - 1 >= 0)  // (r - 1, c - 1).
			{
				const float &pix2 = in_float.at<float>(r - 1, c - 1);
				if (pix <= pix2) continue;
			}

			out_uint8.at<unsigned char>(r, c) = 255;
		}
	}
}

// FiXME [fix] >> Not correctly working.
void NonMaximumSuppression::findMountainChain(const cv::Mat &in_float, cv::Mat &out_uint8)
{
	const int &rows = in_float.rows;
	const int &cols = in_float.cols;

	cv::Mat visit_flag(in_float.size(), CV_8UC1, cv::Scalar::all(0));
	for (int r = 0; r < rows; ++r)
		for (int c = 0; c < cols; ++c)
			if (255 == out_uint8.at<unsigned char>(r, c))
				checkMountainPeak(r, c, rows, cols, in_float, out_uint8, visit_flag, true);
}

// FiXME [fix] >> Not correctly working.
/*static*/ void NonMaximumSuppression::checkMountainPeak(const int ridx, const int cidx, const int rows, const int cols, const cv::Mat &in_float, cv::Mat &peak_flag, cv::Mat &visit_flag, const bool start_flag)
{
/*
	const int r1 = ridx - 1 < 0 ? 0 : ridx - 1;
	const int r2 = ridx + 1 >= rows ? rows - 1 : ridx + 1;
	const int c1 = cidx - 1 < 0 ? 0 : cidx - 1;
	const int c2 = cidx + 1 >= cols ? cols - 1 : cidx + 1;
*/
	const int r1 = ridx - 1 < 0 ? ridx : ridx - 1;
	const int r2 = ridx + 1 >= rows ? ridx : ridx + 1;
	const int c1 = cidx - 1 < 0 ? cidx : cidx - 1;
	const int c2 = cidx + 1 >= cols ? cidx : cidx + 1;
	assert(r1 <= r2 && c1 <= c2);

	visit_flag.at<unsigned char>(ridx, cidx) = 255;

	const float &pix = in_float.at<float>(ridx, cidx);
	bool near_peak_flag = false;
	if (!start_flag)
	{
		for (int r = r1; r <= r2; ++r)
		{
			for (int c = c1; c <= c2; ++c)
			{
				if (ridx == r && cidx == c) continue;
				if (255 == peak_flag.at<unsigned char>(r, c))
				{
					near_peak_flag = true;
					continue;
				}

				if (in_float.at<float>(r, c) > pix) return;
			}
		}
	}

	if (start_flag || near_peak_flag)
	{
		peak_flag.at<unsigned char>(ridx, cidx) = 255;
		for (int r = r1; r <= r2; ++r)
			for (int c = c1; c <= c2; ++c)
				if ((ridx != r || cidx != c) && 255 != peak_flag.at<unsigned char>(r, c) && 0 == visit_flag.at<unsigned char>(r, c))
					checkMountainPeak(r, c, rows, cols, in_float, peak_flag, visit_flag, false);
	}
}

}  // namespace swl
