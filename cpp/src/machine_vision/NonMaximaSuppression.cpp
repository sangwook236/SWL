/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    nms.cpp
 *  Author:  Hilton Bristow
 *  Created: Jul 19, 2012
 */

#include <stdio.h>
#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "nms.hpp"
#include "swl/machine_vision/NonMaximumSuppression.h"
//#include "NonMaximaSuppression.hpp"

namespace swl {

using namespace std;
using namespace cv;

/*! @brief suppress non-maximal values
 *
 * nonMaximaSuppression produces a mask (dst) such that every non-zero
 * value of the mask corresponds to a local maxima of src. The criteria
 * for local maxima is as follows:
 *
 * 	For every possible (sz x sz) region within src, an element is a
 * 	local maxima of src iff it is strictly greater than all other elements
 * 	of windows which intersect the given element
 *
 * Intuitively, this means that all maxima must be at least sz+1 pixels
 * apart, though the spacing may be greater
 *
 * A gradient image or a constant image has no local maxima by the definition
 * given above
 *
 * The method is derived from the following paper:
 * A. Neubeck and L. Van Gool. "Efficient Non-Maximum Suppression," ICPR 2006
 *
 * Example:
 * \code
 * 	// Create a random test image.
 * 	Mat random(Size(2000,2000), DataType<float>::type);
 * 	randn(random, 1, 1);
 *
 * 	// Only look for local maxima above the value of 1.
 * 	Mat mask = (random > 1);
 *
 * 	// Find the local maxima with a window of 50.
 * 	Mat maxima;
 * 	nonMaximaSuppression(random, 50, maxima, mask);
 *
 * 	// Optionally set all non-maxima to zero.
 * 	random.setTo(0, maxima == 0);
 * \endcode
 *
 * @param src The input image/matrix, of any valid cv type.
 * @param sz The size of the window.
 * @param dst The mask of type CV_8U, where non-zero elements correspond to local maxima of the src.
 * @param mask An input mask to skip particular elements.
 */
/*static*/ void NonMaximumSuppression::computeNonMaximumSuppression(const cv::Mat& src, const int sz, cv::Mat& dst, const cv::Mat mask)
//void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask)
{
	// Initialise the block mask and destination.
	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask.empty();
	Mat block = 255*Mat::ones(Size(2*sz+1,2*sz+1), CV_8UC1);
	dst = Mat::zeros(src.size(), CV_8UC1);

	// Iterate over image blocks.
	for (int m = 0; m < M; m += sz + 1)
	{
		for (int n = 0; n < N; n += sz + 1)
		{
			Point  ijmax;
			double vcmax, vnmax;

			// Get the maximal candidate within the block.
			Range ic(m, min(m+sz+1,M));
			Range jc(n, min(n+sz+1,N));
			minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : noArray());
			Point cc = ijmax + Point(jc.start,ic.start);

			// Search the neighbours centered around the candidate for the true maxima.
			Range in(max(cc.y-sz,0), min(cc.y+sz+1,M));
			Range jn(max(cc.x-sz,0), min(cc.x+sz+1,N));

			// Mask out the block whose maxima we already know.
			Mat blockmask;
			block(Range(0,in.size()), Range(0,jn.size())).copyTo(blockmask);
			Range iis(ic.start-in.start, min(ic.start-in.start+sz+1, in.size()));
			Range jis(jc.start-jn.start, min(jc.start-jn.start+sz+1, jn.size()));
			blockmask(iis, jis) = Mat::zeros(Size(jis.size(),iis.size()), CV_8UC1);

			minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
			Point cn = ijmax + Point(jn.start, in.start);

			// If the block centre is also the neighbour centre, then it's a local maxima.
			if (vcmax > vnmax)
			{
				dst.at<unsigned char>(cc.y, cc.x) = 255;
			}
		}
	}
}

}  // namespace swl
