#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <boost/math/constants/constants.hpp>
#include <algorithm>


namespace swl {
	
cv::Rect get_bounding_rect(const cv::Mat &img)
{
	std::vector<cv::Point> pts;
	pts.reserve(img.rows * img.cols);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			if (!img.at<unsigned char>(i, j))
				pts.push_back(cv::Point(j, i));

	return cv::boundingRect(pts);
}

void compute_phase_distribution_from_neighborhood(const cv::Mat &depth_map, const int radius)
{
	const int width = depth_map.cols;
	const int height = depth_map.rows;

	unsigned short dep0, dep;
	int lowerx, upperx, lowery, uppery, lowery2, uppery2;
	int i, j;

	const int num_pixels = (2*radius + 1) * (2*radius + 1) - 1;
	int num;

	// FIXME [enhance] >> speed up.
	//cv::Mat phase(height, width, CV_32FC1, cv::Scalar::all(0)), mag(height, width, CV_32FC1, cv::Scalar::all(0));
	float *phases = new float [num_pixels], *mags = new float [num_pixels];
	float *ptr_phase, *ptr_mag;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			dep0 = depth_map.at<unsigned short>(y, x);
			if (0 == dep0) continue;  // invalid depth.

			memset(phases, 0, num_pixels * sizeof(float));
			memset(mags, 0, num_pixels * sizeof(float));
			ptr_phase = phases;
			ptr_mag = mags;

			num = 0;
			for (int r = 1; r <= radius; ++r)
			{
				lowerx = std::max(0, x-r);
				upperx = std::min(x+r, width-1);
				lowery = std::max(0, y-r);
				uppery = std::min(y+r, height-1);
				lowery2 = std::max(0, y-r+1);
				uppery2 = std::min(y+r-1, height-1);

				if (y-r == lowery)
				{
					// upper horizontal pixels (rightward)
					for (i = lowerx; i <= upperx; ++i)
					{
						dep = depth_map.at<unsigned short>(lowery, i);
						// FIXME [check] >> does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // invalid depth.
						if (0 == dep || dep0 == dep) continue;  // invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - lowery), float(i - x)) : std::atan2(float(lowery - y), float(x - i));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(lowery - y), float(i - x)) : std::atan2(float(y - lowery), float(x - i));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
				if (x+r == upperx)
				{
					// right vertical pixels (downward)
					for (j = lowery2; j <= uppery2; ++j)
					{
						dep = depth_map.at<unsigned short>(j, upperx);
						// FIXME [check] >> does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // invalid depth.
						if (0 == dep || dep0 == dep) continue;  // invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - j), float(upperx - x)) : std::atan2(float(j - y), float(x - upperx));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(j - y), float(upperx - x)) : std::atan2(float(y - j), float(x - upperx));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
				if (y+r == uppery)
				{
					// lower horizontal pixels (leftward)
					for (i = upperx; i >= lowerx; --i)
					{
						dep = depth_map.at<unsigned short>(uppery, i);
						// FIXME [check] >> does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // invalid depth.
						if (0 == dep || dep0 == dep) continue;  // invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - uppery), float(i - x)) : std::atan2(float(uppery - y), float(x - i));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(uppery - y), float(i - x)) : std::atan2(float(y - uppery), float(x - i));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
				if (x-r == lowerx)
				{
					// left vertical pixels (upward)
					for (j = uppery2; j >= lowery2; --j)
					{
						dep = depth_map.at<unsigned short>(j, lowerx);
						// FIXME [check] >> does we consider the case of dep0 == dep?
						//if (0 == dep) continue;  // invalid depth.
						if (0 == dep || dep0 == dep) continue;  // invalid depth.

						*ptr_phase++ = (dep >= dep0) ? std::atan2(float(y - j), float(lowerx - x)) : std::atan2(float(j - y), float(x - lowerx));
						//*ptr_phase++ = (dep >= dep0) ? std::atan2(float(j - y), float(lowerx - x)) : std::atan2(float(y - j), float(x - lowerx));
						*ptr_mag++ = std::fabs(float(dep - dep0));
						++num;
					}
				}
			}

#if 0
			// for checking.
			std::cout << "phases (" << y << ", " << x << ") = " << std::endl;
			for (int ii = 0; ii < num_pixels; ++ii)
				std::cout << (phases[ii] * 180.0f / boost::math::constants::pi<float>()) << ", ";
			std::cout << std::endl;
			std::cout << "magnitude (" << y << ", " << x << ") = " << std::endl;
			for (int ii = 0; ii < num_pixels; ++ii)
				std::cout << mags[ii] << ", ";
			std::cout << std::endl;
			std::cout << "num (" << y << ", " << x << ") = " << num << std::endl;
#endif
		}
	}

	delete [] phases;
	delete [] mags;
}

// [ref] snake() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_hand_pose_estimation.cpp
void snake(IplImage *srcImage, IplImage *grayImage)
{
	const int NUMBER_OF_SNAKE_POINTS = 50;
	const int threshold = 90;

	float alpha = 3;
	float beta = 5;
	float gamma = 2;
	const int use_gradient = 1;
	const CvSize win = cvSize(21, 21);
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	IplImage *tmp_img = cvCloneImage(grayImage);
	IplImage *img = cvCloneImage(grayImage);

	// make a average filtering
	cvSmooth(tmp_img, img, CV_BLUR, 31, 15);
	//iplBlur(tmp_img, img, 31, 31, 15, 15);  // don't use IPL

	// thresholding
	cvThreshold(img, tmp_img, threshold, 255, CV_THRESH_BINARY);
	//iplThreshold(img, tmp_img, threshold);  // distImg is thresholded image (tmp_img)  // don't use IPL

	// expand the thressholded image of ones -smoothing the edge.
	// and move start position of snake out since there are no ballon force
	cvDilate(tmp_img, img, NULL, 3);

	cvReleaseImage(&tmp_img);

	// find the contours
	CvSeq *contour = NULL;
	CvMemStorage *storage = cvCreateMemStorage(0);
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// run through the found coutours
	CvPoint *points = new CvPoint [NUMBER_OF_SNAKE_POINTS];
	while (contour)
	{
		if (contour->total >= NUMBER_OF_SNAKE_POINTS)
		{
			//memset(points, 0, NUMBER_OF_SNAKE_POINTS * sizeof(CvPoint));

			cvSmooth(grayImage, img, CV_BLUR, 7, 3);
			//iplBlur(grayImage, img, 7, 7, 3, 3);  // put blured image in TempImg  // don't use IPL

#if 0
			CvPoint *pts = new CvPoint [contour->total];
			cvCvtSeqToArray(contour, pts, CV_WHOLE_SEQ);  // copy the contour to a array

			// number of jumps between the desired points (downsample only!)
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				points[i].x = pts[int(i * stride)].x;
				points[i].y = pts[int(i * stride)].y;
			}

			delete [] pts;
			pts = NULL;
#else
			const int stride = int(contour->total / NUMBER_OF_SNAKE_POINTS);
			for (int i = 0; i < NUMBER_OF_SNAKE_POINTS; ++i)
			{
				CvPoint *pt = CV_GET_SEQ_ELEM(CvPoint, contour, i * stride);
				points[i].x = pt->x;
				points[i].y = pt->y;
			}
#endif

			// snake
			cvSnakeImage(img, points, NUMBER_OF_SNAKE_POINTS, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);

			// draw snake on image
			cvPolyLine(srcImage, (CvPoint **)&points, &NUMBER_OF_SNAKE_POINTS, 1, 1, CV_RGB(255, 0, 0), 3, 8, 0);
		}

		// get next contours
		contour = contour->h_next;
	}

	//
	//free(contour);
	delete [] points;

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
}

// [ref] fit_contour_by_snake() in ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void fit_contour_by_snake(const cv::Mat &gray_img, const std::vector<cv::Point> &contour, const size_t numSnakePoints, std::vector<cv::Point> &snake_contour)
{
	snake_contour.clear();
	if (contour.empty()) return;

/*
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> hierarchy;
	{
		const int threshold = 90;

		cv::Mat binary_img;

		// make a average filtering
		cv::blur(gray_img, binary_img, cv::Size(31, 15));

		// thresholding
		cv::threshold(binary_img, binary_img, threshold, 255, cv::THRESH_BINARY);

		// expand the thressholded image of ones -smoothing the edge.
		// and move start position of snake out since there are no ballon force
		{
			const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
			cv::dilate(binary_img, binary_img, selement, cv::Point(-1, -1), 3);
		}

		cv::findContours(binary_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	}
*/

	float alpha = 3;
	float beta = 5;
	float gamma = 2;
	const int use_gradient = 1;
	const CvSize win = cvSize(21, 21);
	const CvTermCriteria term_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1.0);

	// run through the found coutours
	const size_t &numPts = contour.size();
	const size_t numSnakePts = 0 == numSnakePoints ? numPts : numSnakePoints;
	if (numPts >= numSnakePts)
	{
		CvPoint *points = new CvPoint [numSnakePts];

		cv::Mat blurred_img;
		cv::blur(gray_img, blurred_img, cv::Size(7, 3));

		const int stride = int(numPts / numSnakePts);
		for (size_t i = 0; i < numSnakePts; ++i)
		{
			const cv::Point &pt = contour[i * stride];
			points[i] = cvPoint(pt.x, pt.y);
		}

		// snake
#if defined(__GNUC__)
        IplImage blurred_img_ipl = (IplImage)blurred_img;
		cvSnakeImage(&blurred_img_ipl, points, numSnakePts, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);
#else
		cvSnakeImage(&(IplImage)blurred_img, points, numSnakePts, &alpha, &beta, &gamma, CV_VALUE, win, term_criteria, use_gradient);
#endif

		snake_contour.assign(points, points + numSnakePts);
		delete [] points;
	}
}

}  // namespace swl
