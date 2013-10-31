//#include "stdafx.h"
#include "DepthGuidedMap.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/timer/timer.hpp>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace swl {

void run_grabcut_using_depth_guided_mask(const cv::Mat &rgb_image, const cv::Mat &depth_guided_mask)
{
	// PPP [] >>
	//	5. 추출된 superpixel index들로부터 foreground & background 영역을 추출.
	//		선택된 depth range로부터 얻은 영역을 1로, 그외 영역을 0으로 지정한 후, 추출된 superpixel index와 bit-and operation.
	//		1을 가지는 영역의 boundary를 GrabCut의 foreground seed로 사용.
	//		선택된 depth range로부터 얻은 영역을 0로, 그외 영역을 1으로 지정한 후, 추출된 superpixel index와 bit-and operation.
	//		1을 가지는 영역의 boundary를 GrabCut의 background seed로 사용.

	cv::Mat grabCut_mask(rgb_image.size(), CV_8UC1);
	cv::Mat grabCut_bgModel, grabCut_fgModel;

#if 1
	// GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
	//grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_BGD));
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_PR_FGD));
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_FGD), SWL_FGD == depth_guided_mask);  // foreground.
	grabCut_mask.setTo(cv::Scalar::all(cv::GC_BGD), SWL_BGD == depth_guided_mask);  // background.

	cv::grabCut(rgb_image, grabCut_mask, cv::Rect(), grabCut_bgModel, grabCut_fgModel, 1, cv::GC_INIT_WITH_MASK);
#else
	// FIXME [enhance] >> too slow. speed up.
	const cv::Rect grabCut_rect(swl::get_bounding_rect(depth_guided_mask > 0));

	cv::grabCut(rgb_image, grabCut_mask, grabCut_rect, grabCut_bgModel, grabCut_fgModel, 1, cv::GC_INIT_WITH_RECT);
#endif

	cv::Mat tmp_image;
#if 0
	// show foreground & background masks.
	//cv::imshow("foreground mask", SWL_FGD == depth_guided_mask);  // foreground.
	//cv::imshow("background mask", SWL_BGD == depth_guided_mask);  // background.

	// show GrabCut mask.
	grabCut_mask.convertTo(tmp_image, CV_8UC1, 255.0 / cv::GC_PR_FGD, 0.0);
	//cv::rectangle(tmp_image, grabCut_rect, cv::Scalar::all(255), 2);
	cv::imshow("GrabCut mask", tmp_image);
#endif

	cv::grabCut(rgb_image, grabCut_mask, cv::Rect(), grabCut_bgModel, grabCut_fgModel, 1, cv::GC_EVAL);

#if 1
	rgb_image.copyTo(tmp_image, cv::Mat(grabCut_mask & 1));
	cv::imshow("GrabCut result", tmp_image);
#endif

#if 0
	{
		static int idx = 0;
		std::ostringstream strm;
		strm << "../data/kinect_segmentation/grabcut_result_" << idx++ << ".png";
		cv::imwrite(strm.str(), tmp_image);
	}
#endif
}
	
}  // namespace swl
