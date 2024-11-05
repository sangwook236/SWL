#if !defined(__SWL_PATTERN_RECOGNITION__MOTION_SEGMENTER__H_)
#define __SWL_PATTERN_RECOGNITION__MOTION_SEGMENTER__H_ 1


#include "swl/pattern_recognition/ExportPatternRecognition.h"
#include <vector>

namespace cv {

class Mat;
template<typename _Tp> class Rect_;
typedef Rect_<int> Rect;

}


namespace swl {

//-----------------------------------------------------------------------------
//

struct SWL_PATTERN_RECOGNITION_API MotionSegmenter
{
	static void segmentUsingMHI(const double timestamp, const double mhiTimeDuration, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &processed_mhi, cv::Mat &component_label_map, std::vector<cv::Rect> &component_rects);
};


}  // namespace swl


#endif  // __SWL_PATTERN_RECOGNITION__MOTION_SEGMENTER__H_
