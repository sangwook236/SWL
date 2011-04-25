#if !defined(__SWL_PATTERN_RECOGNITION__GESTURE_CLASSIFIER_INTERFACE__H_)
#define __SWL_PATTERN_RECOGNITION__GESTURE_CLASSIFIER_INTERFACE__H_ 1


#include "swl/pattern_recognition/GestureType.h"

namespace cv {

class Mat;
typedef Mat MatND;
template<typename _Tp> class Rect_;
typedef Rect_<int> Rect;

}


namespace swl {

//-----------------------------------------------------------------------------
//

struct IGestureClassifier
{
	virtual ~IGestureClassifier()  {}

	virtual bool analyzeOrientation(const int gestureClassToApply, const cv::Mat &orientation) = 0;

	virtual bool classifyGesture() = 0;
	virtual GestureType::Type getGestureType() const = 0;

	virtual void clearGestureHistory(const int gestureClassToApply) = 0;
};


}  // namespace swl


#endif  // __SWL_PATTERN_RECOGNITION__GESTURE_CLASSIFIER_INTERFACE__H_
