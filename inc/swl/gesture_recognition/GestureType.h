#if !defined(__SWL_GESTURE_RECOGNITION__GESTURE_TYPE__H_)
#define __SWL_GESTURE_RECOGNITION__GESTURE_TYPE__H_ 1


#include <string>


namespace swl {

//-----------------------------------------------------------------------------
//

struct GestureType
{
	enum Type
	{
		GT_UNDEFINED = 0,
		GT_LEFT_MOVE, GT_RIGHT_MOVE, GT_UP_MOVE, GT_DOWN_MOVE,
		GT_LEFT_FAST_MOVE, GT_RIGHT_FAST_MOVE, 
		GT_HORIZONTAL_FLIP, GT_VERTICAL_FLIP,
		GT_JAMJAM, GT_SHAKE,
		GT_LEFT_90_TURN, GT_RIGHT_90_TURN,
		GT_CW, GT_CCW,
		GT_INFINITY, GT_TRIANGLE,
		GT_HAND_OPEN, GT_HAND_CLOSE,
		GT_TWO_HAND_DOWN, GT_TWO_HAND_UP
	};

	static std::string getGestureName(const Type &type);
};

}  // namespace swl


#endif  // __SWL_GESTURE_RECOGNITION__GESTURE_TYPE__H_
