#include "swl/pattern_recognition/GestureType.h"


namespace swl {

//-----------------------------------------------------------------------------
//

/*static*/ std::string GestureType::getGestureName(const Type &type)
{
	switch (type)
	{
	case GT_LEFT_MOVE:
		return "Left Move";
	case GT_RIGHT_MOVE:
		return "Right Move";
	case GT_UP_MOVE:
		return "Up Move";
	case GT_DOWN_MOVE:
		return "Down Move";
	case GT_LEFT_FAST_MOVE:
		return "Left Fast Move";
	case GT_RIGHT_FAST_MOVE:
		return "Right Fast Move";
	case GT_HORIZONTAL_FLIP:
		return "Horizontal Flip";
	case GT_VERTICAL_FLIP:
		return "Vertical Flip";
	case GT_JAMJAM:
		return "JamJam";
	case GT_LEFT_90_TURN:
		return "Left 90 Turn";
	case GT_RIGHT_90_TURN:
		return "Right 90 Turn";
	case GT_CW:
		return "CW";
	case GT_CCW:
		return "CCW";
	case GT_INFINITY:
		return "Infinity";
	case GT_TRIANGLE:
		return "Triangle";
	case GT_HAND_OPEN:
		return "Hand Open";
	case GT_HAND_CLOSE:
		return "Hand Close";
	case GT_TWO_HAND_DOWN:
		return "Two Hand Down";
	case GT_TWO_HAND_UP:
		return "Two Hand Up";
	default:
		return "-----";
	}
}

}  // namespace swl
