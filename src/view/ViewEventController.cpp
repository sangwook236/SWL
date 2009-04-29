#include "swl/view/ViewEventController.h"

#if defined(WIN32) && defined(_DEBUG)
#include "swl/common/Config.h"
#define new new(__FILE__, __LINE__)
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// 

void ViewEventController::pressMouse(const MouseEvent &evt) const
{
	pressMouse_(evt);
}

void ViewEventController::releaseMouse(const MouseEvent &evt) const
{
	releaseMouse_(evt);
}

void ViewEventController::moveMouse(const MouseEvent &evt) const
{
	moveMouse_(evt);
}

void ViewEventController::clickMouse(const MouseEvent &evt) const
{
	clickMouse_(evt);
}

void ViewEventController::doubleClickMouse(const MouseEvent &evt) const
{
	doubleClickMouse_(evt);
}

void ViewEventController::pressKey(const KeyEvent &evt) const
{
	pressKey_(evt);
}

void ViewEventController::releaseKey(const KeyEvent &evt) const
{
	releaseKey_(evt);
}

}  // namespace swl
