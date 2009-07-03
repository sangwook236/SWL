#include "swl/view/ViewEventController.h"


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
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

void ViewEventController::wheelMouse(const MouseEvent &evt) const
{
	wheelMouse_(evt);
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

void ViewEventController::hitKey(const KeyEvent &evt) const
{
	hitKey_(evt);
}

}  // namespace swl
