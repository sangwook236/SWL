#include "swl/view/ViewEventController.h"

#if defined(WIN32) && defined(_DEBUG)
#include "swl/common/Config.h"
#define new new(__FILE__, __LINE__)
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// 

bool ViewEventController::addMousePressHandler(const mouse_event_handler_type &handler)
{
	pressMouse_.connect(handler);
	return true;
}

bool ViewEventController::removeMousePressHandler(const mouse_event_handler_type &handler)
{
	pressMouse_.disconnect(handler);
	return true;
}

bool ViewEventController::addMouseReleaseHandler(const mouse_event_handler_type &handler)
{
	releaseMouse_.connect(handler);
	return true;
}

bool ViewEventController::removeMouseReleaseHandler(const mouse_event_handler_type &handler)
{
	releaseMouse_.disconnect(handler);
	return true;
}

bool ViewEventController::addMouseMoveHandler(const mouse_move_event_handler_type &handler)
{
	moveMouse_.connect(handler);
	return true;
}

bool ViewEventController::removeMouseMoveHandler(const mouse_move_event_handler_type &handler)
{
	moveMouse_.disconnect(handler);
	return true;
}

bool ViewEventController::addMouseClickHandler(const mouse_event_handler_type &handler)
{
	clickMouse_.connect(handler);
	return true;
}

bool ViewEventController::removeMouseClickHandler(const mouse_event_handler_type &handler)
{
	clickMouse_.disconnect(handler);
	return true;
}

bool ViewEventController::addMouseDoubleClickHandler(const mouse_event_handler_type &handler)
{
	doubleClickMouse_.connect(handler);
	return true;
}

bool ViewEventController::removeMouseDoubleClickHandler(const mouse_event_handler_type &handler)
{
	doubleClickMouse_.disconnect(handler);
	return true;
}

bool ViewEventController::addKeyPressHandler(const key_event_handler_type &handler)
{
	pressKey_.connect(handler);
	return true;
}

bool ViewEventController::removeKeyPressHandler(const key_event_handler_type &handler)
{
	pressKey_.disconnect(handler);
	return true;
}

bool ViewEventController::addKeyReleaseHandler(const key_event_handler_type &handler)
{
	releaseKey_.connect(handler);
	return true;
}

bool ViewEventController::removeKeyReleaseHandler(const key_event_handler_type &handler)
{
	releaseKey_.disconnect(handler);
	return true;
}

void ViewEventController::pressMouse(const int x, const int y, const EMouseButton button /*= MB_LEFT*/, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	pressMouse_(x, y, button, controlKey, msg);
}

void ViewEventController::releaseMouse(const int x, const int y, const EMouseButton button /*= MB_LEFT*/, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	releaseMouse_(x, y, button, controlKey, msg);
}

void ViewEventController::moveMouse(const int x, const int y, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	moveMouse_(x, y, controlKey, msg);
}

void ViewEventController::clickMouse(const int x, const int y, const EMouseButton button /*= MB_LEFT*/, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	clickMouse_(x, y, button, controlKey, msg);
}

void ViewEventController::doubleClickMouse(const int x, const int y, const EMouseButton button /*= MB_LEFT*/, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	doubleClickMouse_(x, y, button, controlKey, msg);
}

void ViewEventController::pressKey(const int key, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	pressKey_(key, controlKey, msg);
}

void ViewEventController::releaseKey(const int key, const EControlKey controlKey /*= CK_NONE*/, const void * const msg /*= 0L*/) const
{
	releaseKey_(key, controlKey, msg);
}

}  // namespace swl
