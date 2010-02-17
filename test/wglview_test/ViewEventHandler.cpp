#include "stdafx.h"
#include "swl/Config.h"
#include "ViewEventHandler.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// 

void MousePressHandler::operator()(const MouseEvent &evt) const
{
	if (swl::MouseEvent::BT_LEFT == evt.button)
	{
	}
	else if (swl::MouseEvent::BT_RIGHT == evt.button)
	{
	}
}

//-----------------------------------------------------------------------------------
// 

void MouseReleaseHandler::operator()(const MouseEvent &evt) const
{
	if (swl::MouseEvent::BT_LEFT == evt.button)
	{
	}
	else if (swl::MouseEvent::BT_RIGHT == evt.button)
	{
	}
}

//-----------------------------------------------------------------------------------
// 

void MouseMoveHandler::operator()(const MouseEvent &evt) const
{
	if (swl::MouseEvent::BT_LEFT == evt.button)
	{
	}
	else if (swl::MouseEvent::BT_RIGHT == evt.button)
	{
	}
}

//-----------------------------------------------------------------------------------
// 

void MouseWheelHandler::operator()(const MouseEvent &evt) const
{
	if (swl::MouseEvent::BT_LEFT == evt.button)
	{
	}
	else if (swl::MouseEvent::BT_RIGHT == evt.button)
	{
	}
}

//-----------------------------------------------------------------------------------
// 

void MouseClickHandler::operator()(const MouseEvent &evt) const
{
	if (swl::MouseEvent::BT_LEFT == evt.button)
	{
	}
	else if (swl::MouseEvent::BT_RIGHT == evt.button)
	{
	}
}

//-----------------------------------------------------------------------------------
// 

void MouseDoubleClickHandler::operator()(const MouseEvent &evt) const
{
	if (swl::MouseEvent::BT_LEFT == evt.button)
	{
	}
	else if (swl::MouseEvent::BT_RIGHT == evt.button)
	{
	}
}

//-----------------------------------------------------------------------------------
// 

void KeyPressHandler::operator()(const KeyEvent &evt) const
{
}

//-----------------------------------------------------------------------------------
// 

void KeyReleaseHandler::operator()(const KeyEvent &evt) const
{
}

//-----------------------------------------------------------------------------------
// 

void KeyHitHandler::operator()(const KeyEvent &evt) const
{
}

}  // namespace swl
