#include "stdafx.h"
#include "ViewEventHandler.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"

#if defined(WIN32) && defined(_DEBUG)
#include "swl/common/Config.h"
#define new new(__FILE__, __LINE__)
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

}  // namespace swl
