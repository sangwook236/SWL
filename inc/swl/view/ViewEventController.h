#if !defined(__SWL_VIEW__VIEW_CONTROLLER__H_)
#define __SWL_VIEW__VIEW_CONTROLLER__H_ 1


#include "swl/view/ExportView.h"
#include "swl/common/MvcController.h"
#include <boost/signal.hpp>


namespace swl {

//-----------------------------------------------------------------------------------
// 

class SWL_VIEW_API ViewEventController: public MvcController
{
public:
	typedef MvcController base_type;

public:
	enum EControlKey { CK_NONE = 0x0000, CK_CTRL = 0x0001, CK_ALT = 0x0002, CK_SHIFT = 0x0003 };
	enum EMouseButton { MB_NONE = 0x0000, MB_LEFT = 0x0001, MB_MIDDLE = 0x0002, MB_RIGHT = 0x0004 };

public:
	typedef boost::signal<void (const int x, const int y, const EMouseButton button, const EControlKey controlKey, const void * const msg)> mouse_event_publisher_type;
	typedef boost::signal<void (const int x, const int y, const EControlKey controlKey, const void * const msg)> mouse_move_event_publisher_type;
	typedef boost::signal<void (const int key, const EControlKey controlKey, const void * const msg)> key_event_publisher_type;

	//typedef mouse_event_publisher_type::slot_function_type mouse_event_handler_type;
	//typedef mouse_move_event_publisher_type::slot_function_type mouse_move_event_handler_type;
	//typedef key_event_publisher_type::slot_function_type key_event_handler_type;

	typedef void (*mouse_event_handler_type)(const int x, const int y, const EMouseButton button, const EControlKey controlKey, const void * const msg);
	typedef void (*mouse_move_event_handler_type)(const int x, const int y, const EControlKey controlKey, const void * const msg);
	typedef void (*key_event_handler_type)(const int key, const EControlKey controlKey, const void * const msg);

protected:
	ViewEventController()  {}
public:
	virtual ~ViewEventController()  {}

private:
	ViewEventController(const ViewEventController&);
	ViewEventController& operator=(const ViewEventController&);

public:
	bool addMousePressHandler(const mouse_event_handler_type &handler);
	bool removeMousePressHandler(const mouse_event_handler_type &handler);

	bool addMouseReleaseHandler(const mouse_event_handler_type &handler);
	bool removeMouseReleaseHandler(const mouse_event_handler_type &handler);

	bool addMouseMoveHandler(const mouse_move_event_handler_type &handler);
	bool removeMouseMoveHandler(const mouse_move_event_handler_type &handler);

	bool addMouseClickHandler(const mouse_event_handler_type &handler);
	bool removeMouseClickHandler(const mouse_event_handler_type &handler);

	bool addMouseDoubleClickHandler(const mouse_event_handler_type &handler);
	bool removeMouseDoubleClickHandler(const mouse_event_handler_type &handler);

	bool addKeyPressHandler(const key_event_handler_type &handler);
	bool removeKeyPressHandler(const key_event_handler_type &handler);

	bool addKeyReleaseHandler(const key_event_handler_type &handler);
	bool removeKeyReleaseHandler(const key_event_handler_type &handler);

	//
	virtual void pressMouse(const int x, const int y, const EMouseButton button = MB_LEFT, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;
	virtual void releaseMouse(const int x, const int y, const EMouseButton button = MB_LEFT, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;
	virtual void moveMouse(const int x, const int y, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;

	virtual void clickMouse(const int x, const int y, const EMouseButton button = MB_LEFT, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;
	virtual void doubleClickMouse(const int x, const int y, const EMouseButton button = MB_LEFT, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;

	virtual void pressKey(const int key, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;
	virtual void releaseKey(const int key, const EControlKey controlKey = CK_NONE, const void * const msg = 0L) const;

protected:
	mouse_event_publisher_type pressMouse_;
	mouse_event_publisher_type releaseMouse_;
	mouse_move_event_publisher_type moveMouse_;

	mouse_event_publisher_type clickMouse_;
	mouse_event_publisher_type doubleClickMouse_;

	key_event_publisher_type pressKey_;
	key_event_publisher_type releaseKey_;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_CONTROLLER__H_
