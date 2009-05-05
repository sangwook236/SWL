#if !defined(__SWL_VIEW__VIEW_CONTROLLER__H_)
#define __SWL_VIEW__VIEW_CONTROLLER__H_ 1


#include "swl/view/ExportView.h"
#include "swl/common/MvcController.h"
#include <boost/signal.hpp>


namespace swl {

struct MouseEvent;
struct KeyEvent;


//-----------------------------------------------------------------------------------
// 

class SWL_VIEW_API ViewEventController: public MvcController
{
public:
	typedef MvcController base_type;

public:
	ViewEventController()  {}
	virtual ~ViewEventController()  {}

private:
	ViewEventController(const ViewEventController&);
	ViewEventController& operator=(const ViewEventController&);

public:
	template<typename Functor>
	bool addMousePressHandler(const Functor &handler)
	{
		pressMouse_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeMousePressHandler(const Functor &handler)
	{
		pressMouse_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addMouseReleaseHandler(const Functor &handler)
	{
		releaseMouse_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeMouseReleaseHandler(const Functor &handler)
	{
		releaseMouse_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addMouseMoveHandler(const Functor &handler)
	{
		moveMouse_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeMouseMoveHandler(const Functor &handler)
	{
		moveMouse_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addMouseWheelHandler(const Functor &handler)
	{
		wheelMouse_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeMouseWheelHandler(const Functor &handler)
	{
		wheelMouse_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addMouseClickHandler(const Functor &handler)
	{
		clickMouse_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeMouseClickHandler(const Functor &handler)
	{
		clickMouse_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addMouseDoubleClickHandler(const Functor &handler)
	{
		doubleClickMouse_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeMouseDoubleClickHandler(const Functor &handler)
	{
		doubleClickMouse_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addKeyPressHandler(const Functor &handler)
	{
		pressKey_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeKeyPressHandler(const Functor &handler)
	{
		pressKey_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addKeyReleaseHandler(const Functor &handler)
	{
		releaseKey_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeKeyReleaseHandler(const Functor &handler)
	{
		releaseKey_.disconnect(handler);
		return true;
	}

	template<typename Functor>
	bool addKeyHitHandler(const Functor &handler)
	{
		hitKey_.connect(handler);
		return true;
	}
	template<typename Functor>
	bool removeKeyHitHandler(const Functor &handler)
	{
		hitKey_.disconnect(handler);
		return true;
	}

	//
	virtual void pressMouse(const MouseEvent &evt) const;
	virtual void releaseMouse(const MouseEvent &evt) const;
	virtual void moveMouse(const MouseEvent &evt) const;
	virtual void wheelMouse(const MouseEvent &evt) const;

	virtual void clickMouse(const MouseEvent &evt) const;
	virtual void doubleClickMouse(const MouseEvent &evt) const;

	virtual void pressKey(const KeyEvent &evt) const;
	virtual void releaseKey(const KeyEvent &evt) const;
	virtual void hitKey(const KeyEvent &evt) const;

protected:
	boost::signal<void (const MouseEvent &)> pressMouse_;
	boost::signal<void (const MouseEvent &)> releaseMouse_;
	boost::signal<void (const MouseEvent &)> moveMouse_;
	boost::signal<void (const MouseEvent &)> wheelMouse_;

	boost::signal<void (const MouseEvent &)> clickMouse_;
	boost::signal<void (const MouseEvent &)> doubleClickMouse_;

	boost::signal<void (const KeyEvent &)> pressKey_;
	boost::signal<void (const KeyEvent &)> releaseKey_;

	boost::signal<void (const KeyEvent &)> hitKey_;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_CONTROLLER__H_
