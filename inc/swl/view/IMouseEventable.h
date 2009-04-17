#if !defined(__SWL_VIEW__MOUSE_EVENTABLE_INTERFACE__H_)
#define __SWL_VIEW__MOUSE_EVENTABLE_INTERFACE__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct IMouseEventable
{
public:
	//typedef IMouseEventable base_type;

public:
	enum EButtonType { BT_LEFT = 0x0001, BT_MIDDLE = 0x0002, BT_RIGHT = 0x0004 };

protected:
	IMouseEventable()  {}
public:
	virtual ~IMouseEventable()  {}

private:
	IMouseEventable(const IMouseEventable&);
	IMouseEventable& operator=(const IMouseEventable&);

public:
	virtual void pressMouse(const int x, const int y, const EButtonType button = BT_LEFT) = 0;
	virtual void releaseMouse(const int x, const int y, const EButtonType button = BT_LEFT) = 0;
	virtual void moveMouse(const int x, const int y) = 0;

	virtual void clickMouse(const int x, const int y, const EButtonType button = BT_LEFT) = 0;
	virtual void doubleClickMouse(const int x, const int y, const EButtonType button = BT_LEFT) = 0;
};

}  // namespace swl


#endif  // __SWL_VIEW__MOUSE_EVENTABLE_INTERFACE__H_
