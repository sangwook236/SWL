#if !defined(__SWL_VIEW__MOUSE_EVENT__H_)
#define __SWL_VIEW__MOUSE_EVENT__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct MouseEvent
{
public:
	//typedef MouseEvent base_type;

public:
	enum EControlKey {
		CK_NONE = 0x0000,
		CK_LEFT_CTRL = 0x0001, CK_RIGHT_CTRL = 0x0002, CK_CTRL = 0x0003,
		CK_LEFT_ALT = 0x0004, CK_RIGHT_ALT = 0x0008, CK_ALT = 0x000C,
		CK_LEFT_SHIFT = 0x0010, CK_RIGHT_SHIFT = 0x0020, CK_SHIFT = 0x0030
	};
	enum EButton {
		BT_NONE = 0x0000,
		BT_LEFT = 0x0001, BT_MIDDLE = 0x0002, BT_RIGHT = 0x0004,
		BT_FORWARD = 0x0010, BT_BACKWARD = 0x0020,
	};
	enum EScroll {
		SC_NONE = 0x0000,
		SC_HORZ_LEFT = 0x0001, SC_HORZ_RIGHT = 0x0002,
		SC_VERT_UP = 0x0010, SC_VERT_DOWN = 0x0020,
	};

public:
	MouseEvent(const int _x, const int _y, const EButton _button = BT_NONE, const EControlKey _controlKey = CK_NONE, const EScroll _scroll = SC_NONE)
	: x(_x), y(_y), button(_button), controlKey(_controlKey), scroll(_scroll)
	{}
	explicit MouseEvent(const MouseEvent &rhs)
	: x(rhs.x), y(rhs.y), button(rhs.button), controlKey(rhs.controlKey), scroll(rhs.scroll)
	{}
	virtual ~MouseEvent()  {}

private:
	MouseEvent & operator=(const MouseEvent &rhs);

public:
	const int x;
	const int y;
	const EButton button;
	const EControlKey controlKey;
	const EScroll scroll;
};

}  // namespace swl


#endif  // __SWL_VIEW__MOUSE_EVENT__H_
