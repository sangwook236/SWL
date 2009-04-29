#if !defined(__SWL_VIEW__KEY_EVENT__H_)
#define __SWL_VIEW__KEY_EVENT__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------
// 

struct KeyEvent
{
public:
	//typedef KeyEvent base_type;

public:
	enum EControlKey {
		CK_NONE = 0x0000,
		CK_LEFT_CTRL = 0x0001, CK_RIGHT_CTRL = 0x0002, CK_CTRL = 0x0003,
		CK_LEFT_ALT = 0x0004, CK_RIGHT_ALT = 0x0008, CK_ALT = 0x000C,
		CK_LEFT_SHIFT = 0x0010, CK_RIGHT_SHIFT = 0x0020, CK_SHIFT = 0x0030
	};

public:
	KeyEvent(const int _key, const size_t _repetitionCount = 1, const EControlKey _controlKey = CK_NONE)
	: key(_key), repetitionCount(_repetitionCount), controlKey(_controlKey)
	{}
	explicit KeyEvent(const KeyEvent &rhs)
	: key(rhs.key), repetitionCount(rhs.repetitionCount), controlKey(rhs.controlKey)
	{}
	virtual ~KeyEvent()  {}

private:
	KeyEvent & operator=(const KeyEvent &rhs);

public:
	const int key;
	const size_t repetitionCount;
	const EControlKey controlKey;
};

}  // namespace swl


#endif  // __SWL_VIEW__KEY_EVENT__H_
