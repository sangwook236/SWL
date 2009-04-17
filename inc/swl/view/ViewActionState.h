#if !defined(__SWL_VIEW__VIEW_ACTION_STATE__H_)
#define __SWL_VIEW__VIEW_ACTION_STATE__H_ 1


#include "swl/view/IMouseEventable.h"
#include "swl/common/MvcController.h"


namespace swl {

//-----------------------------------------------------------------------------------
// 

class ViewActionState: public MvcController, public IMouseEventable
{
public:
	typedef MvcController		base_type;
	typedef IMouseEventable		interface_type;

protected:
	ViewActionState()  {}
public:
	virtual ~ViewActionState()  {}

private:
	ViewActionState(const ViewActionState&);
	ViewActionState& operator=(const ViewActionState&);

public:
	/*virtual*/ void clickMouse(const int /*x*/, const int /*y*/, const EButtonType /*button = BT_LEFT*/)  {}
	/*virtual*/ void doubleClickMouse(const int /*x*/, const int /*y*/, const EButtonType /*button = BT_LEFT*/)  {}

private:
	virtual void invokeAction() = 0;
};

}  // namespace swl


#endif  // __SWL_VIEW__VIEW_ACTION_STATE__H_
