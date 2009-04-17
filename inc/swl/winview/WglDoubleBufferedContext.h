#if !defined(__SWL_WIN_VIEW__WGL_DOUBLE_BUFFERED_CONTEXT__H_ )
#define __SWL_WIN_VIEW__WGL_DOUBLE_BUFFERED_CONTEXT__H_ 1


#include "swl/winview/WglContextBase.h"


namespace swl {

//-----------------------------------------------------------------------------------
//  Double-buffered Context for OpenGL

class SWL_WIN_VIEW_API WglDoubleBufferedContext: public WglContextBase
{
public:
	typedef WglContextBase base_type;

public:
	WglDoubleBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated = true);
	WglDoubleBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated = true);
	virtual ~WglDoubleBufferedContext();

private:
	WglDoubleBufferedContext(const WglDoubleBufferedContext &);
	WglDoubleBufferedContext & operator=(const WglDoubleBufferedContext &);

public:
	/// redraw the context
	/*virtual*/ bool redraw();

	/// activate the context
	/*virtual*/ bool activate();
	/// de-activate the context
	/*virtual*/ bool deactivate();

private:
	/// a window handle
	HWND hWnd_;
	/// a target context
	HDC hDC_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_DOUBLE_BUFFERED_CONTEXT__H_
