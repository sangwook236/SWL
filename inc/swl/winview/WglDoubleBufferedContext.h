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
	/// swap buffers
	/*virtual*/ bool swapBuffer();

	/// get the native window handle
	/*virtual*/ boost::any getNativeWindowHandle()  {  return NULL != hWnd_ ? boost::any(&hWnd_) : boost::any();  }
	/*virtual*/ const boost::any getNativeWindowHandle() const  {  return NULL != hWnd_ ? boost::any(&hWnd_) : boost::any();  }

	/// get the native context
	/*virtual*/ boost::any getNativeContext()  {  return isActivated() ? boost::any(&hDC_) : boost::any();  }
	/*virtual*/ const boost::any getNativeContext() const  {  return isActivated() ? boost::any(&hDC_) : boost::any();  }

private:
	/// activate the context
	/*virtual*/ bool activate();
	/// de-activate the context
	/*virtual*/ bool deactivate();

	bool createOffScreen();

private:
	/// a window handle
	HWND hWnd_;
	/// a target context
	HDC hDC_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_DOUBLE_BUFFERED_CONTEXT__H_
