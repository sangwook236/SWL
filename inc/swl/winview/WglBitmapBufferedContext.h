#if !defined(__SWL_WIN_VIEW__WGL_BITMAP_BUFFERED_CONTEXT__H_ )
#define __SWL_WIN_VIEW__WGL_BITMAP_BUFFERED_CONTEXT__H_ 1


#include "swl/winview/WglContextBase.h"


namespace swl {

//-----------------------------------------------------------------------------------
//  Bitmap-buffered Context for OpenGL

class SWL_WIN_VIEW_API WglBitmapBufferedContext: public WglContextBase
{
public:
	typedef WglContextBase base_type;

public:
	WglBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated = true);
	WglBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated = true);
	virtual ~WglBitmapBufferedContext();

private:
	WglBitmapBufferedContext(const WglBitmapBufferedContext &);
	WglBitmapBufferedContext & operator=(const WglBitmapBufferedContext &);

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

	/// a buffered context
	HDC memDC_;
	/// a buffered bitmaps
	HBITMAP memBmp_, oldBmp_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_BITMAP_BUFFERED_CONTEXT__H_
