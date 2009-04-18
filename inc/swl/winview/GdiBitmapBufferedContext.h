#if !defined(__SWL_WIN_VIEW__GDI_BITMAP_BUFFERED_CONTEXT__H_ )
#define __SWL_WIN_VIEW__GDI_BITMAP_BUFFERED_CONTEXT__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>


namespace swl {

//-----------------------------------------------------------------------------------
//  Bitmap-buffered Context for GDI in Microsoft Windows

class SWL_WIN_VIEW_API GdiBitmapBufferedContext: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef HDC context_type;

public:
	GdiBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated = true);
	GdiBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated = true);
	virtual ~GdiBitmapBufferedContext();

private:
	GdiBitmapBufferedContext(const GdiBitmapBufferedContext &);
	GdiBitmapBufferedContext & operator=(const GdiBitmapBufferedContext &);

public:
	/// swap buffers
	/*virtual*/ bool swapBuffer();
	/// resize the context
	/*virtual*/ bool resize(const int x1, const int y1, const int x2, const int y2);

	/// activate the context
	/*virtual*/ bool activate();
	/// de-activate the context
	/*virtual*/ bool deactivate();

	/// get the native context
	/*virtual*/ void * getNativeContext()  {  return isActivated() ? (void *)&memDC_ : NULL;  }
	/*virtual*/ const void * const getNativeContext() const  {  return isActivated() ? (void *)&memDC_ : NULL;  }

private:
	bool createOffScreen();
	bool createOffScreenBitmap();

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


#endif  // __SWL_WIN_VIEW__GDI_BITMAP_BUFFERED_CONTEXT__H_
