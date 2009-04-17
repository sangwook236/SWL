#if !defined(__SWL_WIN_VIEW__GDI_PLUS_BITMAP_BUFFERED_CONTEXT__H_ )
#define __SWL_WIN_VIEW__GDI_PLUS_BITMAP_BUFFERED_CONTEXT__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>


namespace Gdiplus {

class Graphics;
class Bitmap;

}  // namespace Gdiplus

namespace swl {

//-----------------------------------------------------------------------------------
//  Bitmap-buffered Context for GDI+ in Microsoft Windows

class SWL_WIN_VIEW_API GdiplusBitmapBufferedContext: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef Gdiplus::Graphics context_type;

public:
	GdiplusBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated = true);
	GdiplusBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated = true);
	virtual ~GdiplusBitmapBufferedContext();

private:
	GdiplusBitmapBufferedContext(const GdiplusBitmapBufferedContext &);
	GdiplusBitmapBufferedContext & operator=(const GdiplusBitmapBufferedContext &);

public:
	/// redraw the context
	bool redraw();

	/// activate the context
	/*virtual*/ bool activate();
	/// de-activate the context
	/*virtual*/ bool deactivate();

	/// get the native context
	/*virtual*/ void * getNativeContext()  {  return isActivated() ? (void *)canvas_ : NULL;  }
	/*virtual*/ const void * const getNativeContext() const  {  return isActivated() ? (void *)canvas_ : NULL;  }

private:
	/// a window handle
	HWND hWnd_;
	/// a target context
	Gdiplus::Graphics *graphics_;

	/// a buffered context
	Gdiplus::Graphics *canvas_;
	/// a buffered bitmaps
	Gdiplus::Bitmap *memBmp_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__GDI_PLUS_BITMAP_BUFFERED_CONTEXT__H_
