#if !defined(__SWL_WIN_VIEW__GDI_PLUS_BITMAP_BUFFERED_CONTEXT__H_ )
#define __SWL_WIN_VIEW__GDI_PLUS_BITMAP_BUFFERED_CONTEXT__H_ 1


#include "swl/winview/GdiplusContextBase.h"


namespace Gdiplus {

class Image;
class Bitmap;

}  // namespace Gdiplus

namespace swl {

//-----------------------------------------------------------------------------------
//  Bitmap-buffered Context for GDI+ in Microsoft Windows

class SWL_WIN_VIEW_API GdiplusBitmapBufferedContext: public GdiplusContextBase
{
public:
	typedef GdiplusContextBase base_type;

public:
	GdiplusBitmapBufferedContext(HWND hWnd, const Region2<int> &drawRegion, const bool isAutomaticallyActivated = true);
	GdiplusBitmapBufferedContext(HWND hWnd, const RECT &drawRect, const bool isAutomaticallyActivated = true);
	virtual ~GdiplusBitmapBufferedContext();

private:
	GdiplusBitmapBufferedContext(const GdiplusBitmapBufferedContext &);
	GdiplusBitmapBufferedContext & operator=(const GdiplusBitmapBufferedContext &);

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
	/*virtual*/ boost::any getNativeContext()  {  return isActivated() ? boost::any(canvas_) : boost::any();  }
	/*virtual*/ const boost::any getNativeContext() const  {  return isActivated() ? boost::any(canvas_) : boost::any();  }

    /// get the off-screen surface
	Gdiplus::Image * getOffScreen();
	const Gdiplus::Image * getOffScreen() const;

private:
	bool createOffScreen();
	void deleteOffScreen();

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
