#include "swl/winview/GdiplusBitmapBufferedContext.h"
#include <gdiplus.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
//#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

GdiplusBitmapBufferedContext::GdiplusBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion),
  hWnd_(hWnd), graphics_(NULL), canvas_(NULL), memBmp_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

GdiplusBitmapBufferedContext::GdiplusBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom)),
  hWnd_(hWnd), graphics_(NULL), canvas_(NULL), memBmp_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

GdiplusBitmapBufferedContext::~GdiplusBitmapBufferedContext()
{
	deactivate();
}

bool GdiplusBitmapBufferedContext::redraw()
{
	if (!isActivated() || isDrawing())
		return false;
	if (NULL == graphics_ || NULL == memBmp_) return false;
	setDrawing(true);

	// when all drawing has been completed, a new graphics canvas should be created,
	// but this time it should be associated with the actual output screen or window
	// (1)
	//return Gdiplus::Ok == graphics_->DrawImage(memBmp_, drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight());
	// (2)
	Gdiplus::CachedBitmap cachedBmp(memBmp_, graphics_);
	const bool ret = Gdiplus::Ok == graphics_->DrawCachedBitmap(&cachedBmp, drawRegion_.left, drawRegion_.bottom);

	setDrawing(false);
	return ret;
}

bool GdiplusBitmapBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hWnd_) return false;

	// create graphics for window
	graphics_ = new Gdiplus::Graphics(hWnd_, FALSE);
	if (NULL == graphics_) return false;

	// create an off-screen graphics for double-buffering
	memBmp_ = new Gdiplus::Bitmap(drawRegion_.getWidth(), drawRegion_.getHeight(), graphics_);
	if (NULL == memBmp_)
	{
		delete graphics_;
		graphics_ = NULL;
		return false;
	}

	canvas_ = Gdiplus::Graphics::FromImage(memBmp_);
	if (NULL == canvas_)
	{
		delete memBmp_;
		memBmp_ = NULL;
		delete graphics_;
		graphics_ = NULL;
		return false;
	}

	setActivation(true);

	return true;

	// draw something into canvas_
}

bool GdiplusBitmapBufferedContext::deactivate()
{
	if (!isActivated()) return true;

	setActivation(false);

	// free-up the off-screen graphics
	delete canvas_;
	canvas_ = NULL;
	delete memBmp_;
	memBmp_ = NULL;

	// delete graphics
	delete graphics_;
	graphics_ = NULL;

	return true;
}

}  // namespace swl
