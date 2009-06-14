#include "swl/winview/GdiplusBitmapBufferedContext.h"
#include <gdiplus.h>
#include <cmath>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
//#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

GdiplusBitmapBufferedContext::GdiplusBitmapBufferedContext(HWND hWnd, const Region2<int> &drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion, true, CM_DEFAULT),
  hWnd_(hWnd), graphics_(NULL), canvas_(NULL), memBmp_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

GdiplusBitmapBufferedContext::GdiplusBitmapBufferedContext(HWND hWnd, const RECT &drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), true, CM_DEFAULT),
  hWnd_(hWnd), graphics_(NULL), canvas_(NULL), memBmp_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

GdiplusBitmapBufferedContext::~GdiplusBitmapBufferedContext()
{
	deactivate();

	deleteOffScreen();
}

bool GdiplusBitmapBufferedContext::swapBuffer()
{
	//if (!isActivated() || isDrawing()) return false;
	if (isDrawing()) return false;
	if (NULL == memBmp_ || NULL == graphics_) return false;
	setDrawing(true);

	// when all drawing has been completed, a new graphics canvas should be created,
	// but this time it should be associated with the actual output screen or window
	// method #1: the image is scaled to fit the rectangle
	const bool ret = Gdiplus::Ok == graphics_->DrawImage(memBmp_, drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight());
	//const bool ret = Gdiplus::Ok == graphics_->DrawImage(memBmp_, Gdiplus::Rect(drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight()));
/*
	// method #2: the image is scaled to fit the rectangle
	const Gdiplus::RectF dstRgn((Gdiplus::REAL)drawRegion_.left, (Gdiplus::REAL)drawRegion_.bottom, (Gdiplus::REAL)drawRegion_.getWidth(), (Gdiplus::REAL)drawRegion_.getHeight());
	// caution:
	// all negative coordinate values are ignored.
	// instead, these values are regarded as absolute(positive) values.
	const bool ret = Gdiplus::Ok == graphics_->DrawImage(
		memBmp_,
		dstRgn,
		//(Gdiplus::REAL)std::floor(viewingRegion_.left + 0.5), (Gdiplus::REAL)std::floor(viewingRegion_.bottom + 0.5),
		(Gdiplus::REAL)0, (Gdiplus::REAL)0,
		(Gdiplus::REAL)std::floor(viewingRegion_.getWidth() + 0.5), (Gdiplus::REAL)std::floor(viewingRegion_.getHeight() + 0.5),
		Gdiplus::UnitPixel
	);
*/
	// method #3: the image is not scaled to fit the rectangle
	//Gdiplus::CachedBitmap cachedBmp(memBmp_, graphics_);
	//const bool ret = Gdiplus::Ok == graphics_->DrawCachedBitmap(&cachedBmp, drawRegion_.left, drawRegion_.bottom);

	setDrawing(false);
	return ret;
}

bool GdiplusBitmapBufferedContext::resize(const int x1, const int y1, const int x2, const int y2)
{
	if (isActivated()) return false;
	drawRegion_ = Region2<int>(x1, y1, x2, y2);

	deleteOffScreen();

	return createOffScreen();
}

bool GdiplusBitmapBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == memBmp_ || NULL == graphics_) return false;

	setActivation(true);
	return true;

	// draw something into canvas_
}

bool GdiplusBitmapBufferedContext::deactivate()
{
	if (!isActivated()) return true;
	if (NULL == memBmp_ || NULL == graphics_) return false;

	setActivation(false);
	return true;
}

bool GdiplusBitmapBufferedContext::createOffScreen()
{
	if (NULL == hWnd_) return false;

	// create graphics for window
	graphics_ = new Gdiplus::Graphics(hWnd_, FALSE);
	if (NULL == graphics_) return false;

	// create an off-screen graphics for double-buffering
	memBmp_ = new Gdiplus::Bitmap(drawRegion_.getWidth(), drawRegion_.getHeight(), graphics_);
	//memBmp_ = new Gdiplus::Bitmap((int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5), graphics_);
	if (NULL == memBmp_)
	{
		delete graphics_;
		graphics_ = NULL;
		return false;
	}

	canvas_ = Gdiplus::Graphics::FromImage(memBmp_);
	if (NULL == canvas_)
	{
		delete graphics_;
		graphics_ = NULL;
		delete memBmp_;
		memBmp_ = NULL;
		return false;
	}

	return true;
}

void GdiplusBitmapBufferedContext::deleteOffScreen()
{
	// free-up the off-screen graphics
	if (canvas_)
	{
		delete canvas_;
		canvas_ = NULL;
	}
	if (memBmp_)
	{
		delete memBmp_;
		memBmp_ = NULL;
	}

	// delete graphics
	if (graphics_)
	{
		delete graphics_;
		graphics_ = NULL;
	}

}

Gdiplus::Image * GdiplusBitmapBufferedContext::getOffScreen()
{  return memBmp_;  }

const Gdiplus::Image * GdiplusBitmapBufferedContext::getOffScreen() const
{  return memBmp_;  }

}  // namespace swl
