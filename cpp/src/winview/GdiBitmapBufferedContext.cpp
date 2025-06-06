#include "swl/Config.h"
#include "swl/winview/GdiBitmapBufferedContext.h"
#include <boost/smart_ptr.hpp>
#include <wingdi.h>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

GdiBitmapBufferedContext::GdiBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion, true, CM_DEFAULT),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL), dibBits_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

GdiBitmapBufferedContext::GdiBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), true, CM_DEFAULT),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL), dibBits_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

GdiBitmapBufferedContext::~GdiBitmapBufferedContext()
{
	deactivate();

	deleteOffScreen();
}

bool GdiBitmapBufferedContext::swapBuffer()
{
	//if (!isActivated() || isDrawing()) return false;
	if (isDrawing()) return false;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;
	setDrawing(true);

	// copy off-screen buffer to window's DC
#if 1
	// method #1: [use DDB & DIB] the image is not scaled to fit the rectangle
	const bool ret = TRUE == BitBlt(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(), 
		memDC_,
		0, 0,  //drawRegion_.left, drawRegion_.bottom,
		SRCCOPY
	);
#elif 0
	// method #2: [use DDB & DIB] the image is scaled to fit the rectangle
	// caution:
	// all negative coordinate values are ignored.
	// instead, these values are regarded as absolute(positive) values.
	const bool ret = TRUE == StretchBlt(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(),
		memDC_,
		//(int)std::floor(viewingRegion_.left + 0.5), (int)std::floor(viewingRegion_.bottom + 0.5), (int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5),
		0, 0, (int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5),
		SRCCOPY
	);
#else
	// method #3: [use DIB] the image is scaled to fit the rectangle
	// caution:
	// all negative coordinate values are ignored.
	// instead, these values are regarded as absolute(positive) values.
	const bool ret = TRUE == StretchDIBits(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(),
		//(int)std::floor(viewingRegion_.left + 0.5), (int)std::floor(viewingRegion_.bottom + 0.5), (int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5),
		0, 0, (int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5),
		dibBits_,
		&bmiDIB,
		!isPaletteUsed_ ? DIB_RGB_COLORS : DIB_PAL_COLORS,
		SRCCOPY
	);
#endif
	setDrawing(false);
	return ret;
}

bool GdiBitmapBufferedContext::resize(const int x1, const int y1, const int x2, const int y2)
{
	if (isActivated()) return false;
	drawRegion_ = Region2<int>(x1, y1, x2, y2);

	deleteOffScreen();

	return createOffScreen();
}

bool GdiBitmapBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;

	setActivation(true);
	return true;

	// draw something into memDC_
}

bool GdiBitmapBufferedContext::deactivate()
{
	if (!isActivated()) return true;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;

	setActivation(false);
	return true;
}

bool GdiBitmapBufferedContext::createOffScreen()
{
	if (NULL == hWnd_) return false;

	// get DC for window
	hDC_ = GetDC(hWnd_);
	if (NULL == hDC_) return false;

	// create an off-screen DC for double-buffering
	memDC_ = CreateCompatibleDC(hDC_);
	if (NULL == memDC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
		return false;
	}

	//
	isPaletteUsed_ = (GetDeviceCaps(memDC_, RASTERCAPS) & RC_PALETTE) == RC_PALETTE;
	assert(false == isPaletteUsed_);

	const int colorBitCount = GetDeviceCaps(memDC_, BITSPIXEL);
	assert(colorBitCount > 8);
	const int colorPlaneCount = GetDeviceCaps(memDC_, PLANES);
	assert(1 == colorPlaneCount);

	// use palette: when using 256 color
	if (isPaletteUsed_) createPalette(memDC_, colorBitCount);

	if (createOffScreenBitmap(colorBitCount, colorPlaneCount))
		return true;
	else
	{
		DeleteDC(memDC_);
		memDC_ = NULL;

		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
		return false;
	}
}

bool GdiBitmapBufferedContext::createOffScreenBitmap(const int colorBitCount, const int colorPlaneCount)
{
	// method #1: use DDB
#if 0
	memBmp_ = CreateCompatibleBitmap(hDC_, drawRegion_.getWidth(), drawRegion_.getHeight());
	//memBmp_ = CreateCompatibleBitmap(hDC_, (int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5));
#else
	// method #2: use DIB
	const size_t bufSize = !isPaletteUsed_ ? sizeof(BITMAPINFO) : sizeof(BITMAPINFO) + sizeof(RGBQUAD) * 255;
	const boost::scoped_array<unsigned char> buf(new unsigned char [bufSize]);
	memset(buf.get(), 0, bufSize);
	BITMAPINFO &bmiDIB = *(BITMAPINFO *)buf.get();

	// following routine aligns given value to 4 bytes boundary.
	// the current implementation of DIB rendering in Windows 95/98/NT seems to be free from this alignment
	// but old version compatibility is needed.
#if 1
	const int width = ((drawRegion_.getWidth() + 3) / 4 * 4 > 0) ? drawRegion_.getWidth() : 4;
	const int height = (0 == drawRegion_.getHeight()) ? 1 : drawRegion_.getHeight();
#else
	const int viewingWidth = (int)std::floor(viewingRegion_.getWidth() + 0.5);
	const int viewingHeight = (int)std::floor(viewingRegion_.getHeight() + 0.5);
	const int width = ((viewingWidth + 3) / 4 * 4 > 0) ? viewingWidth : 4;
	const int height = (0 == viewingHeight) ? 1 : viewingHeight;
#endif

	bmiDIB.bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	bmiDIB.bmiHeader.biWidth		= width;
	bmiDIB.bmiHeader.biHeight		= height;
	bmiDIB.bmiHeader.biPlanes		= colorPlaneCount;
	bmiDIB.bmiHeader.biBitCount		= colorBitCount;
	if (!isPaletteUsed_)
	{
		bmiDIB.bmiHeader.biCompression	= BI_RGB;
		bmiDIB.bmiHeader.biSizeImage	= 0;  // for BI_RGB

		// offscreen surface generated by the DIB section
		memBmp_ = CreateDIBSection(memDC_, &bmiDIB, DIB_RGB_COLORS, &dibBits_, NULL, 0);
	}
	else
	{
		// FIXME [check] >>
		bmiDIB.bmiHeader.biCompression	= colorBitCount > 4 ? BI_RLE8 : BI_RLE4;
		bmiDIB.bmiHeader.biSizeImage	= width * height * 3;

		// use palette: when using 256 color
		PALETTEENTRY paletteEntry[256];
		GetPaletteEntries(shPalette_, 0, 256, paletteEntry);
		for (int i = 0; i < 256; ++i)
		{
			bmiDIB.bmiColors[i].rgbRed = paletteEntry[i].peRed;
			bmiDIB.bmiColors[i].rgbGreen = paletteEntry[i].peGreen;
			bmiDIB.bmiColors[i].rgbBlue = paletteEntry[i].peBlue;
			bmiDIB.bmiColors[i].rgbReserved = 0;
		}

		// offscreen surface generated by the DIB section
		memBmp_ = CreateDIBSection(memDC_, &bmiDIB, DIB_PAL_COLORS, &dibBits_, NULL, 0);
	}
	if (NULL == memBmp_) return false;
	oldBmp_ = (HBITMAP)SelectObject(memDC_, memBmp_);
#endif

	return true;
}

void GdiBitmapBufferedContext::deleteOffScreen()
{
	// free-up the off-screen DC
	if (oldBmp_)
	{
		SelectObject(memDC_, oldBmp_);
		oldBmp_ = NULL;
		dibBits_ = NULL;
	}
	if (memBmp_)
	{
		DeleteObject(memBmp_);
		memBmp_ = NULL;
	}

	if (memDC_)
	{
		// use palette: when using 256 color
		if (isPaletteUsed_)	deletePalette(memDC_);

		DeleteDC(memDC_);
		memDC_ = NULL;
	}

	// release DC
	if (hDC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
	}
}

}  // namespace swl
