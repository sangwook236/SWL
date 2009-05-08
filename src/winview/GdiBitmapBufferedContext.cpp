#include "swl/winview/GdiBitmapBufferedContext.h"
#include <wingdi.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

///*static*/ HPALETTE GdiBitmapBufferedContext::shPalette_ = NULL;
///*static*/ int GdiBitmapBufferedContext::sUsedPaletteCount_ = 0;

GdiBitmapBufferedContext::GdiBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion, true),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL), dibBits_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

GdiBitmapBufferedContext::GdiBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), true),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL), dibBits_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

GdiBitmapBufferedContext::~GdiBitmapBufferedContext()
{
	deactivate();

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
		//// delete palette
		//SelectPalette(memDC_, (HPALETTE)GetStockObject(DEFAULT_PALETTE), FALSE);
		//if (shPalette_ && --sUsedPaletteCount_ <= 0)
		//{
		//	DeleteObject(shPalette_);
		//	shPalette_ = NULL;
		//	sUsedPaletteCount_ = 0;
		//}

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

bool GdiBitmapBufferedContext::swapBuffer()
{
	//if (!isActivated() || isDrawing()) return false;
	if (isDrawing()) return false;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;
	setDrawing(true);

	//if (shPalette_)
	//{
	//	SelectPalette(hDC_, shPalette_, FALSE);
	//	RealizePalette(hDC_);
	//}

	// copy off-screen buffer to window's DC
	// method #1: use DDB
	const bool ret = TRUE == BitBlt(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(), 
		memDC_,
		0, 0,  //drawRegion_.left, drawRegion_.bottom,
		SRCCOPY
	);
/*
	const bool ret = TRUE == StretchBlt(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(),
		memDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(),
		SRCCOPY
	);
*/
	// method #2: use DIB
/*
	const bool ret = TRUE == BitBlt(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(), 
		memDC_,
		0, 0,  //drawRegion_.left, drawRegion_.bottom,
		SRCCOPY
	);
*/
/*
	const bool ret = TRUE == StretchDIBits(
		hDC_,
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(),
		drawRegion_.left, drawRegion_.bottom, drawRegion_.getWidth(), drawRegion_.getHeight(),
		dibBits_,
		&bmiDIB_,
		DIB_RGB_COLORS, //DIB_PAL_COLORS
		SRCCOPY
	);
*/
	setDrawing(false);
	return ret;
}

bool GdiBitmapBufferedContext::resize(const int x1, const int y1, const int x2, const int y2)
{
	if (isActivated()) return false;
	drawRegion_ = Region2<int>(x1, y1, x2, y2);

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
		//// delete palette
		//SelectPalette(memDC_, (HPALETTE)GetStockObject(DEFAULT_PALETTE), FALSE);
		//if (shPalette_ && --sUsedPaletteCount_ <= 0)
		//{
		//	DeleteObject(shPalette_);
		//	shPalette_ = NULL;
		//	sUsedPaletteCount_ = 0;
		//}

		DeleteDC(memDC_);
		memDC_ = NULL;
	}

	// release DC
	if (hDC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
	}

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

	return createOffScreenBitmap();
}

bool GdiBitmapBufferedContext::createOffScreenBitmap()
{
	// method #1: use DDB
	memBmp_ = CreateCompatibleBitmap(hDC_, drawRegion_.getWidth(), drawRegion_.getHeight());

	// method #2: use DIB
/*
    // create DIB section
	BITMAPINFO bmiDIB_;
	memset(&bmiDIB_, 0, sizeof(BITMAPINFO));

	// when using 256 color
	//RGBQUAD rgb[255];

	// following routine aligns given value to 4 bytes boundary.
	// the current implementation of DIB rendering in Windows 95/98/NT seems to be free from this alignment
	// but old version compatibility is needed.
	const int width = ((drawRegion_.getWidth()+3)/4*4 > 0) ? drawRegion_.getWidth() : 4;
	const int height = (0 == drawRegion_.getHeight()) ? 1 : drawRegion_.getHeight();

	bmiDIB_.bmiHeader.biSize		= sizeof(BITMAPINFOHEADER);
	bmiDIB_.bmiHeader.biWidth		= width;
	bmiDIB_.bmiHeader.biHeight		= height;
	bmiDIB_.bmiHeader.biPlanes		= 1;
	//bmiDIB_.bmiHeader.biBitCount	= 32;
	bmiDIB_.bmiHeader.biBitCount	= GetDeviceCaps(memDC_, BITSPIXEL);
	bmiDIB_.bmiHeader.biCompression	= BI_RGB;
	bmiDIB_.bmiHeader.biSizeImage	= 0;  // for BI_RGB
	//bmiDIB_.bmiHeader.biSizeImage	= width * height * 3;

	//// when using 256 color
	//PALETTEENTRY aPaletteEntry[256];
	//GetPaletteEntries(shPalette_, 0, 256, aPaletteEntry);
	//
	//for (int i = 0; i < 256; ++i)
	//{
	//	bmiDIB_.bmiColors[i].rgbRed			= aPaletteEntry[i].peRed;
	//	bmiDIB_.bmiColors[i].rgbGreen		= aPaletteEntry[i].peGreen;
	//	bmiDIB_.bmiColors[i].rgbBlue		= aPaletteEntry[i].peBlue;
	//	bmiDIB_.bmiColors[i].rgbReserved	= 0;
	//}

    // offscreen surface generated by the DIB section
	memBmp_ = CreateDIBSection(memDC_, &bmiDIB_, DIB_RGB_COLORS, &dibBits_, NULL, 0);
*/
	if (NULL == memBmp_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
		return false;
	}
	oldBmp_ = (HBITMAP)SelectObject(memDC_, memBmp_);
/*	
	// when using 256 color
	const int nColorBit = GetDeviceCaps(memDC_, BITSPIXEL);
	if (nColorBit <= 8 && !shPalette_)
	{
		// following routines are originated from ogl2 sdk made by Sillicon Graphics and modified for glext
		const int nPalette = 1 << nColorBit;
		LOGPALETTE *logPalette = (LOGPALETTE *)new char [sizeof(LOGPALETTE) + nPalette * sizeof(PALETTEENTRY)];
		
		if (logPalette)
		{
			logPalette->palVersion = 0x300;
			logPalette->palNumEntries = nPalette;
			
			// start with a copy of the current system palette examples/rb/rb.c
			// in ogl2 toolkit made by Sillicon Graphics
			GetSystemPaletteEntries(memDC_, 0, nPalette, logPalette->palPalEntry);

			// fill in a RGBA color palette
			const int rmask = (1 << pfd.cRedBits) - 1;
			const int gmask = (1 << pfd.cGreenBits) - 1;
			const int bmask = (1 << pfd.cBlueBits) - 1;
			
			for (int i = 0; i < nPalette; ++i)
			{
				logPalette->palPalEntry[i].peRed = (((i >> pfd.cRedShift) & rmask) * 255) / rmask;
				logPalette->palPalEntry[i].peGreen = (((i >> pfd.cGreenShift) & gmask) * 255) / gmask;
				logPalette->palPalEntry[i].peBlue = (((i >> pfd.cBlueShift) & bmask) * 255) / bmask;
				logPalette->palPalEntry[i].peFlags = 0;
			}

			shPalette_ = CreatePalette(logPalette);

			delete [] (char *)logPalette;
			logPalette = NULL;
		}
	}

	if (shPalette_)
	{
		SelectPalette(memDC_, shPalette_, FALSE);
		RealizePalette(memDC_);
		++sUsedPaletteCount_;
	}
*/
	return true;
}

}  // namespace swl
