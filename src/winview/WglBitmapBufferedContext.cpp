#include "swl/winview/WglBitmapBufferedContext.h"
#include <boost/smart_ptr.hpp>

#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

WglBitmapBufferedContext::WglBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion, true, CM_DEFAULT),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL), dibBits_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

WglBitmapBufferedContext::WglBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), true, CM_DEFAULT),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL), dibBits_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

WglBitmapBufferedContext::~WglBitmapBufferedContext()
{
	deactivate();

	// delete rendering context
	if (wglRC_)
	{
		wglDeleteContext(wglRC_);
		wglRC_ = NULL;
	}

	deleteOffScreen();
}

bool WglBitmapBufferedContext::swapBuffer()
{
	//if (!isActivated() || isDrawing()) return false;
	if (isDrawing()) return false;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;
	setDrawing(true);

	// copy off-screen buffer to window's DC
	const bool ret = TRUE == BitBlt(
		hDC_,
		drawRegion_.left, drawRegion_.bottom,
		drawRegion_.getWidth(), drawRegion_.getHeight(), 
		memDC_,
		0, 0,  //drawRegion_.left, drawRegion_.bottom,
		SRCCOPY
	);

	setDrawing(false);
	return ret;
}

bool WglBitmapBufferedContext::resize(const int x1, const int y1, const int x2, const int y2)
{
	if (isActivated()) return false;
	drawRegion_ = Region2<int>(x1, y1, x2, y2);

	// delete rendering context
	if (wglRC_)
	{
		wglDeleteContext(wglRC_);
		wglRC_ = NULL;
	}

	deleteOffScreen();

	return createOffScreen();
}

bool WglBitmapBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;

	const bool ret = (wglGetCurrentContext() == wglRC_) ? true : (wglMakeCurrent(memDC_, wglRC_) == TRUE);
	if (ret)
	{
		setActivation(true);
		return true;
	}
	else return false;

	// draw something into rendering context
}

bool WglBitmapBufferedContext::deactivate()
{
	if (!isActivated()) return true;
	if (NULL == memBmp_ || NULL == memDC_ || NULL == hDC_) return false;

	setActivation(false);

	return wglMakeCurrent(NULL, NULL) == TRUE;
}

bool WglBitmapBufferedContext::createOffScreen()
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

	// create OpenGL pixel format descriptor
    PIXELFORMATDESCRIPTOR pfd;
    // clear OpenGL pixel format descriptor
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion			= 1;
    pfd.iPixelType			= PFD_TYPE_RGBA;
    pfd.iLayerType			= PFD_MAIN_PLANE;

    // offscreen was requested
	pfd.dwFlags				= PFD_DRAW_TO_BITMAP | PFD_SUPPORT_OPENGL | PFD_SUPPORT_GDI | PFD_STEREO_DONTCARE;
	pfd.cColorBits			= colorBitCount;
	pfd.cRedBits			= 8;
	pfd.cRedShift			= 16;
	pfd.cGreenBits			= 8;
	pfd.cGreenShift			= 8;
	pfd.cBlueBits			= 8;
	pfd.cBlueShift			= 0;
	pfd.cAlphaBits			= 0;
	pfd.cAlphaShift			= 0;
	//pfd.cAccumBits		= 64;  // consider more flexible configuration
	//pfd.cAccumRedBits		= 16;
	//pfd.cAccumGreenBits	= 16;
	//pfd.cAccumBlueBits	= 16;
	//pfd.cAccumAlphaBits	= 0;
	pfd.cDepthBits			= 32;
	pfd.cStencilBits		= 8;
	pfd.cAuxBuffers			= 0;
	pfd.bReserved			= 0;
	pfd.dwLayerMask			= 0;
	pfd.dwVisibleMask		= 0;
	pfd.dwDamageMask		= 0;

	// use palette: when using 256 color
	if (isPaletteUsed_) createPalette(memDC_, pfd, colorBitCount);

	//
	if (!createOffScreenBitmap(colorBitCount, colorPlaneCount))
	{
		DeleteObject(memBmp_);
		memBmp_ = NULL;

		DeleteDC(memDC_);
		memDC_ = NULL;

		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
		return false;
	}

	// caution:
	//	1. OpenGL에서 PFD_DRAW_TO_BITMAP flag을 사용하여 bitmap에 drawing할 경우
	//		pixel format을 설정하기 전에 OpenGL RC와 연결된 DC에 bitmap이 선택되어져 있어야 한다.
	//	2. off-screen을 위한 bitmap은 CreateCompatibleBitmap()이 아닌 CreateDIBSection()으로 생성하여야 한다.
	//		그렇지 않을 경우, SetPixelFormat() 실행시 error가 발생한다.

	// choose pixel format
	int nPixelFormat = ChoosePixelFormat(memDC_, &pfd);
	if (0 == nPixelFormat)  // choose default
	{
		nPixelFormat = 1;
		if (DescribePixelFormat(memDC_, nPixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd) == 0)
		{
			deleteOffScreen();
			return false;
		}
	}

	if (FALSE == SetPixelFormat(memDC_, nPixelFormat, &pfd))
	{
		deleteOffScreen();
		return false;
	}

	// create rendering context
    wglRC_ = wglCreateContext(memDC_);
	if (NULL == wglRC_)
	{
		deleteOffScreen();
		return false;
	}

	// create & share a display list
	createDisplayList(memDC_);

	return true;
}

bool WglBitmapBufferedContext::createOffScreenBitmap(const int colorBitCount, const int colorPlaneCount)
{
	// method #1
/*
	memBmp_ = CreateCompatibleBitmap(hDC_, drawRegion_.getWidth(), drawRegion_.getHeight());
	//memBmp_ = CreateCompatibleBitmap(hDC_, (int)std::floor(viewingRegion_.getWidth() + 0.5), (int)std::floor(viewingRegion_.getHeight() + 0.5));
*/
	// method #2
	const size_t bufSize = !isPaletteUsed_ ? sizeof(BITMAPINFO) : sizeof(BITMAPINFO) + sizeof(RGBQUAD) * 255;
	const boost::scoped_array<unsigned char> buf(new unsigned char [bufSize]);
	memset(buf.get(), 0, bufSize);
	BITMAPINFO &bmiDIB = *(BITMAPINFO *)buf.get();

	// Following routine aligns given value to 4 bytes boundary.
	// The current implementation of DIB rendering in Windows 95/98/NT seems to be free from this alignment
	// but old version compatibility is needed.
	const int width = ((drawRegion_.getWidth() + 3) / 4 * 4 > 0) ? drawRegion_.getWidth() : 4;
	const int height = (0 == drawRegion_.getHeight()) ? 1 : drawRegion_.getHeight();
	//const int viewingWidth = (int)std::floor(viewingRegion_.getWidth() + 0.5);
	//const int viewingHeight = (int)std::floor(viewingRegion_.getHeight() + 0.5);
	//const int width = ((viewingWidth + 3) / 4 * 4 > 0) ? viewingWidth : 4;
	//const int height = (0 == viewingHeight) ? 1 : viewingHeight;

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

		// when using 256 color
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
	if (NULL == memBmp_ || NULL == dibBits_) return false;
	oldBmp_ = (HBITMAP)SelectObject(memDC_, memBmp_);

	return true;
}

void WglBitmapBufferedContext::deleteOffScreen()
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
