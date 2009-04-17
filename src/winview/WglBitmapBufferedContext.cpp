#include "swl/winview/WglBitmapBufferedContext.h"
#include <iostream>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

WglBitmapBufferedContext::WglBitmapBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

WglBitmapBufferedContext::WglBitmapBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom)),
  hWnd_(hWnd), hDC_(NULL), memDC_(NULL), memBmp_(NULL), oldBmp_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

WglBitmapBufferedContext::~WglBitmapBufferedContext()
{
	deactivate();
}

bool WglBitmapBufferedContext::redraw()
{
	if (!isActivated() || isDrawing()) return false;
	if (NULL == memDC_ || NULL == memBmp_) return false;
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

bool WglBitmapBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hWnd_) return false;

	// get DC for window
	hDC_ = ::GetDC(hWnd_);
	if (NULL == hDC_) return false;

	// create an off-screen DC for double-buffering
	memDC_ = CreateCompatibleDC(hDC_);
	if (NULL == memDC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;

		return false;
	}
	memBmp_ = CreateCompatibleBitmap(memDC_, drawRegion_.getWidth(), drawRegion_.getHeight());
	if (NULL == memBmp_)
	{
		DeleteDC(memDC_);
		memDC_ = NULL;

		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;

		return false;
	}
	oldBmp_ = (HBITMAP)SelectObject(memDC_, memBmp_);

	// create OpenGL pixel format descriptor
    PIXELFORMATDESCRIPTOR pfd;
    // clear OpenGL pixel format descriptor
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.iLayerType = PFD_MAIN_PLANE;

    // offscreen was requested
	pfd.dwFlags				= PFD_DRAW_TO_BITMAP | PFD_SUPPORT_OPENGL | PFD_SUPPORT_GDI | PFD_STEREO_DONTCARE;
	//pfd.cColorBits		= 32;
	pfd.cColorBits			= ::GetDeviceCaps(memDC_, BITSPIXEL);
	pfd.cRedBits			= 8;
	pfd.cRedShift			= 16;
	pfd.cGreenBits			= 8;
	pfd.cGreenShift			= 8;
	pfd.cBlueBits			= 8;
	pfd.cBlueShift			= 0;
	pfd.cAlphaBits			= 0;
	pfd.cAlphaShift			= 0;
	//pfd.cAccumBits		= 64;  //  consider more flexible configuration
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

	if (NULL == memDC_)
	{
		// free-up the off-screen DC
		SelectObject(memDC_, oldBmp_);
		oldBmp_ = NULL;
		DeleteObject(memBmp_);
		memBmp_ = NULL;
		DeleteDC(memDC_);
		memDC_ = NULL;

		// release DC
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;

		return false;
	}

	// caution:
	//	OpenGL에서 PFD_DRAW_TO_BITMAP flag을 사용하여 bitmap에 drawing할 경우
	//	pixel format을 설정하기 전에 OpenGL RC와 연결된 DC에 bitmap이 선택되어져 있어야 한다

	// choose pixel format
	int nPixelFormat = ::ChoosePixelFormat(memDC_, &pfd);
	if (nPixelFormat == 0)  // choose default
	{
		nPixelFormat = 1;
		if (::DescribePixelFormat(memDC_, nPixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd) == 0)
		{
			// free-up the off-screen DC
			SelectObject(memDC_, oldBmp_);
			oldBmp_ = NULL;
			DeleteObject(memBmp_);
			memBmp_ = NULL;
			DeleteDC(memDC_);
			memDC_ = NULL;

			// release DC
			ReleaseDC(hWnd_, hDC_);
			hDC_ = NULL;

			return false;
		}
	}

	if (!::SetPixelFormat(memDC_, nPixelFormat, &pfd))
	{
		// free-up the off-screen DC
		SelectObject(memDC_, oldBmp_);
		oldBmp_ = NULL;
		DeleteObject(memBmp_);
		memBmp_ = NULL;
		DeleteDC(memDC_);
		memDC_ = NULL;

		// release DC
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;

		return false;
	}

	// create rendering context
    wglRC_ = ::wglCreateContext(memDC_);
	if (NULL == wglRC_)
	{
		// free-up the off-screen DC
		SelectObject(memDC_, oldBmp_);
		oldBmp_ = NULL;
		DeleteObject(memBmp_);
		memBmp_ = NULL;
		DeleteDC(memDC_);
		memDC_ = NULL;

		// release DC
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;

		return false;
	}

	// caution:
	//	OpenGL에서 display list를 share하고자 하는 경우 source RC와 destination RC가 동일하여야 한다
	//	예를 들어, source RC의 flag 속성이 PFD_DRAW_TO_WINDOW이고 destination RC의 flag 속성이
	//	PFD_DRAW_TO_BITMAP이라면, display list의 share는 실패할 것이다

	// share display list
	if (!shareDisplayList(wglRC_))
	{
#if defined(WIN32) && defined(_DEBUG)
		LPVOID lpMsgBuf;
		::FormatMessage( 
			FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			::GetLastError(),
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),  //  Default language
			(LPTSTR) &lpMsgBuf,
			0,
			NULL 
		);
		// display the string
		std::cout << "Error : fail to share display lists(" << (LPCTSTR)lpMsgBuf << ") at " << __LINE__ << " in " << __FILE__ << std::endl;
		// free the buffer
		::LocalFree(lpMsgBuf);
#endif
		// need to add the code that the display lists are created
		activate();
		doRecreateDisplayList();
		deactivate();
	}
/*
	// when using 256 color
    int nColorBit = ::GetDeviceCaps(memDC_, BITSPIXEL);
    if (nColorBit <= 8 && !ms_hPalette)
	{
        // following routines are originated from ogl2 sdk made by Sillicon Graphics and modified for glext
        int nPalette = 1 << nColorBit;
        LOGPALETTE* pLogPalette = 0L;
		pLogPalette = (LOGPALETTE*)malloc(sizeof(LOGPALETTE) + nPalette*sizeof(PALETTEENTRY));
		
        if (pLogPalette)
		{
            pLogPalette->palVersion = 0x300;
            pLogPalette->palNumEntries = nPalette;
			
            // start with a copy of the current system palette examples/rb/rb.c
            // in ogl2 toolkit made by Sillicon Graphics
            GetSystemPaletteEntries(memDC_, 0, nPalette, pLogPalette->palPalEntry);
			
            // fill in a rgba color palette
            int rmask = (1 << pfd.cRedBits) - 1;
            int gmask = (1 << pfd.cGreenBits) - 1;
            int bmask = (1 << pfd.cBlueBits) - 1;
			
            for (int i=0 ; i<nPalette ; ++i)
			{
                pLogPalette->palPalEntry[i].peRed = (((i >> pfd.cRedShift) & rmask) * 255) / rmask;
                pLogPalette->palPalEntry[i].peGreen = (((i >> pfd.cGreenShift) & gmask) * 255) / gmask;
                pLogPalette->palPalEntry[i].peBlue = (((i >> pfd.cBlueShift) & bmask) * 255) / bmask;
                pLogPalette->palPalEntry[i].peFlags = 0;
            }
			
            ms_hPalette = ::CreatePalette(pLogPalette);
            if (pLogPalette)
			{
				free(pLogPalette);
				pLogPalette = 0L;
			}
        }
    }
	
    if (ms_hPalette)
	{
        ::SelectPalette(memDC_, ms_hPalette, FALSE);
        ::RealizePalette(memDC_);
        ++ms_nUsedPalette;
    }
*/

	const bool ret = (::wglGetCurrentContext() != wglRC_) ? (::wglMakeCurrent(hDC_, wglRC_) == TRUE) : true;

	setActivation(true);

	return ret;

	// draw something into rendering context
}

bool WglBitmapBufferedContext::deactivate()
{
	if (!isActivated()) return true;

	setActivation(false);

	const bool ret = ::wglMakeCurrent(0L, 0L) == TRUE;

    // delete rendering context
    if (wglRC_)
	{
        ::wglDeleteContext(wglRC_);
        wglRC_ = 0L;
    }

	// free-up the off-screen DC
	SelectObject(memDC_, oldBmp_);
	oldBmp_ = NULL;
    DeleteObject(memBmp_);
	memBmp_ = NULL;
    DeleteDC(memDC_);
	memDC_ = NULL;

	// release DC
	ReleaseDC(hWnd_, hDC_);
	hDC_ = NULL;

	return ret;
}

}  // namespace swl
