#include "swl/winview/WglDoubleBufferedContext.h"
#include <iostream>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl  {

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion),
  hWnd_(hWnd), hDC_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom)),
  hWnd_(hWnd), hDC_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

WglDoubleBufferedContext::~WglDoubleBufferedContext()
{
	deactivate();
}

bool WglDoubleBufferedContext::redraw()
{
	if (!isActivated() || isDrawing()) return false;
	if (NULL == hDC_) return false;
	setDrawing(true);

	const bool ret = TRUE == SwapBuffers(hDC_);

	setDrawing(false);
	return ret;
}

bool WglDoubleBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hWnd_) return false;

	// get DC for window
	hDC_ = ::GetDC(hWnd_);
	if (NULL == hDC_) return false;

	// create OpenGL pixel format descriptor
    PIXELFORMATDESCRIPTOR pfd;
    // clear OpenGL pixel format descriptor
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.iLayerType = PFD_MAIN_PLANE;

	pfd.dwFlags				= PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_STEREO_DONTCARE;
	//pfd.cColorBits		= 32;
	pfd.cColorBits			= GetDeviceCaps(hDC_, BITSPIXEL);
	pfd.cDepthBits			= 32;

	// caution:
	//	OpenGL에서 PFD_DRAW_TO_BITMAP flag을 사용하여 bitmap에 drawing할 경우
	//	pixel format을 설정하기 전에 OpenGL RC와 연결된 DC에 bitmap이 선택되어져 있어야 한다

	// choose pixel format
	int nPixelFormat = ::ChoosePixelFormat(hDC_, &pfd);
	if (nPixelFormat == 0)  // choose default
	{
		nPixelFormat = 1;
		if (::DescribePixelFormat(hDC_, nPixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd) == 0)
		{
			// release DC
			ReleaseDC(hWnd_, hDC_);
			hDC_ = NULL;

			return false;
		}
	}

	if (!::SetPixelFormat(hDC_, nPixelFormat, &pfd))
	{
		// release DC
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;

		return false;
	}

	// create rendering context
    wglRC_ = ::wglCreateContext(hDC_);
	if (NULL == wglRC_)
	{
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
    int nColorBit = ::GetDeviceCaps(hDC_, BITSPIXEL);
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
            GetSystemPaletteEntries(hDC_, 0, nPalette, pLogPalette->palPalEntry);
			
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
        ::SelectPalette(hDC_, ms_hPalette, FALSE);
        ::RealizePalette(hDC_);
        ++ms_nUsedPalette;
    }
*/

	const bool ret = (::wglGetCurrentContext() != wglRC_) ? (::wglMakeCurrent(hDC_, wglRC_) == TRUE) : true;

	setActivation(true);

	return ret;

	// draw something into rendering context
}

bool WglDoubleBufferedContext::deactivate()
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

	// release DC
	ReleaseDC(hWnd_, hDC_);
	hDC_ = NULL;

	return true;
}

}  // namespace swl
