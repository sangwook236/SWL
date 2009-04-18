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
	if (NULL == hWnd_) return;

	// get DC for window
	hDC_ = GetDC(hWnd_);
	if (NULL == hDC_) return;

	if (isAutomaticallyActivated) activate();
}

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom)),
  hWnd_(hWnd), hDC_(NULL)
{
	if (NULL == hWnd_) return;

	// get DC for window
	hDC_ = GetDC(hWnd_);
	if (NULL == hDC_) return;

	if (isAutomaticallyActivated) activate();
}

WglDoubleBufferedContext::~WglDoubleBufferedContext()
{
	deactivate();

	// release DC
	if (hDC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
	}
}

bool WglDoubleBufferedContext::swapBuffer()
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

	// choose pixel format
	int nPixelFormat = ChoosePixelFormat(hDC_, &pfd);
	if (0 == nPixelFormat)  // choose default
	{
		nPixelFormat = 1;
		if (DescribePixelFormat(hDC_, nPixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd) == 0)
			return false;
	}

	if (FALSE == SetPixelFormat(hDC_, nPixelFormat, &pfd))
		return false;

	// create rendering context
    wglRC_ = wglCreateContext(hDC_);
	if (NULL == wglRC_)
		return false;

	// caution:
	//	OpenGL에서 display list를 share하고자 하는 경우 source RC와 destination RC가 동일하여야 한다.
	//	예를 들어, source RC의 flag 속성이 PFD_DRAW_TO_WINDOW이고 destination RC의 flag 속성이
	//	PFD_DRAW_TO_BITMAP이라면, display list의 share는 실패한다.

	// share display list
	if (!shareDisplayList(wglRC_))
	{
#if defined(WIN32) && defined(_DEBUG)
		LPVOID lpMsgBuf;
		FormatMessage( 
			FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			GetLastError(),
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),  //  Default language
			(LPTSTR) &lpMsgBuf,
			0,
			NULL 
		);
		// display the string
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"error : fail to share display lists(" << (LPCTSTR)lpMsgBuf << L") at " << __LINE__ << L" in " << __FILE__ << std::endl;
#else
		std::cout << "Error : fail to share display lists(" << (LPCTSTR)lpMsgBuf << ") at " << __LINE__ << " in " << __FILE__ << std::endl;
#endif
		// free the buffer
		LocalFree(lpMsgBuf);
#endif
		// need to add the code that the display lists are created
		if (wglMakeCurrent(hDC_, wglRC_) == TRUE)
		{
			doRecreateDisplayList();
			wglMakeCurrent(NULL, NULL);
		}
	}
/*
	// when using 256 color
    int nColorBit = GetDeviceCaps(hDC_, BITSPIXEL);
    if (nColorBit <= 8 && !ms_hPalette)
	{
        // following routines are originated from ogl2 sdk made by Sillicon Graphics and modified for glext
        int nPalette = 1 << nColorBit;
        LOGPALETTE* pLogPalette = NULL;
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
			
            ms_hPalette = CreatePalette(pLogPalette);
            if (pLogPalette)
			{
				free(pLogPalette);
				pLogPalette = NULL;
			}
        }
    }
	
    if (ms_hPalette)
	{
        SelectPalette(hDC_, ms_hPalette, FALSE);
        RealizePalette(hDC_);
        ++ms_nUsedPalette;
    }
*/
	const bool ret = (wglGetCurrentContext() == wglRC_) ? true : (wglMakeCurrent(hDC_, wglRC_) == TRUE);
	if (ret)
	{
		setActivation(true);
		return true;
	}
	else
	{
		releaseWglResources();
		return false;
	}

	// draw something into rendering context
}

bool WglDoubleBufferedContext::deactivate()
{
	if (!isActivated()) return true;

	setActivation(false);

	const bool ret = wglMakeCurrent(NULL, NULL) == TRUE;

	releaseWglResources();

	return ret;
}

void WglDoubleBufferedContext::releaseWglResources()
{
	// delete rendering context
	if (wglRC_)
	{
		wglDeleteContext(wglRC_);
		wglRC_ = NULL;
	}
}

}  // namespace swl
