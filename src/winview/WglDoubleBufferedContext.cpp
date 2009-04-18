#include "swl/winview/WglDoubleBufferedContext.h"

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl  {

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion, false),
  hWnd_(hWnd), hDC_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), false),
  hWnd_(hWnd), hDC_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

WglDoubleBufferedContext::~WglDoubleBufferedContext()
{
	deactivate();

	// delete rendering context
	if (wglRC_)
	{
		wglDeleteContext(wglRC_);
		wglRC_ = NULL;
	}

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
	if (NULL == hDC_ || NULL == hWnd_) return false;
	setDrawing(true);

	const bool ret = TRUE == SwapBuffers(hDC_);

	setDrawing(false);
	return ret;
}

bool WglDoubleBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hDC_ || NULL == hWnd_) return false;

	const bool ret = (wglGetCurrentContext() == wglRC_) ? true : (wglMakeCurrent(hDC_, wglRC_) == TRUE);
	if (ret)
	{
		setActivation(true);
		return true;
	}
	else return false;

	// draw something into rendering context
}

bool WglDoubleBufferedContext::deactivate()
{
	if (!isActivated()) return true;
	if (NULL == hDC_ || NULL == hWnd_) return false;

	setActivation(false);

	return wglMakeCurrent(NULL, NULL) == TRUE;
}

bool WglDoubleBufferedContext::createOffScreen()
{
	if (NULL == hWnd_) return false;

	// get DC for window
	hDC_ = GetDC(hWnd_);
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

	// choose pixel format
	int nPixelFormat = ChoosePixelFormat(hDC_, &pfd);
	if (0 == nPixelFormat)  // choose default
	{
		nPixelFormat = 1;
		if (DescribePixelFormat(hDC_, nPixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd) == 0)
		{
			ReleaseDC(hWnd_, hDC_);
			hDC_ = NULL;
			return false;
		}
	}

	if (FALSE == SetPixelFormat(hDC_, nPixelFormat, &pfd))
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
		return false;
	}

	// create rendering context
    wglRC_ = wglCreateContext(hDC_);
	if (NULL == wglRC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
		return false;
	}

	// create & share a display list
	createDisplayList(hDC_);

	// use a palette in 256 color mode
	usePalette(hDC_, pfd);

	return true;
}

}  // namespace swl
