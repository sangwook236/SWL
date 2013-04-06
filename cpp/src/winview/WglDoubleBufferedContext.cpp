#include "swl/Config.h"
#include "swl/winview/WglDoubleBufferedContext.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl  {

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const Region2<int>& drawRegion, const bool isAutomaticallyActivated /*= true*/)
: base_type(drawRegion, false, CM_DEFAULT),
  hWnd_(hWnd), hDC_(NULL)
{
	if (createOffScreen() && isAutomaticallyActivated)
		activate();
}

WglDoubleBufferedContext::WglDoubleBufferedContext(HWND hWnd, const RECT& drawRect, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), false, CM_DEFAULT),
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
	//if (!isActivated() || isDrawing()) return false;
	if (isDrawing()) return false;
	if (NULL == hDC_) return false;
	setDrawing(true);

	const bool ret = TRUE == SwapBuffers(hDC_);

	setDrawing(false);
	return ret;
}

bool WglDoubleBufferedContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hDC_) return false;

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
	if (NULL == hDC_) return false;

	setActivation(false);

	return wglMakeCurrent(NULL, NULL) == TRUE;
}

bool WglDoubleBufferedContext::createOffScreen()
{
	if (NULL == hWnd_) return false;

	// get DC for window
	hDC_ = GetDC(hWnd_);
	if (NULL == hDC_) return false;

	// without this line, wglCreateContext will fail
	wglMakeCurrent(hDC_, 0);

	// create OpenGL pixel format descriptor
    PIXELFORMATDESCRIPTOR pfd;
    // clear OpenGL pixel format descriptor
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion			= 1;
    pfd.iPixelType			= PFD_TYPE_RGBA;
    pfd.iLayerType			= PFD_MAIN_PLANE;

	pfd.dwFlags				= PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_STEREO_DONTCARE;
	//pfd.cColorBits		= 32;
	pfd.cColorBits			= GetDeviceCaps(hDC_, BITSPIXEL);
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
	pfd.cStencilBits		= 16;
	pfd.cAuxBuffers			= 0;
	pfd.bReserved			= 0;
	pfd.dwLayerMask			= 0;
	pfd.dwVisibleMask		= 0;
	pfd.dwDamageMask		= 0;

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

	return true;
}

}  // namespace swl
