#include "swl/winview/GdiplusContext.h"
#include <gdiplus.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
//#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

GdiplusContext::GdiplusContext(HWND hWnd, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>()),
  hWnd_(hWnd), graphics_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

GdiplusContext::~GdiplusContext()
{
	deactivate();
}

bool GdiplusContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hWnd_) return false;

	// create graphics for window
	graphics_ = new Gdiplus::Graphics(hWnd_, FALSE);
	if (NULL == graphics_) return false;

	setActivation(true);

	return true;

	// draw something into graphics_
}

bool GdiplusContext::deactivate()
{
	if (!isActivated()) return true;

	setActivation(false);

	// delete graphics
	delete graphics_;
	graphics_ = NULL;

	return true;
}

}  // namespace swl
