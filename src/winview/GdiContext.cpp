#include "swl/winview/GdiContext.h"
#include <wingdi.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

GdiContext::GdiContext(HWND hWnd, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>()),
  hWnd_(hWnd), hDC_(NULL)
{
	if (isAutomaticallyActivated) activate();
}

GdiContext::~GdiContext()
{
	deactivate();
}

bool GdiContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hWnd_) return false;

	// get DC for window
	hDC_ = ::GetDC(hWnd_);
	if (NULL == hDC_) return false;

	setActivation(true);

	return true;

	// draw something into hDC_
}

bool GdiContext::deactivate()
{
	if (!isActivated()) return true;

	setActivation(false);

	// release DC
	ReleaseDC(hWnd_, hDC_);
	hDC_ = NULL;

	return true;
}

}  // namespace swl
