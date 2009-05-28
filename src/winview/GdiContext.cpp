#include "swl/winview/GdiContext.h"
#include <wingdi.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

GdiContext::GdiContext(HWND hWnd, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(), false, CM_VIEWING),
  hWnd_(hWnd), hDC_(NULL)
{
	if (hWnd_)
	{
		// get DC for window
		hDC_ = GetDC(hWnd_);
		if (hDC_ && isAutomaticallyActivated)
			activate();
	}
}

GdiContext::~GdiContext()
{
	deactivate();

	// release DC
	if (hDC_)
	{
		ReleaseDC(hWnd_, hDC_);
		hDC_ = NULL;
	}
}

bool GdiContext::activate()
{
	if (isActivated()) return true;
	if (NULL == hDC_) return false;

	setActivation(true);

	return true;

	// draw something into hDC_
}

bool GdiContext::deactivate()
{
	if (!isActivated()) return true;
	if (NULL == hDC_) return false;

	setActivation(false);

	return true;
}

}  // namespace swl
