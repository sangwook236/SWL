#include "swl/Config.h"
#include "swl/winview/GdiContext.h"
#include <wingdi.h>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

GdiContext::GdiContext(HWND hWnd, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(), false, CM_DEFAULT),
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
