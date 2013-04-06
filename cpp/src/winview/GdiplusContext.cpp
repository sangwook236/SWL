#include "swl/Config.h"
#include "swl/winview/GdiplusContext.h"
#include <gdiplus.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
//#define new DEBUG_NEW
#endif


namespace swl {

GdiplusContext::GdiplusContext(HWND hWnd, const bool isAutomaticallyActivated /*= true*/)
: base_type(Region2<int>(), false, CM_DEFAULT),
  hWnd_(hWnd), graphics_(NULL)
{
	if (hWnd_)
	{
		// create graphics for window
		graphics_ = new Gdiplus::Graphics(hWnd_, FALSE);
		if (graphics_ && isAutomaticallyActivated)
			activate();
	}
}

GdiplusContext::~GdiplusContext()
{
	deactivate();

	// delete graphics
	if (graphics_)
	{
		delete graphics_;
		graphics_ = NULL;
	}
}

bool GdiplusContext::activate()
{
	if (isActivated()) return true;
	if (NULL == graphics_) return false;

	setActivation(true);
	return true;

	// draw something into graphics_
}

bool GdiplusContext::deactivate()
{
	if (!isActivated()) return true;
	if (NULL == graphics_) return false;

	setActivation(false);
	return true;
}

}  // namespace swl
