#include "swl/winview/WglViewPrintApi.h"
#include "swl/winview/WglPrintContext.h"
#include "swl/winview/WglViewBase.h"
#include "swl/oglview/OglCamera.h"
#include <wingdi.h>
#include <cmath>

#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

bool printWglViewUsingGdi(WglViewBase &view, HDC hPrintDC)
{
	//
	RECT rctPage;
	rctPage.left = 0;
	rctPage.top = 0;
	rctPage.right = GetDeviceCaps(hPrintDC, HORZRES);
	rctPage.bottom = GetDeviceCaps(hPrintDC, VERTRES);

	{
		const boost::shared_ptr<WglViewBase::camera_type> &currCamera = view.topCamera();
 		if (currCamera.get())
		{
			const std::auto_ptr<WglViewBase::context_type> printContext(new WglPrintContext(hPrintDC, rctPage));
			const std::auto_ptr<WglViewBase::camera_type> printCamera(dynamic_cast<WglViewBase::camera_type *>(currCamera->cloneCamera()));
			if (printCamera.get() && printContext.get() && printContext->isActivated())
			{
				view.initializeView();
				printCamera->setViewRegion(currCamera->getCurrentViewRegion());
				printCamera->setViewport(rctPage.left, rctPage.top, rctPage.right, rctPage.bottom);
				view.renderScene(*printContext, *printCamera);
			}
		}
	}

	return true;
}

}  // namespace swl
