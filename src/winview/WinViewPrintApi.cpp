#include "swl/winview/WinViewPrintApi.h"
#include "swl/winview/GdiPrintContext.h"
#include "swl/winview/WinViewBase.h"
#include "swl/view/ViewCamera2.h"
#include <wingdi.h>
#include <cmath>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

bool printWinViewUsingGdi(WinViewBase &view, HDC hPrintDC)
{
	//
	RECT rctPage;
	rctPage.left = 0;
	rctPage.top = 0;
	rctPage.right = GetDeviceCaps(hPrintDC, HORZRES);
	rctPage.bottom = GetDeviceCaps(hPrintDC, VERTRES);

	{
		const boost::shared_ptr<WinViewBase::camera_type> &currCamera = view.topCamera();
 		if (currCamera.get())
		{
			const std::auto_ptr<WinViewBase::context_type> printContext(new GdiPrintContext(hPrintDC, rctPage));
			const std::auto_ptr<WinViewBase::camera_type> printCamera(currCamera->cloneCamera());
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
