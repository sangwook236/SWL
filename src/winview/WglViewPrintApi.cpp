#include "swl/Config.h"
#include "swl/winview/WglViewPrintApi.h"
#include "swl/winview/WglPrintContext.h"
#include "swl/winview/WglViewBase.h"
#include "swl/glutil/GLCamera.h"
#include <wingdi.h>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


#if defined(min)
#undef min
#endif


namespace swl {

bool printWglViewUsingGdi(WglViewBase &view, HDC hPrintDC)
{
	const boost::shared_ptr<WglViewBase::camera_type> &currCamera = view.topCamera();
	if (currCamera.get())
	{
#if 0
		const Region2<int> rctPage(swl::Point2<int>(0, 0), swl::Point2<int>(GetDeviceCaps(hPrintDC, HORZRES), GetDeviceCaps(hPrintDC, VERTRES)));

		{
			const std::auto_ptr<WglViewBase::context_type> printContext(new WglPrintContext(hPrintDC, rctPage));  // error: use rctPage
			const std::auto_ptr<WglViewBase::camera_type> printCamera(dynamic_cast<WglViewBase::camera_type *>(currCamera->cloneCamera()));
			if (printCamera.get() && printContext.get() && printContext->isActivated())
			{
				view.initializeView();
				printCamera->setViewRegion(currCamera->getCurrentViewRegion());
				printCamera->setViewport(rctPage.left, rctPage.top, rctPage.right, rctPage.bottom);
				view.renderScene(*printContext, *printCamera);
			}
		}
#else
		const double eps = 1.0e-20;

		const Region2<int> rctPage(swl::Point2<int>(0, 0), swl::Point2<int>(GetDeviceCaps(hPrintDC, HORZRES), GetDeviceCaps(hPrintDC, VERTRES)));
		const Region2<double> &currViewRegion = currCamera->getCurrentViewRegion();
		const double width = currViewRegion.getWidth() >= eps ? currViewRegion.getWidth() : 1.0;
		const double height = currViewRegion.getHeight() >= eps ? currViewRegion.getHeight() : 1.0;
		const double ratio = std::min(rctPage.getWidth() / width, rctPage.getHeight() / height);

		const double width0 = width * ratio, height0 = height * ratio;
		const int w0 = (int)std::floor(width0), h0 = (int)std::floor(height0);
		const int x0 = rctPage.left + (int)std::floor((rctPage.getWidth() - width0) * 0.5), y0 = rctPage.bottom + (int)std::floor((rctPage.getHeight() - height0) * 0.5);

		{
			const boost::shared_ptr<WglViewBase::context_type> &currContext = view.topContext();

			const std::auto_ptr<WglViewBase::context_type> printContext(new WglPrintContext(hPrintDC, Region2<int>(x0, y0, x0 + w0, y0 + h0)));
			const std::auto_ptr<WglViewBase::camera_type> printCamera(dynamic_cast<WglViewBase::camera_type *>(currCamera->cloneCamera()));

			const bool isDisplayListShared = !currContext ? false : printContext->shareDisplayList(*currContext);

			if (printCamera.get() && printContext.get() && printContext->isActivated())
			{
				const bool doesRecreateDisplayListUsed = !isDisplayListShared && view.isDisplayListUsed();
				// create & push a new name base of OpenGL display list
				if (doesRecreateDisplayListUsed) view.generateDisplayListName(true);

				view.initializeView();
				printCamera->setViewRegion(currCamera->getCurrentViewRegion());
				printCamera->setViewport(0, 0, w0, h0);

				// re-create a OpenGL display list
				if (doesRecreateDisplayListUsed) view.createDisplayList(true);

				view.renderScene(*printContext, *printCamera);

				// pop & delete a new name base of OpenGL display list
				if (doesRecreateDisplayListUsed) view.deleteDisplayListName(true);
			}
		}
#endif

		return true;
	}
	else return true;
}

}  // namespace swl
