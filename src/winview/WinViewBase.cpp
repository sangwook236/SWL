#include "swl/Config.h"
#include "swl/winview/WinViewBase.h"
#include "swl/view/ViewContext.h"
#include "swl/view/ViewCamera2.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class WinViewBase

void WinViewBase::renderScene(context_type &context, camera_type &camera)
{
	// guard the context
	//context_type::guard_type guard(context)

	//
	context.setViewingRegion(camera.getCurrentViewRegion());

	//
	doPrepareRendering(context, camera);
	doRenderStockScene(context, camera);
	doRenderScene(context, camera);

	// swap buffers
	context.swapBuffer();
}

}  // namespace swl
