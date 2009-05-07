#include "swl/winview/WinViewBase.h"
#include "swl/view/ViewContext.h"

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class WinViewBase

void WinViewBase::renderScene(context_type &context, camera_type &camera)
{
	// activate a context
	//context.activate();

	doPrepareRendering(context, camera);
	doRenderStockScene(context, camera);
	doRenderScene(context, camera);

	// swap buffers
	context.swapBuffer();

	// de-activate the context
	//context.deactivate();
}

}  // namespace swl
