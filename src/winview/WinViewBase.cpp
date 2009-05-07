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

void WinViewBase::renderScene(context_type &viewContext, camera_type &viewCamera)
{
	// activate a context
	//viewContext.activate();

	doPrepareRendering(viewContext, viewCamera);
	doRenderStockScene(viewContext, viewCamera);
	doRenderScene(viewContext, viewCamera);

	// swap buffers
	viewContext.swapBuffer();

	// de-activate the context
	//viewContext.deactivate();
}

}  // namespace swl
