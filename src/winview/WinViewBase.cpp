#include "swl/winview/WinViewBase.h"

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class WinViewBase

void WinViewBase::renderScene()
{
	doPrepareRendering();
	doRenderStockScene();
	doRenderScene();
}

}  // namespace swl
