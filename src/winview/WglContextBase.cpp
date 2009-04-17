#include "swl/winview/WglContextBase.h"

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl  {

HGLRC WglContextBase::sSharedRC_ = NULL;

WglContextBase::WglContextBase(const Region2<int>& drawRegion)
: base_type(drawRegion),
  wglRC_(NULL)
{
}

WglContextBase::WglContextBase(const RECT& drawRect)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom)),
  wglRC_(NULL)
{
}

WglContextBase::~WglContextBase()
{
}

bool WglContextBase::shareDisplayList(HGLRC& wglRC)
{
	if (NULL == wglRC) return false;
	if (NULL == sSharedRC_)
	{
		sSharedRC_ = wglRC;
		return true;
	}
	else
		return ::wglShareLists(sSharedRC_, wglRC) == TRUE;
}

}  // namespace swl
