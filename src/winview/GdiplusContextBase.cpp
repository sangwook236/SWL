#include "swl/winview/GdiplusContextBase.h"
#include <gdiplus.h>
#include <iostream>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl  {

/*static*/ HPALETTE GdiplusContextBase::shPalette_ = NULL;
/*static*/ size_t GdiplusContextBase::sUsedPaletteCount_ = 0;

GdiplusContextBase::GdiplusContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed)
: base_type(drawRegion, isOffScreenUsed)
{
}

GdiplusContextBase::GdiplusContextBase(const RECT &drawRect, const bool isOffScreenUsed)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), isOffScreenUsed)
{
}

GdiplusContextBase::~GdiplusContextBase()
{
}

/*static*/ void GdiplusContextBase::createPalette(Gdiplus::Graphics &graphics, const int colorBitCount)
{
	// FIXME [add] >> not implemented
	throw std::runtime_error("GdiplusContextBase::createPalette() is not implemented");
}

/*static*/ void GdiplusContextBase::deletePalette(Gdiplus::Graphics &graphics)
{
	// FIXME [add] >> not implemented
	throw std::runtime_error("GdiplusContextBase::deletePalette() is not implemented");
}

}  // namespace swl
