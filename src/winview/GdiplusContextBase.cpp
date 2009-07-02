#include "swl/winview/GdiplusContextBase.h"
#include <gdiplus.h>
#include <iostream>

#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl  {

/*static*/ HPALETTE GdiplusContextBase::shPalette_ = NULL;
/*static*/ size_t GdiplusContextBase::sUsedPaletteCount_ = 0;

GdiplusContextBase::GdiplusContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed, const EContextMode contextMode)
: base_type(drawRegion, isOffScreenUsed, contextMode)
{
}

GdiplusContextBase::GdiplusContextBase(const RECT &drawRect, const bool isOffScreenUsed, const EContextMode contextMode)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), isOffScreenUsed, contextMode)
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
