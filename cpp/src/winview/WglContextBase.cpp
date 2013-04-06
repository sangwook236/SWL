#include "swl/Config.h"
#include "swl/winview/WglContextBase.h"
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl  {

/*static*/ HPALETTE WglContextBase::shPalette_ = NULL;
/*static*/ size_t WglContextBase::sUsedPaletteCount_ = 0;

WglContextBase::WglContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed, const EContextMode contextMode)
: base_type(drawRegion, isOffScreenUsed, contextMode),
  wglRC_(NULL), isPaletteUsed_(false)
{
}

WglContextBase::WglContextBase(const RECT &drawRect, const bool isOffScreenUsed, const EContextMode contextMode)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), isOffScreenUsed, contextMode),
  wglRC_(NULL), isPaletteUsed_(false)
{
}

WglContextBase::~WglContextBase()
{
}

bool WglContextBase::shareDisplayList(const WglContextBase &srcContext)
{
	//if (isActivated()) return false;  // don't care

	// caution:
	//	OpenGL에서 display list를 share하고자 하는 경우 source RC와 destination RC가 동일하여야 한다.
	//	예를 들어, source RC의 flag 속성이 PFD_DRAW_TO_WINDOW이고
	//	destination RC의 flag 속성이 PFD_DRAW_TO_BITMAP이라면, display list의 share는 실패한다.

	if (NULL == srcContext.wglRC_) return false;
	return srcContext.wglRC_ == wglRC_ ? true : wglShareLists(srcContext.wglRC_, wglRC_) == TRUE;
}

/*static*/ void WglContextBase::createPalette(HDC hDC, const PIXELFORMATDESCRIPTOR &pfd, const int colorBitCount)
{
	// FIXME [check] >> this implementation is not tested
	if (colorBitCount <= 8 && !shPalette_)
	{
		// following routines are originated from ogl2 sdk made by Sillicon Graphics and modified for glext
		const int paletteSize = 1 << colorBitCount;
		LOGPALETTE *logPalette = (LOGPALETTE *)new char [sizeof(LOGPALETTE) + paletteSize * sizeof(PALETTEENTRY)];
		
		if (logPalette)
		{
			logPalette->palVersion = 0x300;
			// start with a copy of the current system palette examples/rb/rb.c
			// in ogl2 toolkit made by Sillicon Graphics
			logPalette->palNumEntries = GetSystemPaletteEntries(hDC, 0, paletteSize, logPalette->palPalEntry);

			// fill in a RGBA color palette
			const int rmask = (1 << pfd.cRedBits) - 1;
			const int gmask = (1 << pfd.cGreenBits) - 1;
			const int bmask = (1 << pfd.cBlueBits) - 1;
			
			for (int i = 0; i < paletteSize; ++i)
			{
				logPalette->palPalEntry[i].peRed = (((i >> pfd.cRedShift) & rmask) * 255) / rmask;
				logPalette->palPalEntry[i].peGreen = (((i >> pfd.cGreenShift) & gmask) * 255) / gmask;
				logPalette->palPalEntry[i].peBlue = (((i >> pfd.cBlueShift) & bmask) * 255) / bmask;
				logPalette->palPalEntry[i].peFlags = 0;
			}

			shPalette_ = CreatePalette(logPalette);

			delete [] (char *)logPalette;
			logPalette = NULL;
		}
	}

	if (shPalette_)
	{
		SelectPalette(hDC, shPalette_, FALSE);
		RealizePalette(hDC);
		++sUsedPaletteCount_;
	}
}

/*static*/ void WglContextBase::deletePalette(HDC hDC)
{
	// delete palette
	SelectPalette(hDC, (HPALETTE)GetStockObject(DEFAULT_PALETTE), FALSE);
	if (shPalette_ && --sUsedPaletteCount_ <= 0)
	{
		DeleteObject(shPalette_);
		shPalette_ = NULL;
		sUsedPaletteCount_ = 0;
	}
}

}  // namespace swl
