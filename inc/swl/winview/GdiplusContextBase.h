#if !defined(__SWL_WIN_VIEW__GDI_PLUS_CONTEXT_BASE__H_ )
#define __SWL_WIN_VIEW__GDI_PLUS_CONTEXT_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>


namespace Gdiplus {

class Graphics;

}  // namespace Gdiplus

namespace swl {

//-----------------------------------------------------------------------------------
//  Base Context for GDI+ in Microsoft Windows

class SWL_WIN_VIEW_API GdiplusContextBase: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef Gdiplus::Graphics context_type;

protected:
	explicit GdiplusContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed, const EContextMode contextMode);
	explicit GdiplusContextBase(const RECT &drawRect, const bool isOffScreenUsed, const EContextMode contextMode);
public:
	virtual ~GdiplusContextBase();

private:
	GdiplusContextBase(const GdiplusContextBase &);
	GdiplusContextBase & operator=(const GdiplusContextBase &);

protected:
	/// create a palette
	static void createPalette(Gdiplus::Graphics &graphics, const int colorBitCount);
	/// delete a palette
	static void deletePalette(Gdiplus::Graphics &graphics);

protected:
	/// a palette handle for the index mode graphics hardware
    static HPALETTE shPalette_;
	/// a palette usage count
    static size_t sUsedPaletteCount_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__GDI_PLUS_CONTEXT_BASE__H_
