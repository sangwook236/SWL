#if !defined(__SWL_WIN_VIEW__GDI_CONTEXT_BASE__H_ )
#define __SWL_WIN_VIEW__GDI_CONTEXT_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>


namespace swl {

//-----------------------------------------------------------------------------------
//  Base Context for GDI in Microsoft Windows

class SWL_WIN_VIEW_API GdiContextBase: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef HDC context_type;

protected:
	explicit GdiContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed);
	explicit GdiContextBase(const RECT &drawRect, const bool isOffScreenUsed);
public:
	virtual ~GdiContextBase();

private:
	GdiContextBase(const GdiContextBase &);
	GdiContextBase & operator=(const GdiContextBase &);

protected:
	/// create a palette
	static void createPalette(HDC hDC, const int colorBitCount);
	/// delete a palette
	static void deletePalette(HDC hDC);

protected:
	/// a palette handle for the index mode graphics hardware
    static HPALETTE shPalette_;
	/// a palette usage count
    static size_t sUsedPaletteCount_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__GDI_CONTEXT_BASE__H_
