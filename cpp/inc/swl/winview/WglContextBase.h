#if !defined(__SWL_WIN_VIEW__WGL_CONTEXT_BASE__H_ )
#define __SWL_WIN_VIEW__WGL_CONTEXT_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#if defined(WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>


namespace swl {

//-----------------------------------------------------------------------------------
//  Base Context for OpenGL

class SWL_WIN_VIEW_API WglContextBase: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef HGLRC context_type;

protected:
	explicit WglContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed, const EContextMode contextMode);
	explicit WglContextBase(const RECT &drawRect, const bool isOffScreenUsed, const EContextMode contextMode);
public:
	virtual ~WglContextBase();

private:
	WglContextBase(const WglContextBase &);
	WglContextBase & operator=(const WglContextBase &);

public:
	/// get the native window handle
	virtual boost::any getNativeWindowHandle() = 0;
	virtual const boost::any getNativeWindowHandle() const = 0;

	/// share an OpenGL display list
	bool shareDisplayList(const WglContextBase &srcContext);

	/// get the OpenGL rendering context
	HGLRC & getRenderingContext()  {  return wglRC_;  }
	const HGLRC & getRenderingContext() const  {  return wglRC_;  }

protected:
	/// create a palette
	static void createPalette(HDC hDC, const PIXELFORMATDESCRIPTOR &pfd, const int colorBitCount);
	/// delete a palette
	static void deletePalette(HDC hDC);

protected:
	/// an OpenGL rendering context
    HGLRC wglRC_;

	// a flag to check if a color palette is used
	bool isPaletteUsed_;

	/// a palette handle for the index mode graphics hardware
    static HPALETTE shPalette_;
	/// a palette usage count
    static size_t sUsedPaletteCount_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_CONTEXT_BASE__H_
