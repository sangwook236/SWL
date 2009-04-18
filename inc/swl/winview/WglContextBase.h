#if !defined(__SWL_WIN_VIEW__WGL_CONTEXT_BASE__H_ )
#define __SWL_WIN_VIEW__WGL_CONTEXT_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>
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
	explicit WglContextBase(const Region2<int>& drawRegion, const bool isOffScreenUsed);
	explicit WglContextBase(const RECT& drawRect, const bool isOffScreenUsed);
public:
	virtual ~WglContextBase();

private:
	WglContextBase(const WglContextBase &);
	WglContextBase & operator=(const WglContextBase &);

public:
	/// create an OpenGL display list
	void createDisplayList(const HDC hDC);

	/// get the OpenGL shared display list
	static HGLRC& getSharedRC()  {  return sSharedRC_;  }
	/// reset the OpenGL shared display list
	static void resetSharedRC()  {  sSharedRC_ = 0L;  }

	/// get the native context
	/*virtual*/ void * getNativeContext()  {  return isActivated() ? (void *)&wglRC_ : NULL;  }
	/*virtual*/ const void * const getNativeContext() const  {  return isActivated() ? (void *)&wglRC_ : NULL;  }

protected:
	/// share an OpenGL display list
	static bool shareDisplayList(HGLRC &glRC);
	// use a palette k
	static void usePalette(const HDC hDC, const PIXELFORMATDESCRIPTOR &pfd);

private:
	/// re-create an OpenGL display list
	virtual bool doRecreateDisplayList() = 0;

protected:
	/// an OpenGL rendering context
    HGLRC wglRC_;

private:
	/// a sharing OpenGL rendering context
    static HGLRC sSharedRC_;

	/// a palette handle for the index mode graphics hardware
    static HPALETTE shPalette_;
	/// a palette usage count
    static size_t sUsedPaletteCount_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_CONTEXT_BASE__H_
