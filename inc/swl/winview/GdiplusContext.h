#if !defined(__SWL_WIN_VIEW__GDI_PLUS_CONTEXT__H_ )
#define __SWL_WIN_VIEW__GDI_PLUS_CONTEXT__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>


namespace Gdiplus {

class Graphics;

}  // namespace Gdiplus

namespace swl {

//-----------------------------------------------------------------------------------
//  Single-buffered Context for GDI+ in Microsoft Windows

class SWL_WIN_VIEW_API GdiplusContext: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef Gdiplus::Graphics context_type;

public:
	GdiplusContext(HWND hWnd, const bool isAutomaticallyActivated = true);
	virtual ~GdiplusContext();

private:
	GdiplusContext(const GdiplusContext &);
	GdiplusContext & operator=(const GdiplusContext &);

public:
	/// swap buffers
	/*virtual*/ bool swapBuffer()  {  return true;  }

	/// activate the context
	/*virtual*/ bool activate();
	/// de-activate the context
	/*virtual*/ bool deactivate();

	/// get the native context
	/*virtual*/ void * getNativeContext()  {  return isActivated() ? (void *)graphics_ : NULL;  }
	/*virtual*/ const void * const getNativeContext() const  {  return isActivated() ? (void *)graphics_ : NULL;  }

private:
	/// a window handle
	HWND hWnd_;
	/// a target context
	Gdiplus::Graphics *graphics_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__GDI_PLUS_CONTEXT__H_
