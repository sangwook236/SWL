#if !defined(__SWL_WIN_VIEW__GDI_CONTEXT__H_ )
#define __SWL_WIN_VIEW__GDI_CONTEXT__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewContext.h"
#include <windows.h>


namespace swl {

//-----------------------------------------------------------------------------------
//  Single-buffered Context for GDI in Microsoft Windows

class SWL_WIN_VIEW_API GdiContext: public ViewContext
{
public:
	typedef ViewContext base_type;
	typedef HDC context_type;

public:
	GdiContext(HWND hWnd, const bool isAutomaticallyActivated = true);
	virtual ~GdiContext();

private:
	GdiContext(const GdiContext &);
	GdiContext & operator=(const GdiContext &);

public:
	/// swap buffers
	/*virtual*/ bool swapBuffer()  {  return true;  }

	/// activate the context
	/*virtual*/ bool activate();
	/// de-activate the context
	/*virtual*/ bool deactivate();

	/// get the native context
	/*virtual*/ boost::any getNativeContext()  {  return isActivated() ? boost::any(&hDC_) : boost::any();  }
	/*virtual*/ const boost::any getNativeContext() const  {  return isActivated() ? boost::any(&hDC_) : boost::any();  }

private:
	/// a window handle
	const HWND hWnd_;
	/// a target context
	HDC hDC_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__GDI_CONTEXT__H_
