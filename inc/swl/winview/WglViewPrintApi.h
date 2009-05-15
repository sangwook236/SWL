#if !defined(__SWL_WIN_VIEW__WGL_VIEW_PRINT_API__H_ )
#define __SWL_WIN_VIEW__WGL_VIEW_PRINT_API__H_ 1


#include "swl/winview/ExportWinView.h"
#include <windows.h>


namespace swl {

struct WglViewBase;

//-----------------------------------------------------------------------------------
//  Print API using GDI for OpenGL

SWL_WIN_VIEW_API bool printWglViewUsingGdi(WglViewBase &view, HDC hPrintDC);

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_PRINT_API__H_
