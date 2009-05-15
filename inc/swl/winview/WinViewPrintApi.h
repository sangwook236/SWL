#if !defined(__SWL_WIN_VIEW__WIN_VIEW_PRINT_API__H_ )
#define __SWL_WIN_VIEW__WIN_VIEW_PRINT_API__H_ 1


#include "swl/winview/ExportWinView.h"
#include <windows.h>


namespace swl {

struct WinViewBase;

//-----------------------------------------------------------------------------------
//  Print API using GDI in Microsoft Windows

SWL_WIN_VIEW_API bool printWinViewUsingGdi(WinViewBase &view, HDC hPrintDC);

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WIN_VIEW_PRINT_API__H_
