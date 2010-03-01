#if !defined(__SWL_WIN_VIEW__WIN_CLIPBOARD_API__H_ )
#define __SWL_WIN_VIEW__WIN_CLIPBOARD_API__H_ 1


#include "swl/winview/ExportWinView.h"
#include <windef.h>


namespace swl {

//-----------------------------------------------------------------------------------
//  Clipboard API using GDI in Microsoft Windows

SWL_WIN_VIEW_API bool copyToClipboardUsingGdi(HWND hWnd);

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WIN_CLIPBOARD_API__H_
