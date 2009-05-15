#if !defined(__SWL_WIN_VIEW__WIN_VIEW_CAPTURE_API__H_ )
#define __SWL_WIN_VIEW__WIN_VIEW_CAPTURE_API__H_ 1


#include "swl/winview/ExportWinView.h"
#include <windows.h>
#include <string>


namespace swl {

struct WinViewBase;

//-----------------------------------------------------------------------------------
//  Capture API using GDI in Microsoft Windows

#if defined(_UNICODE) || defined(UNICODE)
SWL_WIN_VIEW_API bool captureWinViewUsingGdi(const std::wstring &filePathName, WinViewBase &view, HWND hWnd);
SWL_WIN_VIEW_API bool captureWinViewUsingGdiplus(const std::wstring &filePathName, const std::wstring &fileExtName, WinViewBase &view, HWND hWnd);
#else
SWL_WIN_VIEW_API bool captureWinViewUsingGdi(const std::string &filePathName, WinViewBase &view, HWND hWnd);
SWL_WIN_VIEW_API bool captureWinViewUsingGdiplus(const std::string &filePathName, const std::string &fileExtName, WinViewBase &view, HWND hWnd);
#endif

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WIN_VIEW_CAPTURE_API__H_
