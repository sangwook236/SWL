#if !defined(__SWL_WIN_VIEW__WGL_VIEW_CAPTURE_API__H_ )
#define __SWL_WIN_VIEW__WGL_VIEW_CAPTURE_API__H_ 1


#include "swl/winview/ExportWinView.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <windows.h>
#endif
#include <string>


namespace swl {

struct WglViewBase;

//-----------------------------------------------------------------------------------
//  Capture API using GDI for OpenGL

#if defined(_UNICODE) || defined(UNICODE)
SWL_WIN_VIEW_API bool captureWglViewUsingGdi(const std::wstring &filePathName, WglViewBase &view, HWND hWnd);
#else
SWL_WIN_VIEW_API bool captureWglViewUsingGdi(const std::string &filePathName, WglViewBase &view, HWND hWnd);
#endif

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_CAPTURE_API__H_
