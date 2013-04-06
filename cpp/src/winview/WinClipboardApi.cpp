#include "swl/Config.h"
#include <windows.h>
#include "swl/winview/WinClipboardApi.h"
#include <wingdi.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

bool copyToClipboardUsingGdi(HWND hWnd)
{
	HDC hDC = GetDC(hWnd);

	HDC memDC = CreateCompatibleDC(hDC);

	RECT rect;
	GetWindowRect(hWnd, &rect);
	const long width = rect.right - rect.left, height = rect.bottom - rect.top; 

	HBITMAP bitmap = CreateCompatibleBitmap(hDC, width, height);

	HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, bitmap);
	BitBlt(memDC, 0, 0, width, height, hDC, 0, 0, SRCCOPY);

	// clipboard
#if 1
	if (OpenClipboard(hWnd))
	{
		EmptyClipboard();
		SetClipboardData(CF_BITMAP, bitmap);
		CloseClipboard();
	}
#else
	System::Drawing::Image ^image = System::Drawing::Image::FromHbitmap(System::IntPtr(bitmap));
	System::Windows::Forms::Clipboard::SetImage(image);
#endif

	SelectObject(memDC, oldBitmap);
	DeleteObject(bitmap);

	ReleaseDC(hWnd, hDC);

	return true;
}

}  // namespace swl
