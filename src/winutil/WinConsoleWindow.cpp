#include "swl/Config.h"
#include "swl/winutil/WinConsoleWindow.h"
#include <windows.h>
#include <cstdio>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//

WinConsoleWindow::WinConsoleWindow()
: isValid_(false)
{
}

WinConsoleWindow::~WinConsoleWindow()
{
}

/*static*/ WinConsoleWindow & WinConsoleWindow::getInstance()
{
	static WinConsoleWindow console;
	return console;
}

/*static*/ void WinConsoleWindow::initialize()
{
	if (AllocConsole())
	{
		freopen("CONOUT$", "w", stdout);
		freopen("CONIN$", "r", stdin);

		//
#if defined(_UNICODE) || defined(UNICODE)
		SetConsoleTitle(L"console window");
#else
		SetConsoleTitle("console window");
#endif

		WinConsoleWindow::getInstance().isValid_ = true;
	}
}

/*static*/ void WinConsoleWindow::finalize()
{
	if (WinConsoleWindow::getInstance().isValid_)
		FreeConsole();

	WinConsoleWindow::getInstance().isValid_ = false;
}

}  // namespace swl
