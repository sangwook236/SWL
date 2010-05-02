#include "swl/Config.h"
#include "swl/winutil/WinConsoleWindow.h"
#include <windows.h>
#include <cstdio>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//

/*static*/ boost::scoped_ptr<WinConsoleWindow> WinConsoleWindow::singleton_;

WinConsoleWindow::WinConsoleWindow()
: isValid_(false)
{
}

WinConsoleWindow::~WinConsoleWindow()
{
}

/*static*/ WinConsoleWindow & WinConsoleWindow::getInstance()
{
	if (!singleton_)
		singleton_.reset(new WinConsoleWindow());

	return *singleton_;
}

/*static*/ void WinConsoleWindow::clearInstance()
{
	singleton_.reset();
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
