#if !defined(__SWL_WIN_UTIL__WIN_CONSOLE_WINDOW__H_)
#define __SWL_WIN_UTIL__WIN_CONSOLE_WINDOW__H_ 1


#include "swl/winutil/ExportWinUtil.h"

namespace swl {

//-----------------------------------------------------------------------------------
//

class SWL_WIN_UTIL_API WinConsoleWindow
{
private:
	WinConsoleWindow();
public:
	~WinConsoleWindow();

public:
	static WinConsoleWindow & getInstance();

	static void initialize();
	static void finalize();

	bool isValid() const  {  return isValid_;  }

private:
	bool isValid_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__WIN_CONSOLE_WINDOW__H_
