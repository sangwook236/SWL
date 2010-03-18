#include "stdafx.h"
#include <iostream>

int wmain(int argc, wchar_t* argv[])
{
	std::wcout << L"press any key to exit !!!" << std::endl;
	std::wcout.flush();
	std::wcin.get();

    return 0;
}

