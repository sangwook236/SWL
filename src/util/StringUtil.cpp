#include "swl/util/StringUtil.h"
#include <boost/smart_ptr.hpp>
#if defined(WIN32)
#include <windows.h>
#endif


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	string utility

/*static*/ std::string StringUtil::wcs2mbs(const std::wstring &wcstr)
{
	const size_t len = wcstr.length() * 2 + 1;
	boost::scoped_array<char> mbstr(new char [len]);

#if defined(WIN32)
	const int len2 = WideCharToMultiByte(CP_ACP, 0, wcstr.c_str(), -1, mbstr.get(), (int)len, NULL, NULL);
	//assert(wcstr.length() == len2);
#else
	// method 1
	setlocale(LC_ALL, "korea");
	wcstombs(mbstr.get(), wcstr.c_str(), len);

	// method 2
	//_locale_t locale = _create_locale(LC_ALL, "korea");
	//wcstombs_l(mbstr.get(), wcstr.c_str(), len, locale);
#endif

	return std::string(mbstr.get());
}

/*static*/ std::wstring StringUtil::mbs2wcs(const std::string &mbstr)
{
	const size_t len = mbstr.length() * 2 + 1;
	boost::scoped_array<wchar_t> wcstr(new wchar_t [len]);

#if defined(WIN32)
	const int len2 = MultiByteToWideChar(CP_ACP, 0, mbstr.c_str(), -1, wcstr.get(), (int)len);
	//mbstr(wcstr.length() == len2);
#else
	// method 1
	setlocale(LC_ALL, "korea");
	mbstowcs(wcstr.get(), mbstr.c_str(), len);

	// method 2
	//_locale_t locale = _create_locale(LC_ALL, "korea");
	//mbstowcs_l(wcstr.get(), mbstr.c_str(), len, locale);
#endif

	return std::wstring(wcstr.get());
}

}  // namespace swl
