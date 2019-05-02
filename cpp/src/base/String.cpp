#include "swl/Config.h"
#include "swl/base/String.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <windows.h>
#endif
#include <memory>
#include <clocale>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// String.

/*static*/ std::string String::wcs2mbs(const std::wstring &wcstr)
{
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const int len = WideCharToMultiByte(CP_ACP, 0, wcstr.c_str(), -1, NULL, 0, NULL, NULL);
	std::unique_ptr<char []> mbstr(new char [len]);
	const int len2 = WideCharToMultiByte(CP_ACP, 0, wcstr.c_str(), -1, mbstr.get(), (int)len, NULL, NULL);
	//assert(wcstr.length() == len2);
#else
	const size_t len = wcstr.length() * 2 + 1;
	std::unique_ptr<char []> mbstr(new char [len]);

	// Method 1.
	std::setlocale(LC_ALL, "korea");
	std::wcstombs(mbstr.get(), wcstr.c_str(), len);

	// Method 2.
	//_locale_t locale = _create_locale(LC_ALL, "korea");
	//std::wcstombs_l(mbstr.get(), wcstr.c_str(), len, locale);
#endif

	return std::string(mbstr.get());
}

/*static*/ std::wstring String::mbs2wcs(const std::string &mbstr)
{
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const int len = MultiByteToWideChar(CP_ACP, 0, mbstr.c_str(), -1, NULL, 0);
	std::unique_ptr<wchar_t []> wcstr(new wchar_t [len]);
	const int len2 = MultiByteToWideChar(CP_ACP, 0, mbstr.c_str(), -1, wcstr.get(), (int)len);
	//assert(wcstr.length() == len2);
#else
	const size_t len = mbstr.length() * 2 + 1;
	std::unique_ptr<wchar_t []> wcstr(new wchar_t [len]);

	// Method 1.
	std::setlocale(LC_ALL, "korea");
	std::mbstowcs(wcstr.get(), mbstr.c_str(), len);

	// Method 2.
	//_locale_t locale = _create_locale(LC_ALL, "korea");
	//std::mbstowcs_l(wcstr.get(), mbstr.c_str(), len, locale);
#endif

	return std::wstring(wcstr.get());
}

/*static*/ std::string String::wcs2utf8(const std::wstring &wcstr)
{
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const int len = WideCharToMultiByte(CP_UTF8, 0, wcstr.c_str(), -1, NULL, 0, NULL, NULL);
	std::unique_ptr<char []> utf8(new char [len]);
	const int len2 = WideCharToMultiByte(CP_UTF8, 0, wcstr.c_str(), -1, utf8.get(), (int)len, NULL, NULL);
	//assert(wcstr.length() == len2);

	return std::string(utf8.get());
#else
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	return conv.to_bytes(wcstr);
#endif
}

/*static*/ std::wstring String::utf82wcs(const std::string &utf8)
{
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, NULL, 0);
	std::unique_ptr<wchar_t []> wcstr(new wchar_t [len]);
	const int len2 = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wcstr.get(), (int)len);
	//assert(wcstr.length() == len2);

	return std::wstring(wcstr.get());
#else
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	return conv.from_bytes(utf8);
#endif
}

//-----------------------------------------------------------------------------------
// Byte data converter.

/*static*/ unsigned char String::dec2ascii(const unsigned char dec)
{
	if (0 <= dec && dec <= 9)
		return dec + '0';
	else return -1;
}

/*static*/ unsigned char String::ascii2dec(const unsigned char ascii)
{
	if ('0' <= ascii && ascii <= '9')
		return ascii - '0';
	else return -1;
}

/*static*/ unsigned char String::hex2ascii(const unsigned char hex, const bool isUpperCase /*= true*/)
{
	if (0x0 <= hex && hex <= 0x9)
		return hex + '0';
	//else if (0xa <= hex && hex <= 0xf)
	else if (0xA <= hex && hex <= 0xF)
		return hex - 0xA + (isUpperCase ? 'A' : 'a');
	else return -1;
}

/*static*/ unsigned char String::ascii2hex(const unsigned char ascii)
{
	if ('0' <= ascii && ascii <= '9')
		return ascii - '0';
	else if ('a' <= ascii && ascii <= 'f')
		return ascii - 'a' + 10;
	else if ('A' <= ascii && ascii <= 'F')
		return ascii - 'A' + 10;
	else return -1;
}

}  // namespace swl
