#include "swl/common/LogException.h"
#include "swl/common/StringConversion.h"


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	exception for log

LogException::LogException(const unsigned int level, const std::wstring &message, const std::wstring &filePath, const long lineNo, const std::wstring &methodName)
#if defined(UNICODE) || defined(_UNICODE)
: level_(level), message_(message), filePath_(filePath), lineNo_(lineNo), methodName_(methodName)
#else
: level_(level), message_(StringConversion::wcs2mbs(message)), filePath_(StringConversion::wcs2mbs(filePath)), lineNo_(lineNo), methodName_(StringConversion::wcs2mbs(methodName))
#endif
{}

LogException::LogException(const unsigned int level, const std::wstring &message, const std::string &filePath, const long lineNo, const std::string &methodName)
#if defined(UNICODE) || defined(_UNICODE)
: level_(level), message_(message), filePath_(StringConversion::mbs2wcs(filePath)), lineNo_(lineNo), methodName_(StringConversion::mbs2wcs(methodName))
#else
: level_(level), message_(StringConversion::wcs2mbs(message)), filePath_(filePath), lineNo_(lineNo), methodName_(methodName)
#endif
{}

LogException::LogException(const unsigned int level, const std::string &message, const std::string &filePath, const long lineNo, const std::string &methodName)
#if defined(UNICODE) || defined(_UNICODE)
: level_(level), message_(StringConversion::mbs2wcs(message)), filePath_(StringConversion::mbs2wcs(filePath)), lineNo_(lineNo), methodName_(StringConversion::mbs2wcs(methodName))
#else
: level_(level), message_(message), filePath_(filePath), lineNo_(lineNo), methodName_(methodName)
#endif
{}

LogException::~LogException()
{
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getFileName() const
#else
std::string LogException::getFileName() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
#if defined(WIN32)
	const std::wstring delim(L"\\");
#else
	const std::wstring delim(L"/");
#endif
#else
#if defined(WIN32)
	const std::string delim("\\");
#else
	const std::string delim("/");
#endif
#endif

	return filePath_.substr(filePath_.find_last_of(delim) + 1);
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getClassName() const
#else
std::string LogException::getClassName() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring::size_type pos = methodName_.find_first_of(std::wstring(L"::"));
	return (pos == std::wstring::npos) ? std::wstring(L"") : methodName_.substr(0, pos);
#else
	const std::string::size_type pos = methodName_.find_first_of(std::string("::"));
	return (pos == std::string::npos) ? std::string("") : methodName_.substr(0, pos);
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getMethodName() const
#else
std::string LogException::getMethodName() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring::size_type pos = methodName_.find_first_of(std::wstring(L"::"));
	return (pos == std::wstring::npos) ? methodName_ : methodName_.substr(pos + 2);
#else
	const std::string::size_type pos = methodName_.find_first_of(std::string("::"));
	return (pos == std::string::npos) ? methodName_ : methodName_.substr(pos + 2);
#endif
}

}  // namespace swl
