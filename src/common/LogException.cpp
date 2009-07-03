#include "swl/common/LogException.h"
#include "swl/common/StringConversion.h"


#if defined(_MSC_VER) && defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	exception for log

LogException::~LogException()
{
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getFilePath() const
#else
std::string LogException::getFilePath() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return StringConversion::mbs2wcs(filePath_);
#else
	return filePath_;
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getFileName() const
#else
std::string LogException::getFileName() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring filePath(StringConversion::mbs2wcs(filePath_));
#else
	const std::wstring &filePath = filePath_;
#endif

#if defined(WIN32)
	return filePath.substr(filePath.find_last_of(std::wstring(L"\\")) + 1);
#else
	return filePath.substr(filePath.find_last_of(std::wstring(L"/")) + 1);
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getClassName() const
#else
std::string LogException::getClassName() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring methodName(StringConversion::mbs2wcs(methodName_));
	const std::wstring::size_type pos = methodName.find_first_of(std::wstring(L"::"));
#else
	const std::string::size_type pos = methodName_.find_first_of(std::string("::"));
#endif

	return (pos == std::wstring::npos) ? std::wstring(L"") : methodName.substr(0, pos);
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring LogException::getMethodName() const
#else
std::string LogException::getMethodName() const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring methodName(StringConversion::mbs2wcs(methodName_));
	const std::wstring::size_type pos = methodName.find_first_of(std::wstring(L"::"));
#else
	const std::string::size_type pos = methodName_.find_first_of(std::string("::"));
#endif

	return (pos == std::wstring::npos) ? methodName : methodName.substr(pos + 2);
}

}  // namespace swl
