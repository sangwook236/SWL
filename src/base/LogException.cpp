#include "swl/Config.h"
#include "swl/base/LogException.h"
#include "swl/base/String.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	exception for log

#if defined(UNICODE) || defined(_UNICODE)
/*static*/ std::wostream * LogException::logStream_ = NULL;
#else
/*static*/ std::ostream * LogException::logStream_ = NULL;
#endif

LogException::LogException(const unsigned int level, const std::wstring &message, const std::wstring &filePath, const long lineNo, const std::wstring &methodName)
#if defined(UNICODE) || defined(_UNICODE)
: level_(level), message_(message), filePath_(filePath), lineNo_(lineNo), methodName_(methodName)
#else
: level_(level), message_(String::wcs2mbs(message)), filePath_(String::wcs2mbs(filePath)), lineNo_(lineNo), methodName_(String::wcs2mbs(methodName))
#endif
{
	 report();
}

LogException::LogException(const unsigned int level, const std::wstring &message, const std::string &filePath, const long lineNo, const std::string &methodName)
#if defined(UNICODE) || defined(_UNICODE)
: level_(level), message_(message), filePath_(String::mbs2wcs(filePath)), lineNo_(lineNo), methodName_(String::mbs2wcs(methodName))
#else
: level_(level), message_(String::wcs2mbs(message)), filePath_(filePath), lineNo_(lineNo), methodName_(methodName)
#endif
{
	 report();
}

LogException::LogException(const unsigned int level, const std::string &message, const std::string &filePath, const long lineNo, const std::string &methodName)
#if defined(UNICODE) || defined(_UNICODE)
: level_(level), message_(String::mbs2wcs(message)), filePath_(String::mbs2wcs(filePath)), lineNo_(lineNo), methodName_(String::mbs2wcs(methodName))
#else
: level_(level), message_(message), filePath_(filePath), lineNo_(lineNo), methodName_(methodName)
#endif
{
	 report();
}

LogException::~LogException()
{
}

LogException::LogException(const LogException &rhs)
: level_(rhs.level_), message_(rhs.message_), filePath_(rhs.filePath_), lineNo_(rhs.lineNo_), methodName_(rhs.methodName_)
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

void LogException::report() const
{
	if (logStream_)
	{
		if (getClassName().empty())
		{
#if defined(UNICODE) || defined(_UNICODE)
			*logStream_ << L"level: " << level_ <<
				L", message: " << message_ <<
				L", method name: " << getMethodName() <<
				L", line no: " << lineNo_ <<
				L", file name: " << getFileName() <<
				L", file path: " << filePath_ << std::endl;
#else
			*logStream_ << "level: " << level_ <<
				", message: " << message_ <<
				", method name: " << getMethodName() <<
				", line no: " << lineNo_ <<
				", file name: " << getFileName() <<
				", file path: " << filePath_ << std::endl;
#endif
		}
		else
		{
#if defined(UNICODE) || defined(_UNICODE)
			*logStream_ << L"level: " << level_ <<
				L", message: " << message_ <<
				L", class name: " << getClassName() <<
				L", method name: " << getMethodName() <<
				L", line no: " << lineNo_ <<
				L", file name: " << getFileName() <<
				L", file path: " << filePath_ << std::endl;
#else
			*logStream_ << "level: " << level_ <<
				", message: " << message_ <<
				", class name: " << getClassName() <<
				", method name: " << getMethodName() <<
				", line no: " << lineNo_ <<
				", file name: " << getFileName() <<
				", file path: " << filePath_ << std::endl;
#endif
		}
	}
}

}  // namespace swl
