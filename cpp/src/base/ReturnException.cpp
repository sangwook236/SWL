#include "swl/Config.h"
#include "swl/base/ReturnException.h"
#include "swl/base/String.h"
#include <ostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	exception for function's return

ReturnException::ReturnException(const unsigned int level, const std::wstring &message, const std::wstring &methodName)
#if defined(_UNICODE) || defined(UNICODE)
: level_(level), returnVal_(), message_(message), methodName_(methodName)
#else
: level_(level), returnVal_(), message_(String::wcs2mbs(message)), methodName_(String::wcs2mbs(methodName))
#endif
{
}

ReturnException::ReturnException(const unsigned int level, const std::wstring &message, const std::string &methodName)
#if defined(_UNICODE) || defined(UNICODE)
: level_(level), returnVal_(), message_(message), methodName_(String::mbs2wcs(methodName))
#else
: level_(level), returnVal_(), message_(String::wcs2mbs(message)), methodName_(methodName)
#endif
{
}

ReturnException::ReturnException(const unsigned int level, const std::string &message, const std::string &methodName)
#if defined(_UNICODE) || defined(UNICODE)
: level_(level), returnVal_(), message_(String::mbs2wcs(message)), methodName_(String::mbs2wcs(methodName))
#else
: level_(level), returnVal_(), message_(message), methodName_(methodName)
#endif
{
}

ReturnException::ReturnException(const unsigned int level, const boost::any &returnVal, const std::wstring &message, const std::wstring &methodName)
#if defined(_UNICODE) || defined(UNICODE)
: level_(level), returnVal_(returnVal), message_(message), methodName_(methodName)
#else
: level_(level), returnVal_(returnVal), message_(String::wcs2mbs(message)), methodName_(String::wcs2mbs(methodName))
#endif
{
}

ReturnException::ReturnException(const unsigned int level, const boost::any &returnVal, const std::wstring &message, const std::string &methodName)
#if defined(_UNICODE) || defined(UNICODE)
: level_(level), returnVal_(returnVal), message_(message), methodName_(String::mbs2wcs(methodName))
#else
: level_(level), returnVal_(returnVal), message_(String::wcs2mbs(message)), methodName_(methodName)
#endif
{
}

ReturnException::ReturnException(const unsigned int level, const boost::any &returnVal, const std::string &message, const std::string &methodName)
#if defined(_UNICODE) || defined(UNICODE)
: level_(level), returnVal_(returnVal), message_(String::mbs2wcs(message)), methodName_(String::mbs2wcs(methodName))
#else
: level_(level), returnVal_(returnVal), message_(message), methodName_(methodName)
#endif
{
}

ReturnException::ReturnException(const ReturnException &rhs)
: level_(rhs.level_), returnVal_(rhs.returnVal_), message_(rhs.message_), methodName_(rhs.methodName_)
{
}

ReturnException::~ReturnException() throw()
{
}

#if defined(_UNICODE) || defined(UNICODE)
std::wstring ReturnException::getClassName() const
#else
std::string ReturnException::getClassName() const
#endif
{
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring::size_type pos = methodName_.find_first_of(std::wstring(L"::"));
	return (pos == std::wstring::npos) ? std::wstring(L"") : methodName_.substr(0, pos);
#else
	const std::string::size_type pos = methodName_.find_first_of(std::string("::"));
	return (pos == std::string::npos) ? std::string("") : methodName_.substr(0, pos);
#endif
}

#if defined(_UNICODE) || defined(UNICODE)
std::wstring ReturnException::getMethodName() const
#else
std::string ReturnException::getMethodName() const
#endif
{
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring::size_type pos = methodName_.find_first_of(std::wstring(L"::"));
	return (pos == std::wstring::npos) ? methodName_ : methodName_.substr(pos + 2);
#else
	const std::string::size_type pos = methodName_.find_first_of(std::string("::"));
	return (pos == std::string::npos) ? methodName_ : methodName_.substr(pos + 2);
#endif
}

}  // namespace swl
