#if !defined(__SWL_BASE__LOG__H_)
#define __SWL_BASE__LOG__H_ 1


#include "swl/base/ExportBase.h"
#include <string>
#include <climits>


//#define __SWL__DISABLE_LOG_ 1

#if !defined(__SWL__DISABLE_LOG_)

namespace swl {

//-----------------------------------------------------------------------------------
// Log.

struct SWL_BASE_API Log
{
public:
	enum LogSeverityLevel
	{
		OFF = INT_MAX,
		FATAL = 50000,
		ERROR = 40000,
		WARN = 30000,
		INFO = 20000,
		DEBUG = 10000,
		TRACE = 5000,
		ALL = INT_MIN
	};

public:
	static void log(const int level, const std::wstring &message);

private:
	class Initializer
	{
	public:
		Initializer();
	};

private:
	static Initializer initializer_;
};

}  // namespace swl

#define SWL_LOG(level, message) swl::Log::log(level, message)

#define SWL_LOG_ALL(message) swl::Log::log(Log::ALL, message)
#define SWL_LOG_TRACE(message) swl::Log::log(Log::TRACE, message)
#define SWL_LOG_DEBUG(message) swl::Log::log(Log::DEBUG, message)
#define SWL_LOG_INFO(message) swl::Log::log(Log::INFO, message)
#define SWL_LOG_WARN(message) swl::Log::log(Log::WARN, message)
#define SWL_LOG_ERROR(message) swl::Log::log(Log::ERROR, message)
#define SWL_LOG_FATAL(message) swl::Log::log(Log::FATAL, message)

#else

#define SWL_LOG(level, message)

#define SWL_LOG_ALL(message)
#define SWL_LOG_TRACE(message)
#define SWL_LOG_DEBUG(message)
#define SWL_LOG_INFO(message)
#define SWL_LOG_WARN(message)
#define SWL_LOG_ERROR(message)
#define SWL_LOG_FATAL(message)

#endif


#endif  // __SWL_BASE__LOG__H_
