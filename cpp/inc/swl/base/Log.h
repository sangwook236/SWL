#if !defined(__SWL_BASE__LOG__H_)
#define __SWL_BASE__LOG__H_ 1


//#define __SWL__DISABLE_LOG_ 1

#if !defined(__SWL__DISABLE_LOG_)

#include <log4cxx/logger.h>
#include <log4cxx/level.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/xml/domconfigurator.h>

#else

#undef LOG4CXX_LOG
#undef LOG4CXX_LOGLS

#undef LOG4CXX_TRACE
#undef LOG4CXX_DEBUG
#undef LOG4CXX_INFO
#undef LOG4CXX_WARN
#undef LOG4CXX_ERROR
#undef LOG4CXX_ASSERT
#undef LOG4CXX_FATAL

#undef LOG4CXX_L7DLOG
#undef LOG4CXX_L7DLOG1
#undef LOG4CXX_L7DLOG2
#undef LOG4CXX_L7DLOG3

#define LOG4CXX_LOG(logger, level, message)
#define LOG4CXX_LOGLS(logger, level, message)

#define LOG4CXX_DEBUG(logger, message)
#define LOG4CXX_TRACE(logger, message)
#define LOG4CXX_INFO(logger, message)
#define LOG4CXX_WARN(logger, message)
#define LOG4CXX_ERROR(logger, message)
#define LOG4CXX_ASSERT(logger, condition, message)
#define LOG4CXX_FATAL(logger, message)

#define LOG4CXX_L7DLOG(logger, level, key)
#define LOG4CXX_L7DLOG1(logger, level, key, p1)
#define LOG4CXX_L7DLOG2(logger, level, key, p1, p2)
#define LOG4CXX_L7DLOG3(logger, level, key, p1, p2, p3)

#endif


#endif  // __SWL_BASE__LOG__H_
