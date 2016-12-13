#include "swl/Config.h"
#include "swl/base/Log.h"
#define BOOST_LOG_DYN_LINK 1
#include <boost/log/support/date_time.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/attributes/timer.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/utility/setup/from_stream.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/common.hpp>
#include <boost/log/core.hpp>
#include <fstream>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

// Global logger declaration.
BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(swlLogger, boost::log::sources::severity_logger<>)

}  // namespace local
}  // unnamed namespace

namespace swl {

Log::StaticInitializer::StaticInitializer()
{
	// Open a settings file.
	std::ifstream settings("./data/swl_log.settings");
	if (!settings.is_open())
	{
		throw std::runtime_error("Could not open ./data/swl_log.settings file");
		return;
	}

	// Read the settings and initialize logging library.
	boost::log::init_from_stream(settings);

	// Add some attributes.
	//boost::log::add_common_attributes();
	boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
	boost::log::core::get()->add_global_attribute("RecordID", boost::log::attributes::counter<unsigned int>());
	//boost::log::core::get()->add_thread_attribute("Scope", boost::log::attributes::named_scope());
}

/*static*/ Log::StaticInitializer Log::initializer_;

/*static*/ void Log::log(const int level, const std::wstring &message)
{
	BOOST_LOG_SEV(local::swlLogger::get(), level) << message;
}

}  // namespace swl
